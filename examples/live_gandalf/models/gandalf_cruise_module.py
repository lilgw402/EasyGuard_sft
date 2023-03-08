from abc import abstractmethod
import ast
import torch
import numpy as np
import torch.optim.lr_scheduler as lrs
from collections import defaultdict
from cruise import CruiseModule
from cruise.utilities.distributed import DIST_ENV
from cruise.utilities.rank_zero import rank_zero_warn, rank_zero_info
from training.metrics.simple_metrics import AUC, MAE, MSE, RMSE, MPR
from torch.nn import functional as F
from utils.util import merge_into_target_dict
from models.modules.running_metrics.general_cls_metric import GeneralClsMetric

class GandalfCruiseModule(CruiseModule):
    def __init__(self):
        super().__init__()
        self._eval_output_names =  []
        self._extra_metrics = []
        self._summary_interval = 50
        self._gather_val_loss = False
        # self._eval_output_names =  ["acc", "binary_prec", "binary_f1", "binary_recall", "input_neg_ratio", "input_pos_ratio", "output_neg_ratio", "output_pos_ratio"]
        self._parse_eval_output_advanced_metrics()  # check if metrics like AUC/MPR will need to be calculated
    
    def forward(self, batch, batch_idx):
        pass

    def training_step(self, batch, batch_idx):
        inputs = self.pre_process_inputs(batch)
        targets = self.pre_process_targets(batch)

        train_loss_dict, train_output_dict = self.forward(*(inputs + targets))

        # self.track_logging_info(train_loss_dict,{},prefix='loss',is_loss_item=True)
        # self.track_logging_info(train_output_dict,{},prefix='eval',is_loss_item=False)
        for loss_name, loss_val_list in train_loss_dict.items():
            train_loss_dict[loss_name] = loss_val_list.mean()
        train_loss_dict.update(train_output_dict)
        return train_loss_dict

    def validation_step(self, batch, batch_index):
        inputs = self.pre_process_inputs(batch)
        targets = self.pre_process_targets(batch)
        val_loss_dict, val_output_dict = self.forward(*(inputs + targets))
        reduced_output = defaultdict(list)
        for output_name, output_val in val_loss_dict.items():
            reduced_output[output_name] = output_val.mean().item()
        self.track_logging_info({},val_output_dict,prefix='eval',hidden_keys=None,is_loss_item=False)
        # reduced_output['output'] = val_output_dict['output']
        # reduced_output['label'] = targets
        # reduced_output.update(val_output_dict)
        print('reduced_output',reduced_output.keys())
        # reduced_output = self._add_stage_prefix(reduced_output,prefix='eval',is_loss_item=False)
        return reduced_output

    def validation_epoch_end(self, outputs) -> None:
        print('outputs',len(outputs),outputs[0],outputs[0].keys())
        reduced_test_loss = defaultdict(list)
        reduced_test_output = defaultdict(list)
        reduced_all_gather_output = defaultdict(list)
        extra_reduced_outputs = {}  # add global metrics to outputs
        # preds, targets = [], []
        for output in outputs:
            for out_key, out_value in output.items():
                if out_key in self._all_gather_output_names:
                    reduced_all_gather_output[out_key].append(out_value)
                elif out_key not in self._eval_output_names:
                    reduced_test_loss[out_key].append(out_value)
                else:
                    reduced_test_output[out_key].append(out_value)
                # preds.append(output['output'])
                # targets.append(output['label'])
        # preds = torch.stack()
        if self._extra_metrics:
            # need all gather results and calculate metrics
            gathered_reduced_output = DIST_ENV.all_gather_object(reduced_all_gather_output)
            # [{}, {}, {}, {}] -> {}, flat [list of dict of list] to global [dict of list]
            flat_gathered_reduced_output = defaultdict(list)
            for ro in gathered_reduced_output:
                for k, v in ro.items():
                    flat_gathered_reduced_output[k] += v

            # concat numppy arrays
            final_result = {}
            for k, v in flat_gathered_reduced_output.items():
                if isinstance(v[0], np.ndarray):
                    final_result[k] = np.concatenate(v, axis=0)
                elif isinstance(v[0], (list, tuple)):
                    final_result[k] = [vvv for vv in v for vvv in vv]
                else:
                    raise TypeError(f"Unexpected output/label type: {type(v[0])} of {k}")

            for extra_metric in self._extra_metrics:
                assert all(m_key in extra_metric for m_key in ('op', 'name', 'type', 'score_key', 'label_key'))
                op = extra_metric['op']
                score_key = extra_metric['score_key']
                label_key = extra_metric['label_key']
                scores = final_result.get(score_key, None)
                labels = final_result.get(label_key, None)
                if scores is None or labels is None:
                    err_msg = f"{extra_metric['name']} requires {score_key} and {label_key} in output dict."
                    err_msg += f" Got {final_result.keys()}, skip."
                    rank_zero_warn(err_msg)
                    result = {}
                else:
                    res = []
                    for score, label in zip(scores, labels.flatten()):
                        res.append({score_key: score, label_key: label})
                    probs, labels = op.collect_pre_labels(res)
                    if DIST_ENV.rank == 0:
                        result = op.cal_metric(probs, labels)
                    else:
                        result = {}

                if result:
                    rank_zero_info(f"Result of {extra_metric['name']}: \n{result}")
                    # save to callback metrics
                    op_type = extra_metric['type']
                    for k, v in result.items():
                        if op_type in k:
                            new_name = f'{op_type}-{score_key}'
                            self.log(new_name, v, console=True)
                            extra_reduced_outputs[new_name] = v
                            # self._eval_output_names.add(new_name)
                            break
                        elif 'recall' in k or 'precision' in k:
                            # special case MPR
                            new_name = f'{op_type}-{k}'.replace('.', '_')
                            self.log(new_name, v, console=True)
                            extra_reduced_outputs[new_name] = v
                            # self._eval_output_names.add(new_name)

        else:
            # no need to all gather and calculate extra metrics
            reduced_test_output.update(reduced_all_gather_output)
        reduced_test_loss = {k: np.mean(vl) for k, vl in reduced_test_loss.items()}
        reduced_test_output = {k: np.mean(vl) for k, vl in reduced_test_output.items()}
        print('reduced_all_gather_output',reduced_all_gather_output.keys())
        print('reduced_test_loss',reduced_test_loss.keys())
        print('reduced_test_output',reduced_test_output.keys())
        # self._tk_log_dict("eval/custom_metrics", extra_reduced_outputs)
        self.track_logging_info({},reduced_test_loss,prefix='val',is_loss_item=True)
        # self.track_logging_info({},reduced_test_output,prefix='eval',is_loss_item=False)
        
    def on_train_epoch_start(self):
        self._tk_log_dict("monitors/epoch_num", {"epoch": self.trainer.current_epoch})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.00001,eps=1e-3
        )
        return {"optimizer": optimizer}


    def track_logging_info(self,training_info_dict,test_info_dict,prefix,hidden_keys:set = None,is_loss_item:bool=True):
        if training_info_dict:
            for okey, oval in training_info_dict.items():
                if hidden_keys and okey in hidden_keys:
                    continue
                sub_dict = dict()
                merge_into_target_dict(sub_dict, {okey: oval}, prefix, is_loss_item=is_loss_item)
                if okey in test_info_dict:
                    merge_into_target_dict(sub_dict, {okey: test_info_dict[okey]}, "test")
                # self._tk_log_dict(f"{prefix}/{okey}",sub_dict)
                self._tk_log_dict('',sub_dict)
        else:
            try:
                if self._gather_val_loss:
                    all_test_info_dict = DIST_ENV.all_gather_object(test_info_dict)
                    tmp = {k: np.mean([x.get(k, test_info_dict[k]) for x in all_test_info_dict])
                           for k in test_info_dict.keys()}
                    test_info_dict = tmp
            except Exception as ex:
                raise RuntimeError(f"Failed to sync and reduce val outputs from all ranks: {ex}")
            for okey, oval in test_info_dict.items():
                if hidden_keys and okey in hidden_keys:
                    continue
                sub_dict = dict()
                merge_into_target_dict(sub_dict, {okey: oval}, prefix, is_loss_item=is_loss_item)
                # self._tk_log_dict(f"{prefix}/{okey}", sub_dict)
                self._tk_log_dict('',sub_dict)


    def _tk_log_dict(self, tag, sub_dict):
        for k, v in sub_dict.items():
            if isinstance(v, dict):
                self._tk_log_dict(tag + '.' + k, v)
            name = tag + '.' + k if tag else k
            self.log_dict({name: v},tb=False,tracking=True)

    def _parse_eval_output_advanced_metrics(self):
        self._extra_metrics = []
        self._all_gather_output_names = set()
        valid_metrics = {
            'AUC': AUC,
            'MAE': MAE,
            'MSE': MSE,
            'RMSE': RMSE,
            'MPR': MPR
        }

        for eval_out_name in self._eval_output_names:
            assert isinstance(eval_out_name, str), f"Expected `eval_output_names` to be list of str, given {eval_out_name}"
            for m in valid_metrics:
                if eval_out_name.startswith(m + '-'):
                    parts = eval_out_name.split('-')[1:]
                    assert len(parts) >= 2, f"Invalid eval_output_name={eval_out_name} for metric type {m}"
                    score_key = parts[0]
                    label_key = parts[1]
                    # parse kwargs, such as 'AUC-out0-label0-score_idx=1-pr_scope=[0,1,200]'
                    kwargs = {}
                    for kwarg_str in parts[2:]:
                        assert '=' in kwarg_str, f"Expected arg_name=arg_value in kwargs section, given {kwarg_str}"
                        new_kwarg = kwarg_str.split('=')
                        assert len(new_kwarg) == 2, f"Only one = can be used in kwargs section, given {kwarg_str}"
                        kwarg_key = new_kwarg[0]
                        kwarg_value = ast.literal_eval(new_kwarg[1])
                        kwargs[kwarg_key] = kwarg_value
                    metric_op = valid_metrics[m](score_key=score_key, label_key=label_key, **kwargs)
                    assert hasattr(metric_op, 'cal_metric')
                    self._extra_metrics.append({'type': m, 'name': eval_out_name, 'op': metric_op, 'score_key': score_key, 'label_key': label_key})
                    self._all_gather_output_names.add(score_key)
                    self._all_gather_output_names.add(label_key)
                    break
                else:
                    # no-match, can be a normal output name, so just let go
                    pass
        print(self._all_gather_output_names)

    @staticmethod
    def _pre_process(batched_feature_data_items):
        new_inputs = []
        for input in batched_feature_data_items:
            if isinstance(input, tuple) or isinstance(input, list):
                new_input = list(input)
            else:
                new_input = input
            new_inputs.append(new_input)
        return new_inputs

    def _post_process_output(self, output):
        # 业务逻辑，对模型输出做处理
        return self._sigmoid(output)

    def pre_process_inputs(self, batched_feature_data):
        batched_feature_data_items = [value for value in batched_feature_data.values()]
        return self._pre_process(batched_feature_data_items)

    def pre_process_targets(self, batched_feature_data):
        batched_feature_data_items = [batched_feature_data["label"]]
        return self._pre_process(batched_feature_data_items)

