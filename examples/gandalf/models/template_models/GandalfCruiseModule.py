import numpy as np
from collections import defaultdict
from cruise.utilities.distributed import DIST_ENV
from cruise.utilities.rank_zero import rank_zero_warn, rank_zero_info
from .TemplateCruiseModule import TemplateCruiseModule
from utils.util import merge_into_target_dict

class GandalfCruiseModule(TemplateCruiseModule):
    def __init__(self,kwargs):
        super(GandalfCruiseModule,self).__init__(kwargs)
    
    def forward(self, batch, batch_idx):
        pass

    def training_step(self, batch, batch_idx):
        inputs = self.pre_process_inputs(batch)
        targets = self.pre_process_targets(batch)
        train_loss_dict, train_output_dict = self.forward(*(inputs + targets))
        for loss_name, loss_val_list in train_loss_dict.items():
            train_loss_dict[loss_name] = loss_val_list.mean()
        if self._output_name in train_output_dict:
            merge_into_target_dict(train_output_dict,{self._output_name:train_output_dict[self._output_name]})
        # print('train_loss_dict',train_loss_dict.keys(),train_loss_dict['loss'],type(train_loss_dict['loss']))
        # print('train_output_dict',train_output_dict.keys())
        # 'output':[batch,1], 'loss':[],'acc', 'binary_prec', 'binary_recall', 'binary_f1', 'input_neg_ratio', 'input_pos_ratio', 'output_neg_ratio', 'output_pos_ratio', 'binary_auc']
        # print(train_output_dict['output'].shape,train_output_dict['loss'].shape)
        # self.track_logging_info(train_loss_dict,{},prefix='loss',is_loss_item=True)
        # self.track_logging_info(train_output_dict,{},prefix='train',hidden_keys={'loss'},is_loss_item=False)
        train_loss_dict.update(train_output_dict)
        return train_loss_dict

    def validation_step(self, batch, batch_index):
        inputs = self.pre_process_inputs(batch)
        targets = self.pre_process_targets(batch)
        val_loss_dict, val_output_dict = self.forward(*(inputs + targets))
        reduced_output = defaultdict(list)
        for output_name, output_val in val_loss_dict.items():
            reduced_output[output_name] = output_val.mean().item()
        if self._extra_metrics:
            for output_name, output_val in val_output_dict.items():
                if output_name not in self._all_gather_output_names:
                    continue
                reduced_output[output_name] = output_val.cpu().numpy()
        if 'label' in self._all_gather_output_names:
            reduced_output['label'] = targets[0].cpu().numpy()
        return reduced_output

    def validation_epoch_end(self, outputs) -> None:
        reduced_test_loss = defaultdict(list)
        reduced_test_output = defaultdict(list)
        reduced_all_gather_output = defaultdict(list)
        extra_reduced_outputs = {}  # add global metrics to outputs
        for output in outputs:
            for out_key, out_value in output.items():
                if out_key in self._all_gather_output_names:
                    reduced_all_gather_output[out_key].append(out_value)
                elif out_key in self._eval_output_names:
                    reduced_test_loss[out_key].append(out_value)
                else:
                    reduced_test_output[out_key].append(out_value)
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
                    # save to callback metrics
                    rank_zero_info(f"Result of {extra_metric['name']}: \n{result}")
                    op_type = extra_metric['type']
                    for k, v in result.items():
                        new_name = f'{op_type}.{k}'
                        extra_reduced_outputs[new_name] = v
        else:
            # no need to all gather and calculate extra metrics
            reduced_test_output.update(reduced_all_gather_output)
        reduced_test_loss = {k: np.mean(vl) for k, vl in reduced_test_loss.items()}
        reduced_test_output = {k: np.mean(vl) for k, vl in reduced_test_output.items()}
        self._tk_log_dict("", extra_reduced_outputs)

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

if __name__ == '__main__':
    module = GandalfCruiseModule()