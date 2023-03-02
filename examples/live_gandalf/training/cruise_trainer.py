import os
import time
import ast
import torch
import numpy as np
import warnings
from collections import defaultdict
from cruise import CruiseModule, CruiseTrainer
from cruise.utilities.rank_zero import rank_zero_warn, rank_zero_info
from cruise.trainer.logger.tensorboard import DummySummaryWriter
from cruise.utilities.distributed import DIST_ENV
from cruise.trainer.logger import TensorBoardLogger
from cruise.trainer.callback import EarlyStopping
from .optimizers import build_optimizer
from .lr_schedulers import build_lr_scheduler
from utils.util import push_files, check_hdfs_exist
from training.misc.checkpoint_saver import LAST_CHECKPOINT
from training.metrics.simple_metrics import AUC, MAE, MSE, RMSE, MPR
from examples.live_gandalf.builder import TRAINERS


@TRAINERS.register_module("CruiseSingleTask")
class CruiseTaskTrainer(object):
    def __init__(
            self,
            model,
            local_output_dir,
            hdfs_output_dir,
            train_data_loader,
            test_data_loader,
            train_kwargs,
    ):
        # wrap the vanila model into CruiseModule
        self.orig_kwargs = train_kwargs
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.summary_interval = train_kwargs.summary_interval
        self.auto_resume = train_kwargs.auto_resume

        # construct callback list
        callback_list = []
        # checkpoint saver
        checkpoint_kwargs = train_kwargs.get('checkpoint_saver', {})
        hdfs_ckpt_path = os.path.join(hdfs_output_dir, 'checkpoints')
        metric_keys = checkpoint_kwargs.get('metric_key', ['loss'])
        if not isinstance(metric_keys, list): 
            metric_keys = [metric_keys]
        metric_directions = checkpoint_kwargs.get('metric_direction', ['desc'])
        if not isinstance(metric_directions, list): 
            metric_directions = [metric_directions]
        # epoch ckpt will be saved by Wrapper after each train, checkpoint callback only takes care of best ckpt
        for idx in range(len(metric_keys)):
            checkpoint_callback = MaxwellModelCheckpoint(
                monitor=metric_keys[idx],
                mode='min' if metric_directions[idx] == 'desc' else 'max',
                save_top_k=checkpoint_kwargs.get('save_checkpoint_top_k', 2),
                save_last=True,
            )
            if self.auto_resume and check_hdfs_exist(os.path.join(hdfs_ckpt_path, LAST_CHECKPOINT)):
                pass
            else:
                self.auto_resume = False
                checkpoint_callback.init_from_maxwell_ckpt(model, 
                                                        local_output_dir=local_output_dir,
                                                        hdfs_output_dir=hdfs_output_dir,
                                                        resume_checkpoint=train_kwargs.resume_checkpoint,
                                                        partial_pretrain_prefix_changes=train_kwargs.get(
                                                            "partial_pretrain_prefix_changes", [])
                                                        )
            callback_list.append(checkpoint_callback)
        # early_stopper
        early_stop_kwargs = train_kwargs.get('early_stopper', {})
        if early_stop_kwargs:
            assert len(metric_keys) == 1 and len(metric_directions) == 1, "only support one metric for earlystopping"
            early_stopper = EarlyStopping(
                monitor=metric_keys[0],
                # ignore changes less than min_delta
                min_delta=early_stop_kwargs.get('min_delta', 0.001),
                # if not going better in 10 checks, stop early
                patience=early_stop_kwargs.get('patience', 10),
                # print the reasons in detail
                verbose=early_stop_kwargs.get('verbose', True),
                # min means smaller better
                mode='min' if metric_directions[0] == 'desc' else 'max',
                # stop as soon as the loss < 0.5
                stopping_threshold=early_stop_kwargs.get('stopping_threshold', None),
                # once loss hit 0.6 but get larger next, stop
                divergence_threshold=early_stop_kwargs.get('divergence_threshold', None),
                # usually check on validation end, not train end
                check_on_train_epoch_end=early_stop_kwargs.get('check_on_train_epoch_end', False),
            )
        else:
            early_stopper = None

        self.model = SingleTaskModule(model, train_kwargs)

        log_path = './events_log'
        if DIST_ENV.rank == 0:
            if os.path.exists(log_path):
                rank_zero_info(f"Clean up tensorboard log folder:{log_path}")
                os.system(f"rm -rf {log_path}/*")
            rank_zero_info("Tensorboard_log_path: {}".format(log_path))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cruise_loggers = ['console', TensorBoardLogger(
                save_dir=log_path, hdfs_dir=None, name='', flush_logs_every_n_steps=max(1, 100 // self.summary_interval),
                version='', ignore_keys=["__elapsed__"])]

        # for backward compatibility, use attribute rather than init arguments
        if hasattr(cruise_loggers[-1], '_allow_keys'):
            cruise_loggers[-1]._allow_keys = ['training/grad_norm']
        try:
            from cruise.trainer.logger.tracking import TrackingLogger
            import re
            # add tracking logger
            if train_kwargs.get('tracking_project_name', ''):
                tracking_project_name = train_kwargs['tracking_project_name']
                if "/" in tracking_project_name:
                    project, name = tracking_project_name.rsplit("/", 1)
                    project = project.replace('/', '_')
                    # remove special chars
                    name = "_".join(re.findall(r'[a-zA-Z0-9\u4E00-\u9FA5-_./@]{1,128}',  name))
                else:
                    project = tracking_project_name
                    name = ""
                cruise_loggers.append(
                    TrackingLogger(
                        project=project,
                        name=name,
                        config={'trainer': train_kwargs},
                        version='', ignore_keys=["__elapsed__"],
                        allow_keys=['training/grad_norm']))
                rank_zero_info("Tracking enabled with name: {}".format(tracking_project_name))
        except ImportError:
            rank_zero_info("Tracking not enabled")

        if early_stopper is not None and isinstance(early_stopper, EarlyStopping):
            callback_list.append(early_stopper)

        # disable warnings
        try:
            from cruise.options import CommonTrainerOption
            CommonTrainerOption.ENABLE_DUP_METRIC_WARNING = 0
        except Exception:
            pass

        max_steps = train_kwargs.get("max_total_iter", -1)
        if max_steps == -1:
            # default max_total_iter equals to epochs * train_max_iteration if not set
            if train_kwargs.get("epochs", 1) > 0 and train_kwargs.get("train_max_iteration", -1) > 0:
                max_steps = train_kwargs.get("epochs", 1) * train_kwargs.get("train_max_iteration", -1)
            else:
                # unable to set the max_total_iter, set to unlimited
                max_steps = -1

        self._cruise_trainer = CruiseTrainer(
            default_root_dir=local_output_dir,
            default_hdfs_dir=hdfs_output_dir,
            logger=cruise_loggers,
            log_every_n_steps=min(int(train_kwargs.get('output_iteration', 50)), 100),
            benchmark=True,
            enable_speedmonitor=True,
            enable_versions=False,
            detect_anomaly=train_kwargs.detect_anomaly,
            deterministic=False,
            accelerator='gpu',
            precision=16 if train_kwargs.enable_amp else 32,
            max_epochs=train_kwargs.get("epochs", 1),
            max_steps=max_steps,
            limit_train_batches=train_kwargs.get("train_max_iteration", -1),
            limit_val_batches=train_kwargs.get("test_max_iteration", -1),
            limit_test_batches=None,
            sync_batchnorm=train_kwargs.use_sync_bn,
            val_check_interval=int(train_kwargs.get('output_iteration', 50)),
            accumulate_grad_batches=train_kwargs.get('accum_grad_steps', None),
            gradient_clip_val=train_kwargs.clip_grad_norm,
            seed=None,  # maxwell main workflow set seed already
            summarize_model_depth=3,
            resume_ckpt_path=hdfs_ckpt_path if self.auto_resume else None,
            resume_loader_state=train_kwargs.get('resume_dataloader', False) or self.auto_resume,
            # resume from checkpoints, bin folder is for compatibility
            callbacks=callback_list,
            sync_fit_metrics=None if not train_kwargs.gather_val_loss else 'val',
            find_unused_parameters=train_kwargs.find_unused_parameters,
            enable_qat=train_kwargs.qat_flag,
            qat_kwargs = {"config_file": train_kwargs.qat_yaml}
        )
        self.checkpoint_callback = callback_list

    def train(self):
        self._cruise_trainer.fit(self.model, self.train_data_loader, self.test_data_loader)
        rank_zero_info(f"Finished training, best checkpoints: {self.checkpoint_callback[0].best_k_models}")


class SingleTaskModule(CruiseModule):
    """The specific logic wrappers for maxwell using CruiseModule/CruiseTrainer

    TODO(zhangzhi.joshua):
    - resume_checkpoint: check this behavior
    - partial_pretrain_prefix_changes: implemented in CruiseModule but need verification
    - checkpoint_timestamp: still needed?


    Args:
        model : nn.Module
            the maxwell model created using default configs
        trainer_kwargs: dict
            the trainer kwargs from maxwell configs
    """

    def __init__(self, model, trainer_kwargs):
        super().__init__()
        self._model = model
        self._trainer_kwargs = trainer_kwargs
        self._eval_output_names = set(self._trainer_kwargs.eval_output_names)
        self._parse_eval_output_advanced_metrics()  # check if metrics like AUC/MPR will need to be calculated
        self._cur_iter = 0
        self._summary_interval = trainer_kwargs.summary_interval
        self._gather_val_loss = trainer_kwargs.gather_val_loss
        # allow old version of cruise or if tracking is not installed
        self._tk_writer = DummySummaryWriter()

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
            assert isinstance(eval_out_name, str), \
                f"Expected `eval_output_names` to be list of str, given {eval_out_name}"
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
                    self._extra_metrics.append(
                        {'type': m, 'name': eval_out_name, 'op': metric_op, 'score_key': score_key, 'label_key': label_key})
                    self._all_gather_output_names.add(score_key)
                    self._all_gather_output_names.add(label_key)
                    break
            else:
                # no-match, can be a normal output name, so just let go
                pass

    def dump_checkpoint(self):
        return {'state_dict': self._model.state_dict()}

    def load_state_dict(self, state_dict):
        if not list(state_dict)[0].startswith('_model.'):
            # might resume from stripped checkpoints that are not starting from _model.xxx
            self._model.load_state_dict(state_dict)
        else:
            super().load_state_dict(state_dict)

    def on_save_checkpoint(self, checkpoint):
        # model manager need this
        checkpoint['optimizer'] = checkpoint['optimizer_states']

    def training_step(self, batch, batch_idx):
        inputs = self._model.pre_process_inputs(batch)
        targets = self._model.pre_process_targets(batch)

        train_loss_dict, train_output_dict = self._model(*(inputs + targets))
        self._tb_logging_info(
            train_loss_dict,
            {},
            False,
            prefix="loss",
            show_keys=None,
            is_loss_item=True,
        )
        self._tb_logging_info(
            train_output_dict,
            {},
            False,
            prefix="output",
            show_keys=None,
            is_loss_item=False,
        )
        # save&reduce loss item and output item.
        for loss_name, loss_val_list in train_loss_dict.items():
            train_loss_dict[loss_name] = loss_val_list.mean()
        train_loss_dict.update(train_output_dict)
        self._cur_iter += 1
        return train_loss_dict

    def validation_step(self, batch, batch_index):
        inputs = self._model.pre_process_inputs(batch)
        targets = self._model.pre_process_targets(batch)

        val_loss_dict, val_output_dict = self._model(*(inputs + targets))

        reduced_output = defaultdict(list)
        for output_name, output_val in val_loss_dict.items():
            reduced_output[output_name] = output_val.mean().item()

        if self._extra_metrics:
            # need output in test_mode
            self._model._on_test = True
            val_output = self._model(*inputs)
            self._model._on_test = False
            if isinstance(val_output, torch.Tensor):
                val_output_dict['output0'] = val_output
            elif isinstance(val_output, (list, tuple)):
                for ii, val_output_ii in enumerate(val_output):
                    if not isinstance(val_output_ii, torch.Tensor):
                        msg = f"Returned output without target is of type {type(val_output_ii)} at index={ii}"
                        msg += f'skip test mode `output{ii}` for metric calculation'
                        rank_zero_warn(msg)
                        continue
                    val_output_dict[f'output{ii}'] = val_output_ii
            else:
                raise ValueError(f"Unexpected type of {type(val_output)} from model.forward() without target")

            for output_name, output_val in val_output_dict.items():
                if output_name not in self._eval_output_names and output_name not in self._all_gather_output_names:
                    continue
                reduced_output[output_name] = output_val.cpu().numpy()
            for target_idx, target_val in enumerate(targets):
                reduced_output[f'label{target_idx}'] = target_val.cpu().numpy()
        else:
            for output_name, output_val in val_output_dict.items():
                if output_name not in self._eval_output_names:
                    self._eval_output_names.add(output_name)
                if isinstance(output_val, torch.Tensor):
                    reduced_output[output_name] = output_val.cpu().numpy()
                else:
                    reduced_output[output_name] = output_val
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
                elif out_key not in self._eval_output_names:
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

        # collect val loss
        reduced_test_loss = {k: np.mean(vl) for k, vl in reduced_test_loss.items()}

        # collect val output
        for k, vl in reduced_test_output.items():
            if not vl:
                continue
            if isinstance(vl[0], np.ndarray) and len(vl[0].shape) > 0:
                reduced_test_output[k] = np.mean(np.concatenate(vl))
            else:
                reduced_test_output[k] = np.mean(vl)

        self.logger.experiment.add_scalars(
            "eval/custom_metrics",
            extra_reduced_outputs,
            self.trainer.global_step,
        )
        self._tk_log_dict(
            "eval/custom_metrics", extra_reduced_outputs)
        # reduced_test_output.update(extra_reduced_outputs)  # add global metrics to outputs
        self._tb_logging_info(
            {},
            reduced_test_loss,
            True,
            prefix="loss",
            show_keys=None,
            is_loss_item=True,
        )
        self._tb_logging_info(
            {},
            reduced_test_output,
            True,
            prefix="output",
            show_keys=None,
            is_loss_item=False,
        )

    def on_train_epoch_start(self):
        try:
            from cruise.trainer.logger.tracking import TrackingLogger
            for logger in self.logger._logger_iterable:
                if isinstance(logger, TrackingLogger):
                    # for compatibility, wandb should be used if exists
                    self._tk_writer = getattr(logger, 'wandb', logger.experiment)
                    break
        except ImportError:
            pass
        self._cur_iter = 0
        self.logger.experiment.add_scalars(
            "monitors/epoch_num",
            {"epoch": self.trainer.current_epoch},
            self.trainer.global_step,
        )
        self._tk_log_dict("monitors/epoch_num", {"epoch": self.trainer.current_epoch})

    def on_train_epoch_end(self):
        # save ep end checkpoints
        checkpoint_config = self._trainer_kwargs.get("checkpoint_saver")
        save_at_each_ep = checkpoint_config.get("save_at_each_ep", False)
        if not save_at_each_ep:
            return
        if self.trainer.global_rank == 0:
            output_path = os.path.join(
                self.trainer.default_root_dir, 'bin', f"checkpoint_ep_{self.trainer.current_epoch}.pth")
            ckpt = self.dump_checkpoint()
            ckpt['extra'] = {
                "cur_iter": self._cur_iter,
                "cur_epoch": self.trainer.current_epoch,
                "global_iter": self.trainer.global_step,
                'time': time.asctime()
            }
            torch.save(ckpt, output_path)
            self.print(f"Successfully saved ep end checkpoint {output_path}")
            if self.trainer.default_hdfs_dir:
                hdfs_path = os.path.join(self.trainer.default_hdfs_dir, 'bin')
                push_files(output_path, hdfs_path)
                self.print(f'Successfully push latest checkpoint {output_path} to {hdfs_path}')

    def configure_optimizers(self):
        optimizer_config = self._trainer_kwargs.get("optimizer")
        optimizer = build_optimizer([self._model], optimizer_config)
        lr_scheduler_config = self._trainer_kwargs.get("lr_scheduler", dict())
        lr_scheduler_config["total_step"] = self.trainer.max_steps
        lr_scheduler_config["warmup_steps"] = (
                lr_scheduler_config.get("warmup_steps") or 0
        )
        lr_scheduler = build_lr_scheduler(optimizer, lr_scheduler_config)
        return [optimizer], [lr_scheduler]

    def lr_scheduler_step(self, schedulers, **kwargs) -> None:
        # maxwell lr_schedulers has no step method, but update_lr method
        for scheduler in schedulers:
            scheduler.update_lr(self.trainer.global_step)
            self.logger.experiment.add_scalars(
                "monitors/learning_rate",
                {"lr": scheduler.get_lr()},
                self.trainer.global_step,
            )
            self._tk_log_dict("monitors/learning_rate", {"lr": scheduler.get_lr()})

    def _tk_log_dict(self, tag, sub_dict):
        for k, v in sub_dict.items():
            if isinstance(v, dict):
                self._tk_log_dict(tag + '.' + k, v)
            name = tag + '.' + k
            self._tk_writer.log({name: v}, step=self.trainer.global_step)

    def _tb_logging_info(
            self,
            train_info_dict: dict,
            test_info_dict: dict,
            force: bool,
            prefix: str,
            show_keys: set = None,
            is_loss_item: bool = True,
    ):
        """log train & test info into tensorboard, and decorate with train or test tag"""
        if not force and self.trainer.global_step % self._summary_interval != 0:
            return
        if train_info_dict:
            for okey, oval in train_info_dict.items():
                if show_keys and okey not in show_keys:
                    continue
                sub_dict = dict()
                merge_into_target_dict(sub_dict, {okey: oval}, "train", is_loss_item=is_loss_item)
                if okey in test_info_dict:
                    merge_into_target_dict(sub_dict, {okey: test_info_dict[okey]}, "test")
                self.logger.experiment.add_scalars(
                    f"{prefix}/{okey}",
                    sub_dict,
                    self.trainer.global_step,
                )
                self._tk_log_dict(
                    f"{prefix}/{okey}", sub_dict)
        else:
            # pure validation step
            try:
                if self._gather_val_loss:
                    all_test_info_dict = DIST_ENV.all_gather_object(test_info_dict)
                    tmp = {k: np.mean([x.get(k, test_info_dict[k]) for x in all_test_info_dict])
                           for k in test_info_dict.keys()}
                    test_info_dict = tmp
            except Exception as ex:
                raise RuntimeError(f"Failed to sync and reduce val outputs from all ranks: {ex}")

            for okey, oval in test_info_dict.items():
                if show_keys and okey not in show_keys:
                    continue
                sub_dict = dict()
                merge_into_target_dict(sub_dict, {okey: oval}, "test", is_loss_item=is_loss_item)
                self.logger.experiment.add_scalars(
                    f"{prefix}/{okey}",
                    sub_dict,
                    self.trainer.global_step,
                )
                self._tk_log_dict(
                    f"{prefix}/{okey}", sub_dict)



def merge_into_target_dict(src_dict, sub_dict, prefix, is_loss_item=False):
    # do namespace conversion and merge, cast to standard dtypes if not
    for key, val in sub_dict.items():
        if is_loss_item:
            if isinstance(val, torch.Tensor):
                src_dict[f"{prefix}"] = float(val.mean().item())
            if isinstance(val, np.floating):
                src_dict[f"{prefix}"] = float(val)
            elif isinstance(val, np.integer):
                src_dict[f"{prefix}"] = int(val)
        else:
            if isinstance(val, torch.Tensor):
                if len(val.shape) >= 1:
                    src_dict[f"{prefix}"] = float(val.mean().item())
                else:
                    src_dict[f"{prefix}"] = val.item()  # diff
            if isinstance(val, np.floating):
                src_dict[f"{prefix}"] = float(val)
            elif isinstance(val, np.integer):
                src_dict[f"{prefix}"] = int(val)
