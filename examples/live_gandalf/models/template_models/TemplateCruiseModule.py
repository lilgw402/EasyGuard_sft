import numpy as np
from torch.nn import functional as F
from cruise import CruiseModule
from cruise.utilities.distributed import DIST_ENV
from utils.registry import get_metric_instance,build_optimizer_instance,build_lr_scheduler_instance
from utils.util import merge_into_target_dict


class TemplateCruiseModule(CruiseModule):
    def __init__(self,kwargs):
        super().__init__()
        self.kwargs = kwargs
        self._gather_val_loss = True
        self._metric_params = {'ClsMetric':{'score_key':'output','label_key':'label'}}
        self._output_name = 'output'
        self._eval_output_names =  ["loss"]
        self._extra_metrics = []
        self._parse_eval_output_advanced_metrics()  # check if metrics like AUC/MPR will need to be calculated
    
    def forward(self, batch, batch_idx):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def on_train_epoch_start(self):
        self._tk_log_dict("monitors/epoch_num", {"epoch": self.trainer.current_epoch})

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_index):
        pass

    def validation_epoch_end(self, outputs) -> None:
        pass
        
    def configure_optimizers(self):
        optimizer_config = self.kwargs.get("optimizer",dict())
        optimizer = build_optimizer_instance([self.trainer.model], optimizer_config)
        lr_scheduler_config = self.kwargs.get("lr_scheduler", dict())
        lr_scheduler_config["total_step"] = self.trainer.max_steps
        lr_scheduler_config["warmup_steps"] = (
                lr_scheduler_config.get("warmup_steps") or 0
        )
        lr_scheduler = build_lr_scheduler_instance(optimizer, lr_scheduler_config)
        return [optimizer], [lr_scheduler]

    def lr_scheduler_step(self, schedulers, **kwargs) -> None:
        # lr_schedulers has no step method, but update_lr method
        for scheduler in schedulers:
            scheduler.update_lr(self.trainer.global_step)
            self._tk_log_dict("monitors/learning_rate", {"lr": scheduler.get_lr()})

    def track_logging_info(self,training_info_dict,test_info_dict,prefix,hidden_keys:set = None,is_loss_item:bool=True):
        if training_info_dict:
            for okey, oval in training_info_dict.items():
                if hidden_keys and okey in hidden_keys:
                    continue
                sub_dict = dict()
                # merge_into_target_dict(sub_dict, {okey: oval}, "train", is_loss_item=is_loss_item)
                merge_into_target_dict(sub_dict, {okey: oval}, is_loss_item=is_loss_item)
                if okey in test_info_dict:
                    # merge_into_target_dict(sub_dict, {okey: test_info_dict[okey]}, "test")
                    merge_into_target_dict(sub_dict, {okey: test_info_dict[okey]})
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
                merge_into_target_dict(sub_dict, {okey: oval}, is_loss_item=is_loss_item)
                self._tk_log_dict(f'{prefix}',sub_dict)

    def _tk_log_dict(self, tag, sub_dict):
        for k, v in sub_dict.items():
            if isinstance(v, dict):
                self._tk_log_dict(tag + '.' + k, v)
            name = tag + '.' + k if tag else k
            self.log_dict({name: v},tb=False,tracking=True)

    def _parse_eval_output_advanced_metrics(self):
        self._extra_metrics = []
        self._all_gather_output_names = set()
        for metric_name in self._metric_params:
            metric_op = get_metric_instance(metric_name,self._metric_params[metric_name])
            assert hasattr(metric_op, 'cal_metric')
            score_key = self._metric_params[metric_name].get('score_key','output')
            label_key = self._metric_params[metric_name].get('label_key','label')
            self._extra_metrics.append({'type': metric_name, 'name': metric_name, 'op': metric_op, 'score_key':score_key , 'label_key': label_key})
            self._all_gather_output_names.add(score_key)
            self._all_gather_output_names.add(label_key)
            break

   
if __name__ == '__main__':
    module = TemplateCruiseModule()