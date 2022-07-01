# -*- coding: utf-8 -*-
'''
Created on Jan-05-21 17:33
convert_old_config.py
@author: liuzhen.nlp
Description: 
'''
import os
from absl import flags
from absl import app

from fex.config import cfg, CfgNode


FLAGS = flags.FLAGS
def def_flags():
    flags.DEFINE_string("config_path", '', "config_path")

CONVERT_NAME_LIST = ["RNG_SEED->TRAINER.RNG_SEED", "LOG_FREQUENT->TRAINER.LOG_FREQUENT", "VAL_FREQUENT->TRAINER.VAL_FREQUENT", 
                    "CHECKPOINT_FREQUENT->TRAINER.CHECKPOINT_FREQUENT", "TRAIN.BEGIN_EPOCH->TRAINER.BEGIN_EPOCH", "TRAIN.END_EPOCH->TRAINER.END_EPOCH", 
                    "TRAIN.BATCH_SIZE->TRAINER.TRAIN_BATCH_SIZE", "VAL.BATCH_SIZE->TRAINER.VAL_BATCH_SIZE", "DATASET.TRAIN_SIZE->TRAINER.TRAIN_DATASET_SIZE", 
                    "DATASET.VAL_SIZE->TRAINER.VAL_DATASET_SIZE", "VAL.STEPS->TRAINER.VAL_STEPS", "TRAIN.GRAD_ACCUMULATE_STEPS->ACCELERATOR.GRAD_ACCUMULATE_STEPS", 
                    "TRAIN.CLIP_GRAD_NORM->ACCELERATOR.CLIP_GRAD_NORM", "TRAIN.FP16->ACCELERATOR.FP16", "TRAIN.FP16_OPT_LEVEL->ACCELERATOR.FP16_OPT_LEVEL", 
                    "TRAIN.FP16_LOSS_SCALE->ACCELERATOR.FP16_LOSS_SCALE", "TRAIN.SYNCBN->ACCELERATOR.SYNCBN"]


RM_NAME_LIST = ["OUTPUT_PATH"]

def convert_config_to_new(_):
    none_value = "none"
    cfg.update_cfg(FLAGS.config_path)
    for rm_str in RM_NAME_LIST:
        remove_attr(rm_str, cfg)

    for convert_str in CONVERT_NAME_LIST:
        split_attrs = convert_str.split("->")
        origin_cfg_attr = split_attrs[0]
        dest_cfg_attr = split_attrs[1]
        value = none_value
        origin_attrs = origin_cfg_attr.split(".")
        dest_attrs = dest_cfg_attr.split(".")

        if len(origin_attrs) == 1:
            if getattr(cfg, origin_attrs[0], none_value) != none_value:
                value = cfg.pop(origin_attrs[0])
        else:
            obj = getattr(cfg, origin_attrs[0], none_value)
            if obj != none_value:
                if getattr(obj, origin_attrs[1], none_value) != none_value:
                    value = obj.pop(origin_attrs[1])
        if value != none_value:
            obj = getattr(cfg, dest_attrs[0], none_value)
            if obj != none_value:
                setattr(obj, dest_attrs[1], value)
            else:
                setattr(cfg, dest_attrs[0], CfgNode())
                obj = getattr(cfg, dest_attrs[0])
                setattr(obj, dest_attrs[1], value)
    config_names = os.path.split(FLAGS.config_path)[-1].split(".")
    new_config_name = config_names[0] + "_new_config." + config_names[1]
    cfg.dump(config_name=new_config_name)

def remove_attr(attr: str, obj: CfgNode):
    attrs = attr.split(".")
    if len(attrs) == 1:
        obj.pop(attr)
    else:
        for sub_attr in attrs:
            remove_attr(sub_attr, obj)
            obj = getattr(obj, sub_attr)

if __name__ == "__main__":
    def_flags()
    app.run(convert_config_to_new)

