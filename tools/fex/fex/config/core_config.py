# -*- coding: utf-8 -*-
'''
Created on Nov-10-20 21:06
config.py
@author: liuzhen.nlp
Description: config utils
'''
import os
from copy import deepcopy
from typing import Union, IO, List
import yaml
from yacs.config import CfgNode as _CfgNode

from fex.utils.hdfs_io import hopen

_YAML_EXTS = {"", ".yaml", ".yml"}


class CfgNode(_CfgNode):
    """
    Our own extended version of :class:`yacs.config.CfgNode`.
    It support hdfs config.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        super().__init__(init_dict=init_dict, key_list=key_list, new_allowed=new_allowed)

    def update_cfg(self,
                   cfg_filename: str,
                   cfg_str: str = None,
                   cfg_list: List = None) -> None:
        """
        更新一个本地或者hdfs上的yaml文件.
        Args:
            cfg_filename: 本地或者hdfs上的yaml文件名
            cfg_str: cfg string to merge into this CfgNode,
                可以根据string来更新config，格式如下: `{config1}={params1};{config2}={params2}`
                用 `;` 来分割多个参数，用 `=` 来赋值
                比如 "TRAINER.TRAIN_BATCH_SIZE=500;TRAINER.END_EPOCH=32"
            cfg_list: cfg list to merge into this CfgNode。
        """

        def __load_base_cfg(cfg: CfgNode):
            if hasattr(cfg, "BASE") and cfg.BASE:
                for base_config_item in cfg.BASE:
                    base_cfg = self._load_cfg(base_config_item)
                    base_cfg.set_new_allowed(True)
                    # 如果不先load再merge，嵌套三层配置的时候，就会出问题
                    __load_base_cfg(base_cfg)
                    self.merge_from_other_cfg(base_cfg)

        cfg = self._load_cfg(cfg_filename)
        self.set_new_allowed(True)

        __load_base_cfg(cfg)
        self.set_new_allowed(True)
        self.merge_from_other_cfg(cfg)

        if cfg_str:
            self.merge_from_str(cfg_str)

        if cfg_list is not None:
            self.merge_from_list(cfg_list)

        self.CONFIG_PATH = cfg_filename

    def _load_cfg(self, cfg_filename: str):
        """
        加载本地或者hdfs的yaml文件.
        Args:
            cfg_filename: 本地或者hdfs上的yaml文件名
        """
        if cfg_filename.startswith("hdfs://"):
            _, file_extension = os.path.splitext(cfg_filename)
            if file_extension in _YAML_EXTS:
                with hopen(cfg_filename, "r") as fp:
                    return self._load_cfg_from_yaml_str_remote(fp.read())
            else:
                raise ValueError("Not support file type, please check !")
        else:
            with open(cfg_filename, "r") as fp:
                return super().load_cfg(fp)

    def _load_cfg_from_yaml_str_remote(self, str_obj: Union[bytes, IO[bytes], str, IO[str]]):
        """
        通过file_obj加载hdfs上的yaml文件.
        Args:
            str_obj: 文件流stream
        """
        cfg_as_dict = yaml.safe_load(str_obj)
        return _CfgNode(cfg_as_dict)

    def check_core_attr(self):
        """
        检查trainer中需要的一些核心配置是否在yaml配置文件中设置
        """
        def _check_attr(sub_attr: str, org_attr: str, obj: CfgNode):
            if not getattr(obj, sub_attr):
                raise ValueError(
                    "{} not in train yaml, this value is necessary !".format(org_attr))

        core_attr_list: List[str] = ["TRAINER.CHECKPOINT_FREQUENT", "TRAINER.END_EPOCH",
                                     "TRAINER.TRAIN_BATCH_SIZE", "TRAINER.VAL_BATCH_SIZE",
                                     "TRAINER.TRAIN_DATASET_SIZE", "TRAINER.VAL_DATASET_SIZE", "TRAINER.VAL_STEPS"]
        for attr in core_attr_list:
            attrs = attr.split(".")
            if len(attrs) == 1:
                _check_attr(attrs[0], attr, self)
            else:
                obj: CfgNode = self
                for sub_attr in attrs:
                    _check_attr(sub_attr, attr, obj)
                    obj = getattr(obj, sub_attr)

    def merge_from_str(self, cfg: str) -> None:
        """通过cfg字符串修改当前CfgNode

        Args:
            cfg: cfg字符串，使用"="或者","作分隔符
        """
        cfg_list = cfg.replace('=', ';').split(';')
        for i in range(len(cfg_list)):
            if cfg_list[i].startswith('[') and cfg_list[i].endswith(']'):
                cfg_list[i] = eval(cfg_list[i])
        self.merge_from_list(cfg_list)

    def freeze(self) -> None:
        """
        将当前CfgNode和所有的subCfgNode变成不可以更改的.
        """
        super().freeze()

    def defrost(self) -> None:
        """
        将当前CfgNode和所有的subCfgNode变成可以更改的, 和freeze方法相反.
        """
        super().defrost()

    def clear(self) -> None:
        """
        将当前的CfgNode清空
        """
        if self._is_frozen():
            self.defrost()
        self.clear()

    def dump(self, results_dir: str = "./", config_name: str = "test.yaml") -> None:
        """保存更新后的config到yaml文件中

        Args:
            results_dir (str): 需要保存的yaml文件路径
            config_name (str): 需要保存的yaml文件名
        """
        if not config_name.endswith(".yaml"):
            config_name = config_name + ".yaml"
        if "CONFIG_PATH" in self:
            self.pop("CONFIG_PATH")
        if "MOUDEL_FILE" in self:
            self.pop("MOUDEL_FILE")
        with open(os.path.join(results_dir, f'{config_name}'), 'w') as fw:
            super().dump(stream=fw)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(
                "ERROR: {} not in cfg, please check !".format(name))

    def get(self, names, default=None):
        try:
            names = names.split('.')
            value = self
            for name in names:
                value = value[name]
            return value
        except Exception as e:
            return default

    def _is_frozen(self):
        """ 返回当前cfg是否可以frozen """
        return super().__dict__[CfgNode.IMMUTABLE]

    def flat_keys(self, prefix=''):
        keys = []
        for key, val in self.items():
            full_key = prefix + key
            if isinstance(val, CfgNode):
                keys += val.flat_keys(f'{full_key}.')
            else:
                keys.append(full_key)
        return keys


_C = CfgNode()
cfg = _C

# Support multi base config
_C.BASE = []

# ------------------------------------------------------------------------------------- #
# Common options in trainer
# ------------------------------------------------------------------------------------- #
_C.TRAINER = CfgNode()  # type: ignore
_C.TRAINER.RNG_SEED = 12345
_C.TRAINER.LOG_FREQUENT = 100
_C.TRAINER.VAL_FREQUENT = 5000
_C.TRAINER.CHECKPOINT_FREQUENT = None
_C.TRAINER.BEGIN_EPOCH = 0
_C.TRAINER.END_EPOCH = None
_C.TRAINER.TRAIN_BATCH_SIZE = None
_C.TRAINER.VAL_BATCH_SIZE = None
_C.TRAINER.TRAIN_DATASET_SIZE = None
_C.TRAINER.VAL_DATASET_SIZE = None
_C.TRAINER.VAL_STEPS = None
# 对于包含卷积的model加速: https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html
_C.TRAINER.CHANNELS_LAST = False

# ------------------------------------------------------------------------------------- #
# Common options in accelerators
# ------------------------------------------------------------------------------------- #
_C.ACCELERATOR = CfgNode()  # type: ignore
_C.ACCELERATOR.ACCELERATOR = "ApexDDP"
_C.ACCELERATOR.GRAD_ACCUMULATE_STEPS = 1
_C.ACCELERATOR.CLIP_GRAD_NORM = -1.
_C.ACCELERATOR.FP16_OPT_LEVEL = 'O2'
_C.ACCELERATOR.FP16_LOSS_SCALE = 'dynamic'
_C.ACCELERATOR.FP16_MAX_LOSS_SCALE = 1024.0
_C.ACCELERATOR.FP16_MIN_LOSS_SCALE = 1.0
_C.ACCELERATOR.SYNCBN = False
# By default, allreduce is not delayed (i.e. allreduce and backward computation are
# overlappped to reduce iteration time)
_C.ACCELERATOR.DISABLE_DELAY_ALLREDUCE = True

# ------------------------------------------------------------------------------------- #
# Common network options
# ------------------------------------------------------------------------------------- #
_C.NETWORK = CfgNode()  # type: ignore
_C.NETWORK.NET = ""
_C.NETWORK.PARTIAL_PRETRAIN = None  # can be str or list
_C.NETWORK.PARTIAL_PRETRAIN_PREFIX_CHANGES = []
_C.NETWORK.FREEZE_BN_RUNNING_STATS = False

_C.BERT = CfgNode()  # type: ignore
_C.BERT.visual_type = 'RN50'
_C.BERT.layernorm_eps = 1.0e-5


# ------------------------------------------------------------------------------------- #
# Common xla options
# ------------------------------------------------------------------------------------- #
_C.XLA = CfgNode()  # type: ignore
_C.XLA.ENABLE = False
_C.XLA.AMP = False


# # ------------------------------------------------------------------------------------- #
# # Common dataset options
# # ------------------------------------------------------------------------------------- #
# _C.DATASET = CfgNode() # type: ignore
# _C.DATASET.DATASET = ''
# _C.DATASET.TRAIN_DATASET = ''
# _C.DATASET.VAL_DATASET = ''
# _C.DATASET.TRAIN_PATH = ''
# _C.DATASET.VAL_PATH = ''
# # 训练数据的size, 训练的yaml配置文件必须提供，trainer中会根据world_size和batch_size计算每个epoch的step数. update_config时会检查, 没有提供会报错
# _C.DATASET.TRAIN_SIZE = None
# # validataion数据的size, 训练的yaml配置文件必须提供，trainer中会根据world_size和batch_size计算每个epoch的step数. update_config时会检查, 没有提供会报错
# _C.DATASET.VAL_SIZE = None

# # ------------------------------------------------------------------------------------- #
# # Common training related options
# # ------------------------------------------------------------------------------------- #
# _C.TRAIN = CfgNode() # type: ignore
# _C.TRAIN.SHUFFLE = True
# # 训练的batch_size，训练的yaml配置文件必须提供，trainer中会根据batch_size计算step数. update_config时会检查, 没有提供会报错
# _C.TRAIN.BATCH_SIZE = None
# _C.TRAIN.BEGIN_EPOCH = 0
# # 训练的end_epoch，训练的yaml配置文件必须提供，trainer中会根据end_epoch来控制训练的epoch数. update_config时会检查, 没有提供会报错
# _C.TRAIN.END_EPOCH = None

# _C.TRAIN.OPTIMIZER = 'SGD'
# _C.TRAIN.LR = 0.1
# _C.TRAIN.LR_SCHEDULE = 'step'  # step/triangle/plateau
# _C.TRAIN.LR_FACTOR = 0.1
# _C.TRAIN.LR_STEP = ()
# _C.TRAIN.WARMUP = False
# _C.TRAIN.WARMUP_METHOD = 'linear'
# _C.TRAIN.WARMUP_FACTOR = 0.1
# _C.TRAIN.WARMUP_STEPS = 0
# _C.TRAIN.WD = 0.0001
# _C.TRAIN.MOMENTUM = 0.9

# _C.TRAIN.CLIP_GRAD_NORM = -1.
# _C.TRAIN.GRAD_ACCUMULATE_STEPS = 1
# _C.TRAIN.FP16 = False
# _C.TRAIN.FP16_LOSS_SCALE = 'dynamic'
# _C.TRAIN.FP16_OPT_LEVEL = 'O2'
# _C.TRAIN.MAX_PREFETCH = 2
# # 是否将batch norm的 running states也freeze掉，他们不是parameters，是buffer，只有在eval的时候不更新
# _C.TRAIN.FREEZE_BN_RUNNING_STATS = False


# _C.VAL = CfgNode() # type: ignore
# # validation的batch_size, 训练的yaml配置文件必须提供，trainer中会根据batch_size计算step数. update_config时会检查, 没有提供会报错
# _C.VAL.BATCH_SIZE = None
# # 控制做validation的step数，如果val_dataset数量比较多，可以设置一个相对比较小的数，避免影响训练速度，比如10. default为0，默认validation跑total_step.
# _C.VAL.STEPS = 0


# ------------------------------------------------------------------------------------- #
# 参数注册，需要备注解释每个参数的作用，并且有默认值
# ------------------------------------------------------------------------------------- #

# # 冻结resnet的层数，在ALBertV里使用，[] 为不冻结， [1,2,3,4,5,6] 为全部冻结 （6表示最后分类层）
# _C.NETWORK.FREEZE_RESNET_LAYERS = []
# # 是否要给visual embedding最后的输出做一次layernorm
# _C.NETWORK.v_norm = False
# # 是否需要text embedding，用于训练resnet
# _C.DATASET.NEDD_TEMB = False

# ------------------------------------------------------------------------------------- #
# copy一份默认参数
# ------------------------------------------------------------------------------------- #

default_cfg = deepcopy(_C)


def reset_cfg():
    """ Get default config
    """
    if default_cfg:
        return deepcopy(default_cfg)
