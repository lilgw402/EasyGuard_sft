from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from prettytable import PrettyTable
from torch import Tensor, nn

from cruise.utilities.cloud_io import load as crs_load
from cruise.utilities.rank_zero import rank_zero_info, rank_zero_warn

from ..utils import (  # DUMMY_INPUTS,; FLAX_WEIGHTS_NAME,; SAFE_WEIGHTS_INDEX_NAME,; SAFE_WEIGHTS_NAME,; TF2_WEIGHTS_NAME,; TF_WEIGHTS_NAME,
    load_pretrained_model_weights,
    logging,
)
from .hub import AutoHubClass

# from .utils.versions import require_version_core
# -*- coding: utf-8 -*-
logger = logging.get_logger(__name__)


# cruise
def load_pretrained(load_pretrain, model):
    rank_zero_info(
        f"==============> Loading weight {load_pretrain} for fine-tuning......"
    )
    checkpoint = crs_load(load_pretrain, map_location="cpu")
    state_dict = checkpoint

    # check classifier, if not match, then re-init classifier to zero
    try:
        head_bias_pretrained = state_dict["classifier.bias"]
        Nc1 = head_bias_pretrained.shape[0]
        Nc2 = model.classifier.bias.shape[0]
        if Nc1 != Nc2:
            torch.nn.init.constant_(model.classifier.bias, 0.0)
            torch.nn.init.constant_(model.classifier.weight, 0.0)
            del state_dict["classifier.weight"]
            del state_dict["classifier.bias"]
            rank_zero_warn(
                "Error in loading classifier head, re-init classifier head to 0"
            )
    except:
        rank_zero_warn("Error in loading classifier weights...")

    parsed_state_dict = {}
    non_match_keys = []
    pretrained_keys = []
    for k, v in state_dict.items():
        if k in model.state_dict():
            parsed_state_dict[k] = v
            pretrained_keys.append(k)
        else:
            non_match_keys.append(k + ":" + str(v.shape))
            # raise ValueError('failed to match key of state dict smartly!')

    table = PrettyTable(["Layer Name", "Weight Shape", "Data Type", "Pretrain"])
    for k, v in model.named_parameters():
        table.add_row([k, v.shape, v.dtype, str(k in pretrained_keys)])
    table.align = "l"
    rank_zero_info("\n###### Parameters ######\n{}".format(table.get_string()))
    rank_zero_info(
        "\n###### Not matched keys ######\n{}".format(
            "\n".join(non_match_keys) + "\n"
        )
    )

    new_state_dict = model.state_dict()
    new_state_dict.update(parsed_state_dict)
    msg = model.load_state_dict(new_state_dict, strict=False)
    rank_zero_warn(str(msg))

    rank_zero_info(f"=> loaded successfully '{load_pretrain}'")

    del checkpoint
    torch.cuda.empty_cache()


# TODO (junwei.Dong): 一些基础的通用的方法和对应平台终端的方法可以在ModelBase里进行注册，自己写的模型尽量继承该基类
# EasyGuard Model base class


class ModelBase(AutoHubClass):
    def __init__(self, **kargs) -> None:
        super().__init__()

    def load_pretrained_weights(self, weight_file_path, **kwargs):
        # the default weights load function, we can overload this function in a specific model class and the `AutoModel.from_pretrained` function will call it after loading the architecture of the model
        load_pretrained_model_weights(self, weight_file_path, **kwargs)
