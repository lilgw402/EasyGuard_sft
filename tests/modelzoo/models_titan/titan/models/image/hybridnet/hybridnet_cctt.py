""" CCTT Hybrid architecture searched by weight-sharing EA algorithm
"""
import torch
import torch.nn as nn
from collections import OrderedDict

from timm.models.layers import DropPath, trunc_normal_
from titan.utils.registry import register_model
from titan.utils.helper import (
    download_weights,
    load_pretrained_model_weights,
    get_configs
)

__all__ = ['hybridCCTT_1', 'hybridCCTT_2', 'hybridCCTT_3', 'hybridCCTT_4']

ARCH_CFG_1 = [
    ["ConvBlockX", {"in_channels": 3, "out_channels": 24, "kernel_size": 3, "stride": 2}],
    ["ConvBlockX", {"in_channels": 24, "out_channels": 48, "kernel_size": 5, "stride": 2}],
    ["ResBlock", {"in_channels": 48, "out_channels": 192, "groups": 1, "group_width": 80, "stride": 1, "kernel_size": 3}],
    ["ResBlock", {"in_channels": 192, "out_channels": 192, "groups": 1, "group_width": 80, "stride": 1, "kernel_size": 3}],
    ["ResBlock", {"in_channels": 192, "out_channels": 384, "groups": 1, "group_width": 80, "stride": 2, "kernel_size": 3}],
    ["ResBlock", {"in_channels": 384, "out_channels": 384, "groups": 1, "group_width": 160, "stride": 1, "kernel_size": 3}],
    ["ResBlock", {"in_channels": 384, "out_channels": 384, "groups": 1, "group_width": 80, "stride": 1, "kernel_size": 3}],
    ["ResBlock", {"in_channels": 384, "out_channels": 384, "groups": 1, "group_width": 64, "stride": 1, "kernel_size": 5}],
    ["ResBlock", {"in_channels": 384, "out_channels": 384, "groups": 1, "group_width": 176, "stride": 1, "kernel_size": 5}],
    ["ViTBlock", {"head_num": 7, "head_width": 64, "hid_dims": 1280, "input_resolution": [14, 14], "in_channels": 384, "out_channels": 400, "stride": 2}],
    ["ViTBlock", {"head_num": 10, "head_width": 64, "hid_dims": 800, "input_resolution": [14, 14], "in_channels": 400, "out_channels": 400, "stride": 1}],
    ["ViTBlock", {"head_num": 9, "head_width": 64, "hid_dims": 768, "input_resolution": [14, 14], "in_channels": 400, "out_channels": 400, "stride": 1}],
    ["ViTBlock", {"head_num": 8, "head_width": 64, "hid_dims": 1184, "input_resolution": [14, 14], "in_channels": 400, "out_channels": 400, "stride": 1}],
    ["ViTBlock", {"head_num": 9, "head_width": 64, "hid_dims": 704, "input_resolution": [14, 14], "in_channels": 400, "out_channels": 400, "stride": 1}],
    ["ViTBlock", {"head_num": 9, "head_width": 64, "hid_dims": 672, "input_resolution": [14, 14], "in_channels": 400, "out_channels": 400, "stride": 1}],
    ["ViTBlock", {"head_num": 10, "head_width": 64, "hid_dims": 1216, "input_resolution": [14, 14], "in_channels": 400, "out_channels": 400, "stride": 1}],
    ["ViTBlock", {"head_num": 10, "head_width": 64, "hid_dims": 1440, "input_resolution": [14, 14], "in_channels": 400, "out_channels": 400, "stride": 1}],
    ["ViTBlock", {"head_num": 7, "head_width": 64, "hid_dims": 1408, "input_resolution": [14, 14], "in_channels": 400, "out_channels": 400, "stride": 1}],
    ["ViTBlock", {"head_num": 8, "head_width": 64, "hid_dims": 1376, "input_resolution": [14, 14], "in_channels": 400, "out_channels": 400, "stride": 1}],
    ["ViTBlock", {"head_num": 9, "head_width": 64, "hid_dims": 1344, "input_resolution": [14, 14], "in_channels": 400, "out_channels": 400, "stride": 1}],
    ["ViTBlock", {"head_num": 7, "head_width": 64, "hid_dims": 1216, "input_resolution": [14, 14], "in_channels": 400, "out_channels": 400, "stride": 1}],
    ["ViTBlock", {"head_num": 9, "head_width": 64, "hid_dims": 960, "input_resolution": [14, 14], "in_channels": 400, "out_channels": 400, "stride": 1}],
    ["ViTBlock", {"head_num": 7, "head_width": 64, "hid_dims": 1408, "input_resolution": [14, 14], "in_channels": 400, "out_channels": 400, "stride": 1}],
    ["ViTBlock", {"head_num": 8, "head_width": 64, "hid_dims": 2688, "input_resolution": [7, 7], "in_channels": 400, "out_channels": 768, "stride": 2}],
    ["ViTBlock", {"head_num": 12, "head_width": 64, "hid_dims": 2560, "input_resolution": [7, 7], "in_channels": 768, "out_channels": 768, "stride": 1}],
    ["ViTBlock", {"head_num": 14, "head_width": 64, "hid_dims": 2688, "input_resolution": [7, 7], "in_channels": 768, "out_channels": 768, "stride": 1}],
    ["ViTBlock", {"head_num": 8, "head_width": 64, "hid_dims": 2624, "input_resolution": [7, 7], "in_channels": 768, "out_channels": 768, "stride": 1}]]
ARCH_CFG_2 = [
    ["ConvBlockX", {"in_channels": 3, "out_channels": 16, "kernel_size": 3, "stride": 2}],
    ["ConvBlockX", {"in_channels": 16, "out_channels": 64, "kernel_size": 5, "stride": 2}],
    ["ConvBlockX", {"in_channels": 64, "out_channels": 48, "kernel_size": 5, "stride": 1}],
    ["ResBlock", {"in_channels": 48, "out_channels": 240, "groups": 1, "group_width": 88, "stride": 1, "kernel_size": 3}],
    ["ResBlock", {"in_channels": 240, "out_channels": 240, "groups": 1, "group_width": 32, "stride": 1, "kernel_size": 3}],
    ["ResBlock", {"in_channels": 240, "out_channels": 240, "groups": 1, "group_width": 40, "stride": 1, "kernel_size": 3}],
    ["ResBlock", {"in_channels": 240, "out_channels": 512, "groups": 1, "group_width": 80, "stride": 2, "kernel_size": 5}],
    ["ResBlock", {"in_channels": 512, "out_channels": 512, "groups": 1, "group_width": 112, "stride": 1, "kernel_size": 3}],
    ["ResBlock", {"in_channels": 512, "out_channels": 512, "groups": 1, "group_width": 80, "stride": 1, "kernel_size": 3}],
    ["ResBlock", {"in_channels": 512, "out_channels": 512, "groups": 1, "group_width": 160, "stride": 1, "kernel_size": 3}],
    ["ResBlock", {"in_channels": 512, "out_channels": 512, "groups": 1, "group_width": 80, "stride": 1, "kernel_size": 5}],
    ["ResBlock", {"in_channels": 512, "out_channels": 512, "groups": 1, "group_width": 192, "stride": 1, "kernel_size": 3}],
    ["ViTBlock", {"head_num": 8, "head_width": 64, "hid_dims": 864, "input_resolution": [14, 14], "in_channels": 512, "out_channels": 480, "stride": 2}],
    ["ViTBlock", {"head_num": 7, "head_width": 64, "hid_dims": 736, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 10, "head_width": 64, "hid_dims": 1152, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 7, "head_width": 64, "hid_dims": 1440, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 10, "head_width": 64, "hid_dims": 800, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 10, "head_width": 64, "hid_dims": 1024, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 7, "head_width": 64, "hid_dims": 1440, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 10, "head_width": 64, "hid_dims": 1344, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 7, "head_width": 64, "hid_dims": 704, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 7, "head_width": 64, "hid_dims": 1280, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 8, "head_width": 64, "hid_dims": 1376, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 7, "head_width": 64, "hid_dims": 896, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 8, "head_width": 64, "hid_dims": 1344, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 9, "head_width": 64, "hid_dims": 1152, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 10, "head_width": 64, "hid_dims": 1120, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 6, "head_width": 64, "hid_dims": 768, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 14, "head_width": 64, "hid_dims": 2048, "input_resolution": [7, 7], "in_channels": 480, "out_channels": 960, "stride": 2}],
    ["ViTBlock", {"head_num": 16, "head_width": 64, "hid_dims": 2560, "input_resolution": [7, 7], "in_channels": 960, "out_channels": 960, "stride": 1}],
    ["ViTBlock", {"head_num": 10, "head_width": 64, "hid_dims": 2688, "input_resolution": [7, 7], "in_channels": 960, "out_channels": 960, "stride": 1}],
    ["ViTBlock", {"head_num": 14, "head_width": 64, "hid_dims": 2112, "input_resolution": [7, 7], "in_channels": 960, "out_channels": 960, "stride": 1}]]
ARCH_CFG_3 = [
    ["ConvBlockX", {"in_channels": 3, "out_channels": 24, "kernel_size": 5, "stride": 2}],
    ["ConvBlockX", {"in_channels": 24, "out_channels": 56, "kernel_size": 3, "stride": 2}],
    ["ResBlock", {"in_channels": 56, "out_channels": 224, "groups": 1, "group_width": 40, "stride": 1, "kernel_size": 5}],
    ["ResBlock", {"in_channels": 224, "out_channels": 224, "groups": 1, "group_width": 96, "stride": 1, "kernel_size": 3}],
    ["ResBlock", {"in_channels": 224, "out_channels": 224, "groups": 1, "group_width": 88, "stride": 1, "kernel_size": 3}],
    ["ResBlock", {"in_channels": 224, "out_channels": 512, "groups": 1, "group_width": 80, "stride": 2, "kernel_size": 5}],
    ["ResBlock", {"in_channels": 512, "out_channels": 512, "groups": 1, "group_width": 160, "stride": 1, "kernel_size": 3}],
    ["ResBlock", {"in_channels": 512, "out_channels": 512, "groups": 1, "group_width": 176, "stride": 1, "kernel_size": 3}],
    ["ResBlock", {"in_channels": 512, "out_channels": 512, "groups": 1, "group_width": 144, "stride": 1, "kernel_size": 5}],
    ["ResBlock", {"in_channels": 512, "out_channels": 512, "groups": 1, "group_width": 128, "stride": 1, "kernel_size": 3}],
    ["ViTBlock", {"head_num": 8, "head_width": 64, "hid_dims": 1472, "input_resolution": [14, 14], "in_channels": 512, "out_channels": 480, "stride": 2}],
    ["ViTBlock", {"head_num": 8, "head_width": 64, "hid_dims": 1600, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 10, "head_width": 64, "hid_dims": 1984, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 6, "head_width": 64, "hid_dims": 1600, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 8, "head_width": 64, "hid_dims": 1536, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 9, "head_width": 64, "hid_dims": 1408, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 6, "head_width": 64, "hid_dims": 1664, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 6, "head_width": 64, "hid_dims": 1600, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 9, "head_width": 64, "hid_dims": 2048, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 9, "head_width": 64, "hid_dims": 1792, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 7, "head_width": 64, "hid_dims": 1472, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 6, "head_width": 64, "hid_dims": 1984, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 6, "head_width": 64, "hid_dims": 1792, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 9, "head_width": 64, "hid_dims": 1664, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 8, "head_width": 64, "hid_dims": 1728, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 10, "head_width": 64, "hid_dims": 1600, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 7, "head_width": 64, "hid_dims": 1408, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 9, "head_width": 64, "hid_dims": 1152, "input_resolution": [14, 14], "in_channels": 480, "out_channels": 480, "stride": 1}],
    ["ViTBlock", {"head_num": 14, "head_width": 64, "hid_dims": 2560, "input_resolution": [7, 7], "in_channels": 480, "out_channels": 1024, "stride": 2}],
    ["ViTBlock", {"head_num": 8, "head_width": 64, "hid_dims": 3328, "input_resolution": [7, 7], "in_channels": 1024, "out_channels": 1024, "stride": 1}],
    ["ViTBlock", {"head_num": 10, "head_width": 64, "hid_dims": 3072, "input_resolution": [7, 7], "in_channels": 1024, "out_channels": 1024, "stride": 1}],
    ["ViTBlock", {"head_num": 12, "head_width": 64, "hid_dims": 3072, "input_resolution": [7, 7], "in_channels": 1024, "out_channels": 1024, "stride": 1}]]
ARCH_CFG_4 = [
    ['ConvBlockX', {'in_channels': 3, 'out_channels': 24, 'kernel_size': 5, 'stride': 2}],
    ['ConvBlockX', {'in_channels': 24, 'out_channels': 48, 'kernel_size': 5, 'stride': 2}],
    ['ResBlock', {'in_channels': 48, 'out_channels': 208, 'groups': 1, 'group_width': 80, 'stride': 1, 'kernel_size': 5}],
    ['ResBlock', {'in_channels': 208, 'out_channels': 208, 'groups': 1, 'group_width': 72, 'stride': 1, 'kernel_size': 5}],
    ['ResBlock', {'in_channels': 208, 'out_channels': 208, 'groups': 1, 'group_width': 48, 'stride': 1, 'kernel_size': 3}],
    ['ResBlock', {'in_channels': 208, 'out_channels': 208, 'groups': 1, 'group_width': 88, 'stride': 1, 'kernel_size': 3}],
    ['ResBlock', {'in_channels': 208, 'out_channels': 480, 'groups': 1, 'group_width': 64, 'stride': 2, 'kernel_size': 3}],
    ['ResBlock', {'in_channels': 480, 'out_channels': 480, 'groups': 1, 'group_width': 128, 'stride': 1, 'kernel_size': 3}],
    ['ResBlock', {'in_channels': 480, 'out_channels': 480, 'groups': 1, 'group_width': 128, 'stride': 1, 'kernel_size': 3}],
    ['ResBlock', {'in_channels': 480, 'out_channels': 480, 'groups': 1, 'group_width': 144, 'stride': 1, 'kernel_size': 3}],
    ['ResBlock', {'in_channels': 480, 'out_channels': 480, 'groups': 1, 'group_width': 160, 'stride': 1, 'kernel_size': 3}],
    ['ViTBlock', {'head_num': 9, 'head_width': 64, 'hid_dims': 1472, 'input_resolution': [14, 14], 'in_channels': 480, 'out_channels': 448, 'stride': 2}],
    ['ViTBlock', {'head_num': 10, 'head_width': 64, 'hid_dims': 1664, 'input_resolution': [14, 14], 'in_channels': 448, 'out_channels': 448, 'stride': 1}],
    ['ViTBlock', {'head_num': 9, 'head_width': 64, 'hid_dims': 1152, 'input_resolution': [14, 14], 'in_channels': 448, 'out_channels': 448, 'stride': 1}],
    ['ViTBlock', {'head_num': 7, 'head_width': 64, 'hid_dims': 1216, 'input_resolution': [14, 14], 'in_channels': 448, 'out_channels': 448, 'stride': 1}],
    ['ViTBlock', {'head_num': 8, 'head_width': 64, 'hid_dims': 1280, 'input_resolution': [14, 14], 'in_channels': 448, 'out_channels': 448, 'stride': 1}],
    ['ViTBlock', {'head_num': 9, 'head_width': 64, 'hid_dims': 1984, 'input_resolution': [14, 14], 'in_channels': 448, 'out_channels': 448, 'stride': 1}],
    ['ViTBlock', {'head_num': 7, 'head_width': 64, 'hid_dims': 1920, 'input_resolution': [14, 14], 'in_channels': 448, 'out_channels': 448, 'stride': 1}],
    ['ViTBlock', {'head_num': 8, 'head_width': 64, 'hid_dims': 2048, 'input_resolution': [14, 14], 'in_channels': 448, 'out_channels': 448, 'stride': 1}],
    ['ViTBlock', {'head_num': 9, 'head_width': 64, 'hid_dims': 1984, 'input_resolution': [14, 14], 'in_channels': 448, 'out_channels': 448, 'stride': 1}],
    ['ViTBlock', {'head_num': 9, 'head_width': 64, 'hid_dims': 1664, 'input_resolution': [14, 14], 'in_channels': 448, 'out_channels': 448, 'stride': 1}],
    ['ViTBlock', {'head_num': 10, 'head_width': 64, 'hid_dims': 1536, 'input_resolution': [14, 14], 'in_channels': 448, 'out_channels': 448, 'stride': 1}],
    ['ViTBlock', {'head_num': 10, 'head_width': 64, 'hid_dims': 1600, 'input_resolution': [14, 14], 'in_channels': 448, 'out_channels': 448, 'stride': 1}],
    ['ViTBlock', {'head_num': 7, 'head_width': 64, 'hid_dims': 1984, 'input_resolution': [14, 14], 'in_channels': 448, 'out_channels': 448, 'stride': 1}],
    ['ViTBlock', {'head_num': 10, 'head_width': 64, 'hid_dims': 1536, 'input_resolution': [14, 14], 'in_channels': 448, 'out_channels': 448, 'stride': 1}],
    ['ViTBlock', {'head_num': 10, 'head_width': 64, 'hid_dims': 1728, 'input_resolution': [14, 14], 'in_channels': 448, 'out_channels': 448, 'stride': 1}],
    ['ViTBlock', {'head_num': 10, 'head_width': 64, 'hid_dims': 1344, 'input_resolution': [14, 14], 'in_channels': 448, 'out_channels': 448, 'stride': 1}],
    ['ViTBlock', {'head_num': 8, 'head_width': 64, 'hid_dims': 1792, 'input_resolution': [14, 14], 'in_channels': 448, 'out_channels': 448, 'stride': 1}],
    ['ViTBlock', {'head_num': 8, 'head_width': 64, 'hid_dims': 1600, 'input_resolution': [14, 14], 'in_channels': 448, 'out_channels': 448, 'stride': 1}],
    ['ViTBlock', {'head_num': 9, 'head_width': 64, 'hid_dims': 1152, 'input_resolution': [14, 14], 'in_channels': 448, 'out_channels': 448, 'stride': 1}],
    ['ViTBlock', {'head_num': 14, 'head_width': 64, 'hid_dims': 3328, 'input_resolution': [7, 7], 'in_channels': 448, 'out_channels': 800, 'stride': 2}],
    ['ViTBlock', {'head_num': 10, 'head_width': 64, 'hid_dims': 3968, 'input_resolution': [7, 7], 'in_channels': 800, 'out_channels': 800, 'stride': 1}],
    ['ViTBlock', {'head_num': 16, 'head_width': 64, 'hid_dims': 3456, 'input_resolution': [7, 7], 'in_channels': 800, 'out_channels': 800, 'stride': 1}],
    ['ViTBlock', {'head_num': 8, 'head_width': 64, 'hid_dims': 3456, 'input_resolution': [7, 7], 'in_channels': 800, 'out_channels': 800, 'stride': 1}],
    ['ViTBlock', {'head_num': 12, 'head_width': 64, 'hid_dims': 3584, 'input_resolution': [7, 7], 'in_channels': 800, 'out_channels': 800, 'stride': 1}]]


def _init_conv_weight(op, weight_init="kaiming_normal"):
    assert weight_init in [None, "kaiming_normal"]
    if weight_init is None:
        return
    if weight_init == "kaiming_normal":
        nn.init.kaiming_normal_(op.weight, mode="fan_out", nonlinearity="relu")
        if hasattr(op, "bias") and op.bias is not None:
            nn.init.constant_(op.bias, 0.0)


class Identity(nn.Module):
    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class BatchNorm2d(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(BatchNorm2d, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels, **kwargs)

    def forward(self, x):
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(in_channels, **kwargs)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class AvgPool(nn.Module):
    def __init__(self,
                 kernel_size,
                 in_channels,
                 out_channels,
                 stride=None,
                 **kwargs):
        super(AvgPool, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(
            kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.avgpool(x)


class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride, padding, **kwargs):
        super(MaxPool, self).__init__()
        self.maxpool = torch.nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.maxpool(x)


class ConvBlock(nn.Module):
    # conv-bn-act block
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=0,
                 groups=1,
                 bias=True,
                 bn=True,
                 act='relu',
                 zero_gamma=False):
        super(ConvBlock, self).__init__()

        conv = nn.Conv2d(in_channels, out_channels,
                         kernel_size=kernel_size, stride=stride,
                         padding=padding, groups=groups, bias=bias)
        _init_conv_weight(conv)
        self.conv = conv
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

        if zero_gamma:
            nn.init.constant_(self.bn.weight, 0.0)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        if x.dtype != self.conv.weight.dtype and self.conv.weight.dtype == torch.float16:
            x = x.half()
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ConvBlockX(nn.Module):
    # conv-bn-act block
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 bias=False,
                 bn=True,
                 act='relu',
                 zero_gamma=False):
        super(ConvBlockX, self).__init__()

        conv = nn.Conv2d(in_channels, out_channels,
                         kernel_size=kernel_size, stride=stride,
                         padding=kernel_size//2, groups=groups, bias=bias)
        _init_conv_weight(conv)
        self.conv = conv
        self.bn = BatchNorm2d(out_channels) if bn else None

        if zero_gamma:
            nn.init.constant_(self.bn.weight, 0.0)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        if x.dtype != self.conv.weight.dtype and self.conv.weight.dtype == torch.float16:
            x = x.half()
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModule, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Conv2d(channel, channel // reduction,
                      kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel,
                      kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excite(y)
        return x * y


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 group_width,
                 stride,
                 kernel_size=3,
                 use_se=False,
                 groups=1,
                 act='relu',
                 drop_path=0.):
        super(ResBlock, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_width = group_width
        mid_channels = group_width * groups
        self.mid_channels = mid_channels
        self.stride = stride
        self.groups = groups
        self.use_se = use_se

        # 1x1 conv
        self.conv_bn_relu1 = ConvBlock(in_channels=in_channels,
                                       out_channels=mid_channels, bias=False,
                                       kernel_size=1, stride=1, padding=0, act='relu')
        # 3x3 conv
        self.conv_bn_relu2 = ConvBlock(in_channels=mid_channels,
                                       out_channels=mid_channels, bias=False,
                                       kernel_size=kernel_size, stride=stride,
                                       padding=kernel_size//2, groups=groups, act='relu')
        # se module
        self.se = SEModule(mid_channels) if use_se else None
        # 1x1 conv
        self.conv_bn_relu3 = ConvBlock(in_channels=mid_channels,
                                       out_channels=out_channels, bias=False,
                                       kernel_size=1, stride=1, padding=0, act='none')

        self.res_flag = (in_channels == out_channels) and (stride == 1)
        if self.res_flag:
            self.downsample = Identity()
        else:
            self.downsample = ConvBlock(in_channels=in_channels,
                                        out_channels=out_channels, bias=False,
                                        kernel_size=1, stride=stride, padding=0, act='none')
        self.final_act = nn.ReLU(inplace=True)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        res = x
        sc = self.downsample(x)
        res = self.conv_bn_relu1(res)
        res = self.conv_bn_relu2(res)
        if self.se is not None:
            res = self.se(res)
        res = self.conv_bn_relu3(res)
        out = self.final_act(sc + self.drop_path(res))
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer="GELU", drop=0., layer_scale_init_value=1e-6):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.fc1 = nn.Conv2d(in_features, hidden_features,
                             kernel_size=1, stride=1, bias=True)
        assert hasattr(nn, act_layer)
        self.act = getattr(nn, act_layer)()
        self.fc2 = nn.Conv2d(hidden_features, out_features,
                             kernel_size=1, stride=1, bias=True)
        # self.drop = nn.Dropout(drop, inplace=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_features)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        if self.gamma is not None:
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            if self.gamma.dtype != x.dtype and x.dtype == torch.float16:
                x = self.gamma.half() * x
            else:
                x = self.gamma * x
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x


class Attention(nn.Module):
    """ Multi-head Self-Attention (MSA) module with relative position bias"""

    def __init__(self, in_dim, out_dim, input_resolution, num_heads, head_dim, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., use_relative_position_bias=True, layer_scale_init_value=1e-6,
                 stride=1, use_pool_attn=False, pool_norm_layer="LayerNorm"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mid_dim = head_dim * num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.use_relative_position_bias = use_relative_position_bias
        self.use_pool_attn = use_pool_attn
        pos_heads = num_heads
        if self.use_relative_position_bias:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * input_resolution[0] - 1) * (2 * input_resolution[1] - 1), pos_heads))  # 2*H-1 * 2*W-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.input_resolution[0])
            coords_w = torch.arange(self.input_resolution[1])
            coords = torch.stack(torch.meshgrid(
                [coords_h, coords_w]))  # 2, H, W
            coords_flatten = torch.flatten(coords, 1)  # 2, H*W
            # 2, H*W, H*W
            relative_coords = coords_flatten[:, :,
                                             None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(
                1, 2, 0).contiguous()  # H*W, H*W, 2
            # shift to start from 0
            relative_coords[:, :, 0] += self.input_resolution[0] - 1
            relative_coords[:, :, 1] += self.input_resolution[1] - 1
            relative_coords[:, :, 0] *= 2 * self.input_resolution[1] - 1
            relative_position_index = relative_coords.sum(-1)  # H*W, H*W
            self.register_buffer("relative_position_index",
                                 relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        # self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.q = nn.Conv2d(in_dim, self.mid_dim, kernel_size=1, bias=qkv_bias)
        self.kv = nn.Conv2d(in_dim, self.mid_dim * 2,
                            kernel_size=1, bias=qkv_bias)
        if use_pool_attn:
            self.pool_q = nn.Conv2d(head_dim, head_dim, \
                                    kernel_size=3, stride=stride, \
                                    padding=1, groups=head_dim, bias=False)
            self.pool_norm_q = globals()[pool_norm_layer](head_dim)
            self.pool_k = nn.Conv2d(head_dim, head_dim, \
                                    kernel_size=3, stride=stride, \
                                    padding=1, groups=head_dim, bias=False)
            self.pool_norm_k = globals()[pool_norm_layer](head_dim)
            self.pool_v = nn.Conv2d(head_dim, head_dim, \
                                    kernel_size=3, stride=stride, \
                                    padding=1, groups=head_dim, bias=False)
            self.pool_norm_v = globals()[pool_norm_layer](head_dim)
        #self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(self.mid_dim, out_dim, kernel_size=1)
        #self.proj_drop = nn.Dropout(proj_drop)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        q = self.q(x).reshape(B, self.num_heads, self.head_dim,
                              N).permute(0, 1, 3, 2).contiguous()  # B, nH, N, C//nH
        kv = self.kv(x).reshape(B, 2, self.num_heads, self.head_dim, -
                                1).permute(1, 0, 2, 4, 3).contiguous()  # 2, B, nH, N, C//nH
        # make torchscript happy (cannot use tensor as tuple)
        k, v = kv[0], kv[1]
        if self.use_pool_attn:
            q = q.reshape(B*self.num_heads, H, W, self.head_dim).permute(0, 3, 1, 2).contiguous()
            q = self.pool_norm_q(self.pool_q(q))
            H_, W_ = q.shape[-2:]
            N = H_ * W_
            q = q.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2).contiguous()
            k = k.reshape(B*self.num_heads, H, W, self.head_dim).permute(0, 3, 1, 2).contiguous()
            k = self.pool_norm_k(self.pool_k(k))
            k = k.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2).contiguous()
            v = v.reshape(B*self.num_heads, H, W, self.head_dim).permute(0, 3, 1, 2).contiguous()
            v = self.pool_norm_v(self.pool_v(v))
            v = v.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2).contiguous()
            H, W = H_, W_
        N_ = k.shape[-2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if torch.isinf(attn).any():
            clamp_value = torch.finfo(attn.dtype).max - 1000.0
            attn = torch.clamp(attn, min=-clamp_value, max=clamp_value)

        if self.use_relative_position_bias:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(
                -1)].view(N, N_, -1)  # H*W, H*W,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, H*W, H*W
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        out = (attn @ v).permute(0, 1, 3,
                                 2).contiguous().reshape(B, self.mid_dim, H, W)
        if self.use_pool_attn:
            out = out + q.permute(0, 1, 3, 2).contiguous().reshape(B, self.mid_dim, H, W)
        out = self.proj(out)

        if self.gamma is not None:
            out = out.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            if self.gamma.dtype != out.dtype and out.dtype == torch.float16:
                out = self.gamma.half() * out
            else:
                out = self.gamma * out
            out = out.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return out


class ViTBlock(nn.Module):
    def __init__(self, head_num, head_width, hid_dims, input_resolution, in_channels, out_channels, stride, out_norm=False,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer="GELU",
                 norm_layer="LayerNorm", use_relative_position_bias=True, use_avgdown=False, skip_lam=1.,
                 layer_scale_init_value=1e-6, use_pool_attn=False):
        super(ViTBlock, self).__init__()
        self.head_num = head_num
        self.hid_dims = hid_dims
        self.input_resolution = input_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_avgdown = use_avgdown
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.out_norm = out_norm
        self.skip_lam = skip_lam
        self.use_pool_attn = use_pool_attn

        self.pool_skip = None
        self.proj = None
        if stride != 1 or in_channels != out_channels:
            if use_pool_attn:
                self.pool_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            else:
                if stride != 1 and use_avgdown:
                    self.proj = nn.Sequential(OrderedDict([
                        ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2)),
                        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1))]))
                else:
                    self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=stride, stride=stride)

        if self.pool_skip is not None:
            self.norm1 = globals()[self.norm_layer](in_channels)
            self.attn = Attention(in_channels, out_channels, input_resolution, head_num, head_width, qkv_bias, qk_scale, attn_drop=attn_drop,
                                  proj_drop=drop, use_relative_position_bias=use_relative_position_bias,
                                  layer_scale_init_value=layer_scale_init_value,
                                  stride=stride, use_pool_attn=True, pool_norm_layer=norm_layer)
        else:
            self.norm1 = globals()[self.norm_layer](out_channels)
            self.attn = Attention(out_channels, out_channels, input_resolution, head_num, head_width, qkv_bias, qk_scale, attn_drop=attn_drop,
                                  proj_drop=drop, use_relative_position_bias=use_relative_position_bias,
                                  layer_scale_init_value=layer_scale_init_value,
                                  stride=1, use_pool_attn=False, pool_norm_layer=norm_layer)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = globals()[norm_layer](out_channels)
        mlp_hidden_dim = hid_dims
        self.mlp = Mlp(in_features=out_channels, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, layer_scale_init_value=layer_scale_init_value)
        self.out_norm = globals()[norm_layer](
            out_channels) if out_norm else None

    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x)
        if self.pool_skip is not None:
            x = self.pool_skip(x) + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        else:
            x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        if self.out_norm is not None:
            x = self.out_norm(x)
        return x


class hybridCCTT(nn.Module):
    def __init__(
            self,
            arch_cfg,
            drop_path_rate=0.3,
            vit_block_norm='BatchNorm2d',
            vit_block_act='ReLU',
            res_block_use_drop_path=False,
            global_pool='avg',
            num_classes=1000,
            features_only=False,
            name=None,
            **kwargs):
        super(hybridCCTT, self).__init__()

        self.name = name
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.features_only = features_only
        self.out_channels = []

        dpr = [x.item() for x in torch.linspace(
            0.0, drop_path_rate, len(arch_cfg))]
        layers = []
        for idx, [layer_name, params] in enumerate(arch_cfg):
            if "ViT" in layer_name:
                params.update({'drop_path': dpr[idx]})
                params.update({'norm_layer': vit_block_norm})
                params.update({'act_layer': vit_block_act})
                if idx == len(arch_cfg) - 1:
                    params.update({'out_norm': True})
            elif "ResBlock" in layer_name:
                if res_block_use_drop_path:
                    params.update({'drop_path': dpr[idx]})
                else:
                    params.update({'drop_path': 0.})
            layers.append(globals()[layer_name](**params))
            self.out_channels.append(arch_cfg[idx][-1]['out_channels'])
        self.out_indices = kwargs.pop('out_indices', None)
        if self.out_indices is None:
            self.out_indices = [i for i in range(len(arch_cfg))]
        self.last_out_channels = arch_cfg[-1][-1]['out_channels']

        self.layers = nn.ModuleList(layers)
        if self.global_pool == 'avg':
            self.avgpool = torch.nn.AvgPool2d(kernel_size=7, stride=None)
        elif self.global_pool is not None and self.global_pool != '':
            raise ValueError(
                f'Unsupported global pool type: {self.global_pool}.')
        if self.num_classes > 0:
            self.fc = nn.Linear(self.last_out_channels, self.num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        # kaiming normal for Bottleneck blocks; trunc_normal for ViT blocks
        for n, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                if 'ViTBlock' in n:
                    trunc_normal_(m.weight, std=.02)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # N = x.size()[0]
        outs = OrderedDict()
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx in self.out_indices:
                outs[idx] = x
        if self.features_only:
            return list(outs.values())

        if self.global_pool == 'avg':
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            if self.num_classes > 0:
                x = self.fc(x)
        return x


@register_model
def hybridCCTT_1(pretrained=False, **kwargs):
    model_name = 'hybridCCTT_1'
    pretrained_config, model_config = get_configs(**kwargs)

    model = hybridCCTT(ARCH_CFG_1, **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
    return model


@register_model
def hybridCCTT_2(pretrained=False, **kwargs):
    model_name = 'hybridCCTT_2'
    pretrained_config, model_config = get_configs(**kwargs)

    model = hybridCCTT(ARCH_CFG_2, **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
    return model


@register_model
def hybridCCTT_3(pretrained=False, **kwargs):
    model_name = 'hybridCCTT_3'
    pretrained_config, model_config = get_configs(**kwargs)

    model = hybridCCTT(ARCH_CFG_3, **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
    return model


@register_model
def hybridCCTT_4(pretrained=False, **kwargs):
    model_name = 'hybridCCTT_4'
    pretrained_config, model_config = get_configs(**kwargs)

    model = hybridCCTT(ARCH_CFG_4, **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
    return model
