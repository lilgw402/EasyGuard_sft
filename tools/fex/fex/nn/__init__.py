#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Huang Wenguan (huangwenguan@bytedance.com)
Date: 2020-11-09 20:35:09
LastEditTime: 2020-11-13 19:08:55
LastEditors: Huang Wenguan
Description:
'''

from .backbone.with_ptx import USE_PTX_TRANSFORMER

from .backbone.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .backbone.shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
from .backbone.swin_transformer import SwinTransformer, swin_transformer_tiny, swin_transformer_small, swin_transformer_base

if not USE_PTX_TRANSFORMER:
    from .backbone.vit import VisualTransformer, visual_transformer_B32, visual_transformer_B16
    from .backbone.albert import ALBert
else:
    from .backbone.vit_v2 import VisualTransformer, visual_transformer_B32, visual_transformer_B16
    from .backbone.albert_v2 import ALBert

from .backbone.albertv import ALBertV
from .backbone.vlbert import VisualLinguisticBert
from .backbone.falbert import FrameALBert
from .backbone.vision_deberta_encoder import VisionDeberta
from .backbone.albef import ALBEF

from .backbone.swin_vidt import swin_base_win7_vidt

BACKBONE_MAP = {
    'falbert': FrameALBert,
    'vlbert': VisualLinguisticBert,
    'VisionDeberta': VisionDeberta,
    'ALBEF': ALBEF
}
