# -*- coding: utf-8 -*-
'''
Created on Jan-12-21 18:04
__init__.py
Description:
'''
from .convert_network import apex_convert_network
from .faster_transformer import cast_to_fast_transformer, cast_to_fast_mha
from .summary_parameters import summary_parameters

from .scriptable_text_utils import load_vocab, MultiDomainConcator, \
    ScriptBertTokenizer, clip_pad_1d, clip_pad_2d, tail_first_truncate, make_mask_1d
