#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Huang Wenguan (huangwenguan@bytedance.com)
Date: 2020-11-12 11:52:23
LastEditTime: 2020-11-18 15:40:42
LastEditors: Huang Wenguan
Description: 和模型load相关的函数
'''

import os
import io
import math
import warnings
import collections
from prettytable import PrettyTable

import torch
from torch import nn

from fex.utils.distributed import rank_zero_info
from fex.utils.torch_io import load as torch_io_load
from fex.utils.hdfs_io import hopen
from fex import _logger as log


USE_PTX_TRANSFORMER = bool(int(os.getenv('FEX_USE_PTX_TSFM', '0')))


def rearange_pos_id(pos_ids, target_length, *args, **kwargs):
    return torch.arange(target_length).expand((1, -1))


def interpolate_pos_encoding(pos_embed, patch_size, w, h, has_class=True, *args, **kwargs):
    """
    pos_embed: [n_patch, dim]
    target_shape
    """
    #pos_embed = pos_embed.squeeze(0)
    N, dim = pos_embed.shape
    if has_class:
        N = N - 1
        class_pos_embed = pos_embed[:1]
        patch_pos_embed = pos_embed[1:]
    else:
        patch_pos_embed = pos_embed

    w0 = w // patch_size
    h0 = h // patch_size
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        mode='bicubic',
    )
    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(-1, dim)
    if has_class:
        return torch.cat((class_pos_embed, patch_pos_embed), dim=0)  # .unsqueeze(0)
    else:
        return patch_pos_embed


def truncate1(param, len, *args, **kwargs):
    return param[:, :len]


def load_pretrained_state_dict(model, state_dict):
    """Load state dict of pretrained model
    Args:
        state_dict (dict): state dict to load
    """

    if USE_PTX_TRANSFORMER:
        state_dict = conv_resblocks_state_to_ptx(state_dict)
    else:
        state_dict = conv_resblocks_state_from_ptx(state_dict)

    new_state_dict = model.state_dict()
    miss_keys = []
    for k in new_state_dict.keys():
        if k in state_dict.keys():
            new_state_dict[k] = state_dict[k]
        else:
            miss_keys.append(k)
    if len(miss_keys) > 0:
        warnings.warn('miss keys: {}'.format(miss_keys))
    model.load_state_dict(new_state_dict)


def smart_load_pretrained_state_dict(model, state_dict):
    """ load 模型的时候还会打印一下哪些参数有哪些参数无，相比上面那个好看一点 """

    if USE_PTX_TRANSFORMER:
        state_dict = conv_resblocks_state_to_ptx(state_dict)
    else:
        state_dict = conv_resblocks_state_from_ptx(state_dict)

    parsed_state_dict = {}
    non_match_keys = []
    pretrained_keys = []
    for k, v in state_dict.items():
        if k in model.state_dict():
            parsed_state_dict[k] = v
            pretrained_keys.append(k)
        else:
            non_match_keys.append(k + ':' + str(v.shape))
            # raise ValueError('failed to match key of state dict smartly!')

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank == 0:
        table = PrettyTable(
            ['Layer Name', 'Weight Shape', 'Data Type', 'Pretrain'])
        for k, v in model.named_parameters():
            table.add_row([k, v.shape, v.dtype, str(
                k in pretrained_keys)])
        table.align = 'l'
        log.info(
            '\n###### Parameters ######\n{}'.format(table.get_string()))
        log.info("\n###### Not matched keys ######\n{}".format(
            '\n'.join(non_match_keys) + '\n'))

    new_state_dict = model.state_dict()
    new_state_dict.update(parsed_state_dict)
    model.load_state_dict(new_state_dict)


def load_from_pretrain(model, pretrain_paths, prefix_changes=[]):
    """
    先看有没有partial_pretrain，如果有就直接加载。
    否则就分别看语言模型和image模型是否要加载预训练
    语义：
    1. {a}->{b}  命中{a}前缀的，都替换成{b}
    2. {a}->{b},{c}  生成两份参数
    3. {a}->{b}:{func}({params})
    """

    pretrain_state_dict_parsed = {}

    def _load_from_pretrain_one_file(pretrain_path):
        pretrain_state_dict = torch_io_load(
            pretrain_path, map_location=lambda storage, loc: storage)
        pretrain_state_dict = pretrain_state_dict['state_dict'] if 'state_dict' in pretrain_state_dict else pretrain_state_dict
        prefix_change = [prefix_change.split('->') for prefix_change in prefix_changes]  # 规则1，根据 -> 拆分前后

        # 配置函数
        prefix_to_func = {}
        for i, (porigin, prefix) in enumerate(prefix_change):
            if ':' in prefix:  # 规则3
                new_prefixs, func_name = prefix.split(':')
                if '(' in func_name:
                    func_name, params = func_name.split('(')
                    params = params[:-1]  # 最后一个 )
                    params = [eval(p) for p in params.split(',')]
                else:
                    params = []
                func = eval(func_name)
                prefix_change[i] = [porigin, new_prefixs]
                prefix_to_func[new_prefixs] = (func, params)

        for k, v in pretrain_state_dict.items():
            if k.startswith('module.'):
                k = k.replace('module.', '')
            no_match = True
            for pretrain_prefix, new_prefixs in prefix_change:
                if k.startswith(pretrain_prefix):
                    for new_prefix in new_prefixs.split(','):  # 规则2 , 支持一对多的映射关系。举例： a->b,c 则是将a 同时映射到b,c
                        kk = new_prefix + k[len(pretrain_prefix):]
                        if new_prefixs in prefix_to_func:
                            func, params = prefix_to_func[new_prefixs]
                            v = func(v, *params)
                        pretrain_state_dict_parsed[kk] = v
                    no_match = False
                    break
            if no_match:
                pretrain_state_dict_parsed[k] = v

    # partial load pretrain state dict
    if isinstance(pretrain_paths, list):
        for i, pretrain_path in enumerate(pretrain_paths):
            if pretrain_path != "":
                rank_zero_info('loading from pretrain path %s: %s ' % (i, pretrain_path))
                _load_from_pretrain_one_file(pretrain_path)
        smart_load_pretrained_state_dict(model, pretrain_state_dict_parsed)
    elif isinstance(pretrain_paths, str):
        pretrain_path = pretrain_paths
        if pretrain_path != "":
            rank_zero_info('loading from pretrain path %s ' % pretrain_path)
            _load_from_pretrain_one_file(pretrain_path)
        smart_load_pretrained_state_dict(model, pretrain_state_dict_parsed)
    else:
        rank_zero_info("Warning: no pretrained model found, training from scratch!!!")
        raise Exception("pretrained model is not load in the right way")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    if vocab_file.startswith('hdfs://'):
        with hopen(vocab_file, "r") as reader:
            accessor = io.BytesIO(reader.read())
            while True:
                token = accessor.readline()
                token = token.decode('utf-8')  # 要解码使得数据接口类型一致
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
            del accessor
            return vocab
    else:
        with open(vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
            return vocab


def conv_resblocks_state_to_ptx(state_dict):
    for k in list(state_dict.keys()):
        if k.endswith('.attn.in_proj_weight'):
            proj_q_w, proj_k_w, proj_v_w = state_dict[k].split(state_dict[k].size(0) // 3)
            del state_dict[k]
            state_dict[k.replace('.attn.in_proj_weight', '.attn.proj_q.weight')] = proj_q_w
            state_dict[k.replace('.attn.in_proj_weight', '.attn.proj_k.weight')] = proj_k_w
            state_dict[k.replace('.attn.in_proj_weight', '.attn.proj_v.weight')] = proj_v_w
        elif k.endswith('.attn.in_proj_bias'):
            proj_q_b, proj_k_b, proj_v_b = state_dict[k].split(state_dict[k].size(0) // 3)
            del state_dict[k]
            state_dict[k.replace('.attn.in_proj_bias', '.attn.proj_q.bias')] = proj_q_b
            state_dict[k.replace('.attn.in_proj_bias', '.attn.proj_k.bias')] = proj_k_b
            state_dict[k.replace('.attn.in_proj_bias', '.attn.proj_v.bias')] = proj_v_b
        elif k.endswith('.attn.out_proj.weight'):
            state_dict[k.replace('.attn.out_proj.weight', '.proj.weight')] = state_dict.pop(k)
        elif k.endswith('.attn.out_proj.bias'):
            state_dict[k.replace('.attn.out_proj.bias', '.proj.bias')] = state_dict.pop(k)
        elif k.endswith('.ln_1.weight'):
            state_dict[k.replace('.ln_1.weight', '.norm1.weight')] = state_dict.pop(k)
        elif k.endswith('.ln_1.bias'):
            state_dict[k.replace('.ln_1.bias', '.norm1.bias')] = state_dict.pop(k)
        elif k.endswith('.ln_2.weight'):
            state_dict[k.replace('.ln_2.weight', '.norm2.weight')] = state_dict.pop(k)
        elif k.endswith('.ln_2.bias'):
            state_dict[k.replace('.ln_2.bias', '.norm2.bias')] = state_dict.pop(k)
        elif k.endswith('.mlp.c_fc.weight'):
            state_dict[k.replace('.mlp.c_fc.weight', '.pwff.fc1.weight')] = state_dict.pop(k)
        elif k.endswith('.mlp.c_fc.bias'):
            state_dict[k.replace('.mlp.c_fc.bias', '.pwff.fc1.bias')] = state_dict.pop(k)
        elif k.endswith('.mlp.c_proj.weight'):
            state_dict[k.replace('.mlp.c_proj.weight', '.pwff.fc2.weight')] = state_dict.pop(k)
        elif k.endswith('.mlp.c_proj.bias'):
            state_dict[k.replace('.mlp.c_proj.bias', '.pwff.fc2.bias')] = state_dict.pop(k)
    return state_dict


def conv_resblocks_state_from_ptx(state_dict):
    for k in list(state_dict.keys()):
        if '.resblocks.' not in k:
            continue
        if k.endswith('.attn.proj_q.weight'):
            proj_q_w = state_dict.pop(k)
            proj_k_w = state_dict.pop(k.replace('.attn.proj_q.weight', '.attn.proj_k.weight'))
            proj_v_w = state_dict.pop(k.replace('.attn.proj_q.weight', '.attn.proj_v.weight'))
            state_dict[k.replace('.attn.proj_q.weight', '.attn.in_proj_weight')] = torch.cat([
                proj_q_w,
                proj_k_w,
                proj_v_w,
            ])
        elif k.endswith('.attn.proj_q.bias'):
            proj_q_b = state_dict.pop(k)
            proj_k_b = state_dict.pop(k.replace('.attn.proj_q.bias', '.attn.proj_k.bias'))
            proj_v_b = state_dict.pop(k.replace('.attn.proj_q.bias', '.attn.proj_v.bias'))
            state_dict[k.replace('.attn.proj_q.bias', '.attn.in_proj_bias')] = torch.cat([
                proj_q_b,
                proj_k_b,
                proj_v_b,
            ])
        elif k.endswith('.proj.weight'):
            state_dict[k.replace('.proj.weight', '.attn.out_proj.weight')] = state_dict.pop(k)
        elif k.endswith('.proj.bias'):
            state_dict[k.replace('.proj.bias', '.attn.out_proj.bias')] = state_dict.pop(k)
        elif k.endswith('.norm1.weight'):
            state_dict[k.replace('.norm1.weight', '.ln_1.weight')] = state_dict.pop(k)
        elif k.endswith('.norm1.bias'):
            state_dict[k.replace('.norm1.bias', '.ln_1.bias')] = state_dict.pop(k)
        elif k.endswith('.norm2.weight'):
            state_dict[k.replace('.norm2.weight', '.ln_2.weight')] = state_dict.pop(k)
        elif k.endswith('.norm2.bias'):
            state_dict[k.replace('.norm2.bias', '.ln_2.bias')] = state_dict.pop(k)
        elif k.endswith('.pwff.fc1.weight'):
            state_dict[k.replace('.pwff.fc1.weight', '.mlp.c_fc.weight')] = state_dict.pop(k)
        elif k.endswith('.pwff.fc1.bias'):
            state_dict[k.replace('.pwff.fc1.bias', '.mlp.c_fc.bias')] = state_dict.pop(k)
        elif k.endswith('.pwff.fc2.weight'):
            state_dict[k.replace('.pwff.fc2.weight', '.mlp.c_proj.weight')] = state_dict.pop(k)
        elif k.endswith('.pwff.fc2.bias'):
            state_dict[k.replace('.pwff.fc2.bias', '.mlp.c_proj.bias')] = state_dict.pop(k)
    return state_dict


def device_mapping(cuda_device: int):
    """
    In order to `torch.load()` a GPU-trained model onto a CPU (or specific GPU),
    you have to supply a `map_location` function. Call this with
    the desired `cuda_device` to get the function that `torch.load()` needs.
    """

    def inner_device_mapping(storage: torch.Storage, location) -> torch.Storage:  # pylint: disable=unused-argument
        if cuda_device >= 0:
            return storage.cuda(cuda_device)
        else:
            return storage

    return inner_device_mapping
