# -*- coding: utf-8 -*-
'''
Created on Jan-12-21 17:57
convert_network_to_half.py
@author: liuzhen.nlp
Description: 
'''
import collections
import functools

import torch


__all__ = ['apex_convert_network']


def _apex_to_type(dtype, t):
    if isinstance(t, torch.Tensor):
        if t.is_floating_point():
            return t.to(dtype)
        return t
    else:
        # Trust the user's custom batch type, that's all I can do here.
        return t.to(dtype)


def _apex_apply(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    elif isinstance(value, str):
        return value
    elif isinstance(value, collections.abc.Mapping):
        return {_apex_apply(k, fn): _apex_apply(v, fn) for k, v in value.items()}
    elif isinstance(value, collections.abc.Iterable):
        return type(value)(_apex_apply(v, fn) for v in value)
    elif hasattr(value, 'to'):  # Allow handling of custom batch classes
        return fn(value)
    else:
        return value


def _apex_convert_module(module, dtype):
    """
    Converts a module's immediate parameters and buffers to dtype.
    """
    for param in module.parameters(recurse=False):
        if param is not None:
            if param.data.dtype.is_floating_point:
                param.data = param.data.to(dtype=dtype)
            if param._grad is not None and param._grad.data.dtype.is_floating_point:
                param._grad.data = param._grad.data.to(dtype=dtype)

    for buf in module.buffers(recurse=False):
        if buf is not None and buf.data.dtype.is_floating_point:
            buf.data = buf.data.to(dtype=dtype)


def _apex_convert_network(network, dtype):
    """
    Converts a network's parameters and buffers to dtype.
    """
    for module in network.modules():

        if hasattr(module, '_fixed_dtype'):
            continue

        if isinstance(module, torch.nn.LayerNorm):

            def patch_forward(self):
                old_forward = self.forward

                def new_forward(x):
                    y = x.to(dtype=self.weight.dtype)
                    z = old_forward(y)
                    return z.to(dtype=x.dtype)

                return new_forward

            module.forward = patch_forward(module)

        if (isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True) or \
                isinstance(module, torch.nn.LayerNorm):
            continue
        _apex_convert_module(module, dtype)
        if isinstance(module, torch.nn.RNNBase) or isinstance(module, torch.nn.modules.rnn.RNNBase):
            module.flatten_parameters()
    return network


def apex_convert_network(model, dtype, hack_forward=False, cast_output=False):
    model = _apex_convert_network(model, dtype)
    if not hack_forward:
        return model

    input_caster = functools.partial(_apex_to_type, dtype)
    if cast_output:
        output_caster = functools.partial(_apex_to_type, dtype)
    else:
        output_caster = functools.partial(_apex_to_type, torch.float32)

    def patch_forward(old_forward):

        def new_forward(*args, **kwargs):
            output = old_forward(
                *_apex_apply(args, input_caster), **_apex_apply(kwargs, input_caster))
            return _apex_apply(output, output_caster)

        return new_forward

    model.forward = patch_forward(model.forward)
    return model
