# -*- coding: utf-8 -*-

import torch
from fex.utils.load import load_from_pretrain
from fex.trace.trace_engine import trace_net_wrapper
from fex.trace.utils import apex_convert_network, summary_parameters, cast_to_fast_transformer, cast_to_fast_mha


class ModelConvertToFTAndFP16:
    def __init__(self,
                 use_fp16: bool = False,
                 use_ft: bool = False,
                 return_all_hidden_states: bool = False):
        self.use_fp16 = use_fp16
        self.use_ft = use_ft
        self.return_all_hidden_states = return_all_hidden_states

    def convert_model(self, model: torch.nn.Module):
        if hasattr(model, 'trace'):
            model.forward = model.trace

        if not hasattr(model, 'forward'):
            raise ValueError(
                "Model not have trace or forward method, please check ! ")

        return convert_to_ft_and_fp16(model=model,
                                      use_fp16=self.use_fp16,
                                      use_ft=self.use_ft,
                                      return_all_hidden_states=self.return_all_hidden_states)


def convert_to_ft_and_fp16(model,
                           use_fp16: bool = False,
                           use_ft: bool = False,
                           use_fast_mha: bool = False,
                           return_all_hidden_states: bool = False,
                           use_deberta: bool = False):
    if use_fp16:
        model = apex_convert_network(model,
                                     dtype=torch.float16,
                                     hack_forward=True)
    if use_ft:
        model = cast_to_fast_transformer(
            model,
            fp16=use_ft,
            return_all_hidden_states=return_all_hidden_states)

    if use_fast_mha:
        model = cast_to_fast_mha(
            model, fp16=use_fp16
        )

    if use_deberta:
        from ptx.model.deberta.ft_inference import replace_deberta_ft_in_model
        replace_deberta_ft_in_model(model, fp16=True, return_all_hidden_states=True, rm_padding=False)
        print(model)

    model.eval()
    # summary_parameters(model)

    print('finish convert model to ft and fp16.')
    return model
