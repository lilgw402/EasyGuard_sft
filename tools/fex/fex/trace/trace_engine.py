# -*- coding: utf-8 -*-
"""
Created on Jan-12-21 17:03
trace_engine.py
@author: liuzhen.nlp
Description:
"""
from fex.trace.scriptable_data_parse_engine import ScriptableDataParseEngine
from fex.trace.utils import apex_convert_network, summary_parameters, cast_to_fast_transformer
from fex.core.net import Net
from torch import nn
import torch
import numpy as np
from typing import Dict, List, Tuple


def trace_net_wrapper(net: Net):
    class TraceNet(net):
        """
        torch.nn.Module 只有forward function可以被trace成model用来serving.
        TraceNet继承自自定义model, 重写了forward方法.
        forward直接调用Net中的trace方法, trace前需要实现Net.trace().
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.traced_wrapper = True

        def forward(self, *args, **kwargs):
            return self.trace(*args, **kwargs)

    return TraceNet


class TracerEngine(nn.Module):
    """[summary]
    """

    def __init__(self,
                 net: Net,
                 data_parse_engine: ScriptableDataParseEngine,
                 device_id: int = 0,
                 use_fp16: bool = False,
                 use_ft: bool = False,
                 is_script: bool = True,
                 is_remove_padding: bool = False,
                 return_all_hidden_states: bool = False):
        """[summary]

        Args:
            net (Net): [description]
            data_parse_engine (ScriptableDataParseEngine): [description]
            is_remove_padding (bool, optional): [description]. Defaults to False.
            device_id (int, optional): [description]. Defaults to 0.
            use_fp16 (bool, optional): [description]. Defaults to False.
            use_ft (bool, optional): [description]. Defaults to False.
            is_script (bool, optional): [description]. Defaults to True.
        """
        super().__init__()
        self.traced_data_parse = data_parse_engine
        print('finish tracing preprocess pipeline.')

        if not getattr(net, "traced_wrapper", None):
            raise ValueError(
                "Net not wraper with function trace_net_wrapper, please check!")

        self.model_device = torch.device(
            'cuda', device_id) if device_id >= 0 else torch.device('cpu')
        self.model = net.to(self.model_device)
        self.model.eval()
        self.use_fp16 = use_fp16
        self.use_ft = use_ft

        if use_fp16:
            self.model = apex_convert_network(
                self.model, dtype=torch.float16, hack_forward=True)

        if use_ft:
            self.model = cast_to_fast_transformer(self.model,
                                                  fp16=use_fp16,
                                                  is_remove_padding=is_remove_padding,
                                                  return_all_hidden_states=return_all_hidden_states)

        summary_parameters(self.model)
        print('finish load module weights.')

        if is_script:
            self.traced_data_parse = torch.jit.script(data_parse_engine)
            self.model = torch.jit.trace(
                self.model, self.mock_input(), check_trace=True)
        print('finish tracing model.')

    def move_to_device(self, tensor_dict: Dict[str, torch.Tensor]):
        tensor_cuda_dict: Dict[str, torch.Tensor] = {}
        for key, value in tensor_dict.items():
            tensor_cuda_dict[key] = value.to(self.model_device)
        return tensor_cuda_dict

    def mock_input(self) -> Tuple[torch.Tensor]:
        """ mock_input方法的主要作用是产生mock_data, 该mock_data会在torch.jit.trace的时候用.

        Returns:
            Dict[str, torch.Tensor]: 模型输入的所有字段, key需要和Net.trace()方法中的参数名一致.

        Examples:
            .. code-block:: python
            def mock_input(self):
                batch_size = 8
                field_names = ['titles', 'user_nicknames', 'challenges']

                queries = ["hello world"] * batch_size

                doc_items: List[Dict[str, str]] = []
                doc_info: Dict[str, str] = {}

                for field in field_names:
                    doc_info[field] = ["hello world"]
                doc_items.extend([doc_info] * batch_size)

                vision_embed = torch.as_tensor(np.random.rand(8, 128)).float()
                vision_embed_list = [vision_embed for i in range(batch_size)]

                tensor_dict = self.traced_data_parse(
                    queries, doc_items, vision_embed_list)
                tensor_cuda_dict = self.move_to_device(tensor_dict)
                return tensor_cuda_dict["input_ids"], tensor_cuda_dict["segment_ids"],
                       tensor_cuda_dict["input_masks"], tensor_cuda_dict["vision_embeds"], tensor_cuda_dict["vision_masks"]

        """
        raise ValueError(
            "Mock_input method not implement in TraceEngine, please check! ")

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        """ forward function 是TraceEngine默认编译的方法,
            该方法主要是包含怎么做数据的预处理、做model的forward、对forward的结果做后处理.

        Examples:
            .. code-block:: python
            @torch.no_grad()
            def forward(self, queries: List[str],
                        titles: List[str], user_nicknames: List[str],
                        challenges: List[str], vision_embeds: List[torch.Tensor]):
                doc_items: List[Dict[str, str]] = []

                for i in range(len(titles)):
                    doc: Dict[str, str] = {}
                    doc['titles'] = titles[i]
                    doc['user_nicknames'] = user_nicknames[i]
                    doc['challenges'] = challenges[i]
                    doc_items.append(doc)

                tensor_dict = self.traced_data_parse(queries, doc_items, vision_embeds)

                tensor_cuda_dict = self.move_to_device(tensor_dict)
                logits = self.model(tensor_dict["input_ids"], tensor_dict["segment_ids"], tensor_dict["input_masks"],
                                    tensor_dict["vision_embeds"], tensor_dict["vision_masks"])
                logits = logits.float().cpu()

                return logits
        """
        raise ValueError(
            "Forward method not implement in TraceEngine, please check! ")
