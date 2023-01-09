#converting bert .pt to Lego Optimized .pt
import os.path
from collections import Counter, OrderedDict
from itertools import islice
from pathlib import Path

import lego
import torch
import torch.nn as nn
import typing as T

from anyon.utils.logger import logging
from anyon.details.xperf_lego.convert_torch_by_lego import LegoConverter as LegoConverterBase

from anyon.details.xperf_lego.tf_patch import *
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertModel


class BertWrapper(nn.Module):
    def __init__(self, bert, post_model=None):
        super().__init__()
        self.bert = bert
        self.post_model = post_model

    def forward(self, *args, **kwargs):
        pooler_output = self.bert.forward(*args, **kwargs).pooler_output
        if self.post_model is not None:
            return self.post_model(pooler_output)
        return pooler_output


class LegoConverter(LegoConverterBase):
    pretrained_model_name_or_path: str # eg: "bert-base-chinese"  # -p; bert pretrained model name, only required when checkpoint is a state dict
    max_batch_size: int = 8  # -b; batch size for validation
    sequence_length: int = 128  # -s; sequence length, default 128
    customized: bool = False  # is pretrained bert model, otherwise use parameters below to construct BERT model
    num_hidden_layers: int = 12  # number of hidden layers
    num_heads: int = 12  # number of heads
    hidden_size: int = 768  # hidden size
    intermediate_size: int = 3072  # intermediate size
    vocab_size: int = 21128  # embedding output size
    postprocess_json: str # eg: ''  # post process definition by JSON

    def _validate(self):
        assert 0 < self.max_batch_size < 128
        assert 0 < self.sequence_length < 1024

        if len(self.input_values) > 0:
            return

        self.input_shapes = [[self.max_batch_size, self.sequence_length]
                             for _ in range(2)]
        self.input_dtypes = [torch.int32, torch.int32]
        self.input_values = [0, 1]

    def prepare_ts_model(self):
        m = torch.load(self.checkpoint)
        if isinstance(m, nn.Module):
            logging.info("is module")
            wrapper = m.cuda().eval()
        else:
            # m is state dict
            logging.info("is state dict")
            if 'state_dict' in m:
                # legacy compatibility
                m = m['state_dict']
            state_dict: OrderedDict = m

            if not self.customized:
                bert_model = BertModel.from_pretrained(
                    self.pretrained_model_name_or_path)
                logging.info(
                    f"loaded pretrained BERT model {self.pretrained_model_name_or_path}"
                )
            else:
                bert_model = BertModel(
                    BertConfig(
                        num_hidden_layers=self.num_hidden_layers,
                        num_attention_heads=self.num_heads,
                        hidden_size=self.hidden_size,
                        intermediate_size=self.intermediate_size,
                        vocab_size=self.vocab_size,
                    ))

            logging.info(f"loaded state dict to BERT model")
            bert_model.load_state_dict(m, False)

            if len(self.postprocess_json) > 0:
                # compatibility to legacy bert models
                from anyon.utils.proto_utils import string_to_proto
                from anyon.proto.anyon_core_pb2 import BERTPostprocess

                post = string_to_proto(self.postprocess_json,
                                       BERTPostprocess())

                fc_sizes = list(post.fc_sizes)

                if len(post.post_layer_names) > 0:
                    layers = list(post.post_layer_names)
                elif len(post.post_weight_names) > 0:
                    layers = list(
                        Counter(
                            i.split('.')[0]
                            for i in post.post_weight_names).keys())
                else:
                    raise ValueError(
                        "either post_layer_names or post_weight_names should be given"
                    )

                # matched_layers = Counter(k.split('.')[0] for k in state_dict.keys() if k.split('.')[0] in layers)
                matched_layers = OrderedDict()
                state_dict = OrderedDict()
                for k, v in m.items():
                    name = k.split('.')[0]
                    if name in layers:
                        if name in matched_layers:
                            matched_layers[name] += 1
                        else:
                            matched_layers[name] = 1
                        state_dict[k] = v

                if len(matched_layers) < len(layers):
                    err_msg = f"mismatched layer names: [{','.join(set(layers) - set(matched_layers.keys()))}]"
                    logging.error(f"{err_msg}, please check your config")
                    raise ValueError(err_msg)

                matched_layer_list: T.List[T.Tuple[str, int]] = list(
                    matched_layers.items())

                if post.use_bn:
                    arr = [(fc, bn)
                           for fc, bn in zip(
                               islice(matched_layer_list, 0,
                                      len(matched_layer_list), 2),
                               islice(matched_layer_list, 1,
                                      len(matched_layer_list), 2))]
                    result_layers = arr
                else:
                    result_layers = [(i, None) for i in matched_layer_list]

                expected_layer_count = len(result_layers)
                if expected_layer_count != len(fc_sizes):
                    err_msg = f"fc sizes given {len(fc_sizes)}, but expected {expected_layer_count}"
                    logging.error(f"{err_msg}, please check your config")
                    raise ValueError(err_msg)

                fc_sizes = [self.hidden_size] + fc_sizes
                post_model = nn.Sequential()

                for idx, (fc, bn) in enumerate(result_layers):
                    assert fc[
                        1] == 2, f'fc should have 2 params, found {fc[1]} params'
                    assert bn is None or bn[
                        1] == 4, f'fc should have 2 params, found {bn[1]} params'
                    post_model.add_module(
                        fc[0], nn.Linear(fc_sizes[idx], fc_sizes[idx + 1]))
                    if bn is not None:
                        post_model.add_module(bn[0], nn.BatchNorm1d(idx + 1))
                    if post.act == 'sigmoid':
                        post_model.add_module(f'sigmoid_{idx}', nn.Sigmoid())
                    elif post.act == 'relu':
                        post_model.add_module(f'relu_{idx}', nn.ReLU())

                logging.info("post process model: %s", post_model)
                post_model.load_state_dict(state_dict)
                wrapper = BertWrapper(bert_model, post_model).cuda().eval()
            else:
                logging.info(f'constructed BERT model manually')
                wrapper = BertWrapper(bert_model).cuda().eval()

        m_half = wrapper.half()
        m_traced = torch.jit.trace(m_half, self.get_sample_inputs())
        torch.jit.save(m_traced, self.tmp_ts)
        return m_traced, self.tmp_ts


if __name__ == '__main__':
    LegoConverter().convert()
