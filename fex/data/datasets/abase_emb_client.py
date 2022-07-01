# -*- coding: utf-8 -*-
'''
Created on Dec-08-20 17:44
abase_emb_dataset.py
@author: liuzhen.nlp
Description: embedding dataset for train with abase
'''
from typing import List, Union, Dict
import random
import numpy as np
import torch


from fex import _logger as log

try:
    import bytedabase
except Exception as e:
    log.warning('baytedabase is not installed, you can install by: pip install bytedabase --index-url=https://bytedpypi.byted.org/simple/')

from fex.utils.distributed import rank_zero_warn

try:
    import numpy as np
    import tensorflow as tf
    from tensorflow.core.framework import tensor_pb2
except Exception as e:
    print(str(e))


class AbaseEmbClient:

    def __init__(self, prefix='dy_14b_filter_nonsp_ernie_novl', update_ttl=True):
        self.abase_client = bytedabase.Client(psm="abase_douyin_visual_feature_offline.service.hl", table="visual_feature")
        self.ABASE_PREFIX = prefix
        self.cnt = 0
        self.last_cnt = 0
        self.nokey_cnt = 0
        self.ttl = 8640000  # 100 days
        self.update_ttl = update_ttl

    def get_embeddings(self, keys: List[Union[str, int]]) -> Dict[Union[str, int], torch.Tensor]:
        """ input is gids, output is dict of gids to tensor """
        abase_key: List[str] = []
        if not isinstance(keys, list):
            raise ValueError("Keys not list, please check!")

        keys = list(set(keys))

        for key in keys:
            abase_key.append('{},{}'.format(self.ABASE_PREFIX, key))
        emb_raw_res = self.abase_client.mget(abase_key)
        key_emb_map = self.parse_proto(keys, emb_raw_res)
        if self.update_ttl:
            self.mset_ttl(key_emb_map)
        self.cnt += len(keys)
        self.nokey_cnt += len(keys) - len(key_emb_map)
        if (self.cnt - self.last_cnt) > 10000:
            self.last_cnt = self.cnt
            rank_zero_warn('NO ABASE_KEY RATE [%s = %s / %s], %s' % (
                round(self.nokey_cnt / self.cnt, 4), self.nokey_cnt, self.cnt, keys))
        return key_emb_map

    def mset_ttl(self, key_emb_map):
        for key in key_emb_map.keys():
            self.abase_client.expire(f"{self.ABASE_PREFIX},{key}", self.ttl)

    def exists(self, key):
        abase_key = '{},{}'.format(self.ABASE_PREFIX, key)
        return self.abase_client.exists(abase_key)

    def exists_and_set_ttl(self, key, ttl):
        abase_key = '{},{}'.format(self.ABASE_PREFIX, key)
        ret = self.abase_client.exists(abase_key)
        if ret:
            return self.abase_client.expire(key, ttl)
        return ret

    def set_with_ttl(self, key, value, ttl):
        """ set key value pair to abase with ttl, ttl is seconds"""
        abase_key = '{},{}'.format(self.ABASE_PREFIX, key)
        return self.abase_client.setex(abase_key, ttl, value)

    def parse_proto(self, keys: List[Union[str, int]], embs: List[str]) -> Dict[Union[str, int], torch.Tensor]:
        res_embs_dict: Dict[str, torch.Tensor] = {}
        for i, (value, key) in enumerate(zip(embs, keys)):
            if not value:
                continue
            try:
                proto = tensor_pb2.TensorProto()
                proto.ParseFromString(value)
                try:
                    emb_num = tf.make_ndarray(proto)
                except ValueError:
                    emb_num = tf.io.parse_tensor(value, out_type=tf.dtypes.as_dtype(proto.dtype))
            except Exception:
                emb_num = self.parse_byte_to_numpy(value)
            if emb_num is not None:
                res_embs_dict[key] = torch.from_numpy(emb_num).float()

        return res_embs_dict

    def parse_byte_to_numpy(self, byte_str: bytes) -> np.array:
        try:
            sep = '|'.encode('utf-8')
            i_0 = byte_str.find(sep)
            i_1 = byte_str.find(sep, i_0 + 1)
            arr_dtype = byte_str[:i_0].decode('utf-8')
            arr_shape = tuple([int(a) for a in byte_str[i_0 + 1:i_1].decode('utf-8').split(',')])
            arr_str = byte_str[i_1 + 1:]
            num_data = np.frombuffer(arr_str, dtype=arr_dtype).reshape(arr_shape)
            return num_data
        except Exception:
            return None
