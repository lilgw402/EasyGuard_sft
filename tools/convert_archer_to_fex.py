
""" Load a checkpoint file of pretrained transformer to a model in pytorch """

import re
from collections import OrderedDict
from functools import lru_cache

import numpy as np
import tensorflow as tf
import torch


def get_scalar_sample(tensor):
    while True:
        try:
            tensor = tensor[0]
        except IndexError:
            return float(tensor)


@lru_cache()
def ls_variables(ckpt_dir_or_file, p=False, t=False, p_ignore=None):
    if t:
        reader = tf.train.load_checkpoint(ckpt_dir_or_file)
    else:
        reader = None

    vlist = tf.train.list_variables(
        ckpt_dir_or_file
    )
    vdict = OrderedDict()
    for vname, vsize in vlist:
        vdict[vname] = vsize
        if p:
            if p_ignore is not None and any([pi.strip() in vname for pi in p_ignore.split('|')]):
                continue
            if not t:
                print('%s\t%s' % (vname, vsize))
            else:
                print('%s\t%s %.8f' % (vname, vsize, get_scalar_sample(reader.get_tensor(vname))))
    return vdict


def dirs(state_dict: str, **kwargs):
    """
    Print flattened dirs of state_dict (filename), with tensor size.
    """
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, map_location='cpu')
    for k, v in state_dict.items():
        print('%s%s%.8f\t%s %s %s' % (k, (50 - len(k)) * ' ', get_scalar_sample(v),
                                      list(v.shape), str(v.dtype).replace('torch.', ''), v.device))


def exists_variable(ckpt_dir_or_file, var_name_substr, by_re=False):
    for name in ls_variables(ckpt_dir_or_file):
        if by_re:
            if re.match(var_name_substr, name):
                return name
            continue
        if var_name_substr in name:
            return name


def state_dict_from_ckpt(ckpt_dir_or_file, conversion_table, use_fp32=False, expand_token_type=False):
    """
    Load a pytorch state dict from a tensorflow checkpoint,
    according to a name conversion table.

    Args:
        ckpt_dir_or_file (str): pretrained tensorflow checkpoint model file path
        conversion_table (dict): { checkpoint variable name : pytorch parameter name }

    Returns:
        dict
    """

    state_dict = {}

    for tf_param_name, pyt_param in conversion_table.items():
        if use_fp32 and 'layer_norm' not in tf_param_name:
            tf_param_name = 'FP32-master-copy/' + tf_param_name

        tf_param = tf.train.load_variable(ckpt_dir_or_file, tf_param_name)


        if 'layer_norm' in tf_param_name:
            print(tf_param.shape, 'layern') 

        # For weight(kernel), should transpose
        if tf_param_name.endswith('kernel'):
            tf_param = np.transpose(tf_param)

        # Assign pytorch tensor from tensorflow param
        if expand_token_type and 'token_type_embeddings' in tf_param_name:
            token_type_emb = torch.normal(torch.tensor(0.), torch.tensor(0.02), [16, 768])
            token_type_emb[:tf_param.shape[0], :] = torch.from_numpy(tf_param)
            print(token_type_emb.shape, 'token_type_emb')
            state_dict[pyt_param] = token_type_emb
        else:
            state_dict[pyt_param] = torch.from_numpy(tf_param)

    return state_dict


def conv(src, dst='', embedding_proj_first=False, use_fp32=False, expand_token_type=False):
    """ Load the pytorch model from checkpoint file """

    ckpt_dir_or_file = src
    state_dict = {}

    # Embedding & Pooler
    partial_states = state_dict_from_ckpt(ckpt_dir_or_file, {
        'al_bert_model/embedding_lookup/word_embeddings': 'embedding.token_embedder_tokens.weight',
        'al_bert_model/embedding_post_processor/position_embeddings': 'embedding.token_embedder_positions.weight',
        'al_bert_model/embedding_post_processor/token_type_embeddings': 'embedding.token_embedder_segments.weight',
        #'al_bert_model/embedding_post_processor/post_layer_process/apply_norm/layer_normalization/gamma': 'embedding.norm.gamma',
        #'al_bert_model/embedding_post_processor/post_layer_process/apply_norm/layer_normalization/beta': 'embedding.norm.beta',
        'al_bert_model/embedding_post_processor/post_layer_process/apply_norm/layer_normalization/gamma': 'embedding.norm.weight',
        'al_bert_model/embedding_post_processor/post_layer_process/apply_norm/layer_normalization/beta': 'embedding.norm.bias',
        
        # if not embedding_proj_first else 'embedding.token_embedder_tokens._projection.weight',
        'al_bert_model/embedding_hidden_dense/kernel': 'embedding.proj_embedding_hidden.weight',
        # if not embedding_proj_first else 'embedding.token_embedder_tokens._projection.bias',
        'al_bert_model/embedding_hidden_dense/bias': 'embedding.proj_embedding_hidden.bias',
        'al_bert_model/pooler/dense/kernel': 'pooler.dense.weight',
        'al_bert_model/pooler/dense/bias': 'pooler.dense.bias',
    }, use_fp32=False, expand_token_type=expand_token_type)
    state_dict.update(partial_states)

    # Encoder
    i = 0
    while exists_variable(ckpt_dir_or_file, f'/encoder/layer_{i}/'):
        partial_states = state_dict_from_ckpt(ckpt_dir_or_file, {
            f'al_bert_model/encoder/layer_{i}/self_attention/query/kernel': f'encoder.blocks.{i}.attn.proj_q.weight',
            f'al_bert_model/encoder/layer_{i}/self_attention/query/bias': f'encoder.blocks.{i}.attn.proj_q.bias',
            f'al_bert_model/encoder/layer_{i}/self_attention/key/kernel': f'encoder.blocks.{i}.attn.proj_k.weight',
            f'al_bert_model/encoder/layer_{i}/self_attention/key/bias': f'encoder.blocks.{i}.attn.proj_k.bias',
            f'al_bert_model/encoder/layer_{i}/self_attention/value/kernel': f'encoder.blocks.{i}.attn.proj_v.weight',
            f'al_bert_model/encoder/layer_{i}/self_attention/value/bias': f'encoder.blocks.{i}.attn.proj_v.bias',

            f'al_bert_model/encoder/layer_{i}/attention_output_dense/kernel': f'encoder.blocks.{i}.proj.weight',
            f'al_bert_model/encoder/layer_{i}/attention_output_dense/bias': f'encoder.blocks.{i}.proj.bias',

            #f'al_bert_model/encoder/layer_{i}/post_layer_process/apply_norm/layer_normalization/gamma': f'encoder.blocks.{i}.norm1.gamma',
            #f'al_bert_model/encoder/layer_{i}/post_layer_process/apply_norm/layer_normalization/beta': f'encoder.blocks.{i}.norm1.beta',
            f'al_bert_model/encoder/layer_{i}/post_layer_process/apply_norm/layer_normalization/gamma': f'encoder.blocks.{i}.norm1.weight',
            f'al_bert_model/encoder/layer_{i}/post_layer_process/apply_norm/layer_normalization/beta': f'encoder.blocks.{i}.norm1.bias',


            f'al_bert_model/encoder/layer_{i}/transformer_ffn/intermediate_dense/kernel': f'encoder.blocks.{i}.pwff.fc1.weight',
            f'al_bert_model/encoder/layer_{i}/transformer_ffn/intermediate_dense/bias': f'encoder.blocks.{i}.pwff.fc1.bias',
            f'al_bert_model/encoder/layer_{i}/transformer_ffn/output_dense/kernel': f'encoder.blocks.{i}.pwff.fc2.weight',
            f'al_bert_model/encoder/layer_{i}/transformer_ffn/output_dense/bias': f'encoder.blocks.{i}.pwff.fc2.bias',

            #f'al_bert_model/encoder/layer_{i}/transformer_ffn/post_layer_process/apply_norm/layer_normalization/gamma': f'encoder.blocks.{i}.norm2.gamma',
            #f'al_bert_model/encoder/layer_{i}/transformer_ffn/post_layer_process/apply_norm/layer_normalization/beta': f'encoder.blocks.{i}.norm2.beta',
            f'al_bert_model/encoder/layer_{i}/transformer_ffn/post_layer_process/apply_norm/layer_normalization/gamma': f'encoder.blocks.{i}.norm2.weight',
            f'al_bert_model/encoder/layer_{i}/transformer_ffn/post_layer_process/apply_norm/layer_normalization/beta': f'encoder.blocks.{i}.norm2.bias',
        
        }, use_fp32)
        state_dict.update(partial_states)
        i += 1

    # Classifier
    """
    partial_states = state_dict_from_ckpt(ckpt_dir_or_file, {
        'output_weights': 'classifier.weight',
        'output_bias': 'classifier.bias',
    }, use_fp32=False)
    state_dict.update(partial_states)
    """

    if dst:
        torch.save(state_dict, dst)
    else:
        for k, v in state_dict.items():
            print(f'{k}\t{list(v.shape)}')

    return state_dict


if __name__ == '__main__':
    import sys

    location_dat = sys.argv[1]
    location_res = None if len(sys.argv) < 3 else sys.argv[2]

    conv(location_dat, dst=location_res, embedding_proj_first=True, use_fp32=True, expand_token_type=True)

    # tensorflow list var
    ls_variables(location_dat, True, True, p_ignore='_slot_')

    # pytorch list var
    #dirs(location_res)
