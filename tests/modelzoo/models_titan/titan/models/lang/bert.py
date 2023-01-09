r"""
Build standard BERT style of models from hugging face.
"""

import torch.nn as nn
import torch

from titan.utils.registry import register_model
from titan.utils.helper import (
    download_weights,
    get_configs
)
from titan.utils.misc import get_lang_checkpoint_file
from titan.utils.logger import logger

try:
    from transformers import BertConfig, BertModel, XLMRobertaModel
except ImportError:
    logger.warning('=> ImportError: can not import transformers>=4.6.1. '
                   'Establishing Bert series of models would raise error.')


__all__ = ['bert', 'bert_chilu', 'roberta', 'bert_cm']


class BertWrapper(nn.Module):
    __model_type__ = 'bert'

    def __init__(self,
                 transformer,
                 mlm_enable,
                 embedder_only=False,
                 with_hidden_states=False,
                 out_channels=768,
                 name=None):
        r"""
        Args:
            transformer: the original bert style transformer from hugging face,
                like BERT, Roberta, etc. with pretrained weight loaded.
            mlm_enable: if True, use Bert MLM for training
            out_channels: last out_channels of bert
            embedder_only: if True, only use the embedding step
            with_hidden_states: if True, return the hidden states together
                with the features
        """
        super(BertWrapper, self).__init__()
        logger.info(f'Bert prams: mlm_enable={mlm_enable}; '
                    f'embedder_only={embedder_only}; '
                    f'with_hidden_states={with_hidden_states}; '
                    f'out_channels={out_channels}')

        self.name = name
        self.model = transformer
        self.embedder_only = embedder_only
        self.with_hidden_states = with_hidden_states
        self.last_out_channels = out_channels
        self.mlm_enable = mlm_enable


        if self.embedder_only:
            self.model = self.model.embeddings

    def forward(self, input_ids, attention_mask):
        if self.embedder_only:
            features = self.model(input_ids=input_ids)
            return features, attention_mask

        bert_output = self.model(
            input_ids=input_ids, attention_mask=attention_mask)

        # compatibility with transformers < 4.0
        if isinstance(bert_output, tuple):
            assert len(bert_output) == 2, bert_output
            hidden_states, features = bert_output
        else:
            hidden_states = bert_output.last_hidden_state
            features = bert_output.pooler_output

        if self.with_hidden_states:
            return hidden_states, features
        else:
            return features


@register_model
def bert(pretrained=False, **kwargs):
    mlm_enable = kwargs.pop('mlm_enable', False)
    model_name = 'bert'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise RuntimeError('features_only is not supported for bert model.')
    model_config.pop('features_only')

    transformer = None
    if pretrained:
        model_dir = download_weights(
            model_name,
            **pretrained_config)
        transformer = BertModel.from_pretrained(model_dir)
        logger.info(
            f'=> Model arch: using {BertModel.__name__} '
            f'with config from {model_dir}.')
    model = BertWrapper(
        transformer,
        mlm_enable=mlm_enable,
        name=model_name,
        **model_config)
    return model


@register_model
def bert_chilu(pretrained=False, **kwargs):
    mlm_enable = kwargs.pop('mlm_enable', False)
    model_name = 'bert_chilu'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise RuntimeError('features_only is not supported for bert model.')
    model_config.pop('features_only')

    transformer = None
    if pretrained:
        model_dir = download_weights(
            model_name,
            **pretrained_config)
        # create model from pretrained config
        bert_config = BertConfig.from_pretrained(model_dir)
        transformer = BertModel(bert_config)
        # change pooler dense layer
        transformer.pooler.dense = nn.Linear(768, 128)
        # load pretrained weights from given path
        state_dict = torch.load(
            get_lang_checkpoint_file(model_dir),
            map_location=model_config.pop('map_location', 'cpu'))
        BertModel._load_state_dict_into_model(
            model=transformer,
            state_dict=state_dict,
            pretrained_model_name_or_path=model_dir)
        logger.info(
            f'=> Model arch: using {BertModel.__name__} '
            f'with config from {model_dir}.')
    model = BertWrapper(transformer, mlm_enable=mlm_enable,
                        out_channels=128, name=model_name, **model_config)
    return model


@register_model
def bert_cm(pretrained=False, **kwargs):
    mlm_enable = kwargs.pop('mlm_enable', False)
    model_name = 'bert_cm'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise RuntimeError('features_only is not supported for bert model.')
    model_config.pop('features_only')

    transformer = None
    if pretrained:
        model_dir = download_weights(
            model_name,
            **pretrained_config)
        # create model from pretrained config
        bert_config = BertConfig.from_pretrained(model_dir)
        transformer = BertModel(bert_config)
        # load pretrained weights from given path
        state_dict = torch.load(
            get_lang_checkpoint_file(model_dir),
            map_location=model_config.pop('map_location', 'cpu'))
        BertModel._load_state_dict_into_model(
            model=transformer,
            state_dict=state_dict,
            pretrained_model_name_or_path=model_dir)
        logger.info(
            f'=> Model arch: using {BertModel.__name__} '
            f'with config from {model_dir}.')
    model = BertWrapper(transformer, mlm_enable=mlm_enable,
                         out_channels=768, **model_config)
    return model


@register_model
def roberta(pretrained=False, **kwargs):
    mlm_enable = kwargs.pop('mlm_enable', False)
    model_name = 'roberta'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise RuntimeError('features_only is not supported for bert model.')
    model_config.pop('features_only')

    transformer = None
    if pretrained:
        model_dir = download_weights(
            model_name,
            **pretrained_config)
        transformer = XLMRobertaModel.from_pretrained(model_dir)
        logger.info(
            f'=> Model arch: using {XLMRobertaModel.__name__} '
            f'with config from {model_dir}.')
    model = BertWrapper(
        transformer,
        mlm_enable=mlm_enable,
        name=model_name,
        **model_config)
    return model
