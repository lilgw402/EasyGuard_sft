""" ALBERT + Visual """

import torch
import torch.nn as nn
import torch.nn.functional as F

from fex.nn.backbone.with_ptx import USE_PTX_TRANSFORMER

if not USE_PTX_TRANSFORMER:
    from fex.nn.backbone import albert
else:
    from fex.nn.backbone import albert_v2 as albert

from fex.nn.backbone.resnetv1 import resnet50

# from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
from torch.nn import LayerNorm as BertLayerNorm


class VisualLinguisticBert(nn.Module):
    """ 复现fex v1.0的 VisualLinguisticBert """

    def __init__(self, config):
        super().__init__()
        self.config = config

        if hasattr(config.BERT, 'visual_token_first'):
            self.visual_token_first = config.BERT.visual_token_first
        else:
            self.visual_token_first = False
            config.BERT.visual_token_first = False

        self.embedding = VLBertEmbedding(config.BERT, padding_index=2)
        self.encoder = albert.Transformer(config.BERT)
        self.visual_tokenizer = ResnetMean(config, config.BERT.hidden_size)

        if self.config.BERT.with_pooler:
            self.pooler = albert.BertPooler(config.BERT)

        # if self.config.BERT.word_embedding_frozen:
        #     for p in self.word_embeddings.parameters():
        #         p.requires_grad = False

        # if config.BERT.frozen_layers is not None and config.BERT.frozen_layers >= 1:
        #     self.frozen_parameters(config.BERT.frozen_layers)

    def forward(self, *args, **kwargs):
        """ 通过一个triger  'is_text_visual' 来判断是否是文本only的forward """
        if kwargs.pop("is_visual_only", False):
            return self.visual_only_forward(*args, **kwargs)

        if kwargs.pop('is_text_visual', True):
            return self.text_visual_forward(*args, **kwargs)

        return self.text_only_forward(*args, **kwargs)

    def text_visual_forward(self,
                            input_ids,
                            input_segment_ids,
                            input_mask,
                            image=None,
                            image_mask=None,
                            visual_embs=None,
                            *args,
                            **kwargs):
        """ 可以直接输入image或者输入视觉embedding """
        if image is not None:
            visual_embs = self.visual_tokenizer(image, *args, **kwargs)

        embeddings = self.embedding(input_ids=input_ids,
                                    token_type_ids=input_segment_ids,
                                    visual_embs=visual_embs,
                                    image_mask=image_mask)
        if self.visual_token_first:
            input_mask = torch.cat([input_mask[:, :1], image_mask, input_mask[:, 1:]], dim=1)
        else:
            input_mask = torch.cat([input_mask, image_mask], dim=1)
        out = self.encoder(embeddings, m_input_mask)
        if isinstance(out, tuple):
            encoded_layers, attention_probs = out
        else:
            encoded_layers = out
            attention_probs = None
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output) if self.config.BERT.with_pooler else None

        return {'encoded_layers': encoded_layers,
                'pooled_output': pooled_output,
                'embeddings': embeddings,
                "embedding_masks": input_mask,
                'attention_probs': attention_probs}

    def text_only_forward(self, input_ids, input_segment_ids, input_mask, *args, **kwargs):
        embeddings = self.embedding(input_ids=input_ids, token_type_ids=input_segment_ids, *args, **kwargs)

        # attention_mask = input_mask
        out = self.encoder(embeddings, m_input_mask)
        if isinstance(out, tuple):
            encoded_layers, attention_probs = out
        else:
            encoded_layers = out
            attention_probs = None

        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output) if self.config.BERT.with_pooler else None
        return {'encoded_layers': encoded_layers, 'pooled_output': pooled_output, 'embeddings': embeddings, "embedding_masks": input_mask}

    def visual_only_forward(self, visual_embs, image_mask):
        embeddings = self.embedding(visual_embs=visual_embs, image_mask=image_mask)

        out = self.encoder(embeddings, m_input_mask)
        if isinstance(out, tuple):
            encoded_layers, attention_probs = out
        else:
            encoded_layers, attention_probs = out, None
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output) if self.config.BERT.with_pooler else None
        return {'encoded_layers': encoded_layers,
                'pooled_output': pooled_output,
                'embeddings': embeddings,
                'embedding_masks': image_mask,
                'attention_probs': attention_probs}


class VLBertEmbedding(nn.Module):
    """
    https://code.byted.org/nlp/ptx/blob/master/ptx/core/bert.py#L340
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, padding_index=2):
        super().__init__()
        self.config = config
        self.token_embedder_tokens = torch.nn.Embedding(config.vocab_size,
                                                        config.embedding_size,
                                                        padding_idx=padding_index)
        self.token_embedder_positions = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_embedder_segments = torch.nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-8)    # TODO: configable eps
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob, inplace=True)

        # visual transform
        self.visual_ln_text = BertLayerNorm(config.hidden_size, eps=1e-8)
        self.visual_ln_object = BertLayerNorm(config.hidden_size, eps=1e-8)

        self.visual_token_type_id = config.visual_token_type_id
        self.visual_token_first = config.get('visual_token_first', True)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # self.register_buffer('position_ids', torch.arange(max_len).expand((1, -1)))

    def forward(self, input_ids: torch.Tensor = None, token_type_ids: torch.Tensor = None,
                position_ids: torch.Tensor = None, visual_embs: torch.Tensor = None,
                image_mask: torch.Tensor = None) -> torch.Tensor:
        """
        支持传inputs_embeds，来代替token-embedding，这个不错
        如果input_ids为空，则只有visual emb
        """
        if input_ids is None:
            # assert visual_embs is not None, "visual_emb can't be empty"
            if self.config.visual_ln:
                visual_embs = self.visual_ln_object(visual_embs)
            token_embeddings = visual_embs

            bsz, length = token_embeddings.size()[:2]
            if position_ids is None:
                position_ids = torch.arange(0, length, dtype=torch.long, device=visual_embs.device).expand(bsz, length)
            position_embeddings = self.token_embedder_positions(position_ids)

            token_type_ids = torch.ones_like(image_mask) * self.visual_token_type_id
            token_type_embeddings = self.token_embedder_segments(token_type_ids)
            embeddings = token_embeddings + position_embeddings + token_type_embeddings
            embeddings = self.norm(embeddings)
            embeddings = self.dropout(embeddings)

            return embeddings
        else:
            # text token embedding
            inputs_embeds = self.token_embedder_tokens(input_ids)

            # visual embedding
            if visual_embs is not None and visual_embs.shape[1] != 0:
                if self.config.visual_ln:
                    visual_embs = self.visual_ln_object(visual_embs)
                if self.visual_token_first:
                    token_embeddings = torch.cat([inputs_embeds[:, :1, :], visual_embs, inputs_embeds[:, 1:, :]], dim=1)
                else:
                    token_embeddings = torch.cat([inputs_embeds, visual_embs], dim=1)
            else:
                token_embeddings = inputs_embeds
            bsz, length = token_embeddings.size()[:2]

            # position
            if position_ids is None:
                position_ids = torch.arange(0, length, dtype=torch.long, device=input_ids.device).expand(bsz, length)
            position_embeddings = self.token_embedder_positions(position_ids)

            # segment
            if visual_embs is not None:
                if self.visual_token_first:
                    token_type_ids = torch.cat(
                        [token_type_ids[:, :1], torch.ones_like(image_mask, dtype=torch.long) * self.visual_token_type_id, token_type_ids[:, 1:]], dim=1).long()
                else:
                    token_type_ids = torch.cat(
                        [token_type_ids, torch.ones_like(image_mask) * self.visual_token_type_id], dim=1).long()
            token_type_embeddings = self.token_embedder_segments(token_type_ids)

            # post process
            embeddings = token_embeddings + position_embeddings + token_type_embeddings
            embeddings = self.norm(embeddings)
            embeddings = self.dropout(embeddings)

            return embeddings


class ResnetMean(nn.Module):

    def __init__(self, config, final_dim=768):
        """
        顾名思义，就是resnet最后一个feature map求平均
        """
        super(ResnetMean, self).__init__()

        # backbone
        self.resnet = resnet50(expose_stages=[1, 2, 3, 4, 5], stride_in_1x1=True)
        fm_dimmap = {1: 64, 2: 256, 3: 512, 4: 1024, 5: 2048}
        self.du_sample = torch.nn.ModuleDict({str(t): nn.Linear(fm_dimmap[t], final_dim) for t in [1, 2, 3, 4, 5]})

        self.image_encoder = torch.nn.Linear(768, 128)
        self.image_decoder = torch.nn.Linear(128, 768)

    def forward(self, images, *args, **kwargs):
        img_feats = self.resnet(images)
        tokens = []
        for k, v in sorted(img_feats.items(), reverse=True):    # reverse 让后面的fm放前面
            tokens.append(self.du_sample[k[-1]](torch.mean(v, dim=[-1, -2])))
        tokens = torch.stack(tokens, dim=1)
        tokens = self.image_decoder(self.image_encoder(tokens))
        return tokens
