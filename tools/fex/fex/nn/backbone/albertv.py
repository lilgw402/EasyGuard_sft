""" ALBERT + Visual """


import torch
import torch.nn as nn
import torch.nn.functional as F

from fex.nn.backbone.with_ptx import USE_PTX_TRANSFORMER

if not USE_PTX_TRANSFORMER:
    from fex.nn.backbone import albert
else:
    from fex.nn.backbone import albert_v2 as albert

from fex.nn.backbone.resnet_token import ResnetTokenizer


class ALBertV(nn.Module):
    """ ALBert + Visual Backbone """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual_tokenizer = ResnetTokenizer(config)  # 把图片变成embedding
        self.visual_embedding = VisualEmbedding(config.BERT)  # 给图片embedding加一些position等的东西 # TODO: 看看要不要合到下面
        self.embedding = ALBertVEmbedding(config.BERT, padding_index=2)
        self.encoder = albert.Transformer(config.BERT)

        if self.config.BERT.with_pooler:
            self.pooler = albert.BertPooler(config.BERT)

        # init weights
        self.apply(self.init_weights)

        if self.config.BERT.word_embedding_frozen:
            for p in self.word_embeddings.parameters():
                p.requires_grad = False

        if config.BERT.frozen_layers is not None and config.BERT.frozen_layers >= 1:
            self.frozen_parameters(config.BERT.frozen_layers)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.BERT.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, *args, **kwargs):
        """ 通过一个triger  'is_text_visual' 来判断是否是文本only的forward """
        if kwargs.pop('is_text_visual', True):
            return self.text_visual_forward(*args, **kwargs)
        else:
            return self.text_only_forward(*args, **kwargs)

    def text_visual_forward(self, input_ids, input_segment_ids, input_mask,
                            image, image_mask, *args, **kwargs):

        visual_embs = self.visual_tokenizer(image, *args, **kwargs)
        visual_embs = self.visual_embedding(visual_embs)  # [bsz, v_len, dim]

        embeddings = self.embedding(input_ids=input_ids,
                                    token_type_ids=input_segment_ids,
                                    visual_embs=visual_embs)
        input_mask = torch.cat([input_mask, image_mask], dim=1)
        out = self.encoder(embeddings, input_mask)
        if isinstance(out, tuple):
            encoded_layers, attention_probs = out
        else:
            encoded_layers = out
            attention_probs = None
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output) if self.config.BERT.with_pooler else None

        return {'encoded_layers': encoded_layers,
                'pooled_output': pooled_output,
                'attention_probs': attention_probs}

    def text_only_forward(self, input_ids, input_segment_ids, input_mask, *args, **kwargs):

        embeddings = self.embedding(input_ids=input_ids,
                                    token_type_ids=input_segment_ids,
                                    *args, **kwargs)

        # attention_mask = input_mask
        encoded_layers, attention_probs = self.encoder(embeddings, input_mask)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output) if self.config.BERT.with_pooler else None
        return {'encoded_layers': encoded_layers,
                'pooled_output': pooled_output,
                'attention_probs': attention_probs}


class VisualEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.project_embedding_first = config.project_embedding_first
        dim = config.hidden_size if self.project_embedding_first else config.embedding_size
        self.token_embedder_positions = torch.nn.Embedding(16, dim)
        self.token_embedder_segments = torch.nn.Embedding(2, dim)

        # TODO: 一次内部norm?

    def forward(self, visual_embs):
        bsz, length = visual_embs.size()[:2]
        position_ids = torch.arange(0, length, dtype=torch.long, device=visual_embs.device).expand(bsz, length)
        token_type_ids = torch.zeros([bsz, length], dtype=torch.long, device=visual_embs.device)

        position_embeddings = self.token_embedder_positions(position_ids)
        token_type_embeddings = self.token_embedder_segments(token_type_ids)
        embeddings = visual_embs + position_embeddings + token_type_embeddings
        return embeddings


class ALBertVEmbedding(nn.Module):
    """
    https://code.byted.org/nlp/ptx/blob/master/ptx/core/bert.py#L340
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, padding_index=2):
        super().__init__()

        self.project_embedding_first = config.project_embedding_first
        dim = config.hidden_size if self.project_embedding_first else config.embedding_size
        self.token_embedder_tokens = torch.nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=padding_index)
        self.token_embedder_positions = torch.nn.Embedding(config.max_position_embeddings, dim)
        self.token_embedder_segments = torch.nn.Embedding(config.type_vocab_size, dim)

        self.norm = nn.LayerNorm(dim, eps=1e-12)  # TODO: configable eps
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        if config.embedding_size != config.hidden_size:
            self.proj_embedding_hidden = torch.nn.Linear(config.embedding_size, config.hidden_size)
        else:
            self.proj_embedding_hidden = None

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # self.register_buffer('position_ids', torch.arange(max_len).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, visual_embs=None):
        """
        支持传inputs_embeds，来代替token-embedding，这个不错
        """
        # text token embedding
        inputs_embeds = self.token_embedder_tokens(input_ids)
        if self.project_embedding_first and self.proj_embedding_hidden:
            inputs_embeds = self.proj_embedding_hidden(inputs_embeds)

        bsz, length = inputs_embeds.size()[:2]
        if position_ids is None:
            position_ids = torch.arange(0, length, dtype=torch.long, device=input_ids.device).expand(bsz, length)

        position_embeddings = self.token_embedder_positions(position_ids)
        token_type_embeddings = self.token_embedder_segments(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        if visual_embs is not None:
            embeddings = torch.cat([embeddings, visual_embs], dim=1)

        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)

        # 后 project
        if not self.project_embedding_first and self.proj_embedding_hidden:
            embeddings = self.proj_embedding_hidden(embeddings)

        return embeddings
