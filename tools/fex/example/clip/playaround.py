#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" play around with some toy case """

import io
import requests
from PIL import Image

import torch
import torch.nn.functional as F

from fex.config import cfg
from fex.utils.load import load_from_pretrain
from fex.utils.hdfs_io import hopen
from fex.data import BertTokenizer

from .model import CLIPNet
from .dataset import get_transform

# image model, ViT
CONFIG_PATH = "hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/abvt_b32_ts1b_20210915/config.yaml"
CKPT_PATH = "hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/abvt_b32_ts1b_20210915/model.th"

# image model, Swin
CONFIG_PATH = "hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/swinb224_dycover_20211206/config.yaml"
CKPT_PATH = "hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/swinb224_dycover_20211206/model.th"

SAMPLE_IMAGES = [
    'http://tosv.byted.org/obj/labis/5f44c5e4f0cf29de17a95b40d0c420e3',  # 安妮海瑟薇
    'http://tosv.byted.org/obj/labis/5fcea54b8e38a61f07f6bc6df635a628',  # 奥黛丽赫本
    'http://tosv.byted.org/obj/labis/df24820e51830434a40e80e3654e5bab',  # 范冰冰
]

SAMPLE_TEXTS = [
    '安妮 海瑟薇',
    '奥黛丽 赫本',
    '范冰冰'
]


def play():
    # load model
    cfg.update_cfg(CONFIG_PATH)
    model = CLIPNet(config=cfg)
    load_from_pretrain(model, CKPT_PATH, [])
    model.eval()
    model = model.cuda()

    tokenizer = BertTokenizer(cfg.network.vocab_file, do_lower_case=False)
    transform = get_transform('val')

    # prepare text
    input_ids, segment_ids, input_mask = text_preprocess(SAMPLE_TEXTS, tokenizer)
    input_ids = input_ids.cuda()
    segment_ids = segment_ids.cuda()
    input_mask = input_mask.cuda()

    # prepare image
    image = image_preprocess(SAMPLE_IMAGES, transform)
    image = image.cuda()

    text_emb = model.encode_text(input_ids=input_ids, input_segment_ids=segment_ids, input_mask=input_mask)
    visual_emb = model.encode_image(image=image)
    text_emb = F.normalize(text_emb, dim=1)
    visual_emb = F.normalize(visual_emb, dim=1)
    logit_scale = 30.
    logits = torch.mm(visual_emb, text_emb.t()) * logit_scale  # [bsz, bsz]
    v2t_prob = logits.softmax(dim=-1)
    print('image:')
    for i, m in enumerate(SAMPLE_IMAGES):
        print(i, m)
    print('text:')
    for i, t in enumerate(SAMPLE_TEXTS):
        print(i, t)
    print('score matrix:')
    print(v2t_prob)


def text_preprocess(texts, tokenizer):
    max_len = 16
    token_ids_batch = []
    segment_ids_batch = []
    input_mask_batch = []
    for text in texts:
        tokens = tokenizer.tokenize(text)[:max_len - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        tokens_pad = tokens + ['[PAD]'] * (max_len - len(tokens))
        token_ids = tokenizer.convert_tokens_to_ids(tokens_pad)
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        segment_ids = torch.zeros_like(token_ids, dtype=torch.long)
        input_mask = torch.tensor([1] * len(tokens) + [0] * (max_len - len(tokens)), dtype=torch.long)
        token_ids_batch.append(token_ids)
        segment_ids_batch.append(segment_ids)
        input_mask_batch.append(input_mask)
    token_ids = torch.stack(token_ids_batch, dim=0)
    segment_ids = torch.stack(segment_ids_batch, dim=0)
    input_mask = torch.stack(input_mask_batch, dim=0)
    # print(token_ids.shape, segment_ids.shape, input_mask.shape)
    return token_ids, segment_ids, input_mask


def image_preprocess(image_paths, transform):
    images = []
    for path in image_paths:
        image_str = requests.get(path).content
        # with hopen(path) as f:
        #image_str = f.read()
        image = Image.open(io.BytesIO(image_str)).convert("RGB")
        images.append(transform(image))
    images = torch.stack(images, dim=0)
    # print(images.shape)
    return images


if __name__ == '__main__':
    play()
