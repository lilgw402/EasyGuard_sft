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

from .model import VideoCLIPNet
from example.clip.dataset import get_transform


CONFIG_PATH = "hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/videoclip_swin_dy_20211206/config.yaml"
CKPT_PATH = "hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/videoclip_swin_dy_20211206/model.th"

SAMPLE_VIDEOS = [
    [   # 北海公园溜冰车
        'http://image-dy-his-hl.byted.org/storage/v1/tos-cn-vfp-0015/39c768108ada576a6d0f931ad00df7d1?VideoID=v0200f5b0000bo82p1tahtmdcrj4mpe0&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296191%3AZPGS0CV1cQnTE1GkhNIFihalKOc%3D',
        'http://image-dy-his-hl.byted.org/storage/v1/tos-cn-vfp-0015/8aedea02890b1615c332c2a82eccd9bf?VideoID=v0200f5b0000bo82p1tahtmdcrj4mpe0&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296191%3A%2FZ%2B1YQW3T2w1K%2FCUiRw%2FxufBrMw%3D',
        'http://image-dy-his-lf.byted.org/storage/v1/tos-cn-vfp-0015/6762235c30908b2d0bd2bb195f2bd5a0?VideoID=v0200f5b0000bo82p1tahtmdcrj4mpe0&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296191%3A3SNvdPqPJJOtjmHVyt3g8lm4dzM%3D',
        'http://image-dy-his-lq.byted.org/storage/v1/tos-cn-vfp-0015/4f717c8d56611e93a2a41fb8f10e5251?VideoID=v0200f5b0000bo82p1tahtmdcrj4mpe0&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296191%3A1fpx3BhajXPp%2Fyu%2BxUwHiAUHjEI%3D',
        'http://image-dy-his-lq.byted.org/storage/v1/tos-cn-vfp-0015/e51e88e10af64cf31054fa201756787b?VideoID=v0200f5b0000bo82p1tahtmdcrj4mpe0&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296191%3A9VSn63o2hzGcVdlg5NkFMQt%2Fuhw%3D',
        'http://image-dy-his-lq.byted.org/storage/v1/tos-cn-vfp-0015/6c6521d4cd6968cb3ad5e6324f158342?VideoID=v0200f5b0000bo82p1tahtmdcrj4mpe0&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296191%3AdUkX1Owh5VkuRnWSZjgYtJSBLX4%3D',
        'http://image-dy-his-lq.byted.org/storage/v1/tos-cn-vfp-0015/9b079cb573a316875838353a50b93d40?VideoID=v0200f5b0000bo82p1tahtmdcrj4mpe0&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296191%3A8KypkT%2F%2BwYk%2FgL9hJrosVbGDt94%3D',
        'http://image-dy-his-lf.byted.org/storage/v1/tos-cn-vfp-0015/4a78dfc36d1053c3d0a09c95bfb367e7?VideoID=v0200f5b0000bo82p1tahtmdcrj4mpe0&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296191%3Av7VuzWYN61UDcy4nWpFTXqVOEAo%3D'
    ],
    [   # 花样 滑冰
        'http://image-dy-his-lf.byted.org/storage/v1/tos-cn-vfp-0015/bc185a140b98fb2b5293683deaea39e0?VideoID=v0d00fg10000c5tb3lbc77u8s0cosv80&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296593%3A3qKIGxo%2FeZpqh0uoF3C58cNkQIQ%3D',
        'http://image-dy-his-lq.byted.org/storage/v1/tos-cn-vfp-0015/8417e88ae967cd1f524a5c19376d89bb?VideoID=v0d00fg10000c5tb3lbc77u8s0cosv80&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296593%3AT%2BaDTGkK1px8NO%2Br4l0tabYd6h8%3D',
        'http://image-dy-his-lq.byted.org/storage/v1/tos-cn-vfp-0015/c90a90e6da4594debc68163da065120e?VideoID=v0d00fg10000c5tb3lbc77u8s0cosv80&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296593%3AZ5sv1U9FV2qV%2F%2FO2LuNlQsgtLNE%3D',
        'http://image-dy-his-hl.byted.org/storage/v1/tos-cn-vfp-0015/a73c752f416a78bba7bcee35087f2aca?VideoID=v0d00fg10000c5tb3lbc77u8s0cosv80&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296593%3AhELBpbZrGyUxIp7aC99JPoR6lts%3D',
        'http://image-dy-his-lf.byted.org/storage/v1/tos-cn-vfp-0015/969b557b35964baf0e33e3cd9dfa7ade?VideoID=v0d00fg10000c5tb3lbc77u8s0cosv80&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296593%3AwEN4ezp3KRSh8nM0qV8TQpdOW%2B0%3D',
        'http://image-dy-his-lq.byted.org/storage/v1/tos-cn-vfp-0015/27642971c8bbcc0caf8812de9d14c42d?VideoID=v0d00fg10000c5tb3lbc77u8s0cosv80&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296593%3AHnx6PhusiQXs6%2BF7Q2Og5Z5upFE%3D',
        'http://image-dy-his-lf.byted.org/storage/v1/tos-cn-vfp-0015/f0279aeba390bf4f4331e1eb696f9d33?VideoID=v0d00fg10000c5tb3lbc77u8s0cosv80&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296593%3A2hYUBe6uliL0j2yJxWt%2B4d1pt2M%3D',
        'http://image-dy-his-lq.byted.org/storage/v1/tos-cn-vfp-0015/42ab41dada24d9ae6052825db790c567?VideoID=v0d00fg10000c5tb3lbc77u8s0cosv80&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296593%3AGIOEMHqydaVkOmb%2BL%2FnB2wprGhg%3D'
    ],
    [
        # 坝上洒水成冰
        'http://image-dy-lq.byted.org/storage/v1/tos-cn-vfp-0015/aeadf11caf9b26b87d98c5c5ee89cb19?VideoID=v0300fg10000c67530rc77u0i6llv1og&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296760%3Akjnq8KvUdPXnNQl%2FK10YhY1wHXE%3D',
        'http://image-dy-lq.byted.org/storage/v1/tos-cn-vfp-0015/d641955afe478d6025a590f7f0deb62a?VideoID=v0300fg10000c67530rc77u0i6llv1og&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296760%3AXhJf%2F2734KcAbpg1twgHqFG%2Fg5Y%3D',
        'http://image-dy-lq.byted.org/storage/v1/tos-cn-vfp-0015/d4ff4df3cbd1d4ccb89c00bc5fdb11cd?VideoID=v0300fg10000c67530rc77u0i6llv1og&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296760%3ANH9WpMfJRTFS1xs%2BSgKBZYtj8eo%3D',
        'http://image-dy-hl.byted.org/storage/v1/tos-cn-vfp-0015/92aca8b54d435f287b55169e4e9aa084?VideoID=v0300fg10000c67530rc77u0i6llv1og&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296760%3AYU6Ijj6wav%2FFp024GULTsHlFl1Y%3D',
        'http://image-dy-hl.byted.org/storage/v1/tos-cn-vfp-0015/ab85e99a0fd614f952b24c7e2bc54d1b?VideoID=v0300fg10000c67530rc77u0i6llv1og&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296760%3A50J0kMai5EeSxfL%2FbKb%2BbraluSQ%3D',
        'http://image-dy-lq.byted.org/storage/v1/tos-cn-vfp-0015/9112f845c3c0bb00d26e88a826f8c8bb?VideoID=v0300fg10000c67530rc77u0i6llv1og&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296760%3AFvW5QIrZnXybHtJtcorKboOEyUY%3D',
        'http://image-dy-lf.byted.org/storage/v1/tos-cn-vfp-0015/78cd76b5707fad311b6da00c244d8dd2?VideoID=v0300fg10000c67530rc77u0i6llv1og&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296760%3A1GJO4wnpP05f5L1mXSimhAbR0GI%3D',
        'http://image-dy-lq.byted.org/storage/v1/tos-cn-vfp-0015/dcf5ae88c38f14333a4cb64f16e036db?VideoID=v0300fg10000c67530rc77u0i6llv1og&caller=toutiao.videoarch.guldan&provider=aweme&session=scene_cut&signature=VARCH1-HMAC-SHA1%3Afcfc52beaf8e39b8880ca8a84b38b74a%3A1670296760%3AV1NUj%2FczOWjtFTmHrZ%2FGTYiGZms%3D'
    ]
]

SAMPLE_TEXTS = [
    '北海 公园 溜冰车',
    '花样 滑冰',
    '坝上 洒水 成冰'
]


def play():
    # load model
    cfg.update_cfg(CONFIG_PATH)
    model = VideoCLIPNet(config=cfg)
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
    images = video_preprocess(SAMPLE_VIDEOS, transform)
    images = images.cuda()  # [bsz, frame_num, c, h, w]

    text_emb = model.encode_text(input_ids=input_ids, input_segment_ids=segment_ids, input_mask=input_mask)['pooled_output']
    visual_emb = model.encode_image(images=images)['pooled_output']

    text_emb = F.normalize(text_emb, dim=1)
    visual_emb = F.normalize(visual_emb, dim=1)
    logit_scale = 30.
    logits = torch.mm(text_emb, visual_emb.t()) * logit_scale  # [bsz, bsz]
    t2v_prob = logits.softmax(dim=-2)
    print('video:')
    for i, m in enumerate(SAMPLE_VIDEOS):
        print(i, m)
    print('text:')
    for i, t in enumerate(SAMPLE_TEXTS):
        print(i, t)
    print('score matrix:')
    print(t2v_prob)


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


def video_preprocess(video_paths, transform):
    video_tensor = []
    for video in video_paths:
        one_video_tensor = []
        for path in video:
            image_str = requests.get(path).content
            # with hopen(path) as f:
            #image_str = f.read()
            image = Image.open(io.BytesIO(image_str)).convert("RGB")
            one_video_tensor.append(transform(image))
        one_video_tensor = torch.stack(one_video_tensor, dim=0)
        video_tensor.append(one_video_tensor)
    video_tensor = torch.stack(video_tensor, dim=0)
    print(video_tensor.shape)
    return video_tensor


if __name__ == '__main__':
    play()
