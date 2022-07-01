#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" play around with some toy case """

import io
from PIL import Image

import requests

from fex.config import cfg
from fex.utils.load import load_from_pretrain
from fex.utils.hdfs_io import hopen
from fex.data import BertTokenizer


from .igpt import IGPTNet
from .dataset import get_transform
from .utils import sample

CONFIG_PATH = "./example/vision2text/config.yaml"
CKPT_PATH = "hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/clue/igpt_ts1kw_2/model_state_epoch_20000.th"

SAMPLE_IMAGES = [
    'hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/imagenet_val/n02025239/ILSVRC2012_val_00033421.JPEG', # 鸽子
    'hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/imagenet_val/n01882714/ILSVRC2012_val_00018026.JPEG', # 考拉
    'hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/imagenet_val/n01734418/ILSVRC2012_val_00017967.JPEG'  # 蛇
]

def play():
    # load model
    cfg.update_cfg(CONFIG_PATH)
    model = IGPTNet(config=cfg)
    load_from_pretrain(model, CKPT_PATH, [])
    model.eval()
    model = model.cuda()

    vocab_file = cfg.DATASET.VOCAB_FILE
    tokenizer = BertTokenizer(vocab_file, do_lower_case=False)

    trial_num = 5
    for image_path in SAMPLE_IMAGES:
        image_tensor = image_preprocess(image_path)
        image_tensor = image_tensor.cuda().unsqueeze(0)
        for i in range(trial_num):
            print('trial %s' % i)
            y = sample(model, image=image_tensor, x=None,
                       steps=15, temperature=0.9, sample=True,
                       top_k=5)
            y = y.tolist()[0]
            completion = ''.join(tokenizer.convert_ids_to_tokens(y))
            print(completion)
        print('---' * 10)


def image_preprocess(path):
    transform = get_transform('val')
    with hopen(path) as f:
        image_str = f.read()
        image = Image.open(io.BytesIO(image_str)).convert("RGB")
        image_tensor = transform(image)
    return image_tensor



if __name__ == '__main__':
    play()