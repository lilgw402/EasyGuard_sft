# -*- coding: utf-8 -*-
import os
import io
import json
import time
import torch
import base64
import numpy as np
from tqdm import tqdm
from PIL import ImageFile, Image
from ptx.matx.pipeline import Pipeline

ImageFile.LOAD_TRUNCATED_IMAGES = True

from cruise import CruiseCLI, CruiseTrainer
from cruise.utilities.hdfs_io import hopen, hlist_files
from easyguard.appzoo.framealbert_classification.data import FacDataModule
from easyguard.appzoo.framealbert_classification.model import FrameAlbertClassify

from easyguard.appzoo.framealbert_classification.data import text_concat, get_transform


max_len = 128
preprocess = get_transform(mode='val')
gec = np.load('./example/classify/GEC_cat.npy', allow_pickle=True).item()
pipe = Pipeline.from_option(f'file:/opt/tiger/fex/example/classify/m_albert_h512a8l12')


def image_preprocess(image_str):
    image = _load_image(b64_decode(image_str))
    image_tensor = preprocess(image)
    return image_tensor


def b64_decode(string):
    if isinstance(string, str):
        string = string.encode()
    return base64.decodebytes(string)


def _load_image(buffer):
    img = Image.open(io.BytesIO(buffer))
    img = img.convert('RGB')
    return img


def process(data_item: dict):
    # cid = data_item['leaf_cid']
    # label = gec[cid]['label']

    if 'text' in data_item:
        title = data_item['text']
        desc = None
    else:
        title = data_item['title']
        desc = data_item['desc']

    text = text_concat(title, desc)

    token_ids = pipe.preprocess([text])[0]
    token_ids = token_ids.asnumpy()
    token_ids = torch.from_numpy(token_ids)

    token_mask = token_ids.clone()
    token_mask[token_ids != 0] = 1

    input_segment_ids = torch.zeros([1, max_len], dtype=torch.int64)

    frames = []

    if 'image' in data_item:
        # get image by b64
        try:
            image_tensor = image_preprocess(data_item['image'])
            frames.append(image_tensor)
        except:
            print(f"load image base64 failed -- {data_item['pid']}")
            return None

    frames = torch.stack(frames, dim=0)
    frames = frames.reshape([1, 1, 3, 224, 224])
    frames_mask = torch.tensor([[1]])

    return token_ids, input_segment_ids, token_mask, frames, frames_mask


if __name__ == "__main__":
    cli = CruiseCLI(FrameAlbertClassify,
                    trainer_class=CruiseTrainer,
                    datamodule_class=FacDataModule,
                    trainer_defaults={
                    })
    cfg, trainer, model, datamodule = cli.parse_args()

    # load ckpt
    config_path = './example/classify/config.yaml'
    ckpt = ''
    model.partial_load_from_checkpoints(
        checkpoints=ckpt,
        rename_params=None,
        map_location='cpu',
    )

    model.setup("val")
    datamodule.setup("val")

    # files = [
    #     'hdfs://harunava/home/byte_magellan_va/user/wangxian/datasets/TTS_KG_TEST/test_jsonl_1013/VN_1013.test.jsonl',
    #     'hdfs://harunava/home/byte_magellan_va/user/wangxian/datasets/TTS_KG_TEST/test_jsonl_1013/ID_1013.test.jsonl',
    #     'hdfs://harunava/home/byte_magellan_va/user/wangxian/datasets/TTS_KG_TEST/test_jsonl_1013/MY_1013.test.jsonl',
    #     'hdfs://harunava/home/byte_magellan_va/user/wangxian/datasets/TTS_KG_TEST/test_jsonl_1013/TH_1013.test.jsonl',
    #     'hdfs://harunava/home/byte_magellan_va/user/wangxian/datasets/TTS_KG_TEST/test_jsonl_1013/GB_1013.test.jsonl'
    # ]
    #
    # for file in files:
    #     print(f'testing on {file}')
    #     filename = file.split('/')[-1]
    #     with hopen(file, 'r') as f:
    #         lines = f.readlines()
    #
    #     #
    #     allres = []
    #
    #     for line in tqdm(lines):
    #         sample = json.loads(line)
    #         data = process(sample)
    #
    #         if data is not None:
    #             input_ids, input_segment_ids, input_mask, frames, frames_mask = data
    #             # print(input_ids.shape, input_segment_ids.shape, input_mask.shape,
    #             # frames.shape, frames_mask.shape)
    #             # print(input_ids.dtype, input_segment_ids.dtype, input_mask.dtype,
    #             # frames.dtype, frames_mask.dtype)
    #             input_ids = input_ids.cuda()
    #             input_segment_ids = input_segment_ids.cuda()
    #             input_mask = input_mask.cuda()
    #             frames = frames.cuda()
    #             frames_mask = frames_mask.cuda()
    #         else:
    #             continue
    #
    #         logits = model.infer_step(
    #             input_ids=input_ids, input_segment_ids=input_segment_ids, input_mask=input_mask,
    #             frames=frames, frames_mask=frames_mask
    #         )
    #
    #         # print(logits)
    #         _, pred = logits.topk(1, 1, True, True)
    #         pred = int(pred.cpu().detach().data.numpy()[0][0])
    #
    #         sample['pred'] = pred
    #         sample['image'] = ''
    #         allres.append(json.dumps(sample))
    #
    #     with open(f'./example/classify/infer_res/{filename}', 'w') as f:
    #         f.writelines('\n'.join(allres))
