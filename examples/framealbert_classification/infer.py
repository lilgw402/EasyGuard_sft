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
gec = np.load('/opt/tiger/easyguard/GEC_cat.npy', allow_pickle=True).item()
pipe = Pipeline.from_option(f'file:/opt/tiger/easyguard/m_albert_h512a8l12')
country2idx = {'GB': 0, 'TH': 1, 'ID': 2, 'VN': 3, 'MY': 4, }


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
    if 'text' in data_item:
        title = data_item['text']
        desc = None
        country = data_item['country']
        country_idx = country2idx[country]
    else:
        title = data_item['title']
        desc = data_item['desc']
        country = data_item['country']
        country_idx = country2idx[country]

    text = text_concat(title, desc)

    token_ids = pipe.preprocess([text])[0]
    token_ids = token_ids.asnumpy()
    token_ids = torch.from_numpy(token_ids)

    token_mask = token_ids.clone()
    token_mask[token_ids != 0] = 1

    input_segment_ids = torch.zeros([1, max_len], dtype=torch.int64)

    head_mask = torch.zeros(1, 5, dtype=torch.long)
    head_mask[0, country_idx] = 1

    frames = []
    if 'image' in data_item:
        try:
            image_tensor = image_preprocess(data_item['image'])
            frames.append(image_tensor.half())
        except:
            print(f"load image base64 failed -- {data_item['pid']}")
            return None

    frames = torch.stack(frames, dim=0)
    frames = frames.reshape([1, 1, 3, 224, 224])
    frames_mask = torch.tensor([[1]])

    return token_ids, input_segment_ids, token_mask, frames, frames_mask, head_mask


if __name__ == "__main__":
    cli = CruiseCLI(FrameAlbertClassify,
                    trainer_class=CruiseTrainer,
                    datamodule_class=FacDataModule,
                    trainer_defaults={
                    })
    cfg, trainer, model, datamodule = cli.parse_args()
    print(f'finished model config init!')

    # load ckpt
    model.setup("val")
    # datamodule.setup("val")

    print(f'generate random input demo')
    file = 'hdfs://harunava/home/byte_magellan_va/user/wangxian/datasets/TTS_KG_TEST/test_jsonl_high_risk_1013_country/ID_high_risk_1013.jsonl'
    with hopen(file, 'r') as f:
        lines = f.readlines()

    num_ac = 0
    num_all = 0
    for line in lines[:1]:
        sample = json.loads(line)
        data = process(sample)
        input_ids, input_segment_ids, input_mask, frames, frames_mask, head_mask = data
        # token_ids = torch.zeros([1, 128], dtype=torch.long)
        # input_segment_ids = torch.zeros([1, 128], dtype=torch.long)
        # token_mask = torch.zeros([1, 128], dtype=torch.long)
        # frames = torch.zeros([1, 1, 3, 224, 224], dtype=torch.float32)
        # frames_mask = torch.zeros([1, 1], dtype=torch.long)
        # head_mask = torch.ones([1, 5], dtype=torch.long)

        print(f'inferencing')
        res = model.forward_step(input_ids=input_ids,
                                 input_segment_ids=input_segment_ids,
                                 input_mask=input_mask,
                                 frames=frames,
                                 frames_mask=frames_mask,
                                 head_mask=head_mask, )
        logits = res['logits']
        label, pred = logits.topk(1, 1, True, True)
        num_all += 1
        if gec[sample['leaf_cid']]['label'] == label:
            num_ac += 1

    print(f'top1 acc is {num_ac/num_all}')

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
