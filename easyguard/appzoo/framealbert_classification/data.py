import io
import os
import json
import random
import re
import base64
import emoji
import numpy as np
import torch
import torchvision.transforms as transforms
from cruise.data_module import (
    CruiseDataModule,
    create_cruise_loader,
    customized_processor,
)
from ptx.matx.pipeline import Pipeline
from easyguard.appzoo.multimodal_modeling.utils import BertTokenizer

from cruise.utilities.hdfs_io import hopen
from cruise.data_module.preprocess.decode import save_args
from .dist_dataset import DistLineReadingDataset
from .download import get_real_url, get_original_url, download_url_with_exception
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True

_whitespace_re = re.compile(r'\s+')


def rmEmoji(line):
    return emoji.replace_emoji(line, replace=' ')


def rmRepeat(line):
    return re.sub(r'(.)\1{5,}', r'\1', line)


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def text_preprocess(text):
    if text is None:
        return text

    text = collapse_whitespace(rmRepeat(rmEmoji(text)))

    try:
        return re.sub(r"\<.*?\>", "", text).replace("\n", "").replace("\t", "").replace("\"", "").replace("\\",
                                                                                                          "").strip()
    except Exception:
        return text


def text_concat(title, desc=None):
    title = text_preprocess(title)
    desc = text_preprocess(desc)
    if desc is None:
        return title

    if desc.startswith(title):
        return desc

    if len(desc) > 5:
        text = title + ". " + desc
    else:
        text = title

    return text


class TorchvisionLabelDataset(DistLineReadingDataset):
    """
    dataset，继承的dataset是 DistLineReadingDataset，是一个逐行读hdfs数据的IterableDataset。
    """

    def __init__(self, config, data_path, rank=0, world_size=1, shuffle=True, repeat=False,
                 is_training=False):
        super().__init__(data_path, rank, world_size, shuffle, repeat)
        
        self.config = config
        self.is_training = is_training
        self.text_len = config.text_len
        self.frame_len = config.frame_len
        if 'maplabel' in config.exp:
            self.second_map = np.load('/opt/tiger/easyguard/second_map.npy', allow_pickle=True).item()
            print(f'=============== apply label map to finetune on high risk map ===============')
        else:
            self.second_map = None
        self.gec = np.load('/opt/tiger/easyguard/GEC_cat.npy', allow_pickle=True).item()

        self.pipe = Pipeline.from_option(f'file:/opt/tiger/easyguard/m_albert_h512a8l12')
        self.PAD = self.pipe.special_tokens['pad']
        self.preprocess = get_transform(mode='train' if is_training else 'val')

        with hopen('hdfs://harunava/home/byte_magellan_va/user/xuqi/black_image.jpeg', 'rb') as f:
            self.black_frame = self.preprocess(self._load_image(f.read()))

    def __len__(self):
        print(f"os.world_size: {int(os.environ.get('WORLD_SIZE'))}")
        if self.is_training:
            return self.config.train_size // 16 #int(os.environ.get('WORLD_SIZE'))
        else:
            return self.config.val_size // 16 #int(os.environ.get('WORLD_SIZE'))


    def __iter__(self):
        for example in self.generate():
            try:
                data_item = json.loads(example)
                cid = data_item['leaf_cid']
                label = self.gec[cid]['label']
                if self.second_map is not None:
                    label = self.second_map[label]
                # 文本
                if 'translation' in data_item:
                    country = random.choice(['GB', 'TH', 'ID', 'VN', 'MY'])
                    title = data_item['translation'][country]
                    desc = None
                elif 'text' in data_item:
                    title = data_item['text']
                    desc = None
                else:
                    title = data_item['title']
                    desc = data_item['desc']
                text = text_concat(title, desc)

                token_ids = self.pipe.preprocess([text])[0]
                token_ids = token_ids.asnumpy()
                token_ids = torch.from_numpy(token_ids)

                # 图像
                frames = []

                if 'image' in data_item:
                    # get image by b64
                    try:
                        image_tensor = self.image_preprocess(data_item['image'])
                        frames.append(image_tensor)
                    except:
                        print(f"load image base64 failed -- {data_item['pid']}")
                        continue
                elif 'images' in data_item:
                    # get image by url
                    image = None
                    try:
                        for url in data_item['images']:
                            image_str = download_url_with_exception(get_original_url(url), timeout=3)
                            if image_str != b'' and image_str != '':
                                try:
                                    image = Image.open(io.BytesIO(image_str)).convert("RGB")
                                    break
                                except:
                                    continue
                            else:
                                pass
                    except:
                        pass
                    if image is not None:
                        image_tensor = self.preprocess(image)
                        frames.append(image_tensor)
                    else:
                        print(f"No images in data {data_item['pid']} -- zero of {len(data_item['images'])}")
                else:
                    raise Exception(f'cannot find image or images')

                input_dict = {'frames': frames,
                              'label': label,
                              'input_ids': token_ids}

                yield input_dict

            except Exception as e:
                pass

    def collect_fn(self, data):
        frames = []
        frames_mask = []
        labels = []
        input_ids = []
        input_mask = []
        input_segment_ids = []
        text_len = self.text_len

        for ib, ibatch in enumerate(data):
            labels.append(ibatch["label"])
            input_ids.append(ibatch['input_ids'])
            input_mask_id = ibatch['input_ids'].clone()
            input_mask_id[input_mask_id != 0] = 1
            input_mask.append(input_mask_id)
            input_segment_ids.append([0] * text_len)
            img_np = ibatch['frames']
            frames_mask_cur = []
            # 如果不够8帧，要补帧
            if len(img_np) < self.frame_len:
                # print('encouter not %s frames: %s ' % (self.frame_len, len(img_np)))
                for i, img in enumerate(img_np):
                    frames.append(img)
                    frames_mask_cur.append(1)
                for i in range(len(img_np), self.frame_len):
                    # print('*******add black frame**********')
                    frames.append(self.black_frame)  # 如果这个视频没有帧，就用黑帧来替代
                    frames_mask_cur.append(0)
            else:
                # 够的话就冲啊
                for i, img in enumerate(img_np):
                    frames.append(img)
                    frames_mask_cur.append(1)

            frames_mask.append(frames_mask_cur)

        frames_mask = torch.tensor(frames_mask)  # [bsz, frame_num]
        frames = torch.stack(frames, dim=0)  # [bsz * frame_num, c, h, w]
        _, c, h, w = frames.shape
        bsz, frame_num = frames_mask.shape
        frames = frames.reshape([bsz, frame_num, c, h, w])
        labels = torch.tensor(labels)
        input_ids = torch.cat(input_ids, dim=0)
        input_mask = torch.cat(input_mask, dim=0)
        # input_ids = torch.tensor(input_ids)
        # input_mask = torch.tensor(input_mask)
        input_segment_ids = torch.tensor(input_segment_ids)

        res = {"frames": frames, "frames_mask": frames_mask,
               "label": labels,
               "input_ids": input_ids, "input_mask": input_mask,
               "input_segment_ids": input_segment_ids,
               }
        return res

    def image_preprocess(self, image_str):
        image = self._load_image(self.b64_decode(image_str))
        image_tensor = self.preprocess(image)
        return image_tensor

    @staticmethod
    def b64_decode(string):
        if isinstance(string, str):
            string = string.encode()
        return base64.decodebytes(string)

    @staticmethod
    def _load_image(buffer):
        img = Image.open(io.BytesIO(buffer))
        img = img.convert('RGB')
        return img


def get_transform(mode: str = "train"):
    """
    根据不同的data，返回不同的transform
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if mode == "train":
        com_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    elif mode == 'val':
        com_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
    else:
        raise ValueError('mode [%s] is not in [train, val]' % mode)
    return com_transforms


class FacDataModule(CruiseDataModule):
    def __init__(
            self,
            train_files: str = None,
            train_size: int = 1500000,
            val_files: str = None,
            val_size: int = 16000,
            train_batch_size: int = 64,
            val_batch_size: int = 32,
            num_workers: int = 8,
            text_len: int = 128,
            frame_len: int = 1,
            exp: str = 'default',
    ):
        super().__init__()
        self.save_hparams()

    def local_rank_zero_prepare(self) -> None:
        pass

    def setup(self) -> None:
        self.train_dataset = TorchvisionLabelDataset(
            self.hparams,
            data_path=self.hparams.train_files,
            rank=int(os.environ.get('RANK') or 0),
            world_size=int(os.environ.get('WORLD_SIZE') or 1),
            shuffle=True,
            repeat=True,
            is_training=True)
        print(f'len of trainset: {len(self.train_dataset)}')

        self.val_dataset = TorchvisionLabelDataset(
            self.hparams,
            data_path=self.hparams.val_files,
            rank=int(os.environ.get('RANK') or 0),
            world_size=int(os.environ.get('WORLD_SIZE') or 1),
            # world_size=1,
            shuffle=False,
            repeat=False,
            is_training=False)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hparams.train_batch_size,
                                                   num_workers=self.hparams.num_workers,
                                                   pin_memory=True,
                                                   drop_last=True,
                                                   collate_fn=self.train_dataset.collect_fn)
        print(len(train_loader))
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.hparams.val_batch_size,
                                                 num_workers=1,
                                                 pin_memory=True,
                                                 drop_last=False,
                                                 collate_fn=self.train_dataset.collect_fn)
        return val_loader
