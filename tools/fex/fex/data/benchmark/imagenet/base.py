#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import numpy as np
import math
import threading
import traceback
import base64
import io
from PIL import Image
import torch
from torch.utils.data import Dataset

import torchvision
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms

try:
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator
    from nvidia.dali.plugin.pytorch import LastBatchPolicy
except Exception as e:
    print("DALI is not installd, please install it from https://www.github.com/NVIDIA/DALI")

from fex import _logger as logger
from fex.config import CfgNode
from fex.utils.hdfs_io import hopen, hlist_files
from fex.data import PytorchDaliIter
from fex.data.datasets.dist_dataset_v2 import DistLineReadingDatasetV2

from .dali_pipe import ExternalSourcePipeline
from .bytedvision_pipe import BytedvisionLoader


def get_imagenet_dataloader(data_path='hdfs://haruna/home/byte_search_nlp_lq/multimodal/data/academic/imagenet/val_json',
                            batch_size=512,
                            source='hdfs',
                            preprocess_type='torchvision',
                            mode='val',
                            **kwargs):
    """
    folder 只支持torchvision
    hdfs 可以支持各种预处理形式。

    """
    assert source in ['hdfs', 'folder']
    assert preprocess_type in ['torchvision', 'dali', 'bytedvision']
    assert mode in ['train', 'val']
    func_name = f'get_{source}_{preprocess_type}_{mode}_dataloader'
    logger.info(f'using {func_name} ...')
    return eval(func_name)(data_path, batch_size, **kwargs)


def get_folder_torchvision_val_dataloader(data_path, batch_size):
    """
    folder 版本的dataloader，官方的
    """
    preprocessing = ClassificationPresetEval()
    dataset_test = torchvision.datasets.ImageFolder(
        data_path,
        preprocessing,
    )
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, sampler=test_sampler, num_workers=8, pin_memory=True
    )
    return data_loader_test


def get_hdfs_torchvision_val_dataloader(data_path, batch_size, num_workers=2):
    """
    hdfs 版本的dataloader，用torhciviosn
    """
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    val_dataset = ImageNetJsonDataset(data_path, return_dict=True, mode="eval")
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             sampler=val_sampler,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             collate_fn=val_dataset.collate_fn,
                                             pin_memory=True)
    return val_loader


def get_hdfs_torchvision_train_dataloader(data_path, batch_size, num_workers=8):
    """
    hdfs 版本的dataloader，用torhciviosn
    """
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    dataset = ImageNetJsonIterDataset(data_path,
                                      preprocess_type='torchvision',
                                      return_dict=True,
                                      rank=rank,
                                      world_size=world_size,
                                      shuffle=True,
                                      repeat=True,
                                      buffer_size=1024,
                                      mode="train")
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         collate_fn=dataset.collate_fn,
                                         pin_memory=True)
    return loader


def get_hdfs_dali_val_dataloader(data_path, batch_size, num_workers=2, image_size=224):
    """
    hdfs 版本的dataloader，用 dali
    from: https://github.com/NVIDIA/DALI/blob/629c57592b9b4e91b8213e6c77c1af179f7dd079/docs/examples/use_cases/pytorch/resnet50/main.py#L95
    """
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    val_dataset = ImageNetJsonDataset(data_path, preprocess_type='dali')
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             sampler=val_sampler,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             collate_fn=val_dataset.collate_fn,
                                             pin_memory=True)

    pipe = ExternalSourcePipeline(external_data=val_loader,
                                  batch_size=batch_size,
                                  size=256,
                                  crop=224,
                                  is_training=False,
                                  dali_cpu=False,
                                  num_threads=2,
                                  device_id=int(os.environ.get('LOCAL_RANK') or 0),
                                  )
    dali_iter = PyTorchIterator(pipe,
                                output_map=['image', 'label'],
                                last_batch_padded=True,
                                last_batch_policy=LastBatchPolicy.PARTIAL)
    return dali_iter


def get_hdfs_dali_train_dataloader(data_path, batch_size, num_workers=8, image_size=224):
    """
    hdfs 版本的dataloader，用 dali
    from: https://github.com/NVIDIA/DALI/blob/629c57592b9b4e91b8213e6c77c1af179f7dd079/docs/examples/use_cases/pytorch/resnet50/main.py#L95
    """
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dataset = ImageNetJsonIterDataset(data_path,
                                      preprocess_type='dali',
                                      rank=int(os.environ.get('RANK') or 0),
                                      world_size=int(os.environ.get('WORLD_SIZE') or 1),
                                      shuffle=True,
                                      repeat=True,
                                      buffer_size=1024)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         collate_fn=dataset.collate_fn,
                                         pin_memory=True)

    pipe = ExternalSourcePipeline(external_data=loader,
                                  batch_size=batch_size,
                                  crop=image_size,
                                  is_training=True,
                                  dali_cpu=False,
                                  num_threads=2,
                                  device_id=int(os.environ.get('LOCAL_RANK') or 0),
                                  )
    dali_iter = PyTorchIterator(pipe,
                                output_map=['image', 'label'],
                                last_batch_padded=True,
                                last_batch_policy=LastBatchPolicy.PARTIAL)
    return dali_iter


def get_hdfs_bytedvision_val_dataloader(data_path, batch_size, num_workers=2, image_size=224):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    val_dataset = ImageNetJsonDataset(data_path, preprocess_type='bytedvision', return_dict=True)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             sampler=val_sampler,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             collate_fn=val_dataset.collate_fn,
                                             pin_memory=True)

    data_iter = BytedvisionLoader(data_iter=val_loader,
                                  mode='val',
                                  output_map=['image'],
                                  device_id=int(os.environ.get('LOCAL_RANK') or 0)
                                  )
    return data_iter


def get_hdfs_bytedvision_train_dataloader(data_path, batch_size, num_workers=8, image_size=224):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    dataset = ImageNetJsonIterDataset(data_path,
                                      preprocess_type='bytedvision',
                                      return_dict=True,
                                      rank=int(os.environ.get('RANK') or 0),
                                      world_size=int(os.environ.get('WORLD_SIZE') or 1),
                                      shuffle=True,
                                      repeat=True,
                                      buffer_size=1024)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         collate_fn=dataset.collate_fn,
                                         pin_memory=True)
    # meta_data_path=config.get('dataset.train_meta_data'),
    # state_path=state_path
    data_iter = BytedvisionLoader(data_iter=loader,
                                  mode='train',
                                  output_map=['image'],
                                  device_id=int(os.environ.get('LOCAL_RANK') or 0)
                                  )
    return data_iter


class ImageNetJsonDataset(Dataset):
    """
    ImageNet Hdfs Json 的数据集
    """

    def __init__(self, data_path, preprocess_type='torchvision', *args, **kwargs):
        super().__init__()
        self.debug = False
        self.preprocess_type = preprocess_type
        if self.preprocess_type == 'torchvision':
            self.preprocessing = ClassificationPresetEval(mode=kwargs.get('mode', "eval"))
        self._load_all_set(data_path)  # self.datapool
        # self.datapool = sorted(self.datapool, key=lambda x: x['info']['label_idx'])  # 需要sort一下，否则load的时候顺序不同，会造成一些精度问题。
        self.return_dict = kwargs.get('return_dict', False)
        self.need_text = kwargs.get('need_text', False)

    def __len__(self):
        return len(self.datapool)

    def __getitem__(self, index):
        """
        return: image, label
        """
        example = self.datapool[index]
        image = self.process_image(example['image'])  # 总是框图的
        return {'image': image, 'label': int(example['info']['label_idx'])}

    def collate_fn(self, batch):
        image = []
        label = []
        for ibatch in batch:
            image.append(ibatch['image'])
            label.append(ibatch['label'])
        if self.preprocess_type == 'torchvision':
            image = torch.stack(image, dim=0)
        label = torch.tensor(label, dtype=torch.int64)
        if self.return_dict:
            return {'image': image, 'label': label}
        else:
            return image, label

    def process_image(self, image_str):
        '''
        单条图片处理。
        torchvisoin 是现场处理；dali是需要np读一下；bytedvision是啥也不用干
        '''
        image_str = self.b64_decode(image_str)
        if self.preprocess_type == 'torchvision':
            image_str = Image.open(io.BytesIO(image_str)).convert("RGB")
            image = self.preprocessing(image_str)
        elif self.preprocess_type == 'dali':
            image = np.array(np.frombuffer(image_str, dtype=np.uint8))
        elif self.preprocess_type == 'bytedvision':
            image = image_str
        return image

    def b64_decode(self, string):
        if isinstance(string, str):
            string = string.encode()
        return base64.decodebytes(string)

    def _load_all_set(self, data_path):
        """ 加载整个集 """
        self.datapool = []
        st = time.time()
        logger.info('loading images from {} ...'.format(data_path))
        files = hlist_files([data_path])
        # files = [files[1]]
        logger.info(f'hdfs files: {len(files)}, e.g. {files[0]} ...')
        thread_num = min(16, len(files))
        file_per_process = math.ceil(len(files) / thread_num)

        threads = []
        for thread_idx in range(thread_num):
            begin_index = int(file_per_process * thread_idx)
            end_index = int(file_per_process * (thread_idx + 1))
            t = threading.Thread(target=self._load_one_part, args=(files[begin_index:end_index],))
            t.start()
            threads.append(t)

        for index, t in enumerate(threads):
            t.join()
        logger.info('{} data loaded, cost time {} !'.format(len(self.datapool), time.time() - st))

    def _load_one_part(self, files):
        # for path in tqdm.tqdm(files):
        for path in files:
            with hopen(path) as f:
                for l in f:
                    jl = json.loads(l)
                    self.datapool.append(jl)  # list 的 append 是线程安全的
                    if self.debug and len(self.datapool) > 100:
                        break


class ImageNetJsonIterDataset(DistLineReadingDatasetV2):
    """
    iterate 的版本
    """

    def __init__(self,
                 data_path: str,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle: bool = True,
                 repeat: bool = False,
                 preprocess_type: str = 'torchvision',
                 buffer_size: int = -1,
                 meta_data_path: str = None,
                 state_path: str = None,
                 *args, **kwargs):
        super().__init__(data_path, rank, world_size, shuffle, repeat,
                         buffer_size=buffer_size,
                         meta_data_path=meta_data_path,
                         state_path=state_path)
        if preprocess_type not in ['torchvision', 'dali', 'bytedvision']:
            raise ValueError(f"preprocess_type {preprocess_type} not in ['torchvision', 'dali', 'bytedvision']")

        self.preprocess_type = preprocess_type
        if self.preprocess_type == 'torchvision':
            self.preprocessing = ClassificationPresetEval(mode=kwargs.get('mode', "train"))

        self.return_dict = kwargs.get('return_dict', False)

    def __iter__(self):
        for example, filepath, line_id in self.generate():
            try:
                data_item = json.loads(example)
                image = self.process_image(data_item['image'])  # 总是框图的
                yield {'image': image, 'label': int(data_item['info']['label_idx'])}
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error('encounter broken data: %s' % e)

    def process_image(self, image_str):
        '''
        单条图片处理。
        torchvisoin 是现场处理；dali是需要np读一下；bytedvision是啥也不用干
        '''
        image_str = self.b64_decode(image_str)
        if self.preprocess_type == 'torchvision':
            image_str = Image.open(io.BytesIO(image_str)).convert("RGB")
            image = self.preprocessing(image_str)
        elif self.preprocess_type == 'dali':
            image = np.array(np.frombuffer(image_str, dtype=np.uint8))
        elif self.preprocess_type == 'bytedvision':
            image = image_str
        return image

    def b64_decode(self, string):
        if isinstance(string, str):
            string = string.encode()
        return base64.decodebytes(string)

    def collate_fn(self, batch):
        image = []
        label = []
        for ibatch in batch:
            image.append(ibatch['image'])
            label.append(ibatch['label'])
        if self.preprocess_type == 'torchvision':
            image = torch.stack(image, dim=0)
        label = torch.tensor(label, dtype=torch.int64)
        if self.return_dict:
            return {'image': image, 'label': label}
        else:
            return image, label


class ClassificationPresetEval:
    def __init__(
        self,
        crop_size=224,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        mode="train"
    ):
        if mode == "train":
            self.transforms = transforms.Compose(
                [
                    transforms.RandomResizedCrop(crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

        else:
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(resize_size, interpolation=interpolation),
                    transforms.CenterCrop(crop_size),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

    def __call__(self, img):
        return self.transforms(img)
