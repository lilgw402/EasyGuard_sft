"""
将dataloader 构建包装成一个个的函数。
支持3种预处理方式：
1. torchvision
2. dali
3. byted_vision
"""

import os
import torch

from fex.data import KVSampler, worker_init_fn, PytorchDaliIter
from fex.utils.prefetch import DataLoaderX

from example.clip.dataset import ImageDataset, get_transform
from example.clip.dataset_imagenet import ImageNetKVDataset
from example.clip.dali_pipeline import ImagePipeline
from fex.data.benchmark.imagenet.bytedvision_pipe import BytedvisionLoader

__all__ = ['create_torchvision_loader', 'create_dali_loader', 'create_bytedvision_loader']


def create_val_loader(config):
    val_transform = get_transform(mode="val")
    val_dataset = ImageNetKVDataset(
        config,
        config.dataset.val_path,
        num_readers=config.get('val.num_readers', 4),
        transform=val_transform,
        rank=int(os.environ.get('RANK') or 0),
        world_size=int(os.environ.get('WORLD_SIZE') or 1),
        shuffle=False,
        repeat=True)

    val_sampler = KVSampler(dataset=val_dataset,
                            batch_size=config.TRAINER.VAL_BATCH_SIZE,
                            num_replicas=int(
                                os.environ.get('WORLD_SIZE') or 1),
                            rank=int(os.environ.get('RANK') or 0))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=None,
                                             sampler=val_sampler,
                                             num_workers=config.get('val.num_workers', 4),
                                             pin_memory=True,
                                             drop_last=False,
                                             worker_init_fn=worker_init_fn,
                                             collate_fn=lambda x: x)
    return val_loader


def create_torchvision_loader(config):
    """
    构造 dataloader
    使用torchvision 做数据预处理
    train set：使用的是 DistLineReadingDataset，是一个我们内部实现的hdfs dataset封装
    val set：使用的是 KVDataset，底层是使用的 KVReader (https://bytedance.feishu.cn/docs/doccnqFaodw8Tp03v15zVUcwfLf#G8lQDl)
    do_prefetch: 是否使用prefetch的dataloader，只在训练集上生效
    """
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    train_transform = get_transform(mode="train")

    train_dataset = ImageDataset(
        config,
        config.dataset.train_path,
        transform=train_transform,
        rank=int(os.environ.get('RANK') or 0),
        world_size=int(os.environ.get('WORLD_SIZE') or 1),
        shuffle=True,
        repeat=True)

    if config.get('train.do_prefetch', False):
        train_loader = DataLoaderX(local_rank=local_rank,
                                   dataset=train_dataset, batch_size=config.TRAINER.TRAIN_BATCH_SIZE,
                                   num_workers=config.train.num_workers,
                                   pin_memory=True,
                                   drop_last=True,
                                   collate_fn=train_dataset.collect_fn)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAINER.TRAIN_BATCH_SIZE,
                                                   num_workers=config.train.num_workers,
                                                   pin_memory=True,
                                                   drop_last=True,
                                                   collate_fn=train_dataset.collect_fn)

    val_loader = create_val_loader(config)

    return train_loader, val_loader


def create_dali_loader(config):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    train_dataset = ImageDataset(
        config,
        config.dataset.train_path,
        rank=int(os.environ.get('RANK') or 0),
        world_size=int(os.environ.get('WORLD_SIZE') or 1),
        shuffle=True,
        repeat=True,
        preprocess_mode='dali')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAINER.TRAIN_BATCH_SIZE,
        num_workers=config.train.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collect_fn)

    # TODO: 这里一些预处理的参数写死了，后面做成可配置的
    train_pipeline = ImagePipeline(batch_size=config.TRAINER.TRAIN_BATCH_SIZE,
                                   num_threads=2, device_id=int(os.environ.get('LOCAL_RANK') or 0),
                                   external_data=train_loader,
                                   image_size=config.get('dataset.image_size', 224),
                                   prefetch_queue_depth=2,
                                   is_training=True
                                   )
    train_dali_iter = PytorchDaliIter(dali_pipeline=train_pipeline,
                                      output_map=["image"],
                                      auto_reset=True,
                                      last_batch_padded=True,
                                      fill_last_batch=False)

    val_loader = create_val_loader(config)

    return train_dali_iter, val_loader


def create_bytedvision_loader(config):
    """
    构造 dataloader
    使用 byted_vision 做数据预处理，相比 torchvision 会快。
    train set：使用的是 DistLineReadingDataset，是一个我们内部实现的hdfs dataset封装
    val set：使用的是 KVDataset，底层是使用的 KVReader (https://bytedance.feishu.cn/docs/doccnqFaodw8Tp03v15zVUcwfLf#G8lQDl)
    """
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    train_dataset = ImageDataset(
        config,
        config.dataset.train_path,
        rank=int(os.environ.get('RANK') or 0),
        world_size=int(os.environ.get('WORLD_SIZE') or 1),
        shuffle=True,
        repeat=True,
        preprocess_mode='bytedvision')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAINER.TRAIN_BATCH_SIZE,
                                               num_workers=config.train.num_workers,
                                               pin_memory=True,
                                               drop_last=True,
                                               collate_fn=train_dataset.collect_fn)

    data_iter = BytedvisionLoader(data_iter=train_loader,
                                  mode='train',
                                  output_map=['image'],
                                  device_id=int(os.environ.get('LOCAL_RANK') or 0)
                                  )

    val_loader = create_val_loader(config)

    return data_iter, val_loader
