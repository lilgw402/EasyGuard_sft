# -*- coding: utf-8 -*-
import os
from absl import app
from absl import flags

import cv2
import torch
import numpy as np

from fex.data.benchmark.imagenet import get_imagenet_dataloader

FLAGS = flags.FLAGS


def def_flags():
    flags.DEFINE_string('type',
                        'torchvision',
                        'process type')


def get_data_loader(vision_process_type):
    if vision_process_type == "torchvision":
        source, preprocess_type, data_path = 'hdfs', 'torchvision', 'hdfs://haruna/home/byte_search_nlp_lq/multimodal/data/academic/imagenet/train_json/part_55'
    elif vision_process_type == "dali":
        source, preprocess_type, data_path = 'hdfs', 'dali', 'hdfs://haruna/home/byte_search_nlp_lq/multimodal/data/academic/imagenet/train_json/part_55'
    elif vision_process_type == "bytedvision":
        source, preprocess_type, data_path = 'hdfs', 'bytedvision', 'hdfs://haruna/home/byte_search_nlp_lq/multimodal/data/academic/imagenet/train_json/part_55'
    else:
        raise TypeError("Not support process type: {}. ".format(vision_process_type))

    train_loader = get_imagenet_dataloader(
        data_path=data_path,
        batch_size=128,
        source=source,
        preprocess_type=vision_process_type,
        num_workers=0,
        mode='train')

    return train_loader


def run_torchvision_pipeline():
    image_dirs = "./proprocess_images_torchvision"
    if not os.path.exists(image_dirs):
        os.makedirs(image_dirs)

    train_loader = get_data_loader(vision_process_type="torchvision")

    it = iter(train_loader)
    batch_res = next(it)
    images = batch_res["image"]

    mean = torch.tensor(
        np.array([0.485, 0.456, 0.406])[:, None, None])
    std = torch.tensor(
        np.array([0.229, 0.224, 0.225])[:, None, None])
    output_denormalize = ((images * std + mean) * 255).int()
    print('-------> torchvision image shape: ', output_denormalize.shape)
    output_denormalize = output_denormalize.permute([0, 2, 3, 1]).numpy()

    for i in range(output_denormalize.shape[0]):
        cv2.imwrite(
            "./{}/torchvision_index_{}.jpeg".format(image_dirs, i), output_denormalize[i])


def run_dali_pipeline():
    image_dirs = "./proprocess_images_dali"
    if not os.path.exists(image_dirs):
        os.makedirs(image_dirs)

    train_loader = get_data_loader(vision_process_type="dali")

    it = iter(train_loader)
    batch_res = next(it)
    images = batch_res[0]["image"]

    mean = torch.tensor(
        np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])[:, None, None]).cuda()
    std = torch.tensor(
        np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])[:, None, None]).cuda()
    output_denormalize = (images * std + mean).int().cpu()
    print('-------> dali image shape: ', output_denormalize.shape)
    output_denormalize = output_denormalize.permute([0, 2, 3, 1]).numpy()

    for i in range(output_denormalize.shape[0]):
        cv2.imwrite(
            "./{}/dali_index_{}.jpeg".format(image_dirs, i), output_denormalize[i])


def run_bytedvision_pipeline():
    image_dirs = "./proprocess_images_bytedvision"
    if not os.path.exists(image_dirs):
        os.makedirs(image_dirs)

    train_loader = get_data_loader(vision_process_type="bytedvision")

    it = iter(train_loader)
    batch_res = next(it)
    images = batch_res["image"]

    mean = torch.tensor(
        np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])[:, None, None]).cuda()
    std = torch.tensor(
        np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])[:, None, None]).cuda()
    output_denormalize = (images * std + mean).int().cpu()
    print('-------> bytedvision image shape: ', output_denormalize.shape)
    output_denormalize = output_denormalize.permute([0, 2, 3, 1]).numpy()

    for i in range(output_denormalize.shape[0]):
        cv2.imwrite(
            "./{}/bytedvision_index_{}.jpeg".format(image_dirs, i), output_denormalize[i])


def main(_):
    if FLAGS.type == "torchvision":
        run_torchvision_pipeline()
    elif FLAGS.type == "dali":
        run_dali_pipeline()
    elif FLAGS.type == "bytedvision":
        run_bytedvision_pipeline()
    else:
        raise TypeError("Not support process type: {}. ".format(FLAGS.type))


if __name__ == "__main__":
    def_flags()
    app.run(main)
