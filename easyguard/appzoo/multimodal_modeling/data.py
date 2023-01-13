# -*- coding: utf-8 -*-

import json
from typing import List, Union

import numpy as np
import torch

try:
    import cruise
except ImportError:
    print(
        "[ERROR] cruise is not installed! Please refer this doc: https://bytedance.feishu.cn/wiki/wikcnGP7yzZAuKpPfL6jRJKl2ag"
    )

from cruise.data_module import CruiseDataModule
from cruise.data_module.cruise_loader import DistributedCruiseDataLoader
from cruise.utilities.hdfs_io import hexists, hlist_files, hopen

from .utils import ImageProcess, TextProcess


class ImageTextProcessor:
    def __init__(
        self,
        mode,
        vocab_path="hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/zh_old_cut_145607.vocab",
        max_len=256,
        category_path="hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/category_dict_pt.json",
    ):
        self.image_process = ImageProcess(mode)
        self.text_process = TextProcess(vocab_file=vocab_path, max_len=max_len)

        """
        Load category dict for category prediction
        """
        if not hexists(category_path):
            raise ValueError(
                "Category dict {} does not exist!".format(category_path)
            )
        with hopen(category_path, "r") as fp:
            self.category_dict = json.load(fp)

    def transform(self, data_dict: dict):
        image_urls = data_dict["main_images"]

        product_name = data_dict["product_name"]
        image_ocr = data_dict["main_ocr"]
        if isinstance(image_ocr, list):
            image_ocr = " ".join(image_ocr)

        token_ids, text_masks, text_segment_ids = self.text_process(
            product_name, image_ocr
        )
        image = self.image_process(image_urls)

        level1_cid = str(data_dict["first_cid_new"])
        level2_cid = str(data_dict["second_cid_new"])
        label = 0
        label_l1 = 0
        if level1_cid in self.category_dict["level1"]["id2idx"]:
            label_l1 = self.category_dict["level1"]["id2idx"][level1_cid] + 1
        if level2_cid in self.category_dict["level2"]["id2idx"]:
            label = self.category_dict["level2"]["id2idx"][level2_cid] + 1

        return {
            "token_ids": token_ids,
            "image": image,
            "label": label,
            "label_l1": label_l1,
        }

    def batch_transform(self, data):
        keys = list(data[0].keys())
        batch = {k: [] for k in keys}

        for i, ibatch in enumerate(data):
            for k in keys:
                batch[k].append(ibatch[k])

        for k in keys:
            if isinstance(batch[k][0], torch.Tensor):
                batch[k] = torch.stack(batch[k], dim=0)
            elif isinstance(batch[k][0], np.ndarray):
                batch[k] = torch.from_numpy(np.stack(batch[k], axis=0))
            else:
                batch[k] = torch.tensor(batch[k])

        return batch


class MMDataModule(CruiseDataModule):
    def __init__(
        self,
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        paths: Union[
            str, List[str]
        ] = "hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/pretrain_20220802_20220808_train_url",
        data_size: int = 140000000,
        val_step: int = 20,
        num_workers: int = 24,
        max_len: int = 256,
    ):
        super().__init__()
        self.save_hparams()

    def local_rank_zero_prepare(self) -> None:
        pass

    def setup(self, stage) -> None:
        paths = self.hparams.paths
        if isinstance(paths, str):
            paths = [paths]
        files = hlist_files(paths)
        files = [f for f in files if f.find("_SUCCESS") < 0]
        if not files:
            raise RuntimeError(
                f"No valid files can be found matching `paths`: {paths}"
            )
        files = sorted(files)
        self.train_files = files
        self.val_files = files

    def train_dataloader(self):
        return DistributedCruiseDataLoader(
            data_sources=[self.train_files],
            keys_or_columns=[None],
            batch_sizes=[self.hparams.train_batch_size],
            num_workers=self.hparams.num_workers,
            num_readers=[1],
            decode_fn_list=[[]],
            processor=ImageTextProcessor("train"),
            predefined_steps=self.hparams.data_size
            // self.hparams.train_batch_size
            // self.trainer.world_size,
            source_types=["jsonl"],
            shuffle=True,
        )

    def val_dataloader(self):
        return DistributedCruiseDataLoader(
            data_sources=[self.val_files],
            keys_or_columns=[None],
            batch_sizes=[self.hparams.val_batch_size],
            num_workers=self.hparams.num_workers // 2,
            num_readers=[1],
            decode_fn_list=[[]],
            processor=ImageTextProcessor("val"),
            predefined_steps=self.hparams.val_step,
            source_types=["jsonl"],
            shuffle=False,
        )
