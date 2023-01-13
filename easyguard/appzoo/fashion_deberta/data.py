"""An customizable fashion_deberta example"""
import os
import sys
from typing import List, Union

import torch
from text_cutter import Cutter
from text_tokenizer import BpeTokenizer

try:
    import cruise
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from cruise.data_module import CruiseDataModule
from cruise.utilities.hdfs_io import hlist_files

from .data_helpers import build_vocab


class TextProcessor:
    def __init__(
        self,
        text_label_field,
        text_field,
        tokenizer,
        context_length,
        vocab,
        classification_task_enable,
        cutter_enable,
        cutter,
        cl_enable,
        mlm_probability=0.15,
    ):
        self._text_label_field = text_label_field
        self._text_field = text_field
        self._tokenizer = tokenizer
        self._context_length = context_length
        self._mlm_probability = mlm_probability
        self._vocab = vocab
        self._classification_task_enable = classification_task_enable
        self._cutter_enable = cutter_enable
        self._cutter = cutter
        self._cl_enable = cl_enable
        self.PAD_IDX = self._vocab["[PAD]"]
        self.SEP_IDX = self._vocab["[SEP]"]
        self.CLS_IDX = self._vocab["[CLS]"]
        self.MASK_IDX = self._vocab["[MASK]"]

    def transform(self, data_dict: dict):
        # get image by key, order matters
        if not self._text_field in data_dict:
            raise KeyError(
                f"Unable to find text by keys: {self._text_field} available keys: {data_dict.keys()}"
            )
        text = data_dict.get(self._text_field, "")
        text_token = []
        if self._cutter_enable:
            for word in self._cutter.cut(text):
                text_token.extend(self._tokenizer(word))
        else:
            text_token = self._tokenizer(text)
        text_token = text_token[: self._context_length - 2]
        text_token_ids = [self._vocab[token] for token in text_token]
        text_token_ids = [self.CLS_IDX] + text_token_ids + [self.SEP_IDX]
        return_dict = {"input_ids": text_token_ids}
        if self._classification_task_enable:
            if not self._text_label_field in data_dict:
                raise KeyError(
                    f"Unable to find text by keys: {self._text_label_field}, available keys: {data_dict.keys()}"
                )
            label = int(data_dict.get(self._text_label_field))
            # label = int(ccr_label_map[data_dict.get(self._text_label_field)])
            return_dict["classification_labels"] = int(label)
        return return_dict

    def batch_transform(self, batch_data):
        # batch_data: List[Dict[modal, modal_value]]
        input_ids = []
        input_mask = []
        input_segment_ids = []
        classification_labels = []
        special_tokens_mask = []
        max_len = self._context_length

        for ib, ibatch in enumerate(batch_data):
            input_ids.append(
                ibatch["input_ids"][:max_len]
                + [self.PAD_IDX] * (max_len - len(ibatch["input_ids"]))
            )
            input_mask.append(
                [1] * len(ibatch["input_ids"][:max_len])
                + [0] * (max_len - len(ibatch["input_ids"]))
            )
            input_segment_ids.append([0] * max_len)
            special_tokens_mask.append(
                [1]
                + [0] * (len(ibatch["input_ids"][:max_len]) - 2)
                + [1]
                + [1] * (max_len - len(ibatch["input_ids"]))
            )

            if self._classification_task_enable:
                classification_labels.append(ibatch["classification_labels"])

        # for cl, double data
        if self._cl_enable:
            input_ids.extend(input_ids)  # same data extended in the end
            input_mask.extend(input_mask)
            input_segment_ids.extend(input_segment_ids)
            special_tokens_mask.extend(special_tokens_mask)
            if self._classification_task_enable:
                classification_labels.extend(classification_labels)

        special_tokens_mask = torch.tensor(special_tokens_mask)
        input_ids = torch.tensor(input_ids)
        mlm_input_ids, mlm_labels = self.torch_mask_tokens(
            input_ids, special_tokens_mask
        )
        input_mask = torch.tensor(input_mask)
        input_segment_ids = torch.tensor(input_segment_ids)

        res = {
            "mlm_labels": mlm_labels,
            "input_ids": mlm_input_ids,
            "input_masks": input_mask,
            "input_segment_ids": input_segment_ids,
        }
        if self._classification_task_enable:
            res["classification_labels"] = torch.tensor(classification_labels)
        return res

    def torch_mask_tokens(self, inputs, special_tokens_mask):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self._mlm_probability)

        special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool()
            & masked_indices
        )
        inputs[indices_replaced] = self.MASK_IDX

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self._vocab), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class FashionDataModule(CruiseDataModule):
    def __init__(
        self,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        train_paths: Union[
            str, List[str]
        ] = "hdfs://haruna/home/byte_ecom_govern/user/yangzheming/chinese/common_model/trails/ccr_v3_live_0.3b_mlm/traindata/part*",
        val_paths: Union[
            str, List[str]
        ] = "hdfs://haruna/home/byte_ecom_govern/user/yangzheming/chinese/common_model/trails/ccr_v3_live_0.3b_mlm/validdata/part*",
        data_size: int = 1000,
        val_step: int = 10,
        classification_task_enable: bool = False,
        text_label_field: str = "label",
        text_field: str = "text",
        cl_enable: bool = True,
        num_workers: int = 1,
        pretrained_model_dir: str = "hdfs://haruna/home/byte_ecom_govern/user/yangzheming/asr_model/zh_deberta_base_l6_emd_20210720",
        local_pretrained_model_dir_prefix="/opt/tiger/yangzheming/ckpt/",
        cutter_enable: bool = True,
        cutter_resource_dir: str = "hdfs://haruna/home/byte_ecom_govern/user/yangzheming/asr_model/libcut_data_zh_20200827fix2/",
        local_cutter_dir_prefix: str = "/opt/tiger/yangzheming/cutter/",
        context_length: int = 512,
        mlm_probability: float = 0.15,
    ):
        super().__init__()
        self.save_hparams()
        suffix = self.hparams.pretrained_model_dir.strip("/").split("/")[-1]
        self.local_pretrained_model_dir = (
            f"{self.hparams.local_pretrained_model_dir_prefix}/{suffix}"
        )
        if self.hparams.cutter_enable:
            suffix = self.hparams.cutter_resource_dir.strip("/").split("/")[-1]
            self.local_cutter_dir = (
                f"{self.hparams.local_cutter_dir_prefix}/{suffix}"
            )

    def local_rank_zero_prepare(self) -> None:
        # download cutter resource
        if self.hparams.cutter_enable:
            if not os.path.exists(self.local_cutter_dir):
                os.makedirs(self.hparams.local_cutter_dir_prefix, exist_ok=True)
                os.system(
                    f"hdfs dfs -copyToLocal {self.hparams.cutter_resource_dir} {self.local_cutter_dir}"
                )

    def setup(self, stage) -> None:
        train_paths = self.hparams.train_paths
        if isinstance(train_paths, str):
            train_paths = [train_paths]
        val_paths = self.hparams.val_paths
        if val_paths:
            if isinstance(val_paths, str):
                val_paths = [val_paths]
            train_files = hlist_files(train_paths)
            val_files = hlist_files(val_paths)
            if not train_files:
                raise RuntimeError(
                    f"No valid files can be found matching `paths`: {train_paths}"
                )
            if not val_files:
                raise RuntimeError(
                    f"No valid files can be found matching `paths`: {val_paths}"
                )
            self.train_files = train_files
            self.val_files = val_files
        else:
            # split train/val
            files = hlist_files(train_paths)
            if not files:
                raise RuntimeError(
                    f"No valid files can be found matching `paths`: {train_paths}"
                )
            # use the last file as validation
            self.train_files = files[:-2]
            self.val_files = files[-2:]
        self.text_label_field = self.hparams.text_label_field
        self.text_field = self.hparams.text_field
        self.tokenizer = BpeTokenizer(
            self.local_pretrained_model_dir + "/vocab.txt",
            wordpiece_type="bert",
            lower_case=False,
        )
        self.vocab = build_vocab(self.local_pretrained_model_dir + "/vocab.txt")
        self.cutter = (
            Cutter("CRF_LARGE", self.local_cutter_dir)
            if self.hparams.cutter_enable
            else None
        )
        self.cl_enable = self.hparams.cl_enable

    def train_dataloader(self):
        from cruise.data_module.cruise_loader import DistributedCruiseDataLoader

        return DistributedCruiseDataLoader(
            data_sources=[self.train_files],
            keys_or_columns=[None],
            batch_sizes=[self.hparams.train_batch_size],
            num_workers=self.hparams.num_workers,
            num_readers=[1],
            decode_fn_list=[[]],
            processor=TextProcessor(
                self.text_label_field,
                self.text_field,
                self.tokenizer,
                self.hparams.context_length,
                self.vocab,
                self.hparams.classification_task_enable,
                self.hparams.cutter_enable,
                self.cutter,
                self.cl_enable,
                mlm_probability=self.hparams.mlm_probability,
            ),
            predefined_steps=self.hparams.data_size
            // self.hparams.train_batch_size
            // self.trainer.world_size,
            source_types=["jsonl"],
            shuffle=True,
        )

    def val_dataloader(self):
        from cruise.data_module.cruise_loader import DistributedCruiseDataLoader

        return DistributedCruiseDataLoader(
            data_sources=[self.val_files],
            keys_or_columns=[None],
            batch_sizes=[self.hparams.val_batch_size],
            num_workers=self.hparams.num_workers,
            num_readers=[1],
            decode_fn_list=[[]],
            processor=TextProcessor(
                self.text_label_field,
                self.text_field,
                self.tokenizer,
                self.hparams.context_length,
                self.vocab,
                self.hparams.classification_task_enable,
                self.hparams.cutter_enable,
                self.cutter,
                self.cl_enable,
                mlm_probability=self.hparams.mlm_probability,
            ),
            predefined_steps=self.hparams.val_step,
            source_types=["jsonl"],
            shuffle=False,
        )
