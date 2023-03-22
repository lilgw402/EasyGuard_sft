"""Unsupervised datamodule for GPT pretraining"""
import logging
import os
import tempfile
import torch
from typing import Union, List
from cruise import CruiseDataModule
from cruise.data_module.tools import create_dataloader_by_cfg
from torch.utils.data._utils.collate import default_collate
from cruise.data_module import DistributedCruiseDataLoader
from cruise.data_module.gpu_wrapper import GPUPrefetcher
from cruise.utilities.hdfs_io import hlist_files, hcopy
from cruise.utilities import DIST_ENV
from ..tokenization import CasterTokenizer


class RTEProcessor:
    r"""
        Args:
            tokenizer: the name of the pretrained tokenizer, e.g., "bigscience/bloom"
            text_keys: keys that contains text as values in the input.
            max_seq_len: max length that the model accept, if data is not enough,
                         pad_token_id will be used.
    """
    # dataset description: https://github.com/cluebenchmark/OCNLI
    def __init__(self, tokenizer: str, text_keys: Union[str, List[str]], max_seq_len:int, **kwargs):
        # used test_keys: sentence1, setence2, label
        # sentence1: the premise sentence(s) 句子1，即前提。
        # sentence2: the hypothesis sentence(s) 句子2，即假设。
        if not isinstance(text_keys, list):
            text_keys = [text_keys]
        self.text_keys = text_keys
        if not isinstance(tokenizer, str):
            # from created tokenizer object
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_seq_len = max_seq_len
        # We will automatically convert token list to tensor
        kwargs.pop('return_tensors', None)
        self.kwargs = kwargs

    def transform_classification_label(self, classification_label_id):
        classification_label_map = {
            'not_entailment' : 0,
            'entailment' : 1,
            }
        return classification_label_map.get(classification_label_id, 0)

    def transform(self, data_dict):
        text_dict = {}
        text_output = self.tokenizer(data_dict['premise'] + u'$' + data_dict['hypothesis'], **self.kwargs)
        for k, v in text_output.items():
            # append EOS token
            if 'mask' in k:
                eos_token_id = 1  # EOS mask is still 1
            else:
                eos_token_id = self.tokenizer.eos_token_id
            v += [eos_token_id]
            if k not in text_dict:
                text_dict[k] = v
            else:
                text_dict[k] = text_dict[k] + v
        # add classification_labels
        text_dict['classification_labels'] = self.transform_classification_label(data_dict['label'])
        
        return self.post_process(text_dict)

    def batch_transform(self, batch_data):
        return default_collate(batch_data)

    def post_process(self, example):
        total_length = len(example['input_ids'])

        if total_length < self.max_seq_len:
            example['input_ids'] = example['input_ids'][:total_length] + [self.tokenizer.pad_token_id] * (self.max_seq_len - total_length)
            example['attention_mask'] = example['attention_mask'][:total_length] + [0] * (self.max_seq_len - total_length)
        else:
            example['input_ids'] = example['input_ids'][:self.max_seq_len]
            example['attention_mask'] = example['attention_mask'][:self.max_seq_len]
            total_length = self.max_seq_len

        outputs = [
            {
                'input_ids': torch.as_tensor(example['input_ids']),
                'attention_mask': torch.as_tensor(example['attention_mask']),
                'classification_labels': torch.as_tensor([example['classification_labels']]),
                "actual_seq_length": total_length,
            }
        ]
        return outputs

class IFLYTEKProcessor:
    r"""
        Args:
            tokenizer: the name of the pretrained tokenizer, e.g., "bigscience/bloom"
            text_keys: keys that contains text as values in the input.
            max_seq_len: max length that the model accept, if data is not enough,
                         pad_token_id will be used.
    """
    # dataset description: https://github.com/cluebenchmark/OCNLI
    def __init__(self, tokenizer: str, text_keys: Union[str, List[str]], max_seq_len:int, **kwargs):
        # used test_keys: sentence1, setence2, label
        # sentence1: the premise sentence(s) 句子1，即前提。
        # sentence2: the hypothesis sentence(s) 句子2，即假设。
        if not isinstance(text_keys, list):
            text_keys = [text_keys]
        self.text_keys = text_keys
        if not isinstance(tokenizer, str):
            # from created tokenizer object
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_seq_len = max_seq_len
        # We will automatically convert token list to tensor
        kwargs.pop('return_tensors', None)
        self.kwargs = kwargs
        
        int_label = [i for i in range(119)]
        str_label = [str(i) for i in int_label]
        self.classification_label_map = {
            str_label[i]: int_label[i] for i in range(len(int_label))
        }
        # print("IFLYTEK label mapping: {}".format(self.classification_label_map))

    def transform_classification_label(self, classification_label_id):
        return self.classification_label_map.get(classification_label_id, 118)

    def transform(self, data_dict):
        text_dict = {}
        text_output = self.tokenizer(data_dict['sentence'], **self.kwargs)
        for k, v in text_output.items():
            # append EOS token
            if 'mask' in k:
                eos_token_id = 1  # EOS mask is still 1
            else:
                eos_token_id = self.tokenizer.eos_token_id
            v += [eos_token_id]
            if k not in text_dict:
                text_dict[k] = v
            else:
                text_dict[k] = text_dict[k] + v
        # add classification_labels
        text_dict['classification_labels'] = self.transform_classification_label(data_dict['label'])
        
        return self.post_process(text_dict)

    def batch_transform(self, batch_data):
        return default_collate(batch_data)

    def post_process(self, example):
        total_length = len(example['input_ids'])

        if total_length < self.max_seq_len:
            example['input_ids'] = example['input_ids'][:total_length] + [self.tokenizer.pad_token_id] * (self.max_seq_len - total_length)
            example['attention_mask'] = example['attention_mask'][:total_length] + [0] * (self.max_seq_len - total_length)
        else:
            example['input_ids'] = example['input_ids'][:self.max_seq_len]
            example['attention_mask'] = example['attention_mask'][:self.max_seq_len]
            total_length = self.max_seq_len

        outputs = [
            {
                'input_ids': torch.as_tensor(example['input_ids']),
                'attention_mask': torch.as_tensor(example['attention_mask']),
                'classification_labels': torch.as_tensor([example['classification_labels']]),
                "actual_seq_length": total_length,

            }
        ]
        # print("outputs: {}".format(outputs))
        return outputs


class AFQMCProcessor:
    r"""
        Args:
            tokenizer: the name of the pretrained tokenizer, e.g., "bigscience/bloom"
            text_keys: keys that contains text as values in the input.
            max_seq_len: max length that the model accept, if data is not enough,
                         pad_token_id will be used.
    """
    # dataset description: https://github.com/cluebenchmark/OCNLI
    def __init__(self, tokenizer: str, text_keys: Union[str, List[str]], max_seq_len:int, **kwargs):
        # used test_keys: sentence1, setence2, label
        # sentence1: the premise sentence(s) 句子1，即前提。
        # sentence2: the hypothesis sentence(s) 句子2，即假设。
        if not isinstance(text_keys, list):
            text_keys = [text_keys]
        self.text_keys = text_keys
        if not isinstance(tokenizer, str):
            # from created tokenizer object
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_seq_len = max_seq_len
        # We will automatically convert token list to tensor
        kwargs.pop('return_tensors', None)
        self.kwargs = kwargs

    def transform_classification_label(self, classification_label_id):
        classification_label_map = {
            '0' : 0,
            '1' : 1,
        }
        return classification_label_map.get(classification_label_id, 0)

    def transform(self, data_dict):
        text_dict = {}
        text_output = self.tokenizer(data_dict['sentence1'] + u'$' + data_dict['sentence2'], **self.kwargs)
        for k, v in text_output.items():
            # append EOS token
            if 'mask' in k:
                eos_token_id = 1  # EOS mask is still 1
            else:
                eos_token_id = self.tokenizer.eos_token_id
            v += [eos_token_id]
            if k not in text_dict:
                text_dict[k] = v
            else:
                text_dict[k] = text_dict[k] + v
        # add classification_labels
        text_dict['classification_labels'] = self.transform_classification_label(data_dict['label'])
        
        return self.post_process(text_dict)

    def batch_transform(self, batch_data):
        return default_collate(batch_data)

    def post_process(self, example):
        total_length = len(example['input_ids'])

        if total_length < self.max_seq_len:
            example['input_ids'] = example['input_ids'][:total_length] + [self.tokenizer.pad_token_id] * (self.max_seq_len - total_length)
            example['attention_mask'] = example['attention_mask'][:total_length] + [0] * (self.max_seq_len - total_length)
        else:
            example['input_ids'] = example['input_ids'][:self.max_seq_len]
            example['attention_mask'] = example['attention_mask'][:self.max_seq_len]
            total_length = self.max_seq_len

        outputs = [
            {
                'input_ids': torch.as_tensor(example['input_ids']),
                'attention_mask': torch.as_tensor(example['attention_mask']),
                'classification_labels': torch.as_tensor([example['classification_labels']]),
                "actual_seq_length": total_length,

            }
        ]
        # print("outputs: {}".format(outputs))
        return outputs


class OCNLIProcessor:
    r"""
        Args:
            tokenizer: the name of the pretrained tokenizer, e.g., "bigscience/bloom"
            text_keys: keys that contains text as values in the input.
            max_seq_len: max length that the model accept, if data is not enough,
                         pad_token_id will be used.
    """
    # dataset description: https://github.com/cluebenchmark/OCNLI
    def __init__(self, tokenizer: str, text_keys: Union[str, List[str]], max_seq_len:int, **kwargs):
        # used test_keys: sentence1, setence2, label
        # sentence1: the premise sentence(s) 句子1，即前提。
        # sentence2: the hypothesis sentence(s) 句子2，即假设。
        if not isinstance(text_keys, list):
            text_keys = [text_keys]
        self.text_keys = text_keys
        if not isinstance(tokenizer, str):
            # from created tokenizer object
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_seq_len = max_seq_len
        # We will automatically convert token list to tensor
        kwargs.pop('return_tensors', None)
        self.kwargs = kwargs

    def transform_classification_label(self, classification_label_id):
        classification_label_map = {
            'neutral' : 0,
            'entailment' : 1,
            'contradiction' : 2,
        }
        return classification_label_map.get(classification_label_id, 0)

    def transform(self, data_dict):
        text_dict = {}
        text_output = self.tokenizer(data_dict['sentence1'] + u'$' + data_dict['sentence2'], **self.kwargs)
        for k, v in text_output.items():
            # append EOS token
            if 'mask' in k:
                eos_token_id = 1  # EOS mask is still 1
            else:
                eos_token_id = self.tokenizer.eos_token_id
            v += [eos_token_id]
            if k not in text_dict:
                text_dict[k] = v
            else:
                text_dict[k] = text_dict[k] + v
        # add classification_labels
        text_dict['classification_labels'] = self.transform_classification_label(data_dict['label'])
        
        return self.post_process(text_dict)

    def batch_transform(self, batch_data):
        return default_collate(batch_data)

    def post_process(self, example):
        total_length = len(example['input_ids'])

        if total_length < self.max_seq_len:
            example['input_ids'] = example['input_ids'][:total_length] + [self.tokenizer.pad_token_id] * (self.max_seq_len - total_length)
            example['attention_mask'] = example['attention_mask'][:total_length] + [0] * (self.max_seq_len - total_length)
        else:
            example['input_ids'] = example['input_ids'][:self.max_seq_len]
            example['attention_mask'] = example['attention_mask'][:self.max_seq_len]
            total_length = self.max_seq_len

        outputs = [
            {
                'input_ids': torch.as_tensor(example['input_ids']),
                'attention_mask': torch.as_tensor(example['attention_mask']),
                'classification_labels': torch.as_tensor([example['classification_labels']]),
                "actual_seq_length": total_length,

            }
        ]
        # print("outputs: {}".format(outputs))
        return outputs

class TNewsProcessor:
    r"""
        Args:
            tokenizer: the name of the pretrained tokenizer, e.g., "bigscience/bloom"
            text_keys: keys that contains text as values in the input.
            max_seq_len: max length that the model accept, if data is not enough,
                         pad_token_id will be used.
    """
    def __init__(self, tokenizer: str, text_keys: Union[str, List[str]], max_seq_len:int, **kwargs):
        if not isinstance(text_keys, list):
            text_keys = [text_keys]
        self.text_keys = text_keys
        if not isinstance(tokenizer, str):
            # from created tokenizer object
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_seq_len = max_seq_len
        # We will automatically convert token list to tensor
        kwargs.pop('return_tensors', None)
        self.kwargs = kwargs

    def transform_classification_label(self, classification_label_id):
        classification_label_map = {
            '100' : 0,
            '101' : 1,
            '102' : 2,
            '103' : 3,
            '104' : 4,
            '105' : 5,
            '106' : 6,
            '107' : 7,
            '108' : 8,
            '109' : 9,
            '110' : 10,
            '111' : 11,
            '112' : 12,
            '113' : 13,
            '114' : 14,
            '115' : 15,
            '116' : 16,
        }

        return classification_label_map.get(classification_label_id, 0)

    def transform(self, data_dict):
        text_dict = {}
        # add text
        text_output = self.tokenizer(data_dict['sentence'] + u'，' + data_dict['keywords'], **self.kwargs)
        for k, v in text_output.items():
            # append EOS token
            if 'mask' in k:
                eos_token_id = 1  # EOS mask is still 1
            else:
                eos_token_id = self.tokenizer.eos_token_id
            v += [eos_token_id]
            if k not in text_dict:
                text_dict[k] = v
            else:
                text_dict[k] = text_dict[k] + v
        
        # add classification_labels
        text_dict['classification_labels'] = self.transform_classification_label(data_dict['label'])
        # print("text_dict: ".format(text_dict))

        return self.post_process(text_dict)

    def batch_transform(self, batch_data):
        return default_collate(batch_data)

    def post_process(self, example):
        total_length = len(example['input_ids'])

        if total_length < self.max_seq_len:
            example['input_ids'] = example['input_ids'][:total_length] + [self.tokenizer.pad_token_id] * (self.max_seq_len - total_length)
            example['attention_mask'] = example['attention_mask'][:total_length] + [0] * (self.max_seq_len - total_length)
        else:
            example['input_ids'] = example['input_ids'][:self.max_seq_len]
            example['attention_mask'] = example['attention_mask'][:self.max_seq_len]
            total_length = self.max_seq_len

        outputs = [
            {
                'input_ids': torch.as_tensor(example['input_ids']),
                'attention_mask': torch.as_tensor(example['attention_mask']),
                'classification_labels': torch.as_tensor([example['classification_labels']]),
                "actual_seq_length": total_length

            }
        ]

        return outputs
'''
# entailment task
RTE_Task_config = { 
    "train_path": 'hdfs://haruna/home/byte_data_aml_research/user/jiankai.sun/public_dataset/RTE/split_train',
    "val_path": 'hdfs://haruna/home/byte_data_aml_research/user/jiankai.sun/public_dataset/RTE/split_dev',
    "train_size": 2490,            
    "max_seq_len": 1024,
    "text_keys": ['premise', 'hypothesis', 'label'],
    "tokenizer": 'hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/tokenizer/zh_0620_newcut_caster_145665_lowercase',
}

# similarity task
AFQMC_Task_config = {
    "train_path": 'hdfs://haruna/home/byte_data_aml_research/user/jiankai.sun/public_dataset/AFQMC/split_train',
    "val_path": 'hdfs://haruna/home/byte_data_aml_research/user/jiankai.sun/public_dataset/AFQMC/split_dev',
    "train_size": 34334,            
    "max_seq_len": 1024,
    "text_keys": ['sentence1', 'sentence2', 'label'],
}

# entailment classification task
OCNLI_Task_config = { 
    "train_path": 'hdfs://haruna/home/byte_data_aml_research/user/jiankai.sun/public_dataset/OCNLI/split_train',
    "val_path": 'hdfs://haruna/home/byte_data_aml_research/user/jiankai.sun/public_dataset/OCNLI/split_dev',
    "train_size": 50485,            
    "max_seq_len": 1024,
    "text_keys": ['sentence1', 'sentence2', 'label'],
    "tokenizer": 'hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/tokenizer/zh_0620_newcut_caster_145665_lowercase',
}

# entailment classification task
CMNLI_Task_config = { 
    "train_path": 'hdfs://haruna/home/byte_data_aml_research/user/jiankai.sun/public_dataset/CMNLI/split_train',
    "val_path": 'hdfs://haruna/home/byte_data_aml_research/user/jiankai.sun/public_dataset/CMNLI/split_dev',
    "train_size": 391783,            
    "max_seq_len": 1024,
    "text_keys": ['sentence1', 'sentence2', 'label'],
    "tokenizer": 'hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/tokenizer/zh_0620_newcut_caster_145665_lowercase',
}

# classification task
TNews_Task_config = {
    "train_path": 'hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/tnews_public/split_train',
    "val_path": 'hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/tnews_public/split_dev',
    "train_size": 53360,            
    "max_seq_len": 1024,
    "text_keys": ['sentence', 'keywords', 'label', 'label_desc'],
    "tokenizer": 'hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/tokenizer/zh_0620_newcut_caster_145665_lowercase',
}

# classification task
IFLYTEK_Task_config = { 
    "train_path": 'hdfs://haruna/home/byte_data_aml_research/user/jiankai.sun/public_dataset/IFLYTEK/split_train',
    "val_path": 'hdfs://haruna/home/byte_data_aml_research/user/jiankai.sun/public_dataset/IFLYTEK/split_dev',
    "train_size": 12133,            
    "max_seq_len": 1024,
    "text_keys": ['sentence', 'label'],
    "tokenizer": 'hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/tokenizer/zh_0620_newcut_caster_145665_lowercase',
}

'''

class ClassificationGPTDatamodule(CruiseDataModule):
    """GPT classification dataset module.

    It supports reading from raw text dataset and process using pretrained tokenizers.
    """
    def __init__(self,
                 train_path: str = 'hdfs://haruna/home/byte_data_aml_research/user/jiankai.sun/public_dataset/RTE/split_train', #task_config["train_path"],
                 val_path: str = 'hdfs://haruna/home/byte_data_aml_research/user/jiankai.sun/public_dataset/RTE/split_dev', # task_config["val_path"], #
                 train_size: int = 2490, # task_config["train_size"], #for TNews,
                 train_batch_size: int = 4,
                 train_num_workers: int = 4,
                 val_batch_size: int = 4,
                 val_num_workers: int = 1,
                 max_seq_len: int = 1024, #task_config["max_seq_len"], #1024,
                 text_keys: List[str] = ['premise', 'hypothesis', 'label'], #['sentence', 'keywords', 'label', 'label_desc'],#task_config["text_keys"], #['sentence', 'keywords', 'label', 'label_desc'],
                 tokenizer: str =  'hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/tokenizer/zh_0620_newcut_caster_145665_lowercase',
                 gpu_prefetch: bool = False,
                 task_name: str = "RTE",
                 dyn_bsz: bool = False,
                 dyn_bsz_margin: float = 0.0,
                 stride: int = 896,
                 warmup_step_rate: float = -1,
                 bsz_warmup: bool = False,
                 ):
        super().__init__()
        self.save_hparams()
        self.tokenizer = None
        self.task_name = task_name
        self.rank_zero_info(f"Task_name: {self.task_name}.")
        self.rank_zero_info(f"text_keys: {text_keys}.")


    def local_rank_zero_prepare(self) -> None:
        if self.hparams.tokenizer.startswith('hdfs'):
            # try download it to local once per node and load it in setup
            tmp_dir = os.path.join(tempfile.gettempdir(), os.path.basename(self.hparams.tokenizer))
            hcopy(self.hparams.tokenizer, tmp_dir)
        else:
            logging.info(f"Prefetching HF tokenizers {self.hparams.tokenizer} on local rank zero...")
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(self.hparams.tokenizer)

    def setup(self):
        if self.hparams.tokenizer.startswith('hdfs'):
            # try download it to local once per node and load it in setup
            tmp_dir = os.path.join(tempfile.gettempdir(), os.path.basename(self.hparams.tokenizer))
            self.tokenizer = CasterTokenizer.from_pretrained(tmp_dir)
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer)

        print("setting up text_processor for task: {}".format(self.task_name))
        if self.task_name == "TNews":
            self.text_processor = TNewsProcessor(
                tokenizer=self.tokenizer if self.tokenizer is not None else self.hparams.tokenizer,
                text_keys=self.hparams.text_keys,
                max_seq_len=self.hparams.max_seq_len)
        elif self.task_name in  ["OCNLI", "CMNLI"]:
            self.text_processor = OCNLIProcessor(
                tokenizer=self.tokenizer if self.tokenizer is not None else self.hparams.tokenizer,
                text_keys=self.hparams.text_keys,
                max_seq_len=self.hparams.max_seq_len)
        elif self.task_name == "RTE":
            self.text_processor = RTEProcessor(
                tokenizer=self.tokenizer if self.tokenizer is not None else self.hparams.tokenizer,
                text_keys=self.hparams.text_keys,
                max_seq_len=self.hparams.max_seq_len)
        elif self.task_name == "IFLYTEK":
             self.text_processor = IFLYTEKProcessor(
                tokenizer=self.tokenizer if self.tokenizer is not None else self.hparams.tokenizer,
                text_keys=self.hparams.text_keys,
                max_seq_len=self.hparams.max_seq_len)

    def train_dataloader(self):
        train_steps = -1
        if self.hparams.train_size > 0:
            train_steps = self.hparams.train_size // (self.hparams.train_batch_size * DIST_ENV.world_size)
            assert train_steps > 0, f"train_size={self.hparams.train_size} may be too small to split to batch_size * world_size"
        train_files = [x for x in hlist_files([self.hparams.train_path]) if x.endswith('.jsonl')]
        self.rank_zero_info(f"Fetched {len(train_files)} training files.")

        loader = DistributedCruiseDataLoader(
            data_sources=[train_files],
            batch_sizes=[self.hparams.train_batch_size],
            num_workers=self.hparams.train_num_workers,
            predefined_steps=train_steps,
            source_types=['jsonl'],
            shuffle=True,
            pin_memory=True,
            parquet_cache_on=True,
            keys_or_columns=None,
            num_readers=[1],
            decode_fn_list=None,
            processor = self.text_processor,
            # processor=TNewsProcessor(
            #     tokenizer=self.tokenizer if self.tokenizer is not None else self.hparams.tokenizer,
            #     text_keys=self.hparams.text_keys,
            #     max_seq_len=self.hparams.max_seq_len),
            transform_output_many=True,
            dyn_bsz=self.hparams.dyn_bsz,
            dyn_bsz_margin=self.hparams.dyn_bsz_margin,
            num_warmup_steps=int(self.hparams.warmup_step_rate*train_steps) if self.hparams.bsz_warmup else -1,
        )
        if self.hparams.gpu_prefetch:
            loader = GPUPrefetcher(loader)
        return loader

    def val_dataloader(self):
        if not self.hparams.val_path:
            return iter([])
        val_steps = -1
        val_files = [x for x in hlist_files([self.hparams.val_path]) if x.endswith('.jsonl')]
        self.rank_zero_info(f"Fetched {len(val_files)} val files.")
        
        loader = DistributedCruiseDataLoader(
            data_sources=[val_files],
            batch_sizes=[self.hparams.val_batch_size],
            num_workers=self.hparams.val_num_workers,
            predefined_steps=val_steps,
            source_types=['jsonl'],
            shuffle=False,
            pin_memory=True,
            parquet_cache_on=True,
            keys_or_columns=None,
            num_readers=[1],
            decode_fn_list=None,
            processor = self.text_processor,
            # processor=TNewsProcessor(
            #     tokenizer=self.tokenizer if self.tokenizer is not None else self.hparams.tokenizer,
            #     text_keys=self.hparams.text_keys,
            #     max_seq_len=self.hparams.max_seq_len),
            transform_output_many=True,
        )
        if self.hparams.gpu_prefetch:
            loader = GPUPrefetcher(loader)
        return loader

