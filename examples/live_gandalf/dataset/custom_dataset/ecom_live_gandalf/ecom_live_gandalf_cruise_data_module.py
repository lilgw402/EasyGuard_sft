# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-02-27 20:36:34
# Modified: 2023-02-27 20:36:34
import os
import re
import copy
import math
import json
import torch
import pickle
import numpy as np
from addict import Dict
from typing import Optional,List
from easyguard.core import AutoModel,AutoTokenizer
from dataset.transforms.text_transforms.DebertaTokenizer import DebertaTokenizer
from cruise.data_module import CruiseDataModule
from cruise.data_module.cruise_loader import DistributedCruiseDataLoader
from cruise.data_module.preprocess.decode import TFApiExampleDecode
from cruise.data_module.preprocess.create_preprocess import parse_cruise_processor_cfg
from utils.driver import get_logger
from utils.registry import DATASETS,FEATURE_PROVIDERS
from utils.driver import get_logger, init_env,init_device, DIST_CONTEXT
from utils.file_util import hmkdir, check_hdfs_exist
from utils.torch_util import default_collate
from utils.dataset_utils.create_config import create_cruise_process_config
from utils.dataset_utils.parse_files import get_ds_path


@FEATURE_PROVIDERS.register_module()
class EcomLiveGandalfParquetAutoDisFeatureProvider:
	def __init__(
			self,
			feature_num,
			use_high_precision=False,
			filtered_tags=None,
			slot_mask=None,
			feature_norm_info=None,
			embedding_conf=None,
			save_extra=False,
			eval_mode=False,
            trace_mode=False,
			**kwargs
	):
		super(EcomLiveGandalfParquetAutoDisFeatureProvider, self).__init__()
		self._save_extra = save_extra
		self._slot_mask = slot_mask
		self._feature_num = feature_num
		self._use_high_precision = use_high_precision
		self._filtered_tags =  filtered_tags
		self._feature_norm_info =  feature_norm_info
		self._embedding_conf =  embedding_conf
		self._feature_input_num = feature_num - len(slot_mask)
		self._active_slot = [i for i in range(self._feature_num) if i not in self._slot_mask]
		self.asr_model_name = "deberta_base_6l"
		self._asr_encoder = AutoModel.from_pretrained(self.asr_model_name)
		# self._asr_tokenizer = AutoTokenizer.from_pretrained(self.asr_model_name, return_tensors="pt", max_length=512)
		self._asr_tokenizer = DebertaTokenizer('./models/weights/fashion_deberta_asr/deberta_3l/vocab.txt',max_len=512)

	def process_feature_dense(self, features):
		# 加载预处理参数
		active_slot = torch.tensor(self._active_slot, dtype=torch.long).reshape(len(self._active_slot))
		compressed_min = [self._feature_norm_info.get(str(slot_id), [0, 1])[0] for slot_id in self._active_slot]
		compressed_max = [self._feature_norm_info.get(str(slot_id), [0, 1])[1] for slot_id in self._active_slot]
		compressed_min = torch.tensor(compressed_min, dtype=torch.float32).reshape(-1)
		compressed_max = torch.tensor(compressed_max, dtype=torch.float32).reshape(-1)
		compressed_range = compressed_max - compressed_min
		# 并行特征预处理
		features = torch.tensor(features, dtype=torch.float32)
		feature_dense = features[active_slot]
		feature_dense_norm = (feature_dense - compressed_min) / compressed_range
		feature_dense_norm[feature_dense == -1] = 0.0  # 特征为-1, norm值为0
		feature_dense_norm[feature_dense.isnan()] = 0.0  # 特征缺失, norm值为0
		feature_dense_norm = torch.clamp(feature_dense_norm, min=0.0, max=1.0)
		auto_dis_input_list = [feature_dense_norm, feature_dense_norm * feature_dense_norm, torch.sqrt(feature_dense_norm)]
		auto_dis_input = torch.stack(auto_dis_input_list, dim=1)
		return auto_dis_input, feature_dense_norm
	
	def process_asr(self, asr):
		# 加载预处理参数
		asr_inputs = self._asr_tokenizer(asr)
		return asr_inputs

	def process(self, data):
		feature_data = {}
		# 数据mask:是否mask高准数据流
		source = data.get('source','origin')
		if source == 'high_precision' and not self._use_high_precision:
			return None
		# 标签mask:是否mask特定标签数据
		if self._filtered_tags and data.get('verify_reason',''):
			if len(re.findall(self._filtered_tags, data.get('verify_reason','')))>0:
				# get_logger().info("Mask data by filtered tags:{}".format(data.get('verify_reason','')))
				return None
		# parquet数据主要包含labels,features,contents,embeddings字段，而且均已经序列化了
		labels = pickle.loads(data["labels"])
		# numerical features
		features = pickle.loads(data["features"])
		strategy = list(features['strategy'].values())
		# content features
		contents = pickle.loads(data["contents"])
		asr = contents["asr"]
		# embedding features
		embeddings = pickle.loads(data["embeddings"])
		# 根据embedding conf解析embedding
		embeddings_allowed = {}
		if self._embedding_conf and isinstance(self._embedding_conf, dict):
			for key in self._embedding_conf.keys():
				if key in embeddings.keys():
					embeddings_allowed.update({key: embeddings[key]})
				else:
					embeddings_allowed.update({key: np.zeros(self._embedding_conf[key])})
		else:
			embeddings_allowed = embeddings
		embeddings_input = {}
		for k, v in embeddings_allowed.items():
			if isinstance(v, list) and len(v) == self._embedding_conf[k]:
				embeddings_input.update({k: torch.tensor(v, dtype=torch.float32)})
			else:
				embeddings_input.update({k: torch.zeros(self._embedding_conf[k])})
		# 数值特征
		auto_dis_input, feature_dense = self.process_feature_dense(features=strategy)
		asr_input = self.process_asr(asr)
		# 人审召回相关的label
		label = labels.get('label', 0)  # 处罚维度label
		# if label==1 and random.random()<0.01:
		#     print('Sample labels:',labels)
		#     print('Sample contents:',asr,embeddings)
		#     print('Sample features:',features)
		# 更新传入模型的输入
		feature_data.update({
			"auto_dis_input": auto_dis_input,
			"feature_dense": feature_dense,
			'input_ids': asr_input['input_ids'],
			'attention_mask': asr_input['attention_mask'],
			'token_type_ids': asr_input['token_type_ids'],
			"label": self._process_label(label),
		})
		feature_data.update(embeddings_input)
		if self._save_extra:
			extra = {'object_id':data["object_id"]}
			extra.update(features["context"])
			extra.update({
				"label": int(label),
				"uv": int(data['uv']),
				"online_score": data['online_score'],
				'verify_reason': data["verify_reason"]
			})
			extra_str = json.dumps(extra, ensure_ascii=False)
			try:
				feature_data.update({"extra": extra_str})
			except Exception as e:
				get_logger().error("{}, fail to add extra: {}".format(e, extra))
		return feature_data

	@staticmethod
	def _process_label(label):
		if int(label) == 1:
			return torch.tensor([1], dtype=torch.float32)
		else:
			return torch.tensor([0], dtype=torch.float32)

	@staticmethod
	def _process_mls_label(verify_status, label_length):  # multiclass label
		if int(verify_status) == 1:
			return torch.tensor([1], dtype=torch.float32)
		else:
			return torch.tensor([0], dtype=torch.float32)

	def batch_process(self, batch_data: List[dict]) -> List[dict]:
		return [x for x in list(map(self.process, batch_data)) if x is not None]

	def split_batch_process(self, batch_data_dict: dict) -> dict:
		batch_data_list = self._split_array(batch_data_dict)
		return self.__call__(batch_data_list)

	def __call__(self, batch_data: List[dict]) -> dict:
		return default_collate(self.batch_process(batch_data))

@DATASETS.register_module()
class GandalfCruiseDataModule(CruiseDataModule):
	def __init__(self,
				dataset,
				feature_provider,
				data_factory,
				type=None
				 ):
		super(GandalfCruiseDataModule, self).__init__()
		self.save_hparams()

	def setup(self, stage: Optional[str] = None) -> None:
		print(self.hparams)
		print('self.hparams.dataset',type(self.hparams.dataset),self.hparams.dataset)
		self.dataset = Dict(self.hparams.dataset)
		self.feature_provider = Dict(self.hparams.feature_provider)
		self.data_factory = Dict(self.hparams.data_factory)
		self.total_cfg = Dict({'dataset':self.dataset,'feature_provider':self.feature_provider,'data_factory':self.data_factory})
		self.train_predefined_steps = 'max' if self.data_factory.get('train_max_iteration',-1) == -1 else 'max'
		self.val_predefined_steps = 'max' if self.data_factory.get('val_max_iteration',-1) == -1 else 'max'

	def train_dataloader(self):
		return self.create_cruise_dataloader(self.total_cfg,
											 data_input_dir=self.dataset.input_dir,
											 data_folder=self.dataset.train_folder,
											 arg_dict=self.data_factory,
											 mode='train')

	def val_dataloader(self):
		return self.create_cruise_dataloader(self.total_cfg,
											 data_input_dir=self.dataset.val_input_dir,
											 data_folder=self.dataset.val_folder,
											 arg_dict=self.data_factory,
											 mode='val')

	def test_dataloader(self):
		return iter([])

	def predict_dataloader(self):
		return iter([])
	
	def create_cruise_dataloader(self,cfg, data_input_dir, data_folder, arg_dict, mode="val", specific_bz=None):
		# arg_dict_cp = Dict(arg_dict)
		arg_dict_cp = copy.deepcopy(arg_dict)
		print('arg_dict_cp',arg_dict_cp)
		data_sources, data_types = get_ds_path(
			data_input_dir,
			data_folder,
			arg_dict_cp.type,
			arg_dict_cp.filename_pattern,
			arg_dict_cp.file_min_size,
			arg_dict_cp.group_keys,
			arg_dict_cp.shuffle_files
		)
		ds_num = len(data_sources)
		drop_last = arg_dict_cp.get("drop_last", True)
		shuffle = arg_dict_cp.get("shuffle", True)
		fast_resume = arg_dict_cp.get("fast_resume", True)
		parquet_cache_on = arg_dict_cp.get("parquet_cache_on", True)
		batch_size = arg_dict_cp.get('batch_size',128)
		predefined_steps = -1
		use_arnold = True

		if mode == "train":
			predefined_steps = cfg.trainer.train_max_iteration

		if mode == "val":
			drop_last = False
			# trick only in val: half bz to lower mem usage
			if arg_dict_cp.get('batch_size_val',-1) == -1:
				get_logger().info(
					"batch_size_val is not set, use batch_size // 2 as default"
				)
				arg_dict_cp.batch_size_val = arg_dict_cp.batch_size // 2
			batch_size = arg_dict_cp.batch_size_val
			predefined_steps = cfg.trainer.test_max_iteration

		if mode == "test":
			drop_last = False
			shuffle = False
			predefined_steps = cfg.tester.max_iteration

		if mode == "trace":
			if "ParquetDataFactory" in arg_dict_cp.df_type:
				shuffle = False

		# use in trace model
		if specific_bz and isinstance(specific_bz, int):
			batch_size = specific_bz

		num_workers = arg_dict_cp.num_workers
		num_readers = [arg_dict_cp.num_parallel_reads] * ds_num
		multiplex_dataset_weights = arg_dict_cp.multiplex_dataset_weights
		multiplex_mix_batch = arg_dict_cp.multiplex_mix_batch
		is_kv = data_types[0] == "kv"
		if is_kv:
			multiplex_mix_batch = True
			use_arnold = num_workers > 0

		if multiplex_mix_batch:
			# for the case when one batch data is mixed by multiple datasets
			if not multiplex_dataset_weights:
				batch_sizes = [batch_size // ds_num] * ds_num
				remain = batch_size - sum(batch_sizes)
				for i in range(remain):
					batch_sizes[i] += 1
			else:
				batch_sizes = [
					math.floor(batch_size * p) for p in multiplex_dataset_weights
				]
				remain = batch_size - sum(batch_sizes)
				for i in range(remain):
					batch_sizes[i] += 1
			multiplex_dataset_weights = []
		else:
			# for the case when one batch data is from only single dataset each time,
			# while the dataset is chosen randomly from all the given datasets
			if not multiplex_dataset_weights:
				if ds_num > 1:
					# read each dataset with equal probability when multiplex_dataset_weights is not given
					multiplex_dataset_weights = [1 / ds_num] * ds_num
				else:
					# since we only have one dataset, the multiplex_dataset_weights does not affcet the loading logic
					# we make it to be an empty list here to match the original logic for single dataset
					multiplex_dataset_weights = []
			batch_sizes = [batch_size] * ds_num
		print('batch_sizes',batch_sizes)
		process_cfg = create_cruise_process_config(cfg, mode, is_kv)
		print('process_cfg\n',process_cfg)
		cruise_processor = parse_cruise_processor_cfg(process_cfg, "")

		keys_or_columns = []
		last_step = 0
		if ds_num > 1 and predefined_steps == -1:
			predefined_steps = "max"
		# define decode_fn
		if data_types[0] == "tfrecord":
			features = arg_dict_cp.data_schema
			enable_tf_sample_sharding = int(os.getenv("CRUISE_ENABLE_TF_SAMPLE_SHARDING", "0"))
			to_numpy = not enable_tf_sample_sharding
			decode_fn = [TFApiExampleDecode(features=features, key_mapping=dict(), to_numpy=to_numpy)] * ds_num
		elif is_kv and not use_arnold:
			# since kv feature provider would get the index 0 of the given data; while in cruise loader, if reading kv data from 
			# torch loader, the output data would be the data itself, not a list. Here we make it a list by using decode fn, to 
			# ensure it is runnable, but this might be a little bit hacky.
			decode_fn = [lambda x: [x]] * ds_num
		else:
			decode_fn = None
		# Create DistributedCruiseDataLoader
		loader = DistributedCruiseDataLoader(
			data_sources,
			keys_or_columns,
			batch_sizes,
			num_workers,
			num_readers,
			decode_fn,
			cruise_processor,
			predefined_steps,
			data_types,
			last_step,
			shuffle=shuffle,
			multiplex_weights=multiplex_dataset_weights,
			drop_last=drop_last,
			use_arnold=use_arnold,
			transform_replace_all=is_kv,
			fast_resume=fast_resume,
			parquet_cache_on=parquet_cache_on,
		)
		return loader
