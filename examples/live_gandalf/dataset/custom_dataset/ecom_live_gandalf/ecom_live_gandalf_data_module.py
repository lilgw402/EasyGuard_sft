# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-02-27 20:36:34
# Modified: 2023-02-27 20:36:34
import re
import json
import torch
import pickle
import numpy as np
from addict import Dict
from typing import Optional
from cruise.data_module import CruiseDataModule
from cruise.data_module.cruise_loader import DistributedCruiseDataLoader
from cruise.data_module.preprocess.create_preprocess import parse_cruise_processor_cfg
from dataset.dataset_utils.create_config import create_cruise_process_config
from dataset.dataset_utils.parse_files import get_ds_path
from utils.driver import get_logger
from utils.registry import DATASETS

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
			**kwargs
	):
		super(EcomLiveGandalfParquetAutoDisFeatureProvider, self).__init__()
		if embedding_conf is None:
			embedding_conf = {}
		if feature_norm_info is None:
			feature_norm_info = {}
		if slot_mask is None:
			slot_mask = []
		self._save_extra = kwargs.get('save_extra',False)
		self._slot_mask = kwargs.get('slot_mask',False)
		self._feature_num = kwargs.get('feature_num',False)
		self._use_high_precision =  kwargs.get('use_high_precision',False)
		self._filtered_tags =  kwargs.get('filtered_tags',False)
		self._feature_norm_info =  kwargs.get('feature_norm_info', {})
		self._embedding_conf =  kwargs.get('embedding_conf', {})
		self._feature_input_num = feature_num - len(slot_mask)
		self._active_slot = [i for i in range(self._feature_num) if i not in self._slot_mask]

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
			'asr': asr,
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

	def transform(self, data):
		return self.process(data)

	def batch_transform(self, data):
		batch_output = []
		for sub_data in data:
			batch_output.append(self.process(data))
		return batch_output

from pytorch_lightning.core import LightningDataModule
class GandalfDataModule(LightningDataModule):
	def __init__(self,
				dataset,
				feature_provider,
				data_factory,
				**kwargs
				 ):
		super(GandalfDataModule, self).__init__()
		self.save_hparams()

	def setup(self, stage: Optional[str] = None) -> None:
		self.train_predefined_steps = 'max' if self.hparams.kwargs['train_max_iteration'] == -1 else 'max'
		self.val_predefined_steps = 'max' if self.hparams.kwargs['test_max_iteration'] == -1 else 'max'

	def train_dataloader(self):
		return DistributedCruiseDataLoader(
			data_sources=[self.train_files],
			keys_or_columns=[None],
			batch_sizes=[self.hparams.train_batch_size],
			num_workers=self.hparams.num_workers,
			num_readers=[1],
			decode_fn_list=[[]],
			processor=EcomLiveGandalfParquetAutoDisFeatureProvider(
				**self.hparams.kwargs['feature_provider']
			),
			predefined_steps=self.train_predefined_steps,
			source_types=['parquet'],
			shuffle=True,
		)

	def val_dataloader(self):
		return DistributedCruiseDataLoader(
			data_sources=[self.val_files],
			keys_or_columns=[None],
			batch_sizes=[self.hparams.val_batch_size],
			num_workers=self.hparams.num_workers,
			num_readers=[1],
			decode_fn_list=[[]],
			processor=EcomLiveGandalfParquetAutoDisFeatureProvider(
				**self.hparams.feature_provider_config
			),
			predefined_steps=self.val_predefined_steps,
			source_types=['parquet'],
			shuffle=False,
		)

	def test_dataloader(self):
		return iter([])

	def predict_dataloader(self):
		return iter([])

