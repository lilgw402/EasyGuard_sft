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
from typing import Optional,List
from easyguard.core import AutoModel,AutoTokenizer
from dataset.transforms.text_transforms.DebertaTokenizer import DebertaTokenizer
from dataset.gandalf_cruise_data_module import GandalfParquetFeatureProvider,GandalfCruiseDataModule
from utils.driver import get_logger
from utils.registry import DATASETS,FEATURE_PROVIDERS


@FEATURE_PROVIDERS.register_module()
class EcomLiveGandalfParquetAutoDisFeatureProvider(GandalfParquetFeatureProvider):
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

@DATASETS.register_module()
class EcomLiveGandalfParquetAutoDisCruiseDataModule(GandalfCruiseDataModule):
	def __init__(self,
				dataset,
				feature_provider,
				data_factory,
				type=None
				 ):
		super(GandalfCruiseDataModule, self).__init__()
		self.save_hparams()

	def setup(self, stage: Optional[str] = None) -> None:
		# print(self.hparams)
		# print('self.hparams.dataset',type(self.hparams.dataset),self.hparams.dataset)
		self.dataset = Dict(self.hparams.dataset)
		self.feature_provider = Dict(self.hparams.feature_provider)
		self.data_factory = Dict(self.hparams.data_factory)
		self.total_cfg = Dict({'dataset':self.dataset,'feature_provider':self.feature_provider,'data_factory':self.data_factory})
		# self.train_predefined_steps = 'max' if self.data_factory.get('train_max_iteration',-1) == -1 else 'max'
		# self.val_predefined_steps = 'max' if self.data_factory.get('val_max_iteration',-1) == -1 else 'max'

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

	