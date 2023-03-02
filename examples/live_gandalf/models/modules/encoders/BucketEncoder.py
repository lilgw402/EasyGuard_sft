# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-02-09 11:58:09
# Modified: 2023-02-09 11:58:09
import torch
import torch.nn as nn


class BucketPreprocessor(nn.Module):
	def __init__(self,bucket_config,slot_mask):
		super(BucketPreprocessor, self).__init__()
		slot_ids = []
		val_nums = []
		all_vals = []
		self.num_bucket_map = bucket_config['feature_bucket_info']
		self.feature_norm_info = bucket_config['feature_norm_info']
		for slot_id, info in self.num_bucket_map.items():
			slot_id = int(slot_id.replace('slot_', ''))
			if slot_id in slot_mask:
				continue
			slot_ids.append(slot_id)
			vals = info[0]
			if not vals:
				vals = self.init_threshold_meanly(slot_id)
			val_num = len(vals)
			val_nums.append(val_num)
			all_vals.append(vals)

		all_vals = [v for vals in all_vals for v in vals]
		self.register_buffer('slot_ids', torch.IntTensor(slot_ids, device='cpu'))
		self.register_buffer('bucket_nums', torch.IntTensor(val_nums, device='cpu'))
		self.register_buffer('thresholds', torch.FloatTensor(all_vals))

	def init_threshold_meanly(self,slot):
		bucket_num, _ = self.num_bucket_map.get(str(slot),[[], 10,16])[1:]
		min_val, max_val = self.feature_norm_info.get(str(slot),[0,1])
		thresholds = [i * (max_val - min_val) / bucket_num for i in range(bucket_num + 1)]
		return thresholds

	def forward(self, features):
		device = features.device
		outputs = []
		with torch.no_grad():
			offset = 0
			for slot_id, bucket_num in zip(self.slot_ids.tolist(), self.bucket_nums.tolist()):
				one_slot = features[:, slot_id]
				threshold = self.thresholds[offset:offset + bucket_num]
				threshold = threshold.unsqueeze(0)
				comparison = torch.lt(one_slot.unsqueeze(1), threshold)
				is_hit = torch.any(comparison, dim=1)
				tmp = comparison.float() * torch.arange(bucket_num, 0, -1, device=device).view(1, -1)
				indices = torch.argmax(tmp, 1)
				output = torch.where(is_hit, indices,torch.ones_like(one_slot, dtype=torch.long, device=device).fill_(bucket_num))
				output = output.long().view(-1, 1)
				outputs.append(output)
				offset += bucket_num

		return torch.cat(outputs, 1)


class SequenceBucketPreprocessor(nn.Module):
	def __init__(self,bucket_config,slot_mask):
		super(SequenceBucketPreprocessor, self).__init__()
		slot_ids = []
		val_nums = []
		all_vals = []
		self.num_bucket_map = bucket_config['feature_bucket_info']
		self.feature_norm_info = bucket_config['feature_norm_info']
		for slot_id, info in self.num_bucket_map.items():
			slot_id = int(slot_id.replace('slot_', ''))
			if slot_id in slot_mask:
				continue
			slot_ids.append(slot_id)
			vals = info[0]
			if not vals:
				vals = self.init_threshold_meanly(slot_id)
			self.num_bucket_map[str(slot_id)][0] = vals
			val_num = len(vals)
			val_nums.append(val_num)
			all_vals.append(vals)

		all_vals = [v for vals in all_vals for v in vals]
		self.register_buffer('slot_ids', torch.IntTensor(slot_ids, device='cpu'))
		self.register_buffer('bucket_nums', torch.IntTensor(val_nums, device='cpu'))
		self.register_buffer('thresholds', torch.FloatTensor(all_vals))

	def init_threshold_meanly(self,slot):
		bucket_num, _ = self.num_bucket_map.get(str(slot),[[], 10,16])[1:]
		min_val, max_val = self.feature_norm_info.get(str(slot),[0,1])
		thresholds = [i * (max_val - min_val) / bucket_num for i in range(bucket_num+1)]
		return thresholds

	def forward(self, features):
		device = features.device
		# print('device', device)
		outputs = []
		with torch.no_grad():
			offset = 0
			for slot_id, bucket_num in zip(self.slot_ids.tolist(), self.bucket_nums.tolist()):
				one_slot = features[:, :, slot_id]
				threshold = self.thresholds[offset:offset + bucket_num]
				threshold = threshold.unsqueeze(0).unsqueeze(0)
				comparison = torch.lt(one_slot.unsqueeze(-1), threshold)
				is_hit = torch.any(comparison, dim=-1)
				tmp = comparison.float() * torch.arange(bucket_num, 0, -1,device=device).view(1, 1, -1)
				indices = torch.argmax(tmp, -1)
				output = torch.where(is_hit, indices,torch.ones_like(one_slot, dtype=torch.long,device=device).fill_(bucket_num))
				# output = output.long().view(-1, 1)
				outputs.append(output)
				offset += bucket_num
		return torch.stack(outputs, -1)


class SequenceBucketEncoder(nn.Module):
	def __init__(self, bucket_config, slot_mask, max_slot=20, time_steps=20, output_size=128, use_fc=False):
		super(SequenceBucketEncoder, self).__init__()
		self.sequence_bucket_config = bucket_config
		self.max_slot = max_slot
		self.slot_mask = slot_mask
		self.valid_slots = [slot for slot in range(max_slot) if slot not in self.slot_mask]
		self.time_steps = time_steps
		self.sequence_bucket_embedding_layers = nn.ModuleList()
		self.total_input_size,self.total_output_size = 0, 0
		self.init_sequence_bucket_embedding_layer(self.sequence_bucket_config)
		self.fc = nn.Linear(self.total_output_size, output_size, bias=True) if use_fc else nn.Identity()

	def forward(self,sequence_bucket_inputs):
		assert len(sequence_bucket_inputs.shape) == 3 #batch * sequence_length * feature_num
		sequence_bucket_embeddings = self.encode2embeddings(sequence_bucket_inputs)
		sequence_bucket_embeddings = self.fc(sequence_bucket_embeddings)
		return sequence_bucket_embeddings

	def init_sequence_bucket_embedding_layer(self, sequence_bucket_embedding_config):
		for i in range(self.time_steps):
			step_bucket_embeddings = nn.ModuleList()
			for slot in self.valid_slots:
				emb_config = sequence_bucket_embedding_config.get(str(slot),[[],32,128])# Use default 32 vocab size and 128 output dim
				num_embeddings = emb_config[1]+2# important diff for manually split boundaries
				embedding_dim = emb_config[2]
				if i == 0:
					self.total_input_size += num_embeddings
					self.total_output_size += embedding_dim
				step_bucket_embeddings.append(nn.Embedding(num_embeddings,embedding_dim))
			self.sequence_bucket_embedding_layers.append(step_bucket_embeddings)

	def encode2embeddings(self,sequence_bucket_inputs):
		sequence_bucket_features_emb = []
		for i in range(self.time_steps):
			step_bucket_embeddings = []
			for idx, slot in enumerate(self.valid_slots):
				feature_emb = self.sequence_bucket_embedding_layers[i][idx](sequence_bucket_inputs[:, i, int(slot)])
				step_bucket_embeddings.append(feature_emb)
			step_bucket_embeddings = torch.cat(step_bucket_embeddings,dim=1)
			sequence_bucket_features_emb.append(step_bucket_embeddings)
		sequence_bucket_features_emb = torch.stack(sequence_bucket_features_emb,dim=1)
		return sequence_bucket_features_emb


class SharedSequenceBucketEncoder(nn.Module):
	def __init__(self, bucket_config, slot_mask, max_slot=20, output_size=128, use_fc=False):
		super(SharedSequenceBucketEncoder, self).__init__()
		self.sequence_bucket_config = bucket_config
		self.max_slot = max_slot
		self.slot_mask = slot_mask
		self.valid_slots = [slot for slot in range(max_slot) if slot not in self.slot_mask]
		self.sequence_bucket_embedding_layers = nn.ModuleList()
		self.total_input_size,self.total_output_size = 0, 0
		self.init_sequence_bucket_embedding_layer(self.sequence_bucket_config)
		self.fc = nn.Linear(self.total_output_size, output_size, bias=True) if use_fc else nn.Identity()

	def forward(self,sequence_bucket_inputs):
		assert len(sequence_bucket_inputs.shape) == 3 #batch * sequence_length * feature_num
		sequence_bucket_embeddings = self.encode2embeddings(sequence_bucket_inputs)
		sequence_bucket_embeddings = self.fc(sequence_bucket_embeddings)
		return sequence_bucket_embeddings

	def init_sequence_bucket_embedding_layer(self, sequence_bucket_embedding_config):
		for slot in self.valid_slots:
			emb_config = sequence_bucket_embedding_config.get(str(slot),[[],32,128])# Use default 32 vocab size and 128 output dim
			num_embeddings = emb_config[1]+2# important diff
			embedding_dim = emb_config[2]
			self.total_input_size += num_embeddings
			self.total_output_size += embedding_dim
			self.sequence_bucket_embedding_layers.append(nn.Embedding(num_embeddings, embedding_dim))

	def encode2embeddings(self,sequence_bucket_inputs):
		sequence_bucket_features_emb = []
		for idx, slot in enumerate(self.valid_slots):
			feature_emb = self.sequence_bucket_embedding_layers[idx](sequence_bucket_inputs[:, :, int(slot)])
			sequence_bucket_features_emb.append(feature_emb)
		sequence_bucket_features_emb = torch.cat(sequence_bucket_features_emb,dim=-1)
		return sequence_bucket_features_emb






