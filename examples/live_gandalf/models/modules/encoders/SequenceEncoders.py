# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-02-10 14:15:49
# Modified: 2023-02-10 14:15:49
import torch
import torch.nn as nn
from torch.nn import LSTM
from torch.nn.modules.transformer import Transformer,TransformerEncoder,TransformerEncoderLayer


class PositionEmbedding(nn.Module):
	def __init__(self,max_position,dim):
		super(PositionEmbedding, self).__init__()
		self.embedding = nn.Embedding(max_position, dim)

	def forward(self,x):
		return self.embedding(x)


class TextCNNEncoder(nn.Module):
	def __init__(self):
		super(TextCNNEncoder, self).__init__()
		pass

	def forward(self):
		pass

class LSTMEncoder(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers):
		super(LSTMEncoder, self).__init__()
		self._lstm_encoder = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, proj_size=0,batch_first=True)

	def forward(self, embeddings):
		output, (hn, cn) = self._lstm_encoder(embeddings)
		return hn[-1, :, :]  # Last output state

class EcomTransformerEncoder(nn.Module):
	def __init__(self, **kwargs):
		super(EcomTransformerEncoder, self).__init__()
		print(kwargs)
		self.num_hidden_layers = kwargs.get('num_hidden_layers',6)
		self.hidden_size = kwargs.get('hidden_size',768)
		self.num_attention_heads = kwargs.get('num_attention_heads', 6)
		self.encoder_layer = TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.num_attention_heads)
		self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=self.num_hidden_layers)
		

	def forward(self,src, mask=None, src_key_padding_mask=None):
		sequence_output = self.transformer_encoder(src,mask=mask, src_key_padding_mask=src_key_padding_mask)
		return sequence_output
		# pooled_output = torch.mean(sequence_output,dim=1)
		# return pooled_output

class ConformerEncoder(nn.Module):
	def __init__(self):
		super(ConformerEncoder, self).__init__()
		pass

	def forward(self):
		pass

