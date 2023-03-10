# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-02-09 16:37:59
# Modified: 2023-02-09 16:37:59
import torch
import torch.nn as nn

"""
1,2对不同step共享meta-embedding;
专为序列特征设计的共享meta-embedding的2 SharedSequenceAutoDisBucketEncoder经过升/降维的优化后退化为和1 AutodisBucketEncoder一样的结构
3,4对不同step不共享meta-embedding;
不共享meta-embedding3 SequenceAutoDisBucketEncoder进行了矩阵并行计算优化得到 4 SequenceAutoDisBucketEncoderFast,提高了计算速度,
相同timesteps不同特征数GPU推理500次时间对比(s)
                                      10       100        500       1000
1 AutodisBucketEncoder                0.4662   0.3588    0.3698     0.4800
2 SharedSequenceAutoDisBucketEncoder  0.4795   0.7362    4.3180     8.8142
3 SequenceAutoDisBucketEncoder        8.4103   7.5836    7.8420     10.8445
4 SequenceAutoDisBucketEncoderFast    0.3670   0.9178    4.433      8.9472
相同特征数(100)不同timestepsGPU推理500次时间对比(s)
                                      10       20        50         100
1 AutodisBucketEncoder                0.3925   0.7339    1.7677     3.4917
2 SharedSequenceAutoDisBucketEncoder  0.4191   0.7860    1.8976     3.744
3 SequenceAutoDisBucketEncoder        3.831    7.6551    19.7140    41.149
4 SequenceAutoDisBucketEncoderFast    0.4796   0.9159    2.2315     4.438
为了与V2对齐,使用了ResBlock
由于ResBlock的存在,模型对于ResBlock中的nn.Parameter初始化要求较高
因此收敛速度相比V2会变慢,可将add_block设为False,变为直连MLP,自测收敛更快,准确不会下降
"""

class ResBlock(nn.Module):
	def __init__(self, feature_num, input_dim, dropout, alpha=1):
		super(ResBlock, self).__init__()
		self.linear_w = nn.Parameter(torch.randn(feature_num, input_dim, input_dim))
		self.linear_b = nn.Parameter(torch.randn(feature_num, 1, input_dim))
		nn.init.kaiming_uniform_(self.linear_w, mode='fan_in', nonlinearity='leaky_relu')
		self.leaky_relu = nn.LeakyReLU(inplace=True)
		self.dropout = nn.Dropout(p=dropout)
		self.alpha = alpha

	def forward(self, x):
		# print('Resblok input', x.shape)
		h = torch.matmul(x, self.linear_w) + self.linear_b
		h = h + self.alpha * x
		h = self.leaky_relu(h)
		h = self.dropout(h)
		# print('Resblok output', x.shape)
		return h


class SequenceResBlock(nn.Module):
    def __init__(self, time_steps,feature_num, input_dim, dropout, alpha=1):
        super(SequenceResBlock, self).__init__()
        self.linear_w = nn.Parameter(torch.randn(time_steps,feature_num,input_dim, input_dim))
        self.linear_b = nn.Parameter(torch.randn(time_steps,feature_num, 1, input_dim))
        nn.init.kaiming_uniform_(self.linear_w, mode='fan_in', nonlinearity='leaky_relu')
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.alpha = alpha

    def forward(self, x):
        # print('Resblok input', x.shape)
        h = torch.matmul(x, self.linear_w) + self.linear_b
        h = h + self.alpha * x
        h = self.leaky_relu(h)
        h = self.dropout(h)
        # print('Resblok output', x.shape)
        return h


class AutodisBucketEncoder(nn.Module):
    def __init__(self,
        feature_num,  # 必填参数，最终网络输入的特征数量
        bucket_num=8,
        bucket_dim=128,
        layer_conf=[64, 64, 64],
        alpha=1,
        output_size=128,
        use_fc=False,
        dropout=0,
        add_block=True
        ):
        super(AutodisBucketEncoder, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.Dropout = nn.Dropout(p=dropout)
        self.linear1_w = nn.Parameter(torch.randn(feature_num, 3, layer_conf[0]))
        self.linear1_b = nn.Parameter(torch.randn(feature_num, 1, layer_conf[0]))
        self.layer = nn.Sequential(*[ResBlock(feature_num, layer_len, dropout=0, alpha=alpha) for layer_len in
                                     layer_conf]) if add_block else nn.Identity()
        self.linear2_w = nn.Parameter(torch.randn(feature_num, layer_conf[-1], bucket_num))
        self.linear2_b = nn.Parameter(torch.randn(feature_num, 1, bucket_num))

        self.emb = nn.Parameter(torch.randn(feature_num, bucket_num, bucket_dim))
        self._tau_module = nn.Parameter(torch.ones([feature_num, 1, bucket_num]))
        self.Softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(feature_num * bucket_dim, output_size, bias=True) if use_fc else nn.Identity()
        self.output_size = output_size if use_fc else feature_num * bucket_dim

        nn.init.kaiming_uniform_(self.linear1_w, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.linear2_w, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # print(-1,x.shape)
        #  b feature_num 1 3
        x = x.unsqueeze(-2)
        # print(0,x.shape)
        #  b feature_num layer_conf[0]
        x = torch.matmul(x, self.linear1_w) + self.linear1_b
        x = self.LeakyReLU(x)
        x = self.Dropout(x)
        # print(1, x.shape)
        # b feature_num 1 layer_conf[-1]
        x = self.layer(x)
        # print(2, x.shape)
        #  b feature_num 1 bucket_num
        x = torch.matmul(x, self.linear2_w) + self.linear2_b
        x = self.LeakyReLU(x)
        # print(3, x.shape)
        # b feature_num bucket_num
        x = (x * self._tau_module).squeeze(-2)
        # print(4, 'tau', x.shape)
        x = self.Softmax(x)
        # print(5, 'softmax', x.shape)
        # b feature_num bucket_num bucket_dim
        x = x.unsqueeze(-1) * self.emb
        # print(6, 'emb', x.shape)
        # b feature_num bucket_dim
        x = torch.sum(x, dim=-2)
        # print(7, 'reduce', x.shape)
        # b feature_num*bucket_dim
        # x = torch.flatten(x, start_dim=1)
        x = torch.flatten(x, start_dim=-2)#TODO:Use -2 instead of 1(or 2 for sequence input) will result in the same code
        # print(8, 'reshape', x.shape)
        # b output_size if use_fc else feature_num*bucket_num
        x = self.fc(x)
        return x


class SequenceAutoDisBucketEncoder(nn.Module):
	def __init__(
		self,
		feature_num,  # 必填参数，最终网络输入的特征数量
		time_steps=20,
		bucket_num=8,
		bucket_dim=128,
		layer_conf=[64, 64, 64],
		alpha=1,
		output_size=128,
		use_fc=False,
		dropout=0,
		add_block=True
	):
		super(SequenceAutoDisBucketEncoder, self).__init__()
		self.time_steps = time_steps
		self.auto_dis_encoders = nn.ModuleList()
		for i in range(self.time_steps):
			encoder = AutodisBucketEncoder(feature_num,bucket_num,bucket_dim,layer_conf,alpha,output_size,use_fc,dropout,add_block)
			self.auto_dis_encoders.append(encoder)

	def forward(self, x):  # b timesteps feature_num 3
		time_step_outputs = []
		for i in range(self.time_steps):
			time_step_input = x[:,i,:,:]
			time_step_output = self.auto_dis_encoders[i](time_step_input)
			time_step_outputs.append(time_step_output)
		return torch.stack(time_step_outputs,dim=1)


class SequenceAutoDisBucketEncoderFast(nn.Module):
    def __init__(
        self,
        feature_num,  # 必填参数，最终网络输入的特征数量
        time_steps=20,
        bucket_num=8,
        bucket_dim=128,
        layer_conf=[64, 64, 64],
        alpha=1,
        output_size=128,
        use_fc=False,
        dropout=0,
        add_block=True
    ):
        super(SequenceAutoDisBucketEncoderFast, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.Dropout = nn.Dropout(p=dropout)
        self.linear1_w = nn.Parameter(torch.randn(time_steps,feature_num, 3, layer_conf[0]))
        self.linear1_b = nn.Parameter(torch.randn(time_steps,feature_num, 1, layer_conf[0]))
        self.layer = nn.Sequential(*[SequenceResBlock(time_steps,feature_num, layer_len, dropout=0, alpha=alpha) for layer_len in
                                     layer_conf]) if add_block else nn.Identity()
        self.linear2_w = nn.Parameter(torch.randn(time_steps,feature_num, layer_conf[-1], bucket_num))
        self.linear2_b = nn.Parameter(torch.randn(time_steps,feature_num, 1, bucket_num))

        self.emb = nn.Parameter(torch.randn(time_steps,feature_num, bucket_num, bucket_dim))
        self._tau_module = nn.Parameter(torch.ones([time_steps,feature_num, 1, bucket_num]))
        self.Softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(time_steps,feature_num * bucket_dim, output_size, bias=True) if use_fc else nn.Identity()
        self.output_size = output_size if use_fc else feature_num * bucket_dim

        nn.init.kaiming_uniform_(self.linear1_w, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.linear2_w, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):  # b timesteps feature_num 3
        x = x.unsqueeze(-2) #TODO:Use -2 instead of 3(or 2 for non-sequence input) will result in same code
        #  b time_steps feature_num 1 layer_conf[0]
        # print(x.shape, self.linear1_w.shape)
        x = torch.matmul(x, self.linear1_w) + self.linear1_b
        # x = torch.einsum('bsfj,fjl->bsfl',x,self.linear1_w) #+ self.linear1_b
        x = self.LeakyReLU(x)
        x = self.Dropout(x)
        # print(1, x.shape)
        # b time_steps feature_num 1 layer_conf[-1]
        x = self.layer(x)
        # print(2, x.shape)
        #  b time_steps feature_num 1 bucket_num
        x = torch.matmul(x, self.linear2_w) + self.linear2_b
        # x = torch.einsum('bsfl,flb->bsfd',x,self.linear2_w) + self.linear2_b
        x = self.LeakyReLU(x)
        # print(3, x.shape)
        # print(3.1,(x * self._tau_module).shape)
        # b time_steps feature_num bucket_num
        x = (x * self._tau_module).squeeze(-2)
        # print(4, 'tau', x.shape)
        # x = (x * self._tau_module).squeeze(2)
        x = self.Softmax(x)
        # print(5, 'softmax', x.shape)
        # b time_steps feature_num bucket_num bucket_dim
        x = x.unsqueeze(-1) * self.emb
        # print(6, 'emb', x.shape)
        # b time_steps feature_num bucket_dim
        x = torch.sum(x, dim=-2)
        # print(7, 'reduce', x.shape)
        # b time_steps feature_num*bucket_dim
        # x = torch.flatten(x, start_dim=2)
        x = torch.flatten(x, start_dim=-2)##TODO:Use -2 instead of 2(or 1 for non-sequence input) will result in the same code
        # x = torch.reshape(x, [x.shape[0], x.shape[1], -1])
        # print(8, 'faltten', x.shape)
        # b output_size if use_fc else feature_num*bucket_num
        x = self.fc(x)
        # print(-1, x.shape)
        return x


class SharedSequenceAutoDisBucketEncoder(nn.Module):
	def __init__(
		self,
		feature_num,  # 必填参数，最终网络输入的特征数量
		bucket_num=8,
		bucket_dim=128,
		layer_conf=[64, 64, 64],
		alpha=1,
		output_size=128,
		use_fc=False,
		dropout=0,
		add_block=True
	):
		super(SharedSequenceAutoDisBucketEncoder, self).__init__()
		self.LeakyReLU = nn.LeakyReLU(inplace=True)
		self.Dropout = nn.Dropout(p=dropout)
		self.linear1_w = nn.Parameter(torch.randn(feature_num, 3, layer_conf[0]))
		self.linear1_b = nn.Parameter(torch.randn(feature_num, 1, layer_conf[0]))
		self.layer = nn.Sequential(*[ResBlock(feature_num, layer_len, dropout=0, alpha=alpha) for layer_len in
									 layer_conf]) if add_block else nn.Identity()
		self.linear2_w = nn.Parameter(torch.randn(feature_num, layer_conf[-1], bucket_num))
		self.linear2_b = nn.Parameter(torch.randn(feature_num, 1, bucket_num))

		self.emb = nn.Parameter(torch.randn(feature_num, bucket_num, bucket_dim))
		self._tau_module = nn.Parameter(torch.ones([feature_num, 1, bucket_num]))
		self.Softmax = nn.Softmax(dim=-1)
		self.fc = nn.Linear(feature_num * bucket_dim, output_size, bias=True) if use_fc else nn.Identity()
		self.output_size = output_size if use_fc else feature_num * bucket_dim

		nn.init.kaiming_uniform_(self.linear1_w, mode='fan_in', nonlinearity='leaky_relu')
		nn.init.kaiming_uniform_(self.linear2_w, mode='fan_in', nonlinearity='leaky_relu')

	def forward(self, x):  # b timesteps feature_num 3
		# print(0,x.shape)
		#  b time_steps feature_num 1 3
		x = x.unsqueeze(-2) #TODO:Use -2 instead of 3(or 2 for non-sequence input) will result in same code
		#  b time_steps feature_num layer_conf[0]
		x = torch.matmul(x, self.linear1_w) + self.linear1_b
		x = self.LeakyReLU(x)
		x = self.Dropout(x)
		# print(1, x.shape)
		# b time_steps feature_num 1 layer_conf[-1]
		x = self.layer(x)
		# print(2, x.shape)
		#  b time_steps feature_num 1 bucket_num
		x = torch.matmul(x, self.linear2_w) + self.linear2_b
		x = self.LeakyReLU(x)
		# print(3, x.shape)
		# b time_steps feature_num bucket_num
		x = (x * self._tau_module).squeeze(-2)
		# print(4, 'tau', x.shape)
		x = self.Softmax(x)
		# print(5, 'softmax', x.shape)
		# b time_steps feature_num bucket_num bucket_dim
		x = x.unsqueeze(-1) * self.emb
		# print(6, 'emb', x.shape)
		# b time_steps feature_num bucket_dim
		x = torch.sum(x, dim=-2)
		# print(7, 'reduce', x.shape)
		# b time_steps feature_num*bucket_dim
		# x = torch.flatten(x, start_dim=2)
		x = torch.flatten(x, start_dim=-2)##TODO:Use -2 instead of 2(or 1 for non-sequence input) will result in the same code
		# x = torch.reshape(x, [x.shape[0], x.shape[1], -1])
		# print(8, 'reshape', x.shape)
		# b output_size if use_fc else feature_num*bucket_num
		x = self.fc(x)
		# print(-1, x.shape)
		return x

