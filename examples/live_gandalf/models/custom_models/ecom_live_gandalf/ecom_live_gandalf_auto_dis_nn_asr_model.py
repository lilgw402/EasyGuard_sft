# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-03-02 14:23:40
# Modified: 2023-03-02 14:23:40
# coding=utf-8
# Author: jiangxubin
# Create: 2022/8/2 18:15
import torch
import torch.nn as nn
from typing import Optional
from cruise import CruiseModule
from easyguard.core import AutoModel,AutoTokenizer
from examples.live_gandalf.builder import MODELS
from examples.live_gandalf.models.modules.encoders import AutodisBucketEncoder
from examples.live_gandalf.models.modules.losses import BCEWithLogitsLoss
from examples.live_gandalf.models.modules.running_metrics import GeneralClsMetric
from examples.live_gandalf.utils.util import count_params

@MODELS.register_module()
class EcomLiveGandalfAutoDisNNAsrModel(CruiseModule):
    def __init__(
        self,
        **model_instance_kwargs
    ):
        super(EcomLiveGandalfAutoDisNNAsrModel, self).__init__()
        self.save_hparams()

    def setup(self, stage: Optional[str] = None) -> None:
        # Init model components
        model_instance_kwargs = self.hparams.model_instance_kwargs
        # Init model components:float features
        self._enable_asr_embedding = model_instance_kwargs.get("enable_asr_embedding", 0)
        self._feature_num = model_instance_kwargs.get('feature_num', 1024)
        self._bucket_num = model_instance_kwargs.get('bucket_num', 8)
        self._bucket_dim = model_instance_kwargs.get('bucket_dim', 128)
        self._bucket_output_size = model_instance_kwargs.get('bucket_output_size', 1024)
        self._feature_input_num = self._feature_num - len(model_instance_kwargs.get('slot_mask', []))
        self._drop_prob = model_instance_kwargs.get('dropout', 0.3)
        self._bucket_all_emb_dim = self._feature_input_num * self._bucket_dim
        self._auto_dis_bucket_encoder = AutodisBucketEncoder(
                                        feature_num=self._feature_input_num,
                                        bucket_num=self._bucket_num,
                                        bucket_dim=self._bucket_dim,
                                        output_size=self._bucket_output_size,
                                        use_fc=False,
                                        add_block=False)
        # Init model components:asr
        self._asr_encoder_param = model_instance_kwargs.get('asr_encoder_param', {})
        self.asr_model_name = "fashion-deberta-asr-small"
        self._asr_encoder = AutoModel.from_pretrained(self.asr_model_name)
        self._asr_tokenizer = AutoTokenizer.from_pretrained(self.asr_model_name, return_tensors="pt", max_length=self._asr_encoder_param['max_length'])
        self._asr_emb_dropout = self._init_emb_dropout()
        # Init loss weight
        self._loss_weight = model_instance_kwargs.get('loss_weight', None)
        # Init metric
        self._metric = GeneralClsMetric()
        self._init_cls_layer()
        self._init_criterion()
        self._sigmoid = nn.Sigmoid()
        if model_instance_kwargs.get("reset_params", False):
            self._reset_params()
        count_params(self)

    def forward(
        self,
        auto_dis_input,
        feature_dense,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        targets=None,
    ):
        # get auto_dis embedding
        auto_dis_embedding = self._auto_dis_bucket_encoder(auto_dis_input)
        # get concat features
        asr_embedding = self._asr_encoder(input_ids, attention_mask, token_type_ids)
        # Add dropout for embedding
        asr_embedding = self._asr_emb_dropout(asr_embedding)
        # Concat all input features which have been transformed into embeddings
        concat_features = self._bucket_embedding_bottom(torch.cat([
            auto_dis_embedding,
            feature_dense,
            asr_embedding
            ], 1)
        )
        # Get output of model
        output = self._classifier_final(concat_features)
        if targets is not None:
            if not self._loss_weight:
                loss = self._criterion_final(output, targets)
                loss_dict = {"loss": loss}
                output_prob = self._post_process_output(output)
                output_dict = {"output": output_prob}
            else:
                loss = self._criterion_final(output, targets)
                loss_dict = {
                    "loss": loss,
                }
                output_prob = self._post_process_output(output)
                output_dict = {
                    "output": output_prob,
                }
            # 添加评价指标
            eval_dict = self._metric.batch_eval(output_prob, targets, key='eval')
            output_dict.update(eval_dict)
            return loss_dict, output_dict
        else:
            return self._post_process_output(output)

    def training_step(self, batch, batch_idx):
        auto_dis_input, feature_dense,input_ids,attention_mask, token_type_ids = self.pre_process_inputs(batch)
        targets = self.pre_process_targets(batch)
        loss_dict, output_dict = self.forward(auto_dis_input,feature_dense,input_ids,attention_mask, token_type_ids,targets)
        return loss_dict

    def validation_step(self, batch, batch_idx):
        auto_dis_input, feature_dense, input_ids, attention_mask, token_type_ids, targets = self.pre_process_inputs(batch)
        loss_dict, output_dict = self.forward(auto_dis_input, feature_dense, input_ids, attention_mask, token_type_ids,
                                              None)
        self.log_dict(output_dict)

    def test_step(self,batch,batch_idx):
        auto_dis_input, feature_dense, input_ids, attention_mask, token_type_ids, targets = self.pre_process_inputs(
            batch)
        loss_dict, output_dict = self.forward(auto_dis_input, feature_dense, input_ids, attention_mask, token_type_ids,
                                              None)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate
        )
        return {"optimizer": optimizer}

    def _nn_bottom(self, dim: list):  # net
        net = nn.Sequential()
        for i in range(len(dim)-1):
            net.add_module('layer_{}'.format(i), nn.Linear(dim[i], dim[i+1], bias=True))
            net.add_module('activation_{}'.format(i), nn.ReLU(inplace=True))
            net.add_module('dropout_{}'.format(i), nn.Dropout(p=self._drop_prob, inplace=False))
        return net

    def _init_emb_dropout(self):
        hidden_dropout_prob = self._asr_encoder_param.get('emb_dropout_prob', 0)
        emb_dropout = (0 if not  hidden_dropout_prob else hidden_dropout_prob)
        return nn.Dropout(emb_dropout)

    def _init_cls_layer(self):
        # bottom
        self._bucket_embedding_bottom = self._nn_bottom(
                dim=[self._bucket_all_emb_dim + self._feature_input_num + self._asr_encoder_param.get('embedding_dim', 768), 1024, 512]
            )
        # original classifier for pass/ban
        self._classifier_final = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self._drop_prob, inplace=False),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self._drop_prob, inplace=False),
            nn.Linear(128, 1, bias=True),
        )

    def _init_criterion(self):
        self._criterion_final = BCEWithLogitsLoss()

    @staticmethod
    def _pre_process(batched_feature_data_items):
        new_inputs = []
        for input in batched_feature_data_items:
            if isinstance(input, tuple) or isinstance(input, list):
                new_input = list(input)
            else:
                new_input = input
            new_inputs.append(new_input)
        return new_inputs

    def _post_process_output(self, output):
        # 业务逻辑，对模型输出做处理
        return self._sigmoid(output)

    def pre_process_inputs(self, batched_feature_data):
        asr_inputs = self._asr_tokenizer(batched_feature_data["asr"])
        device_num = batched_feature_data["auto_dis_input"].get_device()
        device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
        asr_inputs['input_ids'] = asr_inputs['input_ids'].to(torch.int32).to(device)
        asr_inputs['attention_mask'] = asr_inputs['attention_mask'].to(torch.int32).to(device)
        asr_inputs['token_type_ids'] = asr_inputs['token_type_ids'].to(torch.int32).to(device)
        batched_feature_data_items = [
            batched_feature_data["auto_dis_input"],
            batched_feature_data["feature_dense"],
            asr_inputs['input_ids'],
            asr_inputs['attention_mask'],
            asr_inputs['token_type_ids']
        ]
        return self._pre_process(batched_feature_data_items)

    def pre_process_targets(self, batched_feature_data):
        batched_feature_data_items = [
            batched_feature_data["label"],
        ]
        return self._pre_process(batched_feature_data_items)
