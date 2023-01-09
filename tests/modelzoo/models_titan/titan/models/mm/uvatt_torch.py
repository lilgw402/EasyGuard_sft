# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Universal Video, Audio, and Text Transformer (UVATT)."""

import torch
import torch.nn as nn

def get_shape(x):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape
  dynamic = x.shape
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]

class UniversalVATT_torch(nn.Module):
  """The general Transformer for extracting different features for modalities."""

  def __init__(self,
               # pre-transformer parameters
               vid_temporal_patch_size=4,
               vid_spatial_patch_size=16,
               aud_temporal_patch_size=128,
               txt_vocab_size=100000,
               txt_embedding_dim=300,
               txt_embedding_trainable=False,
               # video & audio input sampling
               random_patch_sampling=False,
               patch_sampling_rate=0.5,
               # transformer head parameters
               d_model=1024,
               d_kv=64,
               d_ff=4096,
               num_layers=24,
               num_heads=16,
               pre_norm=True,
               use_bias=True,
               activation="gelu",
               dropout_rate=0.1,
               layer_norm_epsilon=1e-6,
               # positional embedding parameters
               max_vid_temporal_buckets=8,
               max_vid_spatial_buckets=14,
               max_aud_temporal_buckets=1200,
               max_txt_temporal_buckets=16,
               # final head parameters
               d_post_proj=1024,
               post_proj_activation="gelu",
               name="unified_vat_transformer",
               **kwargs):

    super(UniversalVATT_torch, self).__init__()
    self.d_model = d_model
    # define pre-tx projection
    self.raw_to_embeddings = {
        "video": nn.Conv3d(in_channels=3, 
                            out_channels=d_model, 
                            kernel_size=(vid_temporal_patch_size,
                                        vid_spatial_patch_size,
                                        vid_spatial_patch_size), 
                            stride=(vid_temporal_patch_size,
                                    vid_spatial_patch_size,
                                    vid_spatial_patch_size), 
                            padding=0),
        "audio": nn.Conv1d(in_channels=1, 
                            out_channels=d_model, 
                            kernel_size=aud_temporal_patch_size, 
                            stride=aud_temporal_patch_size, 
                            padding=0),
        "text": nn.Embedding(num_embeddings=txt_vocab_size, 
                            embedding_dim=txt_embedding_dim,) # TODO. further refine when DL is done
    }
    self.pre_proj = {
        "video": 
                nn.Linear(in_features=d_model, 
                            out_features=d_model), # needs activation
        "audio": 
                nn.Linear(in_features=d_model, 
                            out_features=d_model),
        "text": 
                nn.Linear(in_features=txt_embedding_dim, 
                            out_features=d_model),
        "activation": nn.GELU()
        }

    # define sampling-related params
    self.use_random_patches = random_patch_sampling
    self.patch_sampling_rate = patch_sampling_rate
    self.max_buckets = {
        "video": max_vid_temporal_buckets * (max_vid_spatial_buckets ** 2),
        "audio": max_aud_temporal_buckets,
    }
    self.max_num_patches = {
        "video": int(self.patch_sampling_rate * self.max_buckets["video"]),
        "audio": int(self.patch_sampling_rate * self.max_buckets["audio"]),
    }
    assert self.max_buckets["video"] > self.max_num_patches["video"], (
        "Max number of video positional buckets should be bigger than max"
        " number of video input patches"
        )
    assert self.max_buckets["audio"] > self.max_num_patches["audio"], (
        "Max number of audio positional buckets should be bigger than max"
        " number of audio input patches"
        )


    # define transformer head
    encoder_layer=nn.TransformerEncoderLayer(d_model=d_model, 
                                            nhead=num_heads, 
                                            dim_feedforward=2048, 
                                            dropout=dropout_rate, 
                                            activation=activation, 
                                            layer_norm_eps=layer_norm_epsilon, 
                                            batch_first=False, 
                                            norm_first=pre_norm)
    self.tx=nn.TransformerEncoder(encoder_layer=encoder_layer, 
                                    num_layers=num_layers,
                                   )


    # define post-tx projection head - it could be logits or embd space
    self.post_proj = {
        "video": 
                nn.Linear(in_features=d_model, 
                            out_features=d_post_proj), # needs activation
        "audio": 
                nn.Linear(in_features=d_model, 
                            out_features=d_post_proj),
        "text": 
                nn.Linear(in_features=d_model, 
                            out_features=d_post_proj),
        "activation": nn.GELU()
        }

  
  # note that this has been modified for torch implementation
  def _flatten_inputs(self,
                      inputs):

    input_shape = get_shape(inputs)
    bs = input_shape[0]
    d_embd = input_shape[1]

    inputs = inputs.reshape(bs,d_embd, -1)
    inputs=inputs.permute(0,2,1)
    return inputs, inputs.shape

  def _append_special_tokens(self,
                             inputs,
                             modality):

    batch_size = get_shape(inputs)[0]
    special_embd = torch.Tensor(torch.rand(inputs.shape[-1])) #self.agg_token[modality][None, None, :]
    special_embd=special_embd.view(1,1,-1)

    # (batch_size, 1, d_model)
    special_embd = torch.tile(special_embd, [batch_size, 1, 1])

    return torch.cat([special_embd, inputs], dim=1)

  def _random_patch_selection(self,
                              inputs,
                              training,
                              input_shape,
                              modality):
    if training and modality != "text":
      # get inputs dimensions
      batch_size, seq_len, dim = get_shape(inputs)

      # shuffle on temporal axis and gather the first max_num_patches
      temporal_idx = torch.range(0,seq_len)
      temporal_idx = torch.random.shuffle(temporal_idx)[None, :]
      temporal_idx = tf.tile(temporal_idx, [batch_size, 1])

      batch_idx = tf.range(batch_size)[:, None]
      batch_idx = tf.tile(batch_idx, [1, seq_len])

      gather_idx = tf.stack([batch_idx, temporal_idx], axis=2)

      inputs = tf.gather_nd(inputs,
                            gather_idx)[:, :self.max_num_patches[modality], :]
      input_shape = [batch_size, self.max_num_patches[modality], dim]

    return inputs, input_shape

  def _extend_attn_mask(self,
                        attention_mask):
    attn_mask_shape = get_shape(attention_mask)
    if len(attn_mask_shape) > 2:
      raise NotImplementedError

    batch_size = attn_mask_shape[0]
    extention_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype)
    extended_attention_mask = torch.cat([extention_mask, attention_mask], dim=1)
    return extended_attention_mask

  def _modality_call(self,
                     inputs,
                     modality,
                     training=False,
                     attention_mask=None,
                     input_shape=None):

    print("inferencing modality: ", modality)

    # linear projection to d_model
    embeddings = self.raw_to_embeddings[modality](inputs)
    if modality=="video":
      embeddings=embeddings.permute(0,2,3,4,1)
      embeddings = self.pre_proj[modality](embeddings)
      embeddings =self.pre_proj['activation'](embeddings)
      embeddings=embeddings.permute(0,4,1, 2,3)
    if modality=="audio":
      embeddings=embeddings.permute(0,2,1)
      embeddings = self.pre_proj[modality](embeddings)
      embeddings =self.pre_proj['activation'](embeddings)
      embeddings=embeddings.permute(0, 2,1)
    if modality=="text":
      embeddings = self.pre_proj[modality](embeddings)
      embeddings =self.pre_proj['activation'](embeddings)
      embeddings=embeddings.permute(0, 2,1)

    # flatten inputs if not flattened already
    if input_shape is None:
      embeddings, input_shape = self._flatten_inputs(embeddings)

    else:
      is_flattened = len(get_shape(inputs)) == 3
      assert is_flattened, (
          "if input_shape provided, inputs should be flattened and have rank 3")


    ## TODO still needs to replace the following with spatial-temporal embedding
    embeddings=torch.rand(input_shape)

    # randomly choose "max_num_patches" tokens
    # TODO convert and debug this part during training. no need for inference.
    if self.use_random_patches:
      embeddings, input_shape = self._random_patch_selection(
          embeddings,
          training,
          input_shape,
          modality,
          )

    # append modalities special tokens: [vid, aud, txt]
    tx_inputs = self._append_special_tokens(embeddings, modality)

    # extend attention_mask accordingly
    if attention_mask is not None:
      attention_mask = self._extend_attn_mask(attention_mask)

    # call Transformer
    tx_outputs = self.tx(src=tx_inputs,
                          mask=attention_mask,
                          src_key_padding_mask=None)

    # get last hidden states and perform final linear projection
    last_hidden_states = tx_outputs
    modality_outputs = self.post_proj[modality](last_hidden_states)
    # output_shape = input_shape[:-1] + [get_shape(modality_outputs)[-1]]

    features_pooled = modality_outputs[:, 0, :]
    features = modality_outputs[:, 1:, :]#.reshape(output_shape)

    # add token-level Transformer outputs
    outputs = {"features_pooled": features_pooled,
               "features": features}

    return outputs

  
  def forward(self, inputs,training=False):
    outputs = {}

    for modality in ["video", "audio", "text"]:
      modality_inputs = inputs[modality]["data"]
      modality_attn_mask = inputs[modality].get("attention_mask", None)
      outputs[modality] = self._modality_call(inputs=modality_inputs,
                                              modality=modality,
                                              training=training,
                                              attention_mask=modality_attn_mask)
    return outputs



if __name__ == "__main__":
  model=UniversalVATT_torch()
  print(model)

  ## construct dummy input
  inputs={}
  inputs['video']={}
  inputs['video']['data']=torch.rand(1,3,8,224,224)
  inputs['audio']={}
  inputs['audio']['data']=torch.rand(1,1,1024*20) 
  inputs['text']={}
  inputs['text']['data']=torch.randint(low=1, high=100000,size=(1,240))

  out=model(inputs=inputs)
  print('video output shape: ',out['video']['features_pooled'].shape)
  print('audio output shape: ',out['audio']['features_pooled'].shape)
  print('text output shape: ',out['text']['features_pooled'].shape)

