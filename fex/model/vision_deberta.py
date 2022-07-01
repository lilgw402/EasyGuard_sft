#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vision + deberta pretraining
https://arxiv.org/abs/2006.03654
"""

from typing import Dict, List, Optional
from functools import partial

import torch
import torch.nn as nn

from fex import _logger as log
from fex.nn.backbone.vision_deberta_encoder import VisionDeberta

try:
    from ptx.metric import CategoricalAccuracy
    from ptx.model.util import gather_positions
    from ptx.model.bert import init_weights, BertPreTrainingHeads
    from ptx.model.deberta.model import DebertaEMD
    from ptx.model.electra.model import GumbelSampling
except Exception as e:
    log.warning('PTX is not installed!')


class VisionDebertaPretraining(VisionDeberta):
    """
    vision deberta
    在 encoder 的基础上，增加decoder，和预训练loss的计算
    """

    def __init__(
        self,
        max_len: int = 512,
        n_frames: int = 16,
        abs_pos_embedding: bool = False,
        ignore_index: int = -1,
        calc_mlm_accuracy: bool = True,
        tie_embedding: bool = True,
        use_emd: bool = False,
        num_emd_groups: int = 1,
        emd_group_repeat: int = 2,
        layernorm_fp16: bool = False,
        use_fast: bool = False,
        head_layernorm_type: str = 'default',
        omit_other_output: bool = False,
        visual_front: bool = False,
        **option,
    ):
        super().__init__(use_fast=use_fast, layernorm_fp16=layernorm_fp16, omit_other_output=omit_other_output, **option)

        if abs_pos_embedding:
            self.ape = nn.Embedding(max_len, option['dim'])
            self.v_ape = nn.Embedding(n_frames, option['dim'])
        else:
            self.ape = None
        self.vocab_size = option['vocab_size']
        self.cls = BertPreTrainingHeads(dict(
            dim=option['dim'],
            embedding_dim=option['embedding_dim'],
            layer_norm_eps=option['layer_norm_eps'],
            vocab_size=option['vocab_size'],
            act=option['act'],
            layernorm_type=head_layernorm_type,
        ))
        if layernorm_fp16:
            self.cls.predictions.layer_norm._simply_cast = True
        if tie_embedding:
            self._tie_weights()

        self.with_ve_loss = option.get('with_ve_loss', False)

        self._omit_other_output = omit_other_output
        self._calc_mlm_accuracy = calc_mlm_accuracy and not self._omit_other_output
        self.mlm_accuracy = self._calc_mlm_accuracy and CategoricalAccuracy()

        self.ignore_index = ignore_index
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.nsp_loss_function = torch.nn.CrossEntropyLoss()

        self.local_metrics = {}

        self.use_emd = use_emd
        if use_emd:
            self.emd = DebertaEMD(
                self.da_config,
                num_emd_groups=num_emd_groups,
                emd_group_repeat=emd_group_repeat,
                use_fast=use_fast,
            )

        self.sampling = GumbelSampling()  # sampling

        self.apply(partial(init_weights, initializer_range=option.pop('initializer_range')))

    def _tie_weights(self):
        self.cls.predictions.decoder.weight = self.embedding.token_embedder_tokens.weight

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        global_metrics = {}
        if self._calc_mlm_accuracy:
            global_metrics['mlm_accuracy'] = self.mlm_accuracy.get_metric(reset)
        global_metrics.update(self.local_metrics)
        return global_metrics

    def _update_local_metrics(self, mlm_logits, mlm_labels):
        if self._calc_mlm_accuracy:
            total_count, correct_count = float(self.mlm_accuracy.total_count), float(self.mlm_accuracy.correct_count)
            mlm_positions = torch.nonzero(mlm_labels != self.ignore_index, as_tuple=False).view(-1)
            self.mlm_accuracy(mlm_logits[mlm_positions], mlm_labels[mlm_positions])
            local_total_count = float(self.mlm_accuracy.total_count) - total_count
            local_correct_count = float(self.mlm_accuracy.correct_count) - correct_count
            local_accuracy = 0.0 if local_total_count == 0 else (float(local_correct_count) / float(local_total_count))
            self.local_metrics.update({
                'local_mlm_total_count': local_total_count,
                'local_mlm_correct_count': local_correct_count,
                'local_mlm_accuracy': local_accuracy,
            })

    def forward(
        self,
        input_ids,
        position_ids=None,
        input_segment_ids=None,
        input_mask=None,
        masked_tokens=None,
        sentence_label=None,
        masked_lm_positions=None,
        masked_lm_ids=None,
        frames=None,
        frames_mask=None,
        visual_embeds=None,
        mode='tv'
    ):
        # if input_mask is not None:
        #     mask = input_mask
        # else:
        #     mask = input_ids != self.padding_index
        #     mask[:, 0:1] = 1

        output = super().forward(
            input_ids=input_ids,
            input_segment_ids=input_segment_ids,
            input_mask=input_mask,
            # position_ids=position_ids,
            frames=frames,
            frames_mask=frames_mask,
            visual_embeds=visual_embeds,
            output_pooled=True,
            output_rel_pos=self.use_emd,
            mode='tv'
        )

        sequence_output = output['encoded_layers'][-1]
        pooled_output = output['pooled_output']
        mask = output['embedding_masks']

        encoder_last_seq_output = sequence_output
        if self.with_ve_loss:
            vis_target_ids = self.sample_visual_to_text(output['visual_final_output'])

        if self.use_emd:
            if self.ape is not None:
                abs_pos_embeddings = self.ape(position_ids.long())
                bsz, length = frames_mask.size()[:2]
                v_position_ids = torch.arange(0, length, dtype=torch.long, device=position_ids.device).expand(bsz, length)
                v_abs_pos_embeddings = self.v_ape(v_position_ids.long())
                abs_pos_embeddings = torch.cat([abs_pos_embeddings, v_abs_pos_embeddings], dim=1)
                sequence_output = self.emd(
                    abs_pos_embeddings + sequence_output,
                    sequence_output,
                    mask,
                    relative_pos=output['relative_pos'],
                    rel_embedding=self.encoder.rel_embeddings.weight,
                )
            else:
                # TODO: fix
                sequence_output = self.emd(sequence_output, sequence_output)

        decoder_last_seq_output = sequence_output

        # 计算visual token 的 MLM
        if self.with_ve_loss:
            if self.visual_front:
                visual_length = frames_mask.shape[1]  # frame_num
                visual_emd_out = sequence_output[:, 1:visual_length]  # 第0个位置是cls，不要
            else:
                text_length = input_mask.shape[1]
                visual_emd_out = sequence_output[:, text_length:]
            vis_pred_logits, _ = self.cls(visual_emd_out)
            vis_pred_logits = vis_pred_logits.view(-1, self.vocab_size)
            vis_target_ids = vis_target_ids.view(-1)
            ve_loss = self.loss_function(vis_pred_logits, vis_target_ids)

        # MLM Loss
        # Shrink `sequence_output` and `masked_tokens` according to `masked_lm_positions` and `masked_lm_ids`
        positioned = masked_lm_positions is not None

        if (masked_tokens is not None) and positioned:
            masked_lm_positions_dim = masked_lm_positions.dim()
            if masked_lm_positions_dim == 2:
                position_ids = masked_lm_positions
                sequence_output = gather_positions(sequence_output, masked_lm_positions)
                # Well, `ignore_index` may vary with this case
                masked_tokens = masked_lm_ids
            elif masked_lm_positions_dim == 1:
                position_ids = position_ids.view(-1)[masked_lm_positions]
                sequence_output = sequence_output.contiguous().view(-1, sequence_output.size(-1))[masked_lm_positions]
                masked_tokens = masked_lm_ids
            else:
                raise Exception('Invalid dim of masked_lm_positions and masked_lm_ids')

        pred_score, seq_score = self.cls(sequence_output, pooled_output)

        loss = 0.0
        mlm_logits = pred_score.view(-1, self.vocab_size)
        if masked_tokens is not None:
            mlm_labels = masked_tokens.view(-1)
            loss = self.loss_function(mlm_logits, mlm_labels)
            if not self._omit_other_output:
                self._update_local_metrics(mlm_logits, mlm_labels)
                self.local_metrics['local_mlm_loss'] = loss.item()
        if sentence_label is not None:
            next_sentence_loss = self.nsp_loss_function(seq_score.view(-1, 2), sentence_label.view(-1))
            if not self._omit_other_output:
                self.local_metrics['local_nsp_loss'] = next_sentence_loss.item()
            loss = loss + next_sentence_loss

        if self._omit_other_output:
            return {'loss': loss}
        res = {
            'loss': loss,
            'encoder_last_hidden_state': encoder_last_seq_output,
            'decoder_last_hidden_state': decoder_last_seq_output,
            'pooled_output': pooled_output,
            'local_mlm_total_count': self.local_metrics.get('local_mlm_total_count'),
            'local_mlm_correct_count': self.local_metrics.get('local_mlm_correct_count'),
            'local_mlm_accuracy': self.local_metrics.get('local_mlm_accuracy'),
        }
        if self.with_ve_loss:
            res['ve_loss'] = ve_loss
        return res

    def sample_visual_to_text(self, vis_out):
        """
        """
        vis_scores, _ = self.cls(vis_out)
        sample_ids = self.sampling(vis_scores)
        return sample_ids
