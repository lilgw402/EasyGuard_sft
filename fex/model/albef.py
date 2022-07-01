#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
albef
"""
from typing import Tuple
import os
import torch
import torch.nn.functional as F
from torch import nn

from fex.config import CfgNode
from fex.nn.backbone.albef import ALBEF

try:
    from fex.nn.loss.align_uniform import uniform_loss
    from fex.nn.loss.set_criterion import SetCriterionCOS
except:
    print('no uniform and setcri')

from fex.nn.backbone.albert import AlbertLMPredictionHead
from ptx.model.deberta.model import DebertaEMD
from ptx.model.util import gather_positions

from fex.utils.distributed import AllGather
allgather = AllGather.apply


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


@torch.no_grad()
def concat_allgather(tensor):
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class ALBEFPretraining(ALBEF):
    """
    albef
    模型参数：可以分成3部分，
    1. visual encoder。是一个纯纯的visual encoder，只负责将图片（视频）编码成 feature map
    2. text encoder。  是一个纯纯的text encoder，譬如一个bert，只负责将文本编码成 text embedding
    3. cross encoder。 负责建模图像和文本之间的交互。

    训练过程：
    先通过contrastive 寻找一个难负例，这里会产出一个对比学习的loss
    再通过MLM

    """

    def __init__(self, config: CfgNode):
        super().__init__(config)

        if self.text_backbone_mode == 'deberta_cross':
            # deberta 在后面做MLM的时候，还需要一些emd之类的东西
            self.emd = DebertaEMD(
                self.cross_config,
                num_emd_groups=1,
                emd_group_repeat=2,
                use_fast=False,
            )
            self.ape = nn.Embedding(self.cross_config.max_relative_positions, self.cross_config.dim)

        # contrastive
        contrastive_proj_mode = config.get('network.contrastive_proj_mode', 'default')
        if contrastive_proj_mode == 'default':
            self.vision_proj = nn.Linear(config.network.visual_dim, config.network.contrastive_embed_dim)
            self.text_proj = nn.Linear(config.BERT.hidden_size, config.network.contrastive_embed_dim)
        elif contrastive_proj_mode == '1024':
            self.vision_proj = torch.nn.Sequential(
                torch.nn.Linear(config.network.visual_dim, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024)
            )
            self.text_proj = torch.nn.Sequential(
                torch.nn.Linear(config.BERT.hidden_size, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024)
            )

        self.temp = config.network.temp
        self.calc_ce = torch.nn.CrossEntropyLoss(ignore_index=-1)

        # itm
        self.with_margin_loss = config.get('train.with_margin_loss', False)
        if not self.with_margin_loss:
            self.itm_head = nn.Linear(config.BERT.hidden_size, 2)
        else:
            self.relevance_head = RelevanceHead(config.BERT.hidden_size)
            self.margin = config.train.get('margin', 0.2)
        self.global_neg_itm = config.get('train.global_neg_itm', False)
        self.random_neg = config.get('train.random_neg', False)

        # mlm
        self.mlm_head = AlbertLMPredictionHead(config.BERT, self.text.embedding.token_embedder_tokens.weight)
        self.mlm_probability = config.get('network.mlm_probability', 0.15)
        self.PAD_IDX = 2
        self.CLS_IDX = 0
        self.MASK_IDX = 1
        self.vocab_size = config.BERT.vocab_size

        # uniform
        self.with_visual_uniform = config.get('train.with_visual_uniform', False)

        # 是否给 visual 一个match loss
        self.visual_match_loss_mode = config.get('train.visual_match_loss_mode', None)
        if self.visual_match_loss_mode == 'set':
            self.set_criterion = SetCriterionCOS(dim=config.BERT.hidden_size, eos_coef=0.1, matcher='HungarianMatcherCOSBatch')
        self.visual_match_loss_with_aux = config.get('train.visual_match_loss_with_aux', False)

    def forward(self, input_ids, input_segment_ids, input_mask,
                image=None, image_mask=None,
                masked_input_ids=None, mlm_labels=None, mlm_index=None, mlm_pos=None,
                *args, **kwargs):
        """
        """
        # 1. encode
        # image
        image_out = self.visual_encode(image, return_dict=True)
        image_embeds, image_final_token = image_out['feature_map'], image_out['pooled_out']
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        image_atts = image_atts * image_mask.unsqueeze(-1)
        # text
        text_output = self.text_encode(input_ids=input_ids, input_segment_ids=input_segment_ids, input_mask=input_mask)
        text_embeds = text_output['encoded_layers'][-1]

        # visual match loss
        # =================================
        visual_match_loss = 0.
        # filip
        if self.visual_match_loss_mode == 'filip':

            # 构造文本 target， detach & norm & mask cls+sep+pad
            filip_target = F.normalize(text_embeds.detach(), dim=-1)
            filip_target_mask = input_mask.unsqueeze(1)  # [bsz, 1, seq_len]
            filip_target_mask[:, :, 0] = 0  # mask cls
            sep_index = input_mask.sum(1)  # [bsz]
            for i, j in enumerate(sep_index):  # TODO: 后面看看能加速下不
                filip_target_mask[i, :, j] = 0
            visual_match_loss = calc_filip_loss(image_embeds, filip_target, filip_target_mask)
            visual_match_loss_ofd = calc_offdiag_loss(image_embeds)
            visual_match_loss = visual_match_loss + visual_match_loss_ofd

            visual_match_loss_aux = 0.
            if self.visual_match_loss_with_aux and 'all_feature_map' in image_out:
                for i, image_embeds_i in enumerate(image_out['all_feature_map']):
                    visual_match_loss_i = calc_filip_loss(image_embeds_i, filip_target, filip_target_mask)
                    # visual_match_loss_ofd_i = calc_offdiag_loss(image_embeds)
                    visual_match_loss_aux = visual_match_loss_aux + visual_match_loss_i  # + visual_match_loss_ofd_i
            visual_match_loss = visual_match_loss + visual_match_loss_aux * 0.15
        elif self.visual_match_loss_mode == 'uniform':
            # uniform
            for embs in image_embeds:
                visual_match_loss = visual_match_loss + uniform_loss(embs)
            visual_match_loss = visual_match_loss / image_embeds.shape[0]
            visual_match_loss = torch.clamp(visual_match_loss + 10, min=0.) * 0.1
        elif self.visual_match_loss_mode == 'set':
            # set criterion
            targets = text_embeds[:, 1:-1, :].detach()  # text_embeds 掐头去尾 # detach
            visual_match_loss = self.set_criterion(image_embeds, targets)
        elif self.visual_match_loss_mode == 'off_diagonal':

            image_embeds_norm = torch.nn.functional.normalize(image_embeds, dim=-1)
            c = image_embeds_norm @ image_embeds_norm.transpose(1, 2)  # [bsz, 8, 8]
            c.div_(image_embeds_norm.shape[1])
            off_diags = []
            for mat in c:
                off_diag = off_diagonal(mat).pow_(2).sum()
                off_diags.append(off_diag)
            off_diags = torch.stack(off_diags, dim=-1)
            visual_match_loss = off_diags.mean()

        # 2. contrastive
        image_feat = F.normalize(self.vision_proj(image_final_token), dim=-1)
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        try:
            image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
            text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        except:
            image_feat_all = image_feat
            text_feat_all = text_feat

        bsz = image_feat_all.shape[0]
        labels = torch.arange(bsz, device=image.device)  # bsz
        image_feat_all = F.normalize(image_feat_all, dim=1)
        text_feat_all = F.normalize(text_feat_all, dim=1)
        logits = image_feat_all @ text_feat_all.t() / self.temp
        loss_i2t = self.calc_ce(logits, labels)
        loss_t2i = self.calc_ce(logits.t(), labels)
        loss_ita = (loss_i2t + loss_t2i) / 2

        # 3. ITM
        ###============== ITM ===================###
        # forward the positve image-text pair
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.

        # attention_mask = self.get_extended_attention_mask(input_mask, input_ids.size(),
        #                                                   input_mask.device, False)

        output_pos = self.cross_encoder_encode(text_embeds, input_mask, image_embeds, image_atts)

        # find hard negative for ITM
        image_gat = None
        with torch.no_grad():
            bs = image.size(0)
            if self.global_neg_itm:
                # sim_i2t = image_feat @ text_feat_all.t() / self.temp # 本地和全局算
                #sim_t2i = text_feat @ image_feat_all.t() / self.temp
                cur_rank = torch.distributed.get_rank()
                sim_i2t = logits[cur_rank * bs:(cur_rank + 1) * bs]
                sim_t2i = logits.t()[cur_rank * bs:(cur_rank + 1) * bs]
                # 如果是 global negative，需要将输入都gather一下，在取hard negative 的时候取
                #image_embeds_all = allgather(image_embeds, torch.distributed.get_rank(), torch.distributed.get_world_size())
                image_gat = concat_allgather(image)
                input_ids_gat = concat_allgather(input_ids)
                input_mask_gat = concat_allgather(input_mask)
            else:
                sim_i2t = image_feat @ text_feat.t() / self.temp
                sim_t2i = text_feat @ image_feat.t() / self.temp
            sim_i2t_max = torch.max(sim_i2t).detach().clone()  # [:, :bs]
            sim_t2i_max = torch.max(sim_t2i).detach().clone()
            weights_i2t = F.softmax(sim_i2t - sim_i2t_max, dim=1)
            weights_t2i = F.softmax(sim_t2i - sim_t2i_max, dim=1)
            weights_i2t.add_(1e-5).fill_diagonal_(0)
            weights_t2i.add_(1e-5).fill_diagonal_(0)

        # select a negative image for each text
        # TODO: 这里可能可以改成并行的，加速一下
        image_embeds_neg = []
        image_neg = []  # 只有在 self.global_neg_itm 为 true 时用到
        image_neg_idx = []
        for b in range(bs):
            if self.random_neg:
                neg_idx = torch.multinomial(torch.ones_like(weights_t2i[b]), 1).item()
            else:
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_neg_idx.append(neg_idx)
            if self.global_neg_itm:
                image_neg.append(image_gat[neg_idx])
            else:
                image_embeds_neg.append(image_embeds[neg_idx])
        if self.global_neg_itm:  # global neg 的情况，现算
            image_neg = torch.stack(image_neg, dim=0)
            image_embeds_neg, _ = self.visual_encode(image_neg)
        else:
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        input_ids_neg = []
        input_mask_neg = []
        # text_neg_idx = []
        for b in range(bs):
            if self.random_neg:
                neg_idx = torch.multinomial(torch.ones_like(weights_i2t[b]), 1).item()
            else:
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            # text_neg_idx.append(neg_idx)
            if self.global_neg_itm:
                input_ids_neg.append(input_ids_gat[neg_idx])
                input_mask_neg.append(input_mask_gat[neg_idx])
            else:
                text_embeds_neg.append(text_embeds[neg_idx])
                input_mask_neg.append(input_mask[neg_idx])
        if self.global_neg_itm:
            input_ids_neg = torch.stack(input_ids_neg, dim=0)
            input_mask_neg = torch.stack(input_mask_neg, dim=0)
            # TODO: 现在用的 input_segment_ids 是一样的，所以没有同步，是一个偷懒的做法
            text_output_neg = self.text_encode(input_ids=input_ids_neg, input_segment_ids=input_segment_ids, input_mask=input_mask_neg)
            text_embeds_neg = text_output_neg['encoded_layers'][-1]
        else:
            text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
            input_mask_neg = torch.stack(input_mask_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        #text_atts_all = torch.cat([attention_mask, text_atts_neg], dim=0)
        input_mask_all = torch.cat([input_mask, input_mask_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.cross_encoder_encode(text_embeds_all, input_mask_all, image_embeds_all, image_atts_all)

        vl_embeddings = torch.cat([output_pos[:, 0, :], output_neg[:, 0, :]], dim=0)
        vl_embeddings = self.pooler(vl_embeddings)  # cls token pool 一下

        if self.with_margin_loss:
            rel_score = self.relevance_head(vl_embeddings)
            pos_score, neg_score_v, neg_score_t = rel_score.chunk(3, dim=0)
            # print(pos_score.shape, neg_score_v.shape, neg_score_t.shape, 'ddd')
            loss_margin_t = torch.clamp(self.margin + neg_score_t - pos_score, min=0.0)
            loss_margin_v = torch.clamp(self.margin + neg_score_v - pos_score, min=0.0)
            loss_itm = loss_margin_t.mean() + loss_margin_v.mean()
        else:
            vl_output = self.itm_head(vl_embeddings)
            # 一共做 bsz * 3: 第一份是正例，第二份是text正image负，第3份text负image正
            itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                                   dim=0).to(image.device)
            loss_itm = F.cross_entropy(vl_output, itm_labels)

        # 4. MLM
        ##================= MLM ========================##

        if masked_input_ids is None:
            input_ids = input_ids.clone()
            mlm_labels = input_ids.clone()
            probability_matrix = torch.full(mlm_labels.shape, self.mlm_probability)
            masked_input_ids, mlm_labels = self.mask(input_ids, self.vocab_size, image.device, targets=mlm_labels,
                                                     probability_matrix=probability_matrix)

        text_output = self.text_encode(input_ids=masked_input_ids, input_segment_ids=input_segment_ids, input_mask=input_mask)

        if 'concat' in self.text_backbone_mode:  # 如果有concat，就只取前面的文本token
            mlm_output = self.cross_encoder_encode(text_output['encoded_layers'][-1], input_mask, image_embeds, image_atts)
            mlm_output = mlm_output[:, :mlm_labels.shape[1], :]
        elif 'deberta' in self.text_backbone_mode:  # 如果是 deberta，需要emd一下
            mlm_output = self.cross_encoder_encode(text_output['encoded_layers'][-1], input_mask, image_embeds, image_atts, return_dict=True)
            position_ids = torch.arange(input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
            abs_pos_embeddings = self.ape(position_ids)
            mlm_output = self.emd(
                abs_pos_embeddings + mlm_output['sequence_output'],
                mlm_output['sequence_output'],
                input_mask,
                relative_pos=mlm_output['relative_pos'],
                rel_embedding=self.cross.rel_embeddings.weight,
            )
        else:
            mlm_output = self.cross_encoder_encode(text_output['encoded_layers'][-1], input_mask, image_embeds, image_atts)

        if mlm_pos is not None and mlm_index is not None:
            # 一种 gather的实现，先放着
            # mlm_pos_flaten = mlm_pos[:, :, None].expand(-1, -1, self.config.BERT.hidden_size) # [batch_size, max_pred, dim]
            #text_layer_masked = torch.gather(text_layer_out, 1, mlm_pos_flaten)
            mlm_output = gather_positions(mlm_output, mlm_pos)
            mlm_labels = mlm_index

        mlm_logits = self.mlm_head(mlm_output)
        # print(mlm_logits.shape, mlm_labels.shape, 'ddd')
        loss_mlm = F.cross_entropy(mlm_logits.view(-1, mlm_logits.shape[-1]),
                                   mlm_labels.view(-1),
                                   ignore_index=-1)

        loss = loss_mlm + loss_ita + loss_itm + visual_match_loss
        return {'loss_mlm': loss_mlm,
                'loss_ita': loss_ita,
                'loss_itm': loss_itm,
                'visual_match_loss': visual_match_loss,
                'loss': loss,
                'image_neg_idx': image_neg_idx,
                'image': image_gat if image_gat is not None else image
                }

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        """
        TODO: mask 逻辑前移，可以加一些实体词mask
        """
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.PAD_IDX] = False
        masked_indices[input_ids == self.CLS_IDX] = False

        if targets is not None:
            targets[~masked_indices] = -1  # -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.MASK_IDX

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids


class RelevanceHead(torch.nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.projector = torch.nn.Linear(hidden_size, 256)
        self.relevance = torch.nn.Linear(256, 1)
        self.acti = torch.nn.ReLU6(inplace=True)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, emb):
        """
        emb: [bsz, dim]
        """
        proj = self.acti(self.projector(emb))
        relevance_score = self.relevance(proj)
        relevance_score = torch.sigmoid(relevance_score)
        relevance_score = relevance_score.reshape(-1)
        return relevance_score


def calc_filip_loss(image_embeds, filip_target, filip_target_mask):
    """
    计算filip loss，
    """
    image_embeds_norm = torch.nn.functional.normalize(image_embeds, dim=-1)
    element_cross_matrix = image_embeds_norm @ filip_target.transpose(1, 2)
    element_cross_matrix = element_cross_matrix + (1 - filip_target_mask) * -10
    i2t = torch.mean(element_cross_matrix.max(2)[0], 1)  # [bsz]
    visual_match_loss = (1 - i2t.mean())
    return visual_match_loss


def calc_offdiag_loss(image_embeds):
    # offdiag
    image_embeds_norm = torch.nn.functional.normalize(image_embeds, dim=-1)
    c = image_embeds_norm @ image_embeds_norm.transpose(1, 2)  # [bsz, 8, 8]
    c.div_(image_embeds_norm.shape[1])
    off_diags = []
    for mat in c:
        off_diag = off_diagonal(mat).pow_(2).sum()
        off_diags.append(off_diag)
    off_diags = torch.stack(off_diags, dim=-1)
    visual_match_loss_ofd = off_diags.mean()
    return visual_match_loss_ofd
