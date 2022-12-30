import copy
from titan.utils.helper import create_model
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from transformers import BertConfig
from titan.utils.logger import logger
from titan.utils.registry import register_model
from titan.models.mm.xbert import BertForMaskedLM
from titan.utils.helper import download_weights, get_configs


__all__ = [
    'albef_pretrain',
]


@torch.no_grad()
def concat_allgather(tensor):
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(
            dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class ALBEFPretrain(nn.Module):
    def __init__(
            self,
            image_model=None,
            vision_width=768,
            text_encoder='bert-base-uncased',
            embed_dim=256,
            queue_size=65536,
            momentum=0.995,
            temp=0.07,
            temp_min=0.001,
            temp_max=0.5,
            fusion_layer=6,
            conf_path='',
            img_size=256,
            multinomial_offset=0,
            simlarity_offset=0,
            **kwargs) -> None:
        super(ALBEFPretrain, self).__init__()
        if image_model is None:
            logger.info(
                'No image model for ALBEF, create a ViT model by default')
            self.visual_encoder = create_model(
                'vit_base',
                num_classes=0,
                only_cls_token=False,
                img_size=img_size)
        else:
            self.visual_encoder = image_model
        if conf_path:
            user_config = BertConfig.from_json_file(conf_path)
            user_config.fusion_layer = fusion_layer
            user_config.encoder_width = vision_width
            self.text_encoder = BertForMaskedLM.from_pretrained(
                text_encoder, config=user_config)
        else:
            default_bert_config = BertConfig(
                fusion_layer=fusion_layer, encoder_width=vision_width)
            self.text_encoder = BertForMaskedLM.from_pretrained(
                text_encoder, config=default_bert_config)
        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.temp = nn.Parameter(torch.ones([]) * temp)
        self.temp_range = (temp_min, temp_max)
        self.queue_size = queue_size
        self.momentum = momentum
        self.multinomial_offset = multinomial_offset
        self.simlarity_offset = simlarity_offset
        self.itm_head = nn.Linear(text_width, 2)
        self.visual_encoder_m = copy.deepcopy(self.visual_encoder)
        for param in self.visual_encoder_m.parameters():
            param.requires_grad = False
        self.text_encoder_m = copy.deepcopy(self.text_encoder)
        for param in self.text_encoder_m.parameters():
            param.requires_grad = False
        self.vision_proj_m = copy.deepcopy(self.vision_proj)
        for param in self.vision_proj_m.parameters():
            param.requires_grad = False
        self.text_proj_m = copy.deepcopy(self.text_proj)
        for param in self.text_proj_m.parameters():
            param.requires_grad = False
        self.register_buffer(
            "image_queue", torch.randn(
                embed_dim, self.queue_size))
        self.register_buffer(
            "text_queue", torch.randn(
                embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.image_queue = F.normalize(self.image_queue, dim=0)
        self.text_queue = F.normalize(self.text_queue, dim=0)
        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m],
        ]

    def forward(self, image, text, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(self.temp_range[0], self.temp_range[1])
        image_embeds = self.visual_encoder(image)
        # user's image model does not output hidden states
        if image_embeds.dim() == 2:
            image_embeds = image_embeds[:, None, :]
        image_atts = torch.ones(
            image_embeds.size()[
                :-1],
            dtype=torch.long).to(
            image.device)
        image_feat = F.normalize(
            self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        # self.vision_proj(image_embeds), dim=-1)

        text_output = self.text_encoder.bert(
            text['input_ids'],
            attention_mask=text['attention_mask'],
            return_dict=True,
            mode='text')
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image)
            if image_embeds_m.dim() == 2:
                image_embeds_m = image_embeds_m[:, None, :]
            image_feat_m = F.normalize(
                self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            # self.vision_proj_m(image_embeds_m), dim=-1)
            image_feat_all = torch.cat(
                [image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
            text_output_m = self.text_encoder_m.bert(
                text['input_ids'],
                attention_mask=text['attention_mask'],
                return_dict=True,
                mode='text')
            text_feat_m = F.normalize(self.text_proj_m(
                text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat(
                [text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * \
                F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * \
                F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)
                              * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)
                              * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        ###=================================###
        # forward the positve image-text pair
        output_pos = self.text_encoder.bert(encoder_embeds=text_embeds,
                                            attention_mask=text['attention_mask'],
                                            encoder_hidden_states=image_embeds,
                                            encoder_attention_mask=image_atts,
                                            return_dict=True,
                                            mode='fusion',
                                            )
        with torch.no_grad():
            bs = image.size(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1) + self.simlarity_offset
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1) + self.simlarity_offset

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            # fix issue: https://github.com/salesforce/ALBEF/issues/107
            nan_idx = weights_t2i[b].isnan()
            weights_t2i[b][nan_idx] = self.multinomial_offset
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            nan_idx = weights_i2t[b].isnan()
            weights_i2t[b][nan_idx] = self.multinomial_offset
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text['attention_mask'][neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat(
            [text['attention_mask'], text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.text_encoder.bert(
            encoder_embeds=text_embeds_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
            mode='fusion',
        )

        vl_embeddings = torch.cat(
            [output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(
            2 * bs, dtype=torch.long)], dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        ##================= MLM ========================##
        input_ids = text['input_ids'].clone()
        labels = text['mlm_labels']

        with torch.no_grad():
            logits_m = self.text_encoder_m(
                input_ids,
                attention_mask=text['attention_mask'],
                encoder_hidden_states=image_embeds_m,
                encoder_attention_mask=image_atts,
                return_dict=True,
                return_logits=True,
            )
        mlm_output = self.text_encoder(input_ids,
                                       attention_mask=text['attention_mask'],
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       labels=labels,
                                       soft_labels=F.softmax(logits_m, dim=-1),
                                       alpha=alpha
                                       )
        loss_mlm = mlm_output.loss

        return (loss_mlm, loss_ita,
                loss_itm), output_pos.last_hidden_state[:, 0, :]

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                    model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + \
                    param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        if dist.is_initialized():
            image_feats = concat_allgather(image_feat)
            text_feats = concat_allgather(text_feat)
        else:
            image_feats = image_feat
            text_feats = text_feat

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr
    
    def load_pretrained(self, checkpoint_path, load_img_q=False, load_text_q=False):
        checkpoint_path.pop('queue_ptr', None)
        image_queue = checkpoint_path.pop('image_queue', None)
        text_queue = checkpoint_path.pop('text_queue', None)
        if image_queue is not None and load_img_q:
            self.image_queue[:, :] = image_queue[:, :self.queue_size]
        if text_queue is not None and load_text_q:
            self.text_queue[:, :] = text_queue[:, :self.queue_size]
        self.load_state_dict(checkpoint_path, strict=False)


@register_model
def albef_pretrain(pretrained=False, load_image_queue=False, load_text_queue=False, **kwargs):
    model_name = 'albef_pretrain'
    pretrained_config, model_config = get_configs(**kwargs)
    model = ALBEFPretrain(**model_config)
    if pretrained:
        weight_path = download_weights(model_name, **pretrained_config)
        checkpoint = torch.load(weight_path)
        model.load_pretrained(checkpoint, load_image_queue, load_text_queue)
    return model
