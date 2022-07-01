

import torch
from torch import nn

from fex.nn.module.xbert import BertConfig, TransformerCrossEncoder


def detr(visual_backbone, *args, **kwargs):
    return DETR(visual_backbone, *args, **kwargs)


class DETR(nn.Module):
    def __init__(self, visual_backbone, keep_token, mode='decoder', dim=768, mid_dim=128, out_dim=768, has_cls=True, *args, **kwargs):
        """
        has_cls: 是否有cls，如果has_cls=true, 则表示有cls，如果has_cls=false，表示没有
        """
        super().__init__()
        self.backbone = visual_backbone
        self.mode = mode
        keep_token = keep_token - 1 if has_cls else keep_token
        self.keep_token = keep_token
        detr_config = BertConfig.from_dict(kwargs.get('detr', {}))
        self.decoder = TransformerCrossEncoder(detr_config)
        self.query = nn.Embedding(keep_token, mid_dim)
        self.proj_in = nn.Linear(dim, mid_dim)
        self.proj_out = nn.Linear(mid_dim, out_dim)
        self.has_cls = has_cls

    def forward(self, *args, **kwargs):
        if kwargs.get('return_dict'):
            kwargs.pop('return_dict')
        out = self.backbone(return_dict=True, *args, **kwargs)

        fm = out['feature_map']
        cls_emb = out['pooled_out']
        if self.has_cls:  # 如果有cls，不给cls 进detr的机会
            fm = fm[:, 1:]
        fm = self.proj_in(fm)
        bsz = fm.shape[0]

        query = self.query(torch.arange(self.keep_token, device=fm.device))
        query = query.repeat(bsz, 1, 1)
        #attention_mask = torch.ones([bsz, self.keep_token-1], dtype=torch.long, device=query.device)
        #image_atts = torch.ones(fm.size()[:-1], dtype=torch.long, device=fm.device)
        dec_out = self.decoder(hidden_states=query,
                               # attention_mask=attention_mask,
                               encoder_hidden_states=fm,
                               # encoder_attention_mask=image_atts,
                               output_attentions=True,
                               output_hidden_states=True
                               )
        visual_tokens = dec_out.last_hidden_state
        visual_tokens = self.proj_out(visual_tokens)
        if self.has_cls:  # 跳过的cls直接拼接
            cls_emb_aed = self.proj_out(self.proj_in(cls_emb))
            visual_tokens = torch.cat([cls_emb_aed.unsqueeze(1), visual_tokens], dim=1)
        return {'feature_map': visual_tokens,
                'pooled_out': cls_emb,
                'all_feature_map': [self.proj_out(i) for i in dec_out.hidden_states],  # 所有层的
                'attention_weight': dec_out.cross_attentions}
