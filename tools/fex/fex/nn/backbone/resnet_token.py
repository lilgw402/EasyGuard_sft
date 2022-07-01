""" visual tokenizer """

import torch
from torch import nn
from fex.nn.backbone.resnet import resnet50


class ResnetTokenizer(nn.Module):
    """ resnet """

    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.activ_and_nobias = config.NETWORK.activ_and_nobias
        self.add_v_norm = config.NETWORK.v_norm

        self.resnet = resnet50(expose_stages=[5])
        self.v_proj = torch.nn.Linear(2048, 128)
        self.activ = torch.nn.Tanh()
        self.v_proj2 = torch.nn.Linear(128, 768, bias=(not self.activ_and_nobias))
        if self.add_v_norm:
            self.v_layer_norm = torch.nn.LayerNorm(768)    

        # freeze
        if config.NETWORK.FREEZE_RESNET_LAYERS:
            self.resnet.frozen_parameters(config.NETWORK.FREEZE_RESNET_LAYERS, frozen_bn=True)
            

    def forward(self, images, *args, **kwargs):
        """ TODO: capable of frame_mask """
        # 单图片的情况
        if len(images.shape) == 4: # [bsz, C, H, W]
            visual_embs = self.resnet(images)['body5'] # [bsz, 2048, h', w']
            visual_embs = torch.mean(visual_embs, dim=[2, 3])
            visual_embs = self.v_proj(visual_embs)
            if self.activ_and_nobias:
                visual_embs = self.activ(visual_embs)
            visual_embs = self.v_proj2(visual_embs)
            if self.add_v_norm:
                visual_embs = self.v_layer_norm(visual_embs)            
            visual_embs = visual_embs.unsqueeze(1)  # 加 token_length 
        # 多帧的情况
        elif len(images.shape) == 5: # [bsz, frames, C, H, W]
            bsz, frame_num, c, h, w = images.shape
            images = images.reshape([bsz * frame_num, c, h, w])
            visual_embs = self.resnet(images)['body5']
            visual_embs = torch.mean(visual_embs, dim=[2, 3])
            visual_embs = self.v_proj(visual_embs)
            if self.activ_and_nobias:
                visual_embs = self.activ(visual_embs)
            visual_embs = self.v_proj2(visual_embs)
            visual_embs = visual_embs.reshape([bsz, frame_num, -1])

        return visual_embs