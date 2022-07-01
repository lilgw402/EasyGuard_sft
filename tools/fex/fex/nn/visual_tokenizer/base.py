"""
Visual Tokenizer
"""

import torch
import torch.nn as nn

from fex.nn.backbone.resnet import resnet50
from fex.nn.backbone.dino import dino_vit_s16, dino_vit_s8, dino_vit_b8, dino_vit_b16, dino_vit_b32
from fex.nn.backbone.swin_transformer import swin_transformer_base, swin_transformer_tiny, swin_transformer_small, swin_transformer_large
from fex.nn.backbone.swin_transformer_dual import swin_transformer_base as swin_transformer_dual_base
from fex.nn.backbone.swin_transformer_3d import swin_transformer_3d_B244


from fex.nn.backbone.with_ptx import USE_PTX_TRANSFORMER
if not USE_PTX_TRANSFORMER:
    from fex.nn.backbone.vit import visual_transformer_B32, visual_transformer_B16
else:
    from fex.nn.backbone.vit_v2 import visual_transformer_B32, visual_transformer_B16

from fex.nn.backbone.vit_ibot import vit_tiny, vit_small, vit_base, vit_large
from fex.nn.backbone.vit_beit import beit_base_patch16_224
from fex.nn.backbone.vit_clip import clip_vit_b16

from fex.nn.backbone.vit_mae import mae_base_patch16
from fex.nn.backbone.deit import deit_vit_b16

from fex.nn.backbone.swin_vidt import swin_base_win7_vidt

from fex.nn.visual_tokenizer.not_all_patch import not_all_patch
from fex.nn.visual_tokenizer.detr import detr


SUPPORTTED_VISUAL_TYPE_BASE = [
    'RN50',
    'VitB32', 'VitB16',
    'DINO/ViTS16', 'DINO/ViTS8', 'DINO/ViTB16', 'DINO/ViTB8',
    'SwinB224', 'SwinT224', 'SwinS224', 'SwinB384', 'SwinB256',
    'SwinDualB224', 'SwinL224',
    'IBotT16', 'IBotS16', 'IBotB16', 'IBotL16',
    'BeitB16',
    'MaeB16S224', 'MaeB16S384',
    'DeitB16S224', 'DeitB16S256', 'DeitB16S288', 'DeitB16S384',
    'ClipB16S224', 'ClipB16S256', 'ClipB16S288', 'ClipB16S384',
    'VideoSwinB244',
    'SwinVidtBW7',
    # 'NAPClipB16S224', 'NAPDINO/ViTB16', 'NAPIBotB16',
    # 'DETRClipB16S224', 'DETRDINO/ViTB16', 'DETRIBotB16'

]

PREFIX = ['NAP', 'DETR']


SUPPORTTED_VISUAL_TYPE_PREFIX = []
for p in PREFIX:
    for b in SUPPORTTED_VISUAL_TYPE_BASE:
        SUPPORTTED_VISUAL_TYPE_PREFIX.append(f'{p}{b}')

SUPPORTTED_VISUAL_TYPE = SUPPORTTED_VISUAL_TYPE_PREFIX + SUPPORTTED_VISUAL_TYPE_BASE


def create_visual_tokenizer(visual_type: str = 'VitB32',
                            output_dim: int = 512,
                            vit_dropout: float = 0.1,
                            vit_emb_dropout: float = 0.0,
                            patch_length: int = 49,
                            *args, **kwargs
                            ):
    if visual_type not in SUPPORTTED_VISUAL_TYPE:
        raise ValueError(f'visual type: {visual_type} is not supportted, please choose one from {SUPPORTTED_VISUAL_TYPE}')
    if visual_type == 'VitB32':
        return visual_transformer_B32(
            output_dim=output_dim,
            dropout=vit_dropout,
            emb_dropout=vit_emb_dropout,
            patch_length=patch_length
        )
    elif visual_type == 'VitB16':
        return visual_transformer_B16(
            output_dim=output_dim,
            dropout=vit_dropout,
            emb_dropout=vit_emb_dropout,
            patch_length=patch_length
        )
    elif visual_type == 'DINO/ViTS16':
        return dino_vit_s16()
    elif visual_type == 'DINO/ViTS8':
        return dino_vit_s8()
    elif visual_type == 'DINO/ViTB16':
        return dino_vit_b16()
    elif visual_type == 'DINO/ViTB8':
        return dino_vit_b8()
    elif visual_type == 'RN50':
        return resnet50(expose_stages=[5])  # 5是最后一层，6是分类输出
    # elif visual_type == 'DINO/RN50':
    #     return dino_resnet50()
    elif visual_type == 'SwinT224':
        return swin_transformer_tiny(
            output_dim=output_dim,
            img_size=224,
            *args, **kwargs)
    elif visual_type == 'SwinS224':
        return swin_transformer_small(
            output_dim=output_dim,
            img_size=224,
            *args, **kwargs)
    elif visual_type == 'SwinB224':
        return swin_transformer_base(
            output_dim=output_dim,
            img_size=224,
            *args, **kwargs
        )
    elif visual_type == 'SwinDualB224':
        return swin_transformer_dual_base(
            output_dim=output_dim,
            img_size=224,
            *args, **kwargs
        )
    elif visual_type == 'SwinB384':
        return swin_transformer_base(
            output_dim=output_dim,
            img_size=384,
            window_size=12,
            *args, **kwargs
        )
    elif visual_type == 'SwinB256':
        return swin_transformer_base(
            output_dim=output_dim,
            img_size=256,
            window_size=8,
            *args, **kwargs
        )
    elif visual_type == 'SwinVidtBW7':
        return swin_base_win7_vidt(*args, **kwargs)
    elif visual_type == 'VideoSwinB244':
        return swin_transformer_3d_B244()
    elif visual_type == 'SwinL224':
        return swin_transformer_large(
            output_dim=output_dim,
            img_size=224,
            *args, **kwargs)
    elif visual_type == 'IBotT16':
        return vit_tiny()
    elif visual_type == 'IBotS16':
        return vit_small()
    elif visual_type == 'IBotB16':
        return vit_base()
    elif visual_type == 'IBotL16':
        return vit_large()
    elif visual_type == 'BeitB16':
        return beit_base_patch16_224()
    elif visual_type == 'MaeB16S224':
        return mae_base_patch16(image_size=224)
    elif visual_type == 'MaeB16S384':
        return mae_base_patch16(image_size=384)
    elif visual_type == 'DeitB16S224':
        return deit_vit_b16(224)
    elif visual_type == 'DeitB16S256':
        return deit_vit_b16(256)
    elif visual_type == 'DeitB16S288':
        return deit_vit_b16(288)
    elif visual_type == 'DeitB16S384':
        return deit_vit_b16(384)
    elif visual_type == 'ClipB16S224':
        return clip_vit_b16(224)
    elif visual_type == 'ClipB16S256':
        return clip_vit_b16(256)
    elif visual_type == 'ClipB16S288':
        return clip_vit_b16(288)
    elif visual_type == 'ClipB16S384':
        return clip_vit_b16(384)
    elif visual_type.startswith('NAP'):
        raw_visual_type = visual_type.replace('NAP', '')
        return not_all_patch(create_visual_tokenizer(raw_visual_type), *args, **kwargs)
    elif visual_type.startswith('DETR'):
        raw_visual_type = visual_type.replace('DETR', '')
        return detr(create_visual_tokenizer(raw_visual_type), *args, **kwargs)
    else:
        raise ValueError(f'visual type: {visual_type} is no supported ')
