import torch
import torch.nn as nn

from collections import OrderedDict
from copy import deepcopy
from timm.models.layers import DropPath, trunc_normal_

from titan.utils.registry import register_model
from titan.utils.helper import (
    download_weights,
    load_pretrained_model_weights,
    get_configs
)
from titan.utils.logger import logger

__all__ = ['automl_fbnet_v1', 'automl_fbnet_v2']


# search space for each TBS layer
# FBNet V2
SSPACE = {
    "MICRO": [
        # bottleneck block
        ('B1', "Bottleneck", {'kernel_size': 3, 'use_se': 0, 'groups': 1}),
        ('B2', "Bottleneck", {'kernel_size': 3, 'use_se': 0, 'groups': 2}),
        ('B3', "Bottleneck", {'kernel_size': 3, 'use_se': 0, 'groups': 4}),
        ('B4', "Bottleneck", {'kernel_size': 3, 'use_se': 1, 'groups': 1}),
        ('B5', "Bottleneck", {'kernel_size': 3, 'use_se': 1, 'groups': 2}),
        ('B6', "Bottleneck", {'kernel_size': 3, 'use_se': 1, 'groups': 4}),
        # ViTBlock
        ('B7', "ViTBlock", {'num_heads': 24, 'mlp_ratio': 4}),
        ('B8', "ViTBlock", {'num_heads': 24, 'mlp_ratio': 3}),
        ('B9', "ViTBlock", {'num_heads': 24, 'mlp_ratio': 2}),
        ('B10', "ViTBlock", {'num_heads': 12, 'mlp_ratio': 4}),
        ('B11', "ViTBlock", {'num_heads': 12, 'mlp_ratio': 3}),
        ('B12', "ViTBlock", {'num_heads': 12, 'mlp_ratio': 2}),
        # skip op
        ('SKIP', "Identity", {}),
    ],

    # hybridnet search space
    "hybridnet_2344": [
        # stage 0, input 224 x 224 x 3
        ["conv_3x3_1", "ConvBlock", {"kernel_size": 3, "stride": 2,
                                     "padding": 1, "in_channels": 3,
                                     "out_channels": 32, "act": "relu",
                                     "bias": False}],
        ["conv_3x3_2", "ConvBlock", {"kernel_size": 3, "stride": 1,
                                     "padding": 1, "in_channels": 32,
                                     "out_channels": 32, "act": "relu",
                                     "bias": False}],
        ["conv_3x3_3", "ConvBlock", {"kernel_size": 3, "stride": 1,
                                     "padding": 1, "in_channels": 32,
                                     "out_channels": 64, "act": "relu",
                                     "bias": False}],
        ["maxpool_3x3", "MaxPool", {
            "kernel_size": 3, "stride": 2, "padding": 1}],
        # stage 1 --- 2 layers, conv block
        ["TBS1", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                             "expansion":1, "in_channels":64,
                             "out_channels": [64, 128, 16]}],
        ["TBS2", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                             "expansion":1, "in_channels":128,
                             "out_channels":[64, 128, 16]}],
        # stage 2 --- 3 layers, conv block
        ["TBS3", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':2,
                             "expansion":1, "in_channels":128,
                             "out_channels":[128, 256, 32]}],
        ["TBS4", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                             "expansion":1, "in_channels":256,
                             "out_channels":[128, 256, 32]}],
        ["TBS5", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                             "expansion":1, "in_channels":256,
                             "out_channels":[128, 256, 32]}],
        # stage 3 --- 4 layers, conv block
        ["TBS6", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':2,
                             "expansion":1, "in_channels":256,
                             "out_channels":[256, 512, 64]}],
        ["TBS7", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                             "expansion":1, "in_channels":512,
                             "out_channels":[256, 512, 64]}],
        ["TBS8", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                             "expansion":1, "in_channels":512,
                             "out_channels":[256, 512, 64]}],
        ["TBS9", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                             "expansion":1, "in_channels":512,
                             "out_channels":[256, 512, 64]}],
        # stage 4 -- 4 layers
        ["TBS10", "MixedOp", {'bids': [6, 7, 8, 9, 10, 11], 'stride':2,
                              "in_channels":512,
                              "out_channels":[768, 1536, 768],
                              "drop_path":0.075}],
        ["TBS11", "MixedOp", {'bids': [6, 7, 8, 9, 10, 11], 'stride':1,
                              "in_channels":1536,
                              "out_channels":[768, 1536, 768],
                              "drop_path":0.083}],
        ["TBS12", "MixedOp", {'bids': [6, 7, 8, 9, 10, 11], 'stride':1,
                              "in_channels":1536,
                              "out_channels":[768, 1536, 768],
                              "drop_path":0.092}],
        ["TBS13", "MixedOp", {'bids': [6, 7, 8, 9, 10, 11], 'stride':1,
                              "in_channels":1536,
                              "out_channels":[768, 1536, 768],
                              "drop_path":0.1}],
        # post process
        ["batchnorm", "BatchNorm2d", {"in_channels": 1536}],
        ["ap_7x7", "AdaptiveAvgPool", {}]
    ],

    "hybridnet_3463": [
        # stage 0, input 224 x 224 x 3
        ["conv_3x3_1", "ConvBlock", {"kernel_size": 3, "stride": 2,
                                     "padding": 1, "in_channels": 3,
                                     "out_channels": 32, "act": "relu",
                                     "bias": False}],
        ["conv_3x3_2", "ConvBlock", {"kernel_size": 3, "stride": 1,
                                     "padding": 1, "in_channels": 32,
                                     "out_channels": 32, "act": "relu",
                                     "bias": False}],
        ["conv_3x3_3", "ConvBlock", {"kernel_size": 3, "stride": 1,
                                     "padding": 1, "in_channels": 32,
                                     "out_channels": 64, "act": "relu",
                                     "bias": False}],
        ["maxpool_3x3", "MaxPool", {
            "kernel_size": 3, "stride": 2, "padding": 1}],
        # stage 1 --- 3 layers, conv block
        ["TBS1", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                             "expansion":1, "in_channels":64,
                             "out_channels": [64, 128, 16]}],
        ["TBS2", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                             "expansion":1, "in_channels":128,
                             "out_channels":[64, 128, 16]}],
        ["TBS3", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                             "expansion":1, "in_channels":128,
                             "out_channels":[64, 128, 16]}],
        # stage 2 --- 4 layers, conv block
        ["TBS4", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':2,
                             "expansion":1, "in_channels":128,
                             "out_channels":[128, 256, 32]}],
        ["TBS5", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                             "expansion":1, "in_channels":256,
                             "out_channels":[128, 256, 32]}],
        ["TBS6", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                             "expansion":1, "in_channels":256,
                             "out_channels":[128, 256, 32]}],
        ["TBS7", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                             "expansion":1, "in_channels":256,
                             "out_channels":[128, 256, 32]}],
        # stage 3 --- 6 layers, conv block
        ["TBS8", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':2,
                             "expansion":1, "in_channels":256,
                             "out_channels":[256, 512, 64]}],
        ["TBS9", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                             "expansion":1, "in_channels":512,
                             "out_channels":[256, 512, 64]}],
        ["TBS10", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                              "expansion":1, "in_channels":512,
                              "out_channels":[256, 512, 64]}],
        ["TBS11", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                              "expansion":1, "in_channels":512,
                              "out_channels":[256, 512, 64]}],
        ["TBS12", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                              "expansion":1, "in_channels":512,
                              "out_channels":[256, 512, 64]}],
        ["TBS13", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                              "expansion":1, "in_channels":512,
                              "out_channels":[256, 512, 64]}],
        # stage 4 -- 4 layers
        ["TBS14", "MixedOp", {'bids': [6, 7, 8, 9, 10, 11], 'stride':2,
                              "in_channels":512,
                              "out_channels":[768, 1536, 768],
                              "drop_path":0.0867}],
        ["TBS15", "MixedOp", {'bids': [6, 7, 8, 9, 10, 11], 'stride':1,
                              "in_channels":1536,
                              "out_channels":[768, 1536, 768],
                              "drop_path":0.0933}],
        ["TBS16", "MixedOp", {'bids': [6, 7, 8, 9, 10, 11], 'stride':1,
                              "in_channels":1536,
                              "out_channels":[768, 1536, 768],
                              "drop_path":0.1}],
        # post process
        ["batchnorm", "BatchNorm2d", {"in_channels": 1536}],
        ["ap_7x7", "AdaptiveAvgPool", {}]
    ],

    "hybridnet_234": [
        # stage 0, input 224 x 224 x 3
        ["conv_3x3_1", "ConvBlock", {"kernel_size": 3, "stride": 2,
                                     "padding": 1, "in_channels": 3,
                                     "out_channels": 32, "act": "relu",
                                     "bias": False}],
        ["conv_3x3_2", "ConvBlock", {"kernel_size": 3, "stride": 1,
                                     "padding": 1, "in_channels": 32,
                                     "out_channels": 32, "act": "relu",
                                     "bias": False}],
        ["conv_3x3_3", "ConvBlock", {"kernel_size": 3, "stride": 1,
                                     "padding": 1, "in_channels": 32,
                                     "out_channels": 64, "act": "relu",
                                     "bias": False}],
        ["maxpool_3x3", "MaxPool", {
            "kernel_size": 3, "stride": 2, "padding": 1}],
        # stage 1 --- 2 layers, conv block
        ["TBS1", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                             "expansion":1, "in_channels":64,
                             "out_channels": [64, 128, 16]}],
        ["TBS2", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                             "expansion":1, "in_channels":128,
                             "out_channels":[64, 128, 16]}],
        # stage 2 --- 3 layers, conv block
        ["TBS3", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':2,
                             "expansion":1, "in_channels":128,
                             "out_channels":[128, 256, 32]}],
        ["TBS4", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                             "expansion":1, "in_channels":256,
                             "out_channels":[128, 256, 32]}],
        ["TBS5", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                             "expansion":1, "in_channels":256,
                             "out_channels":[128, 256, 32]}],
        # stage 3 --- 4 layers, conv block
        ["TBS6", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':2,
                             "expansion":1, "in_channels":256,
                             "out_channels":[256, 512, 64]}],
        ["TBS7", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                             "expansion":1, "in_channels":512,
                             "out_channels":[256, 512, 64]}],
        ["TBS8", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                             "expansion":1, "in_channels":512,
                             "out_channels":[256, 512, 64]}],
        ["TBS9", "MixedOp", {'bids': [0, 1, 2, 3, 4, 5], 'stride':1,
                             "expansion":1, "in_channels":512,
                             "out_channels":[256, 512, 64]}],
        # stage 4 -- 4 layers
        ["vit10", "ViTBlock", {'num_heads': 24, 'mlp_ratio': 3,
                               'stride': 2, "in_channels": 512,
                               "out_channels": 768}],
        ["vit11", "ViTBlock", {'num_heads': 24, 'mlp_ratio': 3,
                               'stride': 1, "in_channels": 768,
                               "out_channels": 768}],
        ["vit12", "ViTBlock", {'num_heads': 24, 'mlp_ratio': 3,
                               'stride': 1, "in_channels": 768,
                               "out_channels": 768}],
        ["vit13", "ViTBlock", {'num_heads': 24, 'mlp_ratio': 3,
                               'stride': 1, "in_channels": 768,
                               "out_channels": 768}],
        # post process
        ["batchnorm", "BatchNorm2d", {"in_channels": 768}],
        ["ap_7x7", "AdaptiveAvgPool", {}]
    ],
}


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.ReLU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features,
                             kernel_size=1, stride=1, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features,
                             kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            input_resolution,
            num_heads,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.,
            use_relative_position_bias=True,
            refiner_ks=1,
            refiner_dim=None,
            refiner_skip_proj=False,
            refiner_shortcut=False,
            linear=False,
            linear_size=7,
            sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_relative_position_bias = use_relative_position_bias

        pos_heads = num_heads
        if self.use_relative_position_bias:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.zeros(
                (2 * input_resolution[0] - 1) *
                (2 * input_resolution[1] - 1), pos_heads))  # 2*H-1 * 2*W-1, nH

            # get pair-wise relative position index for each token
            # inside the window
            coords_h = torch.arange(self.input_resolution[0])
            coords_w = torch.arange(self.input_resolution[1])
            coords = torch.stack(torch.meshgrid(
                [coords_h, coords_w]))  # 2, H, W
            coords_flatten = torch.flatten(coords, 1)  # 2, H*W
            # 2, H*W, H*W
            relative_coords = coords_flatten[:, :,
                                             None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(
                1, 2, 0).contiguous()  # H*W, H*W, 2
            # shift to start from 0
            relative_coords[:, :, 0] += self.input_resolution[0] - 1
            relative_coords[:, :, 1] += self.input_resolution[1] - 1
            relative_coords[:, :, 0] *= 2 * self.input_resolution[1] - 1
            relative_position_index = relative_coords.sum(-1)  # H*W, H*W
            self.register_buffer("relative_position_index",
                                 relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        # self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        # self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads,
                              N).permute(0, 1, 3, 2)  # B, nH, N, C//nH
        kv = self.kv(x).reshape(B, 2, self.num_heads, C //
                                self.num_heads, -
                                1).permute(1, 0, 2, 4, 3)  # 2, B, nH, N, C//nH
        # make torchscript happy (cannot use tensor as tuple)
        k, v = kv[0], kv[1]
        N_ = k.shape[-2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.use_relative_position_bias:
            # H*W, H*W, nH
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)].view(N, N_, -1)
            # nH, H*W, H*W
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()

            attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        out = (attn @ v).permute(0, 1, 3, 2).contiguous().reshape(B, C, H, W)
        out = self.proj(out)
        # out = self.proj_drop(out)
        return out


class ViTBlock(nn.Module):
    def __init__(self, in_dim, dim, input_resolution, num_heads, proj=None,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., stride=1,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_relative_position_bias=True, use_avgdown=False,
                 skip_lam=1., dwconv=False, dwconv_shortcut=False,
                 refiner_ks=0, refiner_dim=None,
                 refiner_skip_proj=False, refiner_shortcut=False,
                 linear=False, linear_size=1, sr_ratio=1, block_dwconv_ks=0):
        super().__init__()

        self.skip_lam = skip_lam

        if stride != 1 or in_dim != dim:
            proj = nn.Conv2d(in_dim, dim, kernel_size=1, stride=stride)
        self.proj = proj or nn.Identity()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_relative_position_bias=use_relative_position_bias,
            refiner_ks=refiner_ks,
            refiner_dim=refiner_dim,
            refiner_skip_proj=refiner_skip_proj,
            refiner_shortcut=refiner_shortcut,
            linear=linear,
            linear_size=linear_size,
            sr_ratio=sr_ratio)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer)

    def forward(self, x):
        x = self.proj(x)
        if isinstance(self.norm1, nn.BatchNorm2d):
            return self.forward_batchnorm(x)
        elif isinstance(self.norm1, nn.LayerNorm):
            return self.forward_layernorm(x)

    def forward_batchnorm(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x

    def forward_layernorm(self, x):
        B, C, H, W = x.shape
        x0 = x
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        x = self.norm1(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = x0 + self.drop_path(self.attn(x)) / self.skip_lam

        x0 = x
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        x = self.norm2(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = x0 + self.drop_path(self.mlp(x)) / self.skip_lam
        return x


# conv block
class ConvBlock(nn.Module):
    # conv-bn-act block
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=0,
                 groups=1,
                 bias=True,
                 bn=True,
                 act='relu',
                 zero_gamma=False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

        if zero_gamma:
            nn.init.constant_(self.bn.weight, 0.0)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        if x.dtype != self.conv.weight.dtype and \
                self.conv.weight.dtype == torch.float16:
            x = x.half()
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModule, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Conv2d(channel, channel // reduction,
                      kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel,
                      kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excite(y)
        return x * y


class Bottleneck(nn.Module):
    def __init__(self,
                 kernel_size,
                 in_channels,
                 out_channels,
                 stride,
                 expansion=1,
                 groups=1,
                 use_se=False,
                 use_avgdown=False,
                 act='relu'):

        super(Bottleneck, self).__init__()

        mid_channel = out_channels * expansion
        # 1x1 conv
        self.conv_bn_relu1 = ConvBlock(
            in_channels=in_channels,
            out_channels=mid_channel,
            bias=False,
            kernel_size=1,
            stride=1,
            padding=0,
            act='relu')
        # 3x3 conv
        self.conv_bn_relu2 = ConvBlock(in_channels=mid_channel,
                                       out_channels=mid_channel,
                                       kernel_size=kernel_size, stride=stride,
                                       padding=kernel_size // 2, groups=groups,
                                       bias=False, act='relu')
        # se module
        self.se = SEModule(mid_channel) if use_se else None
        # 1x1 conv
        self.conv_bn_relu3 = ConvBlock(
            in_channels=mid_channel,
            out_channels=out_channels,
            bias=False,
            kernel_size=1,
            stride=1,
            padding=0,
            act='none')

        if stride != 1 or in_channels != out_channels:
            self.downsample = ConvBlock(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                act='none')
        else:
            self.downsample = nn.Identity()

        self.final_act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.downsample(x)

        out = self.conv_bn_relu1(x)
        out = self.conv_bn_relu2(out)
        if self.se is not None:
            out = self.se(out)
        out = self.conv_bn_relu3(out)
        out = self.final_act(out + res)
        return out


class HybridNet(nn.Module):
    r""" Hybrid network structure
    """

    def __init__(self, in_chans=3, num_classes=1000, img_size=224,
                 layers=[], blocks=None, transitions=None,
                 embed_dims=[], kernel_sizes=[],
                 groups=[], use_ses=[],
                 num_heads=[], mlp_ratios=[],
                 mode='pytorch', use_avgdown=False,
                 classifier_norm=False, stem_type='conv7', stem_chs=[64],
                 patch_size=16, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 qkv_bias=False, qk_scale=None, is_repeat=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 global_pool='avg', features_only=False, name=None,
                 **kwargs):

        super().__init__()
        self.name = name
        self.in_dim = stem_chs[-1]
        self.img_size = img_size
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.features_only = features_only
        self.out_channels = []

        if stem_type == 'conv7':
            # default stem used in ResNet
            self.layer0 = nn.Sequential(
                *
                [
                    nn.Conv2d(
                        in_chans,
                        stem_chs[0],
                        kernel_size=7,
                        stride=2,
                        padding=3,
                        bias=False),
                    nn.BatchNorm2d(
                        stem_chs[0]),
                    nn.ReLU(
                        inplace=True),
                    nn.MaxPool2d(
                        kernel_size=3,
                        stride=2,
                        padding=1)])
            input_resolution = (img_size // 4, img_size // 4)

        elif stem_type == '3xconv3':
            self.layer0 = nn.Sequential(*[
                nn.Conv2d(
                    in_chans, stem_chs[0], kernel_size=3,
                    stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], kernel_size=3,
                          stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_chs[1], stem_chs[2], kernel_size=3,
                          stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[2]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)])
            input_resolution = (img_size // 4, img_size // 4)

        elif stem_type == 'patch_stem':
            stem_chs = [384]  # FIXME
            patch_size = 16  # FIXME
            self.layer0 = nn.Conv2d(
                in_chans,
                stem_chs[0],
                kernel_size=patch_size,
                stride=patch_size)
            input_resolution = (img_size // patch_size, img_size // patch_size)

        elif stem_type == 'early_conv_stem':
            self.layer0 = nn.Sequential(*[
                nn.Conv2d(
                    in_chans, stem_chs[0], kernel_size=3,
                    stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], kernel_size=3,
                          stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_chs[1], stem_chs[2], kernel_size=3,
                          stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[2]),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_chs[2], stem_chs[3], kernel_size=3,
                          stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[3]),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_chs[3], stem_chs[4], kernel_size=1, stride=1)])
            input_resolution = (img_size // 16, img_size // 16)

        elif stem_type == '4xconv_stride8':
            # 3x conv3x3 followed by 1 conv1x1, output stride is 8
            self.layer0 = nn.Sequential(*[
                nn.Conv2d(
                    in_chans, stem_chs[0], kernel_size=3,
                    stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], kernel_size=3,
                          stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_chs[1], stem_chs[2], kernel_size=3,
                          stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[2]),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_chs[2], stem_chs[3], kernel_size=1, stride=1)])
            input_resolution = (img_size // 8, img_size // 8)

        # build stages
        self.stages = []
        self.out_indices = kwargs.pop('out_indices', None)

        dpr = [
            x.item() for x in torch.linspace(
                0,
                drop_path_rate,
                sum(layers))]  # stochastic depth decay rule

        self.is_repeat = is_repeat
        for i in range(len(layers)):
            # FIXME merge this into function
            # read block-specific hyperparams
            layer = layers[i]
            block = blocks[i]
            transition = transitions[i]
            if is_repeat:
                kernel_size = kernel_sizes[i]
                group = groups[i]
                use_se = use_ses[i]
                dim = embed_dims[i]
                num_head = num_heads[i]
                mlp_ratio = mlp_ratios[i]
            else:
                start = sum(layers[:i])
                end = sum(layers[:i + 1])
                kernel_size = kernel_sizes[start:end]
                group = groups[start:end]
                use_se = use_ses[start:end]
                dim = embed_dims[start:end]
                num_head = num_heads[start:end]
                mlp_ratio = mlp_ratios[start:end]

            # check if dimension is divisible by num_heads
            # if num_head is not None:
            #    assert dim % num_head == 0, f"Dimension ({dim}) \
            #       is not divisible by number of heads ({num_head})"

            if transition:
                stride = 2
                input_resolution = [input_resolution[0] //
                                    2, input_resolution[1] // 2]
            else:
                stride = 1

            if block == 'bottleneck':
                # ResNet Bottleneck block
                stage = self._make_res_stage(
                    dim=dim,
                    layer=layer,
                    block=Bottleneck,
                    mode=mode,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=group,
                    use_se=use_se,
                    use_avgdown=use_avgdown,
                    **kwargs)
            elif block == 'vit':
                # Vision Transformer block
                stage = self._make_stage(
                    dim=dim,
                    layer=layer,
                    block=ViTBlock,
                    input_resolution=input_resolution,
                    num_heads=num_head,
                    stride=stride,
                    use_avgdown=use_avgdown,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(layers[:i]):sum(layers[:i + 1])],
                    act_layer=act_layer,
                    norm_layer=norm_layer)

            if isinstance(dim, list):
                self.out_channels.append(dim[-1])
            else:  # type of dim is int
                self.out_channels.append(dim)

            stage_name = f'layer{i + 1}'
            self.add_module(stage_name, stage)
            self.stages.append(stage_name)
        if self.out_indices is None:
            self.out_indices = [i for i in range(len(self.stages))]
        assert isinstance(self.out_indices, (tuple, list)), \
            'out_indices must be tuple or list.'

        # classifier head
        self.norm = nn.BatchNorm2d(
            self.in_dim) if classifier_norm else nn.Identity()
        if self.global_pool == 'avg':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.global_pool is not None and self.global_pool != '':
            raise ValueError(
                f'Unsupported global pool type: {self.global_pool}.')
        if self.num_classes > 0:
            self.fc = nn.Linear(self.in_dim, self.num_classes)
        self._initialize_weights()
        self.last_out_channels = self.in_dim

    def _initialize_weights(self):
        # kaiming normal for Bottleneck blocks; trunc_normal for ViT blocks
        for n, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                if 'ViTBlock' in n:
                    trunc_normal_(m.weight, std=.02)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_res_stage(
            self,
            dim,
            layer,
            block,
            mode=None,
            kernel_size=3,
            use_se=False,
            groups=1,
            stride=1,
            use_avgdown=False,
            **kwargs):
        layers = OrderedDict()
        kernel_sizes = kernel_size
        groupss = groups
        use_ses = use_se
        dims = dim
        if not self.is_repeat:
            kernel_size = kernel_sizes[0]
            groups = groupss[0]
            use_se = use_ses[0]
            dim = dims[0]
        layers[f'{block.__name__}0'] = block(
            kernel_size,
            self.in_dim,
            dim,
            stride,
            groups=groups,
            use_se=use_se,
            expansion=1,
            **kwargs)

        self.in_dim = dim
        for i in range(1, layer):
            if not self.is_repeat:
                kernel_size = kernel_sizes[i]
                groups = groupss[i]
                use_se = use_ses[i]
                dim = dims[i]
            layers[f'{block.__name__}{i}'] = block(
                kernel_size,
                self.in_dim,
                dim,
                stride=1,
                groups=groups,
                use_se=use_se,
                expansion=1,
                **kwargs)
            self.in_dim = dim
        return nn.Sequential(layers)

    def _make_stage(
            self,
            dim,
            layer,
            block,
            input_resolution,
            num_heads=1,
            stride=1,
            use_avgdown=False,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=None,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm):
        num_headss = num_heads
        mlp_ratios = mlp_ratio
        dims = dim
        if not self.is_repeat:
            num_heads = num_headss[0]
            mlp_ratio = mlp_ratios[0]
            dim = dims[0]

        layers = OrderedDict()
        layers[f'{block.__name__}0'] = block(
            self.in_dim,
            dim,
            input_resolution,
            num_heads,
            stride=stride,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[0],
            act_layer=act_layer,
            norm_layer=norm_layer)

        self.in_dim = dim
        for i in range(1, layer):
            if not self.is_repeat:
                num_heads = num_headss[i]
                mlp_ratio = mlp_ratios[i]
                dim = dims[i]
            layers[f'{block.__name__}{i}'] = block(
                self.in_dim,
                dim,
                input_resolution,
                num_heads,
                stride=1,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i],
                act_layer=act_layer,
                norm_layer=norm_layer)
            self.in_dim = dim
        return nn.Sequential(layers)

    def forward_features(self, x):
        outs = OrderedDict()
        x = self.layer0(x)
        for i, stage_name in enumerate(self.stages):
            stage = getattr(self, stage_name)
            x = stage(x)
            if i in self.out_indices:
                outs[i] = x
        if self.features_only:
            return outs

        x = self.norm(x)
        if self.global_pool == 'avg':
            x = self.avgpool(x)
            return torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if isinstance(x, OrderedDict):
            return list(x.values())
        if self.num_classes > 0:
            x = self.fc(x)
        return x


def get_params(search_cfg, bids):
    # get searched arch parameters from bids
    micro = search_cfg.micro
    is_repeat = search_cfg.is_repeat
    kernel_size_list, use_se_list, groups_list, \
        num_heads_list, mlp_ratio_list, channel_list = [], [], [], [], [], []
    TBS_names = ['TBS1', 'TBS3', 'TBS6', 'TBS10']

    idx = 0
    sspace = deepcopy(SSPACE[micro])
    for name, layer, params in sspace:
        is_skip = is_repeat and name not in TBS_names
        if layer != 'MixedOp' or is_skip:
            continue
        o_index = params['bids'][bids[idx]]
        if isinstance(params['out_channels'], list):
            c_index = bids[idx + 1]
            c_min, _, c_step = params['out_channels']
            channel = c_min + c_index * c_step
            step = 2
        else:
            channel = params['out_channels']
            step = 1
        channel_list.append(channel)
        _, block, micro_args = SSPACE['MICRO'][o_index]
        if block == 'Bottleneck':
            kernel_size_list.append(micro_args['kernel_size'])
            use_se_list.append(micro_args['use_se'])
            groups_list.append(micro_args['groups'])
            num_heads_list.append(None)
            mlp_ratio_list.append(None)
        elif block == 'ViTBlock':
            num_heads_list.append(micro_args['num_heads'])
            mlp_ratio_list.append(micro_args['mlp_ratio'])
            kernel_size_list.append(None)
            use_se_list.append(None)
            groups_list.append(None)
        idx = idx + step
    return kernel_size_list, use_se_list, groups_list, \
        num_heads_list, mlp_ratio_list, channel_list


@register_model
def automl_fbnet_v1(pretrained=False, **kwargs):
    blocks = ['bottleneck', 'bottleneck', 'bottleneck', 'vit']
    transitions = [False, True, True, True]
    layers = [2, 3, 4, 4]
    kernel_size_list = [3, 5, 3, 3, 3, 3, 3, 5, 3, None, None, None, None]
    use_se_list = [0, 0, 0, 1, 1, 1, 1, 1, 1, None, None, None, None]
    groups_list = [4, 4, 1, 8, 4, 4, 1, 1, 1, None, None, None, None]
    num_heads_list = [None, None, None, None, None,
                      None, None, None, None, 12, 24, 24, 12]
    mlp_ratio_list = [None, None, None, None,
                      None, None, None, None, None, 4, 3, 3, 3]
    channel_list = [64, 64, 256, 256, 256,
                    512, 256, 512, 512, 624, 720, 768, 864]
    is_repeat = False
    model_name = 'automl_fbnet_v1'
    pretrained_config, model_config = get_configs(**kwargs)

    model = HybridNet(layers=layers, blocks=blocks, transitions=transitions,
                      embed_dims=channel_list, kernel_sizes=kernel_size_list,
                      groups=groups_list, use_ses=use_se_list,
                      num_heads=num_heads_list, mlp_ratios=mlp_ratio_list,
                      mode='pytorch', use_avgdown=False,
                      classifier_norm=True, stem_type='3xconv3',
                      stem_chs=[32, 32, 64],
                      # act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                      is_repeat=is_repeat,
                      act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                      qkv_bias=True, drop_path_rate=0.1, name=model_name,
                      **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def automl_fbnet_v2(pretrained=False, **kwargs):
    blocks = ['bottleneck', 'bottleneck', 'bottleneck', 'vit']
    transitions = [False, True, True, True]
    layers = [2, 3, 4, 4]
    kernel_size_list = [3, 3, 5, None]
    use_se_list = [0, 1, 1, None]
    groups_list = [4, 8, 4, None]
    num_heads_list = [None, None, None, 16]
    mlp_ratio_list = [None, None, None, 3]
    channel_list = [64, 256, 512, 672]
    is_repeat = True
    model_name = 'automl_fbnet_v2'
    pretrained_config, model_config = get_configs(**kwargs)

    model = HybridNet(layers=layers, blocks=blocks, transitions=transitions,
                      embed_dims=channel_list, kernel_sizes=kernel_size_list,
                      groups=groups_list, use_ses=use_se_list,
                      num_heads=num_heads_list, mlp_ratios=mlp_ratio_list,
                      mode='pytorch', use_avgdown=False,
                      classifier_norm=True, stem_type='3xconv3',
                      stem_chs=[32, 32, 64],
                      # act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                      is_repeat=is_repeat,
                      act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                      qkv_bias=True, drop_path_rate=0.1, name=model_name,
                      **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model
