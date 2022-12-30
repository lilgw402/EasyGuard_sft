r""" Customized ResNet implemented by liujiajun.ljj@bytedance.com """

from typing import List
from collections import OrderedDict
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import torch

from titan.models.components import SEBlock, SEBNBlock
from titan.utils.registry import register_model
from titan.utils.helper import (
    download_weights,
    load_pretrained_model_weights,
    get_configs
)
from titan.utils.logger import logger

__all__ = ['resnet_v70']


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.ReLU,
            drop=0.,
            dwconv=False,
            dwconv_shortcut=False):
        r""" MLP layers with depth-wise convolution.
        Reference:
            Wang et al. "PVTv2: Improved Baselines with Pyramid Vision
                Transformer" - https://arxiv.org/abs/2106.13797
            Xie et al. "SegFormer: Simple and Efficient Design for Semantic
                Segmentation with Transformers" -
                https://arxiv.org/abs/2105.15203
            Li et al. "LocalViT: Bringing Locality to Vision Transformers"
                - https://arxiv.org/abs/2104.05707
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features,
                             kernel_size=1, stride=1, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features,
                             kernel_size=1, stride=1, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features) if dwconv else None
        self.dwconv_shortcut = dwconv_shortcut
        self.norm = nn.BatchNorm2d(hidden_features) if dwconv else None

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.dwconv is not None:
            if self.dwconv_shortcut:
                # x = self.act(identity + self.dwconv(x))
                x = self.act(x + self.dwconv(self.norm(x)))  # FIXED
            else:
                # x = self.act(self.dwconv(x))
                x = self.act(self.dwconv(self.norm(x)))  # FIXED
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    r""" Multi-head Self-Attention (MSA) module with relative position bias,
        attention refiner and spatial downsampling on q, v
    Reference:
        Shazeer et al. "Talking-Heads Attention"
            - https://arxiv.org/abs/2003.02436
        Zhou et al. "Refiner: Refining Self-attention for Vision Transformers"
            - https://arxiv.org/abs/2106.03714
        Zhou et al. "DeepViT: Towards Deeper Vision Transformer"
            - https://arxiv.org/abs/2103.11886
        Wang et al. "PVTv2: Improved Baselines with Pyramid Vision Transformer"
            - https://arxiv.org/abs/2106.13797
        Xie et al. "SegFormer: Simple and Efficient Design for Semantic
            Segmentation with Transformers" - https://arxiv.org/abs/2105.15203
    """

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

        self.refiner = True if refiner_dim is not None else False
        if self.refiner:
            self.refiner_shortcut = refiner_shortcut
            self.refiner_skip_proj = refiner_skip_proj
            if refiner_shortcut or refiner_skip_proj:
                assert num_heads == refiner_dim, \
                    f"num_heads ({num_heads}) does not match " \
                    f"refiner_dim ({refiner_dim})"
            self.refiner_exp = nn.Conv2d(
                num_heads,
                refiner_dim,
                kernel_size=refiner_ks,
                stride=1,
                padding=refiner_ks // 2,
                bias=True)
            self.refiner_proj = nn.Identity() \
                if refiner_skip_proj else nn.Conv2d(
                    refiner_dim, num_heads, kernel_size=1, stride=1, bias=True)
        pos_heads = refiner_dim if self.refiner else num_heads

        # spatial reduction of k, v
        self.linear = linear
        self.sr_ratio = sr_ratio
        # FIXME
        if linear or sr_ratio > 1:
            assert use_relative_position_bias is False, \
                "currently does not support relative position bias " \
                "when downsampling on k, v"
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(
                    dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.BatchNorm2d(dim)
        else:
            # self.pool = nn.AdaptiveAvgPool2d(linear_size) # not supported by
            # ONNX
            self.pool = nn.AvgPool2d(linear_size, stride=linear_size)
            # self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
            self.norm = nn.BatchNorm2d(dim)
            self.act = nn.ReLU(inplace=True)

        if self.use_relative_position_bias:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.zeros(
                (2 * input_resolution[0] - 1) *
                (2 * input_resolution[1] - 1), pos_heads))  # 2*H-1 * 2*W-1, nH

            # get pair-wise relative position index for each token inside the
            # window
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
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        q = self.q(x).reshape(
                B, self.num_heads, C // self.num_heads, N
            ).permute(0, 1, 3, 2)  # B, nH, N, C//nH

        if not self.linear:
            if self.sr_ratio > 1:
                x = self.sr(x)
                x = self.norm(x)
                kv = self.kv(x).reshape(
                        B, 2, self.num_heads, C // self.num_heads, -1
                    ).permute(1, 0, 2, 4, 3)  # 2, B, nH, N, C//nH
            else:
                kv = self.kv(x).reshape(
                        B, 2, self.num_heads, C // self.num_heads, -1
                    ).permute(1, 0, 2, 4, 3)  # 2, B, nH, N, C//nH
        else:
            x = self.sr(self.pool(x))
            x = self.norm(x)
            x = self.act(x)
            kv = self.kv(x).reshape(
                    B, 2, self.num_heads, C // self.num_heads, -1
                ).permute(1, 0, 2, 4, 3)  # 2, B, nH, N, C//nH
        # make torchscript happy (cannot use tensor as tuple)
        k, v = kv[0], kv[1]
        N_ = k.shape[-2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # attention refinement (expansion)
        if self.refiner:
            if self.refiner_shortcut:
                attn = attn + self.refiner_exp(attn)
            else:
                attn = self.refiner_exp(attn)

        if self.use_relative_position_bias:
            # H*W, H*W,nH
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)].view(N, N_, -1)
            # nH, H*W, H*W
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        # attention refinement (re-projection)
        if self.refiner:
            if not self.refiner_skip_proj and self.refiner_shortcut:
                attn = attn + self.refiner_proj(attn)
            else:
                attn = self.refiner_proj(attn)
        out = (attn @ v).permute(0, 1, 3, 2).contiguous().reshape(B, C, H, W)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class ViTBlock(nn.Module):
    r""" Vision Transformer block
    """

    def __init__(
            self,
            dim,
            input_resolution,
            num_heads,
            proj=None,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            use_relative_position_bias=True,
            skip_lam=1.,
            dwconv=None,
            dwconv_shortcut=False,
            refiner_ks=1,
            refiner_dim=None,
            refiner_skip_proj=False,
            refiner_shortcut=False,
            linear=False,
            linear_size=7,
            sr_ratio=1,
            block_dwconv_ks=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.skip_lam = skip_lam
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

        # depth-wise conv between attention and mlp similar to
        # Shuffle-Transformer
        self.block_dwconv_ks = block_dwconv_ks
        if self.block_dwconv_ks > 0:
            self.block_dwconv = nn.Conv2d(
                dim,
                dim,
                kernel_size=block_dwconv_ks,
                padding=block_dwconv_ks // 2,
                stride=1,
                groups=dim,
                bias=qkv_bias)
            self.block_norm = norm_layer(dim)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            dwconv=dwconv,
            dwconv_shortcut=dwconv_shortcut)

    def forward(self, x):
        x = self.proj(x)
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        # FIXME: whether to add skip_lam here?
        if self.block_dwconv_ks > 0:
            x = x + self.block_dwconv(self.block_norm(x))
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))


class DWConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(DWConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(
                in_planes, out_planes, kernel_size, stride,
                padding=padding, groups=out_planes, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))


class SE(nn.Module):
    def __init__(self, in_c, ratio=16):
        super(SE, self).__init__()
        self.ada_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_c, in_c, 1, 1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = self.ada_pool(x)
        se = self.fc(se)
        se = self.sigmoid(se)
        return x * se


class RepVGGBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: nn.Module = None,
                 mode: str = None,
                 use_se: bool = False,
                 se_bn: bool = False,
                 se_reduction: int = 16,
                 expansion: int = 1,
                 groups: int = 1,):
        super(RepVGGBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1_bn_3x3 = ConvBN(inplanes, planes, 3, stride=stride)
        self.conv1_bn_1x1 = ConvBN(inplanes, planes, 1, stride=stride)
        self.conv2_bn_3x3 = ConvBN(planes, planes, 3)
        self.conv2_bn_1x1 = ConvBN(planes, planes, 1)
        self.relu = nn.ReLU(inplace=True)

        self.identity = nn.Identity()
        self.downsample = downsample
        self.stride = stride

        if use_se:
            self.se = SE(planes)
        else:
            self.se = None

    def forward(self, x):
        residual = self.conv1_bn_3x3(x) + self.conv1_bn_1x1(x)  # + x
        if self.inplanes == self.planes:
            residual += x
        residual = self.relu(residual)
        residual = self.conv2_bn_3x3(
            residual) + self.conv2_bn_1x1(residual) + residual
        if self.se is not None:
            residual = self.se(residual)

        out = self.relu(residual)
        return out


class RepVGGBlockDeploy(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: nn.Module = None,
                 mode: str = None,
                 use_se: bool = False,
                 se_bn: bool = False,
                 se_reduction: int = 16,
                 expansion: int = 1,
                 groups: int = 1,):
        super(RepVGGBlockDeploy, self).__init__()
        self.conv1_bn = ConvBN(inplanes, planes, 3, stride=stride)
        self.conv2_bn = ConvBN(planes, planes, 3)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride

    def forward(self, x):
        residual = self.conv1_bn(x)
        residual = self.relu(residual)
        residual = self.conv2_bn(residual)
        out = self.relu(residual)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: nn.Module = None,
                 mode: str = None,
                 use_se: bool = False,
                 se_bn: bool = False,
                 se_reduction: int = 16,
                 expansion: int = 1,
                 groups: int = 1,):
        super(BasicBlock, self).__init__()
        self.conv1_bn = ConvBN(inplanes, planes, 3, stride=stride)
        self.conv2_bn = ConvBN(planes, planes, 3)
        self.relu = nn.ReLU(inplace=True)

        self.identity = nn.Identity()
        self.downsample = downsample
        self.stride = stride

        if use_se:
            self.se = SE(planes)
        else:
            self.se = None

    def forward(self, x):
        identity = self.identity(x)
        if self.downsample is not None:
            identity = self.downsample(identity)

        residual = self.conv1_bn(x)
        residual = self.relu(residual)
        residual = self.conv2_bn(residual)
        if self.se is not None:
            residual = self.se(residual)

        out = self.relu(identity + residual)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: nn.Module = None,
                 mode: str = None,
                 use_se: bool = False,
                 se_bn: bool = False,
                 se_reduction: int = 16,
                 expansion: int = 4,
                 groups: int = 2,):
        super(Bottleneck, self).__init__()
        strides = [stride, 1] if mode == 'default' else [1, stride]
        if groups == 4:
            inv_planes = int(planes * 2)
        else:
            inv_planes = planes
        self.inv_planes = inv_planes
        self.conv1_bn = ConvBN(inplanes, inv_planes, 1, stride=strides[0])
        self.conv2_bn = ConvBN(inv_planes, inv_planes, 3,
                               stride=strides[1], groups=groups)
        self.conv3_bn = ConvBN(inv_planes, planes * expansion, 1)
        self.relu = nn.ReLU(inplace=True)

        self.identity = nn.Identity()
        self.downsample = downsample
        self.stride = stride

        if use_se:
            self.se = SE(self.inv_planes)
        else:
            self.se = None

    def forward(self, x):
        identity = self.identity(x)
        if self.downsample is not None:
            identity = self.downsample(identity)

        residual = self.conv1_bn(x)
        residual = self.relu(residual)
        residual = self.conv2_bn(residual)
        residual = self.relu(residual)
        if self.se is not None:
            residual = self.se(residual)
        residual = self.conv3_bn(residual)

        out = self.relu(identity + residual)
        return out


class BottleneckDW(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: nn.Module = None,
                 mode: str = None,
                 use_se: bool = False,
                 se_bn: bool = False,
                 se_reduction: int = 16,
                 expansion: int = 4,):
        super(BottleneckDW, self).__init__()
        strides = [stride, 1] if mode == 'default' else [1, stride]
        self.conv1_bn = ConvBN(
            inplanes, planes * expansion, 1, stride=strides[0])
        self.conv2_bn = DWConvBN(
            planes * expansion, planes * expansion, 3, stride=strides[1])
        self.conv3_bn = ConvBN(planes * expansion, planes, 1)
        self.relu = nn.ReLU(inplace=True)

        self.identity = nn.Identity()
        self.downsample = downsample
        self.stride = stride

        if use_se:
            if se_bn:
                self.se = SEBNBlock(planes * expansion, reduction=se_reduction)
            else:
                self.se = SEBlock(planes * expansion, reduction=se_reduction)
        else:
            self.se = None

    def forward(self, x):
        identity = self.identity(x)
        if self.downsample is not None:
            identity = self.downsample(identity)

        residual = self.conv1_bn(x)
        residual = self.relu(residual)
        residual = self.conv2_bn(residual)
        residual = self.relu(residual)
        residual = self.conv3_bn(residual)
        if self.se is not None:
            residual = self.se(residual)

        out = self.relu(identity + residual)
        return out


class ResNet(nn.Module):
    r"""
    Build a resnet backbone
    """

    def __init__(self,
                 block: nn.Module,
                 layers: List[int],
                 inplanes: int = 64,
                 num_classes: int = 1000,
                 in_channels: int = 3,
                 mode: str = 'pytorch',
                 zero_gamma: bool = False,
                 global_pool: str = 'avg',
                 features_only: bool = False,
                 out_indices: list = None,
                 verbose: bool = True,
                 input_resolution: List[int] = [224, 224],
                 name: str = None,
                 **kwargs):
        r"""
        Args:
            block: resnet block
            layers: number of blocks at each layer
            inplanes: channel of the first convolution layer
            num_classes: number of classes for classification task
            in_channels: channel of input image
            mode: resnet downsample mode. 'pytorch' for 3x3 downsample
                  while 'default' for 1x1 downsample
            zero_gamma: if True, the gamma of last BN of each block is
                        initialized to zero
            global_pool: type of global pooling, default is 'avg'.
                         If set to None or '', then no pooling will be used.
            features_only (bool): whether to output only feature maps. Default
                        is False.
            out_indices: mark the indices of layer to get their output
            verbose: if True, logging is activated
            name: model name
            **kwargs: extra params which is passed to self._make_layer()
        """
        assert mode in ['pytorch', 'default'], \
            f'Illegal resnet downsample mode {mode}. ' \
            f'Choose from ["pytorch", "default"]'
        if verbose:
            logger.info(f'=> Model arch: using {mode} downsample ResNet')

        super(ResNet, self).__init__()
        self.name = name
        self.inplanes = inplanes
        self.block = block
        self.zero_gamma = zero_gamma
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.features_only = features_only

        self.layer0 = nn.Sequential(OrderedDict([
            ('conv_bn1', ConvBN(in_channels, 32, kernel_size=3, stride=2)),
            ('relu1', nn.ReLU(inplace=True)),
            # ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            ('conv_bn2', ConvBN(32, 64, kernel_size=3, stride=2)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))

        # build residual layers
        self.res_layers = []
        self.out_channels = []
        layers = [1, 1, 3, 3, 4]  # 1, 2]#, 1]
        strides = [1, 2, 2, 1, 2]  # 2, 1]#, 1]
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(layers))]
        for i in range(len(layers)):
            # stride = 1 if i == 0 else 2
            stride = strides[i]
            if i == 0:
                res_layer = self._make_layer_ratio(
                    128,
                    layers[i],
                    mode,
                    stride=stride,
                    inplanes=self.inplanes,
                    expansion=1,
                    block=RepVGGBlock,
                    **kwargs)
                self.out_channels.append(128)
            elif i == 1:
                res_layer = self._make_layer_ratio(
                    256,
                    layers[i],
                    mode,
                    stride=stride,
                    inplanes=self.inplanes,
                    expansion=1,
                    block=RepVGGBlock,
                    **kwargs)
                self.out_channels.append(256)
            elif i == 2:
                res_layer = self._make_layer_ratio(
                    512,
                    layers[i],
                    mode,
                    stride=stride,
                    inplanes=self.inplanes,
                    expansion=1,
                    groups=32,
                    **kwargs)
                self.out_channels.append(512)
            elif i == 3:
                res_layer = self._make_layer_ratio(
                    512,
                    layers[i],
                    mode,
                    stride=stride,
                    inplanes=self.inplanes,
                    expansion=1,
                    groups=32,
                    **kwargs)
                self.out_channels.append(512)
            elif i == 4:
                res_layer = self._make_stage(
                    dim=512,
                    layer=layers[i],
                    block=ViTBlock,
                    input_resolution=(input_resolution[0] // 32,
                                      input_resolution[1] // 32),
                    num_heads=16,
                    stride=stride,
                    mlp_ratio=4,
                    drop_path=dpr[sum(layers[:i]):sum(layers[:i + 1])])
                self.out_channels.append(512)

            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self.last_out_channels = 512
        self.out_indices = [i for i in range(
            len(layers))] if out_indices is None else out_indices
        assert isinstance(self.out_indices, (tuple, list)), \
            'out_indices must be tuple or list.'

        # build classifier is required
        if self.num_classes > 0:
            self.fc = nn.Linear(self.last_out_channels, self.num_classes)

        self._initialize_weights()

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

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def _make_layer_ratio(self,
                          planes: int,
                          blocks: int,
                          mode: str = None,
                          stride: int = 1,
                          use_avgdown: bool = False,
                          force_downsample: bool = False,
                          expansion: int = 4,
                          inplanes: int = 0,
                          block=Bottleneck,
                          groups: int = 2,
                          **kwargs) -> nn.Module:
        r""" Auxiliary function for resnet to make layer of each stage

        Args:
            planes: basic number of channel at current stage
            blocks: resnet block
            mode: resnet downsample mode. 'pytorch' for 3x3 downsample
                  while 'default' for 1x1 downsample
            stride: stride at the first block
            use_avgdown: if True, resnetD architecture is used
            force_downsample: if True, insert dowmsample in all blocks to
                              align with hadron2
            **kwargs: extra params which is passed to block.__init__()

        Returns:
            A sequenced pytorch Module of a certain resnet stage
        """
        downsample = None
        if block != RepVGGBlock:
            if stride != 1 or inplanes != planes * expansion \
                    or force_downsample:
                if stride != 1 and use_avgdown:
                    downsample = nn.Sequential(OrderedDict([
                        ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2)),
                        ('conv_bn', ConvBN(
                            self.inplanes, planes * self.block.expansion,
                            kernel_size=1, stride=1))]))
                else:
                    downsample = ConvBN(inplanes, planes * expansion,
                                        kernel_size=1, stride=stride)

        layers = OrderedDict()
        layers[f'{self.block.__name__}0'] = block(
            inplanes,
            planes,
            stride,
            downsample,
            mode,
            expansion=expansion,
            groups=groups,
            **kwargs)

        self.inplanes = planes * expansion
        for i in range(1, blocks):
            if i == blocks - 1:
                kwargs['use_se'] = True
            layers[f'{self.block.__name__}{i}'] = block(
                self.inplanes, planes, mode=mode,
                expansion=expansion, groups=groups, **kwargs)
            kwargs['use_se'] = False
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
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            use_relative_position_bias=True,
            skip_lam=1.,
            dwconv=None,
            dwconv_shortcut=False,
            refiner_ks=1,
            refiner_dim=None,
            refiner_skip_proj=False,
            refiner_shortcut=False,
            linear=False,
            linear_size=7,
            sr_ratio=1,
            block_dwconv_ks=0):

        proj = None
        if stride != 1 or self.inplanes != dim:
            proj = nn.Sequential(OrderedDict([
                ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2)),
                ('conv', nn.Conv2d(
                    self.inplanes, dim, kernel_size=1, stride=1))]))

        layers = OrderedDict()
        layers[f'{block.__name__}0'] = block(
            dim,
            input_resolution,
            num_heads,
            proj,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[0],
            act_layer=act_layer,
            norm_layer=norm_layer,
            use_relative_position_bias=use_relative_position_bias,
            skip_lam=skip_lam,
            dwconv=dwconv,
            dwconv_shortcut=dwconv_shortcut,
            refiner_ks=refiner_ks,
            refiner_dim=refiner_dim,
            refiner_skip_proj=refiner_skip_proj,
            refiner_shortcut=refiner_shortcut,
            linear=linear,
            linear_size=linear_size,
            sr_ratio=sr_ratio,
            block_dwconv_ks=block_dwconv_ks)

        self.in_dim = dim
        for i in range(1, layer):
            layers[f'{block.__name__}{i}'] = block(
                dim,
                input_resolution,
                num_heads,
                proj=None,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                use_relative_position_bias=use_relative_position_bias,
                skip_lam=skip_lam,
                dwconv=dwconv,
                dwconv_shortcut=dwconv_shortcut,
                refiner_ks=refiner_ks,
                refiner_dim=refiner_dim,
                refiner_skip_proj=refiner_skip_proj,
                refiner_shortcut=refiner_shortcut,
                linear=linear,
                linear_size=linear_size,
                sr_ratio=sr_ratio,
                block_dwconv_ks=block_dwconv_ks)

        return nn.Sequential(layers)

    def forward_features(self, x):
        x = self.layer0(x)  # x: 64*112*112

        outs = OrderedDict()
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs[layer_name] = x
        if self.features_only:
            return outs

        if self.global_pool == 'avg':
            return x.mean([2, 3])  # x: 2048
        return x

    def forward(self, x):   # x: 64*224*224
        x = self.forward_features(x)
        if isinstance(x, OrderedDict):
            return list(x.values())
        if self.num_classes > 0:
            x = self.fc(x)
        return x


class ResNetCifar(ResNet):
    r"""
    A customized ResNet for Cifar classification
    """

    def __init__(self,
                 block: nn.Module,
                 layers: List[int],
                 num_classes: int = 10,
                 inplanes: int = 16,
                 **kwargs):
        r"""
        Args:
            block: resnet Cifar block
            layers: number of blocks at each layer
            num_classes: number of classes
            inplanes: channel of the first convolution layer
            **kwargs: extra params which is passed to self._make_layer()
        """
        super(ResNetCifar, self).__init__(
            block=block,
            layers=layers,
            num_classes=num_classes,
            inplanes=inplanes,
            **kwargs)

        self.layer0 = nn.Sequential(OrderedDict([
            ('conv_bn', ConvBN(3, inplanes, kernel_size=3)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self._initialize_weights()


@register_model
def resnet_v70(pretrained=False, **kwargs):
    model_name = 'resnet_v70'
    pretrained_config, model_config = get_configs(**kwargs)

    model = ResNet(
                Bottleneck,
                [3, 4, 6, 3],
                name=model_name,
                **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model
