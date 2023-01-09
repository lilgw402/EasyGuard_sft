""" Hybrid network architecture
"""

import torch
import torch.nn as nn

from collections import OrderedDict
from timm.models.layers import DropPath, trunc_normal_

from titan.utils.registry import register_model
from titan.utils.helper import (
    download_weights,
    load_pretrained_model_weights,
    get_configs
)

__all__ = ['hybridnet89', 'hybridnet89b']


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.,
                 dwconv=False, dwconv_shortcut=False):
        r""" MLP layers with depth-wise convolution.
        Reference:
            Wang et al. "PVTv2: Improved Baselines with Pyramid Vision Transformer" - https://arxiv.org/abs/2106.13797
            Xie et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers" - https://arxiv.org/abs/2105.15203
            Li et al. "LocalViT: Bringing Locality to Vision Transformers" - https://arxiv.org/abs/2104.05707
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features) if dwconv else None
        self.dwconv_shortcut = dwconv_shortcut
        self.norm = nn.BatchNorm2d(hidden_features) if dwconv else None

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.dwconv is not None:
            if self.dwconv_shortcut:
                # x = self.act(identity + self.dwconv(x))
                x = self.act(x + self.dwconv(self.norm(x))) # FIXED
            else:
                # x = self.act(self.dwconv(x))
                x = self.act(self.dwconv(self.norm(x))) # FIXED
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    r""" Multi-head Self-Attention (MSA) module with relative position bias, attention refiner and spatial downsampling on q, v
    Reference:
        Shazeer et al. "Talking-Heads Attention" - https://arxiv.org/abs/2003.02436
        Zhou et al. "Refiner: Refining Self-attention for Vision Transformers" - https://arxiv.org/abs/2106.03714
        Zhou et al. "DeepViT: Towards Deeper Vision Transformer" - https://arxiv.org/abs/2103.11886
        Wang et al. "PVTv2: Improved Baselines with Pyramid Vision Transformer" - https://arxiv.org/abs/2106.13797
        Xie et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers" - https://arxiv.org/abs/2105.15203
    """
    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 use_relative_position_bias=True, refiner_ks=1, refiner_dim=None, refiner_skip_proj=False, refiner_shortcut=False, 
                 linear=False, linear_size=7, sr_ratio=1):
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
                assert num_heads == refiner_dim, f"num_heads ({num_heads}) does not match refiner_dim ({refiner_dim})"
            self.refiner_exp = nn.Conv2d(num_heads, refiner_dim, kernel_size=refiner_ks, stride=1, padding=refiner_ks//2, bias=True)
            self.refiner_proj = nn.Identity() if refiner_skip_proj else nn.Conv2d(refiner_dim, num_heads, kernel_size=1, stride=1, bias=True)
        pos_heads = refiner_dim if self.refiner else num_heads

        # spatial reduction of k, v
        self.linear = linear
        self.sr_ratio = sr_ratio
        # FIXME
        if linear or sr_ratio > 1:
            assert use_relative_position_bias == False, "currently does not support relative position bias when downsampling on k, v"
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.BatchNorm2d(dim)
        else:
            # self.pool = nn.AdaptiveAvgPool2d(linear_size) # not supported by ONNX
            self.pool = nn.AvgPool2d(linear_size, stride=linear_size)
            # self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
            self.norm = nn.BatchNorm2d(dim)
            self.act = nn.ReLU(inplace=True)

        if self.use_relative_position_bias:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * input_resolution[0] - 1) * (2 * input_resolution[1] - 1), pos_heads))  # 2*H-1 * 2*W-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.input_resolution[0])
            coords_w = torch.arange(self.input_resolution[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, H, W
            coords_flatten = torch.flatten(coords, 1)  # 2, H*W
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, H*W, H*W
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # H*W, H*W, 2
            relative_coords[:, :, 0] += self.input_resolution[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.input_resolution[1] - 1
            relative_coords[:, :, 0] *= 2 * self.input_resolution[1] - 1
            relative_position_index = relative_coords.sum(-1)  # H*W, H*W
            self.register_buffer("relative_position_index", relative_position_index)
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
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, N).permute(0, 1, 3, 2) # B, nH, N, C//nH

        if not self.linear:
            if self.sr_ratio > 1:
                x = self.sr(x)
                x = self.norm(x)
                kv = self.kv(x).reshape(B, 2, self.num_heads, C // self.num_heads, -1).permute(1, 0, 2, 4, 3) # 2, B, nH, N, C//nH
            else:
                kv = self.kv(x).reshape(B, 2, self.num_heads, C // self.num_heads, -1).permute(1, 0, 2, 4, 3) # 2, B, nH, N, C//nH
        else:
            x = self.sr(self.pool(x))
            x = self.norm(x)
            x = self.act(x)
            kv = self.kv(x).reshape(B, 2, self.num_heads, C // self.num_heads, -1).permute(1, 0, 2, 4, 3) # 2, B, nH, N, C//nH
        k, v = kv[0], kv[1] # make torchscript happy (cannot use tensor as tuple)
        N_ = k.shape[-2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # attention refinement (expansion)
        if self.refiner:
            if self.refiner_shortcut:
                attn = attn + self.refiner_exp(attn)
            else:
                attn = self.refiner_exp(attn)

        if self.use_relative_position_bias:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N_, -1)  # H*W, H*W,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, H*W, H*W
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
        return out, attn


class ViTBlock(nn.Module):
    r""" Vision Transformer block
    """
    def __init__(self, dim, input_resolution, num_heads, proj=None,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, use_relative_position_bias=True,
                 skip_lam=1., dwconv=None, dwconv_shortcut=False,
                 refiner_ks=1, refiner_dim=None, refiner_skip_proj=False, refiner_shortcut=False,
                 linear=False, linear_size=7, sr_ratio=1, block_dwconv_ks=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.skip_lam = skip_lam
        self.proj = proj or nn.Identity()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, input_resolution=input_resolution, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            use_relative_position_bias=use_relative_position_bias,
            refiner_ks=refiner_ks, refiner_dim=refiner_dim,
            refiner_skip_proj=refiner_skip_proj, refiner_shortcut=refiner_shortcut,
            linear=linear, linear_size=linear_size, sr_ratio=sr_ratio)

        # depth-wise conv between attention and mlp similar to Shuffle-Transformer
        self.block_dwconv_ks = block_dwconv_ks
        if self.block_dwconv_ks > 0:
            self.block_dwconv = nn.Conv2d(
                dim, dim, kernel_size=block_dwconv_ks, padding=block_dwconv_ks // 2,
                stride=1, groups=dim, bias=qkv_bias)
            self.block_norm = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
            dwconv=dwconv, dwconv_shortcut=dwconv_shortcut)

    def forward(self, x):
        x = self.proj(x)
        x, attn_map = self.attn(self.norm1(x))
        x = x + self.drop_path(x) / self.skip_lam
        # FIXME: whether to add skip_lam here?
        if self.block_dwconv_ks > 0:
            x = x + self.block_dwconv(self.block_norm(x))
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x, attn_map


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))


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
                 se_reduction: int = 16):
        super(BasicBlock, self).__init__()
        self.conv1_bn = ConvBN(inplanes, planes, 3, stride=stride)
        self.conv2_bn = ConvBN(planes, planes, 3)
        self.relu = nn.ReLU(inplace=True)

        self.identity = nn.Identity()
        self.downsample = downsample
        self.stride = stride

        if use_se:
            if se_bn:
                self.se = SEBNBlock(planes * self.expansion, reduction=se_reduction)
            else:
                self.se = SEBlock(planes * self.expansion, reduction=se_reduction)
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
                 se_reduction: int = 16):
        super(Bottleneck, self).__init__()
        strides = [stride, 1] if mode == 'default' else [1, stride]
        self.conv1_bn = ConvBN(inplanes, planes, 1, stride=strides[0])
        self.conv2_bn = ConvBN(planes, planes, 3, stride=strides[1])
        self.conv3_bn = ConvBN(planes, planes * self.expansion, 1)
        self.relu = nn.ReLU(inplace=True)

        self.identity = nn.Identity()
        self.downsample = downsample
        self.stride = stride

        if use_se:
            if se_bn:
                self.se = SEBNBlock(planes * self.expansion, reduction=se_reduction)
            else:
                self.se = SEBlock(planes * self.expansion, reduction=se_reduction)
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


class HybridNet(nn.Module):
    r""" Hybrid network structure
    """
    __model_type__ = 'vit'

    def __init__(
            self,
            in_chans=3,
            num_classes=1000,
            img_size=224,
            layers=[],
            blocks=None,
            transitions=None,
            embed_dims=None,
            num_heads=None,
            mode='pytorch',
            use_avgdown=False,
            classifier_norm=False,
            stem_type='conv7',
            stem_chs=[64],
            patch_size=16,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.ReLU,
            mlp_ratios=None,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            use_relative_position_bias=True,
            skip_lam=1.,
            dwconv=None,
            dwconv_shortcut=False,
            refiner_ks=1,
            refiner_dims=None,
            refiner_skip_proj=False,
            refiner_shortcut=False,
            linear=False,
            linear_sizes=None,
            sr_ratios=None,
            block_dwconv_ks=0,
            global_pool='avg',
            features_only=False,
            name=None,
            use_attn_map=False,
            **kwargs):

        super().__init__()
        self.name = name
        self.in_dim = stem_chs[-1]
        self.img_size = img_size
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.features_only = features_only
        self.out_indices = kwargs.pop('out_indices', None)

        if stem_type == 'conv7':
            # default stem used in ResNet
            self.layer0 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(stem_chs[0]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)])
            input_resolution = (img_size // 4, img_size // 4)

        elif stem_type == '3xconv3':
            # ResNet-C in He et al. "Bag of Tricks for Image Classification with Convolutional Neural Networks"
            # - https://arxiv.org/abs/1812.01187
            # stem_chs = [32, 32, 64] # FIXME
            self.layer0 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_chs[1], stem_chs[2], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[2]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)])
            input_resolution = (img_size // 4, img_size // 4)

        elif stem_type == 'patch_stem':
            # Patchify stem used in ViT, splitting image into tokens of size 16x16
            # Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
            # - https://arxiv.org/abs/2010.11929
            stem_chs = [384] # FIXME
            patch_size = 16 # FIXME
            self.layer0 = nn.Conv2d(in_chans, stem_chs[0], kernel_size=patch_size, stride=patch_size)
            input_resolution = (img_size // patch_size, img_size // patch_size)

        elif stem_type == 'early_conv_stem':
            # Xiao et al. "Early Convolutions Help Transformers See Better" - https://arxiv.org/abs/2106.14881
            # stem_chs = [48, 96, 192, 384, 384] # FIXME
            self.layer0 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_chs[1], stem_chs[2], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[2]),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_chs[2], stem_chs[3], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[3]),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_chs[3], stem_chs[4], kernel_size=1, stride=1)])
            input_resolution = (img_size // 16, img_size // 16)

        elif stem_type == '4xconv_stride8':
            # 3x conv3x3 followed by 1 conv1x1, output stride is 8
            self.layer0 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_chs[1], stem_chs[2], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_chs[2]),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_chs[2], stem_chs[3], kernel_size=1, stride=1)])
            input_resolution = (img_size // 8, img_size // 8)

        # build stages
        self.blocks = blocks
        self.stages = []
        self.out_channels = []

        # FIXME currently do not apply to conv layers
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]  # stochastic depth decay rule

        for i in range(len(layers)):
            # FIXME merge this into function
            # read block-specific hyperparams
            layer = layers[i]
            dim = embed_dims[i]
            num_head = num_heads[i]
            block = blocks[i]
            transition = transitions[i]
            mlp_ratio = mlp_ratios[i]
            refiner_dim = refiner_dims[i]
            sr_ratio = sr_ratios[i]
            position_bias = use_relative_position_bias[i]

            # check if dimension is divisible by num_heads
            if num_head is not None:
                assert dim % num_head == 0, f"Dimension ({dim}) is not divisible by number of heads ({num_head})"

            if transition:
                stride = 2
                input_resolution = [input_resolution[0] // 2, input_resolution[1] // 2]
            else:
                stride = 1

            if block == 'bottleneck':
                # ResNet Bottleneck block
                stage = self._make_res_stage(
                    dim=dim,
                    layer=layer,
                    block=Bottleneck,
                    mode=mode,
                    stride=stride,
                    use_avgdown=use_avgdown,
                    **kwargs)
                # Bottleneck block has expansion of 4
                self.out_channels.append(dim * 4)

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
                    norm_layer=norm_layer,
                    use_relative_position_bias=position_bias,
                    skip_lam=skip_lam,
                    dwconv=dwconv,
                    dwconv_shortcut=dwconv_shortcut,
                    refiner_ks=refiner_ks,
                    refiner_dim=refiner_dim,
                    refiner_skip_proj=refiner_skip_proj,
                    refiner_shortcut=refiner_shortcut,
                    linear=linear[i],
                    linear_size=linear_sizes[i],
                    sr_ratio=sr_ratio,
                    block_dwconv_ks=block_dwconv_ks)
                self.out_channels.append(dim)

            stage_name = f'layer{i + 1}'
            self.add_module(stage_name, stage)
            self.stages.append(stage_name)

        if self.out_indices is None:
            self.out_indices = [i for i in range(len(self.stages))]
        assert isinstance(self.out_indices, (tuple, list)), \
            'out_indices must be tuple or list.'

        # classifier head
        self.norm = norm_layer(self.in_dim) if classifier_norm else nn.Identity()
        if self.global_pool == 'avg':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.global_pool is not None and self.global_pool != '':
            raise ValueError(f'Unsupported global pool type: {self.global_pool}.')
        if self.num_classes > 0:
            self.fc = nn.Linear(self.in_dim, num_classes)
        self.last_out_channels = self.in_dim

        # Whether to output attn_map
        self.use_attn_map = use_attn_map

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
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def _make_res_stage(self, dim, layer, block, mode=None, stride=1, use_avgdown=False, **kwargs):
        downsample = None
        if stride != 1 or self.in_dim != dim * block.expansion:
            if stride != 1 and use_avgdown:
                downsample = nn.Sequential(OrderedDict([
                    ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2)),
                    ('conv_bn', ConvBN(self.in_dim, dim * block.expansion, kernel_size=1, stride=1))]))
            else:
                downsample = ConvBN(self.in_dim, dim * block.expansion, kernel_size=1, stride=stride)

        layers = OrderedDict()
        layers[f'{block.__name__}0'] = block(self.in_dim, dim, stride, downsample, mode, **kwargs)

        self.in_dim = dim * block.expansion
        for i in range(1, layer):
            layers[f'{block.__name__}{i}'] = block(self.in_dim, dim, mode=mode, **kwargs)

        return nn.Sequential(layers)

    def _make_stage(self, dim, layer, block, input_resolution, num_heads=1, stride=1, use_avgdown=False,
                    mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=None,
                    act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, use_relative_position_bias=True,
                    skip_lam=1., dwconv=None, dwconv_shortcut=False,
                    refiner_ks=1, refiner_dim=None, refiner_skip_proj=False, refiner_shortcut=False,
                    linear=False, linear_size=7, sr_ratio=None, block_dwconv_ks=0):

        proj = None
        if stride != 1 or self.in_dim != dim:
            if stride != 1 and use_avgdown:
                proj = nn.Sequential(OrderedDict([
                    ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2)),
                    ('conv', nn.Conv2d(self.in_dim, dim, kernel_size=1, stride=1))]))
            else:
                proj = nn.Conv2d(self.in_dim, dim, kernel_size=1, stride=stride)

        layers = OrderedDict()
        layers[f'{block.__name__}0'] = block(dim, input_resolution, num_heads, proj,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[0],
            act_layer=act_layer, norm_layer=norm_layer, use_relative_position_bias=use_relative_position_bias, skip_lam=skip_lam,
            dwconv=dwconv, dwconv_shortcut=dwconv_shortcut,
            refiner_ks=refiner_ks, refiner_dim=refiner_dim, refiner_skip_proj=refiner_skip_proj, refiner_shortcut=refiner_shortcut,
            linear=linear, linear_size=linear_size, sr_ratio=sr_ratio, block_dwconv_ks=block_dwconv_ks)

        self.in_dim = dim
        for i in range(1, layer):
            layers[f'{block.__name__}{i}'] = block(dim, input_resolution, num_heads, proj=None,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i],
                act_layer=act_layer, norm_layer=norm_layer, use_relative_position_bias=use_relative_position_bias, skip_lam=skip_lam,
                dwconv=dwconv, dwconv_shortcut=dwconv_shortcut,
                refiner_ks=refiner_ks, refiner_dim=refiner_dim, refiner_skip_proj=refiner_skip_proj, refiner_shortcut=refiner_shortcut,
                linear=linear, linear_size=linear_size, sr_ratio=sr_ratio, block_dwconv_ks=block_dwconv_ks)

        return nn.Sequential(layers)

    def forward_features(self, x):
        outs = OrderedDict()
        attn_map = None
        x = self.layer0(x)
        for i, stage_name in enumerate(self.stages):
            stage = getattr(self, stage_name)
            if self.blocks[i] == 'vit':
                for _, layer in enumerate(stage):
                    x, attn_map = layer(x)
            else:
                x = stage(x)
            if i in self.out_indices:
                outs[i] = x
        if self.features_only:
            return outs, attn_map

        x = self.norm(x)
        if self.global_pool == 'avg':
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        return x, attn_map

    def forward(self, x):
        x, attn_map = self.forward_features(x)
        if isinstance(x, OrderedDict):
            if self.use_attn_map and self.training:
                if attn_map is None:
                    raise ValueError(
                        'Model does not include a ViT block. Failed to output attn_map.')
                return list(x.values()), attn_map
            return list(x.values())
        if self.num_classes > 0:
            x = self.fc(x)
        
        if self.use_attn_map and self.training:
            if attn_map is None:
                raise ValueError(
                    'Model does not include a ViT block. Failed to output attn_map.')
            return x, attn_map
        return x


@register_model
def hybridnet89(pretrained=False, **kwargs):
    # CCCT with mlp_ratio=3; layers=[2,3,4,4]
    # qps: 2166.3
    # acc: 81.2
    layers = [2, 3, 4, 4]
    blocks = ['bottleneck', 'bottleneck', 'bottleneck', 'vit']
    transitions = [False, True, True, True]
    embed_dims = [64, 128, 256, 768]
    num_heads = [None, None, None, 24]
    mlp_ratios = [None, None, None, 3]
    refiner_dims = [None, None, None, None]
    sr_ratios = [None, None, None, 1]
    linear = [None, None, None, False]
    linear_sizes = [None, None, None, 1]
    relative_position_bias = [None, None, None, True]
    model_name = 'hybridnet89'
    pretrained_config, model_config = get_configs(**kwargs)

    model = HybridNet(layers=layers, blocks=blocks, transitions=transitions,
                      embed_dims=embed_dims, num_heads=num_heads,
                      mode='pytorch', use_avgdown=False,
                      classifier_norm=True, stem_type='3xconv3', stem_chs=[32, 32, 64],
                      mlp_ratios=mlp_ratios, qkv_bias=True,
                      use_relative_position_bias=relative_position_bias, skip_lam=1.,
                      dwconv=False, dwconv_shortcut=False,
                      refiner_ks=0, refiner_dims=refiner_dims, refiner_skip_proj=False, refiner_shortcut=False,
                      linear=linear, linear_sizes=linear_sizes, sr_ratios=sr_ratios, block_dwconv_ks=0,
                      drop_path_rate=0.1, name=model_name, **model_config) # FIXME: parse drop_path_rate as cfg param
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
    return model


@register_model
def hybridnet89b(pretrained=False, **kwargs):
    # hybridnet89 with use_avgdown=True
    layers = [2, 3, 4, 4]
    blocks = ['bottleneck', 'bottleneck', 'bottleneck', 'vit']
    transitions = [False, True, True, True]
    embed_dims = [64, 128, 256, 768]
    num_heads = [None, None, None, 24]
    mlp_ratios = [None, None, None, 3]
    refiner_dims = [None, None, None, None]
    sr_ratios = [None, None, None, 1]
    linear = [None, None, None, False]
    linear_sizes = [None, None, None, 1]
    relative_position_bias = [None, None, None, True]
    model_name = 'hybridnet89b'
    pretrained_config, model_config = get_configs(**kwargs)

    model = HybridNet(
        layers=layers,
        blocks=blocks,
        transitions=transitions,
        embed_dims=embed_dims,
        num_heads=num_heads,
        mode='pytorch',
        use_avgdown=True,
        classifier_norm=True,
        stem_type='3xconv3',
        stem_chs=[32, 32, 64],
        mlp_ratios=mlp_ratios,
        qkv_bias=True,
        use_relative_position_bias=relative_position_bias,
        skip_lam=1.,
        dwconv=False,
        dwconv_shortcut=False,
        refiner_ks=0,
        refiner_dims=refiner_dims,
        refiner_skip_proj=False,
        refiner_shortcut=False,
        linear=linear,
        linear_sizes=linear_sizes,
        sr_ratios=sr_ratios,
        block_dwconv_ks=0,
        drop_path_rate=0.1,
        name=model_name,
        **model_config)  # FIXME: parse drop_path_rate as cfg param
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
    return model
