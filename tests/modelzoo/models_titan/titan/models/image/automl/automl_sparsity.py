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
from titan.utils.logger import logger

__all__ = ['automl_sparsity_v3', 'automl_sparsity_v4']

ARCH_CFG_1 = [['ConvBlock',
               {'kernel_size': 3, 'stride': 2, 'padding': 1,
                'in_channels': 3, 'out_channels': 32, 'act': 'relu',
                'bias': False}],
              ['ConvBlock',
               {'kernel_size': 3, 'stride': 1, 'padding': 1,
                'in_channels': 32, 'out_channels': 32, 'act': 'relu',
                'bias': False}],
              ['ConvBlock',
               {'kernel_size': 3, 'stride': 1, 'padding': 1,
                'in_channels': 32, 'out_channels': 64, 'act': 'relu',
                'bias': False}],
              ['MaxPool', {'kernel_size': 3, 'stride': 2, 'padding': 1}],
              ['ResBlock', {'in_channels': 64, 'out_channels': 176,
                            'groups': 2, 'group_width': 32, 'stride': 1}],
              ['ResBlock', {'in_channels': 176, 'out_channels': 176,
                            'groups': 1, 'group_width': 32, 'stride': 1}],
              ['ResBlock', {'in_channels': 176, 'out_channels': 288,
                            'groups': 4, 'group_width': 64, 'stride': 2}],
              ['ResBlock', {'in_channels': 288, 'out_channels': 288,
                            'groups': 1, 'group_width': 64, 'stride': 1}],
              ['ResBlock', {'in_channels': 288, 'out_channels': 288,
                            'groups': 4, 'group_width': 64, 'stride': 1}],
              ['ResBlock', {'in_channels': 288, 'out_channels': 768,
                            'groups': 4, 'group_width': 128, 'stride': 2}],
              ['ResBlock', {'in_channels': 768, 'out_channels': 768,
                            'groups': 2, 'group_width': 128, 'stride': 1}],
              ['ResBlock', {'in_channels': 768, 'out_channels': 768,
                            'groups': 1, 'group_width': 128, 'stride': 1}],
              ['ResBlock', {'in_channels': 768, 'out_channels': 768,
                            'groups': 1, 'group_width': 128, 'stride': 1}],
              ['ResBlock', {'in_channels': 768, 'out_channels': 768,
                            'groups': 2, 'group_width': 128, 'stride': 1}],
              ['ViTBlock',
               {'head_num': 16, 'head_width': 32, 'hid_dims': 2016,
                'input_resolution': [7, 7], 'in_channels': 768,
                'out_channels': 864, 'stride': 2}],
              ['ViTBlock',
               {'head_num': 20, 'head_width': 32, 'hid_dims': 2304,
                'input_resolution': [7, 7], 'in_channels': 864,
                'out_channels': 864, 'stride': 1}],
              ['ViTBlock',
               {'head_num': 24, 'head_width': 32, 'hid_dims': 1824,
                'input_resolution': [7, 7], 'in_channels': 864,
                'out_channels': 864, 'stride': 1}],
              ['ViTBlock',
               {'head_num': 24, 'head_width': 32, 'hid_dims': 1728,
                'input_resolution': [7, 7], 'in_channels': 864,
                'out_channels': 864, 'stride': 1}],
              ['ViTBlock',
               {'head_num': 16, 'head_width': 32, 'hid_dims': 960,
                'input_resolution': [7, 7], 'in_channels': 864,
                'out_channels': 864, 'stride': 1}]]
ARCH_CFG_2 = [['ConvBlock',
               {'kernel_size': 3, 'stride': 2, 'padding': 1,
                'in_channels': 3, 'out_channels': 32, 'act': 'relu',
                'bias': False}],
              ['ConvBlock',
               {'kernel_size': 3, 'stride': 1, 'padding': 1,
                'in_channels': 32, 'out_channels': 32,
                'act': 'relu', 'bias': False}],
              ['ConvBlock',
               {'kernel_size': 3, 'stride': 1, 'padding': 1,
                'in_channels': 32, 'out_channels': 64,
                'act': 'relu', 'bias': False}],
              ['MaxPool', {'kernel_size': 3, 'stride': 2, 'padding': 1}],
              ['ResBlock', {'in_channels': 64, 'out_channels': 224,
                            'groups': 4, 'group_width': 32, 'stride': 1}],
              ['ResBlock', {'in_channels': 224, 'out_channels': 224,
                            'groups': 4, 'group_width': 32, 'stride': 1}],
              ['ResBlock', {'in_channels': 224, 'out_channels': 160,
                            'groups': 8, 'group_width': 32, 'stride': 2}],
              ['ResBlock', {'in_channels': 160, 'out_channels': 160,
                            'groups': 4, 'group_width': 32, 'stride': 1}],
              ['ResBlock', {'in_channels': 160, 'out_channels': 768,
                            'groups': 8, 'group_width': 32, 'stride': 2}],
              ['ResBlock', {'in_channels': 768, 'out_channels': 768,
                            'groups': 4, 'group_width': 32, 'stride': 1}],
              ['ResBlock', {'in_channels': 768, 'out_channels': 768,
                            'groups': 4, 'group_width': 32, 'stride': 1}],
              ['ResBlock', {'in_channels': 768, 'out_channels': 768,
                            'groups': 16, 'group_width': 32, 'stride': 1}],
              ['ResBlock', {'in_channels': 768, 'out_channels': 768,
                            'groups': 16, 'group_width': 32, 'stride': 1}],
              ['ViTBlock',
               {'head_num': 20, 'head_width': 32, 'hid_dims': 1536,
                'input_resolution': [7, 7], 'in_channels': 768,
                'out_channels': 960, 'stride': 2}],
              ['ViTBlock',
               {'head_num': 20, 'head_width': 32, 'hid_dims': 1248,
                'input_resolution': [7, 7], 'in_channels': 960,
                'out_channels': 960, 'stride': 1}],
              ['ViTBlock',
               {'head_num': 24, 'head_width': 32, 'hid_dims': 1632,
                'input_resolution': [7, 7], 'in_channels': 960,
                'out_channels': 960, 'stride': 1}],
              ['ViTBlock',
               {'head_num': 12, 'head_width': 32, 'hid_dims': 2304,
                'input_resolution': [7, 7], 'in_channels': 960,
                'out_channels': 960, 'stride': 1}],
              ['ViTBlock',
               {'head_num': 20, 'head_width': 32, 'hid_dims': 2112,
                'input_resolution': [7, 7], 'in_channels': 960,
                'out_channels': 960, 'stride': 1}]]


def _init_conv_weight(op, weight_init="kaiming_normal"):
    assert weight_init in [None, "kaiming_normal"]
    if weight_init is None:
        return
    if weight_init == "kaiming_normal":
        nn.init.kaiming_normal_(op.weight, mode="fan_out", nonlinearity="relu")
        if hasattr(op, "bias") and op.bias is not None:
            nn.init.constant_(op.bias, 0.0)


class Identity(nn.Module):
    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class BatchNorm2d(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(BatchNorm2d, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class AvgPool(nn.Module):
    def __init__(self,
                 kernel_size,
                 in_channels,
                 out_channels,
                 stride=None,
                 **kwargs):
        super(AvgPool, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(
            kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.avgpool(x)


class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride, padding, **kwargs):
        super(MaxPool, self).__init__()
        self.maxpool = torch.nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.maxpool(x)


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
        super(ConvBlock, self).__init__()

        conv = nn.Conv2d(in_channels, out_channels,
                         kernel_size=kernel_size, stride=stride,
                         padding=padding, groups=groups, bias=bias)
        _init_conv_weight(conv)
        self.conv = conv
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


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 group_width,
                 stride,
                 kernel_size=3,
                 use_se=False,
                 groups=1,
                 act='relu',
                 drop_path=0.):
        super(ResBlock, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_width = group_width
        mid_channels = group_width * groups
        self.mid_channels = mid_channels
        self.stride = stride
        self.groups = groups
        self.use_se = use_se

        # 1x1 conv
        self.conv_bn_relu1 = ConvBlock(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=False,
            kernel_size=1,
            stride=1,
            padding=0,
            act='relu')
        # 3x3 conv
        self.conv_bn_relu2 = ConvBlock(
            in_channels=mid_channels,
            out_channels=mid_channels,
            bias=False,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            act='relu')
        # se module
        self.se = SEModule(mid_channels) if use_se else None
        # 1x1 conv
        self.conv_bn_relu3 = ConvBlock(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=False,
            kernel_size=1,
            stride=1,
            padding=0,
            act='none')

        self.res_flag = (in_channels == out_channels) and (stride == 1)
        if self.res_flag:
            self.downsample = Identity()
        else:
            self.downsample = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                bias=False,
                kernel_size=1,
                stride=stride,
                padding=0,
                act='none')
        self.final_act = nn.ReLU(inplace=True)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        res = x
        sc = self.downsample(x)
        res = self.conv_bn_relu1(res)
        res = self.conv_bn_relu2(res)
        if self.se is not None:
            res = self.se(res)
        res = self.conv_bn_relu3(res)
        out = self.final_act(sc + res)
        return out


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer="GELU",
            drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.fc1 = nn.Conv2d(in_features, hidden_features,
                             kernel_size=1, stride=1, bias=True)
        assert hasattr(nn, act_layer)
        self.act = getattr(nn, act_layer)()
        self.fc2 = nn.Conv2d(hidden_features, out_features,
                             kernel_size=1, stride=1, bias=True)
        # self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x


class Attention(nn.Module):
    """ Multi-head Self-Attention (MSA) module with relative position bias"""

    def __init__(
            self,
            dim,
            input_resolution,
            num_heads,
            head_dim,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.,
            use_relative_position_bias=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mid_dim = head_dim * num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
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
        self.q = nn.Conv2d(dim, self.mid_dim, kernel_size=1, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, self.mid_dim * 2,
                            kernel_size=1, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(self.mid_dim, dim, kernel_size=1)
        # self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # B, nH, N, C//nH
        q = self.q(x).reshape(B, self.num_heads, self.head_dim,
                              N).permute(0, 1, 3, 2)
        # 2, B, nH, N, C//nH
        kv = self.kv(x).reshape(
            B, 2, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)

        # make torchscript happy (cannot use tensor as tuple)
        k, v = kv[0], kv[1]
        N_ = k.shape[-2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.use_relative_position_bias:
            # H*W, H*W,nH
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)].view(N, N_, -1)
            # nH, H*W, H*W
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        out = (attn @ v).permute(0, 1, 3,
                                 2).contiguous().reshape(B, self.mid_dim, H, W)
        out = self.proj(out)
        return out


class ViTBlock(nn.Module):
    def __init__(
            self,
            head_num,
            head_width,
            hid_dims,
            input_resolution,
            in_channels,
            out_channels,
            stride,
            out_norm=False,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer="GELU",
            norm_layer="LayerNorm",
            use_relative_position_bias=True,
            use_avgdown=False,
            skip_lam=1.):
        super(ViTBlock, self).__init__()
        self.head_num = head_num
        self.hid_dims = hid_dims
        self.input_resolution = input_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_avgdown = use_avgdown
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.out_norm = out_norm
        self.skip_lam = skip_lam

        if stride != 1 or in_channels != out_channels:
            if stride != 1 and use_avgdown:
                self.proj = nn.Sequential(OrderedDict([
                    ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2)),
                    ('conv', nn.Conv2d(
                        in_channels, out_channels, kernel_size=1, stride=1))]))
            else:
                self.proj = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.proj = None

        self.norm1 = globals()[norm_layer](out_channels)
        self.attn = Attention(
            out_channels,
            input_resolution,
            head_num,
            head_width,
            qkv_bias,
            qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_relative_position_bias=use_relative_position_bias)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = globals()[norm_layer](out_channels)
        mlp_hidden_dim = hid_dims
        self.mlp = Mlp(
            in_features=out_channels,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)
        self.out_norm = globals()[norm_layer](
            out_channels) if out_norm else None

    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x)
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        if self.out_norm is not None:
            x = self.out_norm(x)
        return x


class Hybridnet_retrain(nn.Module):
    def __init__(self,
                 arch_cfg,
                 drop_path_rate=0.1,
                 vit_block_norm='BatchNorm2d',
                 vit_block_act='ReLU',
                 num_classes=1000,
                 global_pool='avg',
                 features_only=False,
                 name=None,
                 **kwargs):
        super(Hybridnet_retrain, self).__init__()
        self.name = name
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.features_only = features_only
        self.out_channels = []

        dpr_layer_idxes = []
        for idx, [layer_name, params] in enumerate(arch_cfg):
            if "ViT" in layer_name:
                params.update({'norm_layer': vit_block_norm})
                params.update({'act_layer': vit_block_act})
                dpr_layer_idxes.append(idx)
        dpr = [x.item() for x in torch.linspace(
            0.05, drop_path_rate, len(dpr_layer_idxes))]
        layers = []
        for idx, [layer_name, params] in enumerate(arch_cfg):
            if "ViT" in layer_name:
                params.update({'drop_path': dpr[dpr_layer_idxes.index(idx)]})
                if idx == len(arch_cfg) - 1:
                    params.update({'out_norm': True})
            layers.append(globals()[layer_name](**params))
            if "Res" in layer_name or "ViT" in layer_name:
                self.out_channels.append(layers[-1].out_channels)
        self.layers = nn.ModuleList(layers)
        self._initialize_weights()

        self.start_indice = len(self.layers) - len(self.out_channels)
        self.out_indices = kwargs.pop('out_indices', None)
        if self.out_indices is None:
            self.out_indices = [i for i in range(len(self.out_channels))]
        assert isinstance(self.out_indices, (tuple, list)), \
            'out_indices must be tuple or list.'

        if self.global_pool == 'avg':
            self.avgpool = torch.nn.AvgPool2d(kernel_size=7, stride=None)
        elif self.global_pool is not None and self.global_pool != '':
            raise ValueError(
                f'Unsupported global pool type: {self.global_pool}.')
        self.last_out_channels = arch_cfg[-1][-1]['out_channels']
        if self.num_classes > 0:
            self.fc = nn.Linear(self.last_out_channels, num_classes)

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

    def forward_features(self, x):
        outs = OrderedDict()
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i - self.start_indice in self.out_indices:
                outs[i - self.start_indice] = x
        if self.features_only:
            return outs

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


@register_model
def automl_sparsity_v3(pretrained=False, **kwargs):
    model_name = 'automl_sparsity_v3'
    pretrained_config, model_config = get_configs(**kwargs)

    model = Hybridnet_retrain(
                ARCH_CFG_1,
                name=model_name,
                **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def automl_sparsity_v4(pretrained=False, **kwargs):
    model_name = 'automl_sparsity_v4'
    pretrained_config, model_config = get_configs(**kwargs)

    model = Hybridnet_retrain(
                ARCH_CFG_2,
                name=model_name,
                **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model
