"""
Neighborhood Attention Transformer.
https://arxiv.org/abs/2204.07143

Modified by: Bhavya Shah
"""
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from titan.utils.registry import register_model
from titan.utils.helper import (
    download_weights,
    load_pretrained_model_weights,
    get_configs
)
from titan.utils.logger import logger
from torch.nn.functional import unfold, pad
from timm.models.layers import trunc_normal_
import warnings


__all__ = [
    'nat_mini',
    'nat_tiny',
    'nat_small',
    'nat_base'
]

class LegacyNeighborhoodAttention(nn.Module):
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mode=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        self.win_size = kernel_size // 2
        self.mid_cell = kernel_size - 1
        self.rpb_size = 2 * kernel_size - 1

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        self.rpb = nn.Parameter(torch.zeros(num_heads, self.rpb_size, self.rpb_size))
        trunc_normal_(self.rpb, std=.02)
        # RPB implementation by @qwopqwop200
        self.idx_h = torch.arange(0, kernel_size)
        self.idx_w = torch.arange(0, kernel_size)
        self.idx_k = ((self.idx_h.unsqueeze(-1) * self.rpb_size) + self.idx_w).view(-1)
        warnings.warn("This is the legacy version of NAT -- it uses unfold+pad to produce NAT, and is highly inefficient.")

    def apply_pb(self, attn, height, width):
        """
        RPB implementation by @qwopqwop200
        https://github.com/qwopqwop200/Neighborhood-Attention-Transformer
        """
        num_repeat_h = torch.ones(self.kernel_size,dtype=torch.long)
        num_repeat_w = torch.ones(self.kernel_size,dtype=torch.long)
        num_repeat_h[self.kernel_size//2] = height - (self.kernel_size-1)
        num_repeat_w[self.kernel_size//2] = width - (self.kernel_size-1)
        bias_hw = (self.idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2*self.kernel_size-1)) + self.idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + self.idx_k
        # Index flip
        # Our RPB indexing in the kernel is in a different order, so we flip these indices to ensure weights match.
        bias_idx = torch.flip(bias_idx.reshape(-1, self.kernel_size**2), [0])
        return attn + self.rpb.flatten(1, 2)[:, bias_idx].reshape(self.num_heads, height * width, 1, self.kernel_size ** 2).transpose(0, 1)

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        num_tokens = int(self.kernel_size ** 2)
        pad_l = pad_t = pad_r = pad_b = 0
        Ho, Wo = H, W
        if N <= num_tokens:
            if self.kernel_size > W:
                pad_r = self.kernel_size - W
            if self.kernel_size > H:
                pad_b = self.kernel_size - H
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            B, H, W, C = x.shape
            N = H * W
            assert N == num_tokens, f"Something went wrong. {N} should equal {H} x {W}!"
        x = self.qkv(x).reshape(B, H, W, 3 * C)
        q, x = x[:, :, :, :C], x[:, :, :, C:]
        q = q.reshape(B, N, self.num_heads, C // self.num_heads, 1).transpose(3, 4) * self.scale
        pd = self.kernel_size - 1
        pdr = pd // 2
        # NAT Implementation mode
        # Mode 0 is more memory efficient because Tensor.unfold is not contiguous, so
        #         it will be almost as if the replicate pad and unfold will allocate the
        #         memory for the new tensor at the same time.
        # Mode 1 is less memory efficient, because F.unfold is contiguous, so unfold will
        #         output an actual tensor once, and replicate will work on that so it'll be
        #         one extra memory allocation. On the other hand, because F.unfold has a CUDA
        #         kernel of its own, and possibly because we don't have to flatten channel
        #         and batch axes to use Tensor.unfold, this will be somewhat faster, but at the
        #         expense of using more memory. It is more feasible for CLS as opposed to DET/SEG
        #         because we're dealing with smaller-res images, but have a lot more images to get
        #         through.
        if self.mode == 0:
            x = x.permute(0, 3, 1, 2).flatten(0, 1)
            x = x.unfold(1, self.kernel_size, 1).unfold(2, self.kernel_size, 1).permute(0, 3, 4, 1, 2)
            x = pad(x, (pdr, pdr, pdr, pdr, 0, 0), 'replicate')
            x = x.reshape(B, 2, self.num_heads, C // self.num_heads, num_tokens, N)
            x = x.permute(1, 0, 5, 2, 4, 3)
        elif self.mode == 1:
            Hr, Wr = H - pd, W - pd
            x = unfold(x.permute(0, 3, 1, 2),
                       kernel_size=(self.kernel_size, self.kernel_size),
                       stride=(1, 1),
                       padding=(0, 0)).reshape(B, 2 * C * num_tokens, Hr, Wr)
            x = pad(x, (pdr, pdr, pdr, pdr), 'replicate').reshape(
                B, 2, self.num_heads, C // self.num_heads, num_tokens, N)
            x = x.permute(1, 0, 5, 2, 4, 3)
        else:
            raise NotImplementedError(f'Mode {self.mode} not implemented for NeighborhoodAttention2D.')
        k, v = x[0], x[1]

        attn = (q @ k.transpose(-2, -1))  # B x N x H x 1 x num_tokens
        attn = self.apply_pb(attn, H, W)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)  # B x N x H x 1 x C
        x = x.reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Ho, :Wo, :]
        return self.proj_drop(self.proj(x))





class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NATLayer(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        self.attn = LegacyNeighborhoodAttention(
            dim, kernel_size=kernel_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class NATBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, kernel_size, downsample=True,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 layer_scale=None):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            NATLayer(dim=dim,
                     num_heads=num_heads, kernel_size=kernel_size,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop, attn_drop=attn_drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer, layer_scale=layer_scale)
            for i in range(depth)])

        self.downsample = None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is None:
            return x
        return self.downsample(x)


class NAT(nn.Module):
    __model_type__ = 'nat'
    def __init__(self,
                 embed_dim,
                 mlp_ratio,
                 depths,
                 num_heads,
                 drop_path_rate=0.2,
                 in_chans=3,
                 kernel_size=7,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 name = None,
                 **kwargs):
        """
        Args:
            embed_dim (int): embedding dimension
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            drop_path_rate (float): stochastic depth rate
            in_chans (int): number of input channels
            kernel_size (int): size of the neighbourhood kernel 
            num_classes (int): number of classes for classification head
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
            Default: None
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            norm_layer: (nn.Module): normalization layer
            name: model name
        """
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_levels - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = ConvTokenizer(in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(dim=int(embed_dim * 2 ** i),
                             depth=depths[i],
                             num_heads=num_heads[i],
                             kernel_size=kernel_size,
                             mlp_ratio=self.mlp_ratio,
                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                             drop=drop_rate, attn_drop=attn_drop_rate,
                             drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                             norm_layer=norm_layer,
                             downsample=(i < self.num_levels - 1),
                             layer_scale=layer_scale)
            self.levels.append(level)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}


    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for level in self.levels:
            x = level(x)

        x = self.norm(x).flatten(1, 2)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def nat_mini(pretrained=False, **kwargs):

    model_name = 'nat_mini'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "NAT does not support features_only.")
    model_config.pop('features_only')

    model_kwargs = dict(depths=[3, 4, 6, 5], 
    num_heads=[2, 4, 8, 16], 
    embed_dim=64, 
    mlp_ratio=3,
    drop_path_rate=0.2, 
    kernel_size=7, 
    name=model_name,
    **model_config)
    
    model = NAT(**model_kwargs)
   
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")

    return model


@register_model
def nat_tiny(pretrained=False, **kwargs):
    model_name = 'nat_tiny'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "NAT does not support features_only.")
    model_config.pop('features_only')

    model_kwargs = dict(depths=[3, 4, 18, 5], 
    num_heads=[2, 4, 8, 16], 
    embed_dim=64, 
    mlp_ratio=3,
    drop_path_rate=0.2, 
    kernel_size=7,
    name=model_name,
    **model_config)
    
    model = NAT(**model_kwargs)

    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")

    return model


@register_model
def nat_small(pretrained=False, **kwargs):

    model_name = 'nat_small'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "NAT does not support features_only.")
    model_config.pop('features_only')

    model_kwargs = dict(depths=[3, 4, 18, 5], 
    num_heads=[3, 6, 12, 24], 
    embed_dim=96, 
    mlp_ratio=2,
    drop_path_rate=0.3, 
    layer_scale=1e-5, 
    kernel_size=7,
    name=model_name,
    **model_config)
    
    model = NAT(**model_kwargs)
   
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")

    return model


@register_model
def nat_base(pretrained=False, **kwargs):

    model_name = 'nat_base'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "NAT does not support features_only.")
    model_config.pop('features_only')

    model_kwargs = dict(depths=[3, 4, 18, 5], 
    num_heads=[4, 8, 16, 32], 
    embed_dim=128, 
    mlp_ratio=2,
    drop_path_rate=0.5, 
    layer_scale=1e-5, 
    kernel_size=7,
    name=model_name,
    **model_config)
    
    model = NAT(**model_kwargs)

    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")

    return model