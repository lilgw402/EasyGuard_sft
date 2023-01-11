r""" This is a self-implemented version of ResNet by Wei Li
       - from `"Deep Residual Learning for Image Recognition"
        <https://arxiv.org/pdf/1512.03385.pdf>`_
       - ResNet for Cifar is included
"""

from multiprocessing.sharedctypes import Value
from typing import List
from collections import OrderedDict
import torch.nn as nn
import torch

from titan.utils.registry import register_model
from titan.utils.helper import (
    download_weights,
    load_pretrained_model_weights,
    get_configs
)
from titan.utils.logger import logger

__all__ = ['resnet_v33_lw']


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
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
        # self.downsample = downsample
        self.stride = stride

        if use_se:
            self.se = SE(planes)
        else:
            self.se = None

    def forward(self, x):
        residual = self.conv1_bn_3x3(x) + self.conv1_bn_1x1(x)  # + x
        residual = self.relu(residual)
        residual = self.conv2_bn_3x3(
            residual) + self.conv2_bn_1x1(residual) + residual
        if self.se is not None:
            residual = self.se(residual)

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
            inv_planes = int(planes*2)
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
                 name: str = None,
                 **kwargs):
        r"""
        Args:
            block: resnet block
            layers: number of blocks at each layer
            inplanes: channel of the first convolution layer
            num_classes: number of classes for classification task
                only useful when `with_classifier` is True
            in_channels: channel of input image
            mode: resnet downsample mode. 'pytorch' for 3x3 downsample
                while 'default' for 1x1 downsample
            zero_gamma: if True, the gamma of last BN of each block is
                initialized to zero
            global_pool: type of global pooling, default is 'avg', meaning that
                an average pooling is added at the end of resnet stage5.
                If set to None or '', then no pooling will be used.
            features_only (bool): whether to output only feature maps. Default
                is False.
            out_indices: mark the indices of layer to get their output
            verbose: if True, logging is activated
            name: model name
            **kwargs: extra params which is passed to self._make_layer()
        """
        assert mode in ['pytorch', 'default'], \
            'Illegal resnet downsample mode {mode}. ' \
            'Choose from ["pytorch", "default"]'
        if verbose:
            pass

        super(ResNet, self).__init__()
        self.name = name
        self.inplanes = inplanes
        self.block = block
        self.zero_gamma = zero_gamma
        self.num_classes = num_classes
        assert global_pool in ['', 'avg', None], \
            f'global_pool type:{global_pool} is not supported.'
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
        strides = [1, 2, 2, 1, 2]  # , 1]
        out_c = [128, 256, 512, 512, 2048]
        for i in range(len(layers)):
            # stride = 1 if i == 0 else 2
            stride = strides[i]
            if i == 0:
                res_layer = self._make_layer_ratio(
                    128, layers[i], mode, stride=stride,
                    inplanes=self.inplanes, expansion=1,
                    block=RepVGGBlock, **kwargs)
            elif i == 1:
                res_layer = self._make_layer_ratio(
                    256, layers[i], mode, stride=stride,
                    inplanes=self.inplanes, expansion=1,
                    block=RepVGGBlock, **kwargs)
            elif i == 2:
                res_layer = self._make_layer_ratio(
                    512, layers[i], mode, stride=stride,
                    inplanes=self.inplanes, expansion=1,
                    groups=32, **kwargs)
            elif i == 3:
                res_layer = self._make_layer_ratio(
                    512, layers[i], mode, stride=stride,
                    inplanes=self.inplanes, expansion=1,
                    groups=32, **kwargs)
            elif i == 4:
                res_layer = self._make_layer_ratio(
                    512, layers[i], mode, stride=stride,
                    inplanes=self.inplanes, expansion=4,
                    **kwargs)
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
            self.out_channels.append(out_c[i])
        self.last_out_channels = 2048
        self.out_indices = [i for i in range(
            len(layers))] if out_indices is None else out_indices
        assert isinstance(self.out_indices, (tuple, list)), \
            'out_indices must be tuple or list.'

        # build classifier is required
        if self.num_classes > 0:
            self.fc = nn.Linear(self.last_out_channels, self.num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.zero_gamma:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.conv3_bn[1].weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.conv2_bn[1].weight, 0)

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
            force_downsample: if True, insert dowmsample in all blocks
                to align with hadron2
            **kwargs: extra params which is passed to block.__init__()

        Returns:
            A sequenced pytorch Module of a certain resnet stage
        """
        downsample = None
        if stride != 1 or inplanes != planes * expansion or force_downsample:
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
            inplanes, planes, stride, downsample, mode,
            expansion=expansion, groups=groups, **kwargs)

        self.inplanes = planes * expansion
        for i in range(1, blocks):
            if i == blocks - 1:
                kwargs['use_se'] = True
            layers[f'{self.block.__name__}{i}'] = block(
                self.inplanes, planes, mode=mode,
                expansion=expansion, groups=groups, **kwargs)
            kwargs['use_se'] = False
        return nn.Sequential(layers)

    def _make_layer(self,
                    planes: int,
                    blocks: int,
                    mode: str = None,
                    stride: int = 1,
                    use_avgdown: bool = False,
                    force_downsample: bool = False,
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
        if stride != 1 or self.inplanes != planes * self.block.expansion \
                or force_downsample:
            if stride != 1 and use_avgdown:
                downsample = nn.Sequential(OrderedDict([
                    ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2)),
                    ('conv_bn', ConvBN(
                        self.inplanes, planes * self.block.expansion,
                        kernel_size=1, stride=1))]))
            else:
                downsample = ConvBN(
                    self.inplanes, planes * self.block.expansion,
                    kernel_size=1, stride=stride)

        layers = OrderedDict()
        layers[f'{self.block.__name__}0'] = self.block(
            self.inplanes, planes, stride, downsample, mode, **kwargs)

        self.inplanes = planes * self.block.expansion
        for i in range(1, blocks):
            layers[f'{self.block.__name__}{i}'] = self.block(
                self.inplanes, planes, mode=mode, **kwargs)

        return nn.Sequential(layers)

    def forward(self, x):   # x: 64*224*224
        if len(x.shape) == 5:
            N, T, C, H, W = x.shape
        else:
            N, C, H, W = x.shape
            T = 1
        x = x.view(N * T, C, H, W)
        x = self.layer0(x)  # x: 64*112*112

        outs = OrderedDict()
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if self.features_only and i in self.out_indices:
                feat_map = x.clone()
                feat_map = feat_map.view((N, T,) + feat_map.size()[1:])
                feat_map = feat_map.mean(dim=1)
                outs[i] = feat_map
        if self.features_only:
            return list(outs.values())

        if self.global_pool == 'avg':
            x = x.mean([2, 3])
            x = torch.flatten(x, 1)  # (B, 2048)
            x = x.view(N, T, -1)
            x = x.mean(dim=1)

        if self.num_classes > 0:
            x = self.fc(x)  # (B, 1024)
            x = x.reshape(N, 4, self.fc.out_features // 4)
            x = torch.max(x, dim=1)[0]

        return x


@register_model
def resnet_v33_lw(pretrained=False, **kwargs):
    model_name = 'resnet_v33_lw'
    pretrained_config, model_config = get_configs(**kwargs)

    model = ResNet(
                Bottleneck,
                [1, 1, 3, 3, 4],
                name=model_name,
                **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model