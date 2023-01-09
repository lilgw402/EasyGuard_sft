r""" This is a self-implemented version of ResNet from `"Deep Residual Learning
     for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
     ResNet for Cifar is included.
"""

from typing import List
from collections import OrderedDict
import torch.nn as nn

from titan.models.components import SEBlock, SEBNBlock
from titan.utils.registry import register_model
from titan.utils.helper import (
    download_weights,
    load_pretrained_model_weights,
    get_configs
)
from titan.utils.logger import logger

__all__ = [
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'resnet_cifar32',
    'resnet_cifar56',
    'resnet_cifar110',
    'resnet_cifar1202',
]


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
                self.se = SEBNBlock(planes * self.expansion,
                                    reduction=se_reduction)
            else:
                self.se = SEBlock(planes * self.expansion,
                                  reduction=se_reduction)
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
                self.se = SEBNBlock(planes * self.expansion,
                                    reduction=se_reduction)
            else:
                self.se = SEBlock(planes * self.expansion,
                                  reduction=se_reduction)
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
            features_only (bool): whether to output only feature maps.
                        Default is False.
            out_indices: mark the indices of layer to get their output
            verbose: if True, logging is activated
            **kwargs: extra params which is passed to self._make_layer()
        """
        assert mode in ['pytorch', 'default'], \
            f'Illegal resnet downsample mode {mode}. ' \
            f'Choose from ["pytorch", "default"]'
        if verbose:
            logger.info(f'=> Model arch: using {mode} downsample ResNet')

        assert 'with_classifier' not in kwargs, \
            'with_classifier no longer supported in backbones, ' \
            'please define fc layers using head'
        assert 'num_classes' not in kwargs, \
            'num_classes no longer supported in backbones, ' \
            'please define fc layers using head'

        super(ResNet, self).__init__()
        self.inplanes = inplanes
        self.block = block
        self.zero_gamma = zero_gamma
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.features_only = features_only
        self.out_indices = [i for i in range(
            len(layers))] if out_indices is None else out_indices
        assert isinstance(self.out_indices, (tuple, list)), \
            'out_indices must be tuple or list.'

        self.layer0 = nn.Sequential(OrderedDict([
            ('conv_bn', ConvBN(
                in_channels, inplanes, kernel_size=7, stride=2)),
            ('relu', nn.ReLU(inplace=True)),
            ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        # build residual layers
        self.res_layers = []
        self.out_channels = []
        for i in range(len(layers)):
            stride = 1 if i == 0 else 2
            res_layer = self._make_layer(
                inplanes * 2**i, layers[i], mode, stride=stride, **kwargs)
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
            self.out_channels.append(self.block.expansion * inplanes * 2**i)
        self.last_out_channels = self.out_channels[-1]

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
        if stride != 1 or self.inplanes != planes * \
                self.block.expansion or force_downsample:
            if stride != 1 and use_avgdown:
                downsample = nn.Sequential(OrderedDict([
                    ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2)),
                    ('conv_bn', ConvBN(
                        self.inplanes, planes * self.block.expansion,
                        kernel_size=1, stride=1))]))
            else:
                downsample = ConvBN(
                    self.inplanes,
                    planes *
                    self.block.expansion,
                    kernel_size=1,
                    stride=stride)

        layers = OrderedDict()
        layers[f'{self.block.__name__}0'] = self.block(
            self.inplanes, planes, stride, downsample, mode, **kwargs)

        self.inplanes = planes * self.block.expansion
        for i in range(1, blocks):
            layers[f'{self.block.__name__}{i}'] = self.block(
                self.inplanes, planes, mode=mode, **kwargs)

        return nn.Sequential(layers)

    def forward_features(self, x):
        x = self.layer0(x)  # x: 64*112*112

        outs = OrderedDict()
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs[i] = x
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
                 inplanes: int = 16,
                 **kwargs):
        r"""
        Args:
            block: resnet Cifar block
            layers: number of blocks at each layer
            inplanes: channel of the first convolution layer
            **kwargs: extra params which is passed to self._make_layer()
        """
        super(ResNetCifar, self).__init__(
            block=block,
            layers=layers,
            inplanes=inplanes,
            **kwargs)

        self.layer0 = nn.Sequential(OrderedDict([
            ('conv_bn', ConvBN(3, inplanes, kernel_size=3)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self._initialize_weights()


@register_model
def resnet18(pretrained=False, **kwargs):
    model_name = 'resnet18'
    pretrained_config, model_config = get_configs(**kwargs)

    model = ResNet(BasicBlock, [2, 2, 2, 2], **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def resnet34(pretrained=False, **kwargs):
    model_name = 'resnet34'
    pretrained_config, model_config = get_configs(**kwargs)

    model = ResNet(BasicBlock, [3, 4, 6, 3], **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def resnet50(pretrained=False, **kwargs):
    model_name = 'resnet50'
    pretrained_config, model_config = get_configs(**kwargs)

    model = ResNet(Bottleneck, [3, 4, 6, 3], **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def resnet50_vd(pretrained=False, **kwargs):
    model_name = 'resnet50_vd'
    pretrained_config, model_config = get_configs(**kwargs)
    model_config['use_avgdown'] = True

    model = ResNet(Bottleneck, [3, 4, 6, 3], **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def resnet101(pretrained=False, **kwargs):
    model_name = 'resnet101'
    pretrained_config, model_config = get_configs(**kwargs)

    model = ResNet(Bottleneck, [3, 4, 23, 3], **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def resnet152(pretrained=False, **kwargs):
    model_name = 'resnet152'
    pretrained_config, model_config = get_configs(**kwargs)

    model = ResNet(Bottleneck, [3, 8, 36, 3], **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def resnet_cifar32(pretrained=False, **kwargs):
    model_name = 'resnet_cifar32'
    pretrained_config, model_config = get_configs(**kwargs)

    model = ResNetCifar(BasicBlock, [5, 5, 5], **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def resnet_cifar56(pretrained=False, **kwargs):
    model_name = 'resnet_cifar56'
    pretrained_config, model_config = get_configs(**kwargs)

    model = ResNetCifar(BasicBlock, [9, 9, 9], **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def resnet_cifar110(pretrained=False, **kwargs):
    model_name = 'resnet_cifar110'
    pretrained_config, model_config = get_configs(**kwargs)

    model = ResNetCifar(BasicBlock, [18, 18, 18], **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def resnet_cifar1202(pretrained=False, **kwargs):
    model_name = 'resnet_cifar1202'
    pretrained_config, model_config = get_configs(**kwargs)

    model = ResNetCifar(BasicBlock, [200, 200, 200], **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model
