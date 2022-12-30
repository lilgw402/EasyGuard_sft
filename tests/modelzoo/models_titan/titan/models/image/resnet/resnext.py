from collections import OrderedDict

import torch.nn as nn

from titan.utils.registry import register_model
from titan.utils.helper import (
    download_weights,
    load_pretrained_model_weights,
    get_configs
)
from titan.utils.logger import logger

__all__ = ['resnext50', 'resnext101']


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C
    (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channel, out_channel, stride, cardinality,
                 base_width, widen_factor, mode=None):
        """ Constructor
        Args:
            in_channel: input channel dimensionality
            out_channel: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality
                before convolution.
            mode: down sample mode, 'default' for 1x1conv and 'pytorch'
                for of 3x3conv.

        Rmk:
            The width_ratio is the expand ratio of D compared to D of the
            first stage. D represents the middle_channel of each sub-branch.
        """

        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channel / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        slist = [stride, 1] if mode == 'default' else [1, stride]

        self.conv_reduce = ConvBN(in_channel, D, 1, stride=slist[0])
        self.conv_conv = ConvBN(D, D, 3, stride=slist[1], groups=cardinality)
        self.conv_expand = ConvBN(D, out_channel, 1)
        self.shortcut = ConvBN(in_channel, out_channel, 1, stride=stride) \
            if in_channel != out_channel or stride != 1 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        residual = self.relu(self.conv_reduce(x))
        residual = self.relu(self.conv_conv(residual))
        residual = self.conv_expand(residual)

        return self.relu(identity + residual)


class ResNeXt(nn.Module):
    r"""
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(
            self,
            block,
            layers,
            cardinality=32,
            mode='pytorch',
            base_width=4,
            widen_factor=4,
            zero_init_residual=False,
            num_classes=1000,
            global_pool='avg',
            features_only=False,
            name=None,
            **kwargs):
        r"""
        :param block: the elementary block of the network
        :param layers: the number of blocks of each stage
        :param cardinality: number of convolution groups.
        :param mode: mode for downsample in bottleneck block.
        :param base_width: base number of channels in each group.
        :param widen_factor: factor to adjust the channel dimensionality.
        :param zero_init_residual: whether to zero the last BN of each
                                   residual block.
        :param num_classes: number of classes for classification task.
        :param global_pool: type of global pooling, default is 'avg'.
                            If set to None or '', then no pooling will be used.
        :param features_only: whether to output only feature maps. Default
                            is False.
        :param name: model name
        """
        super(ResNeXt, self).__init__()
        self.name = name
        self.block = block
        self.layers = layers
        self.cardinality = cardinality
        self.mode = mode
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.zero_init_residual = zero_init_residual
        self.channels = [64, 256, 512, 1024, 2048]
        self.out_channels = self.channels
        self.last_out_channels = self.out_channels[-1]
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.features_only = features_only

        self.layer0 = nn.Sequential(OrderedDict([
            ("conv7x7", nn.Conv2d(3, 64, kernel_size=7,
                                  stride=2, padding=3, bias=False)),
            ("bn", nn.BatchNorm2d(64)),
            ("relu", nn.ReLU(inplace=True)),
            ("pooling", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))
        self.layer1 = self._make_layer(idx=1, stride=1)
        self.layer2 = self._make_layer(idx=2, stride=2)
        self.layer3 = self._make_layer(idx=3, stride=2)
        self.layer4 = self._make_layer(idx=4, stride=2)

        self.out_indices = kwargs.pop('out_indices', None)
        if self.out_indices is None:
            self.out_indices = [i for i in range(5)]
        assert isinstance(self.out_indices, (tuple, list)), \
            'out_indices must be tuple or list.'

        if self.num_classes > 0:
            self.fc = nn.Linear(self.last_out_channels, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNeXtBottleneck):
                    nn.init.constant_(m.conv_expand['bn'].weight, 0)

    def _make_layer(self, idx, stride=1):
        """ Stack n bottleneck modules where n is inferred from the
            depth of the network.
        Args:
            idx: the index of layer
            stride: factor to reduce the spatial dimensionality in the
                first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """

        in_channel = self.channels[idx - 1]
        out_channel = self.channels[idx]
        layer = OrderedDict()
        layer["bottleneck_0"] = self.block(in_channel, out_channel, stride,
                                           cardinality=self.cardinality,
                                           base_width=self.base_width,
                                           widen_factor=self.widen_factor)
        for index in range(1, self.layers[idx - 1]):
            layer[f"bottleneck_{index}"] = self.block(
                out_channel,
                out_channel,
                1,
                cardinality=self.cardinality,
                base_width=self.base_width,
                widen_factor=self.widen_factor)
        return nn.Sequential(layer)

    def forward_features(self, x):
        outs = OrderedDict()

        out = self.layer0(x)
        if 0 in self.out_indices:
            outs[0] = out
        out = self.layer1(out)
        if 1 in self.out_indices:
            outs[1] = out
        out = self.layer2(out)
        if 2 in self.out_indices:
            outs[2] = out
        out = self.layer3(out)
        if 3 in self.out_indices:
            outs[3] = out
        out = self.layer4(out)
        if 4 in self.out_indices:
            outs[4] = out
        if self.features_only:
            return outs

        if self.global_pool == 'avg':
            return out.mean([2, 3])
        return x

    def forward(self, x):
        out = self.forward_features(x)
        if isinstance(out, OrderedDict):
            return list(out.values())
        if self.num_classes > 0:
            out = self.fc(out)
        return out


@register_model
def resnext50(pretrained=False, **kwargs):
    r"""ResNeXt-50 model from
        `"Aggregated Residual Transformation for Deep Neural Networks"
        <https://arxiv.org/pdf/1611.05431.pdf>`_.
    """
    model_name = 'resnext50'
    pretrained_config, model_config = get_configs(**kwargs)

    model = ResNeXt(
                ResNeXtBottleneck,
                [3, 4, 6, 3],
                base_width=4,
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
def resnext101(pretrained=False, **kwargs):
    r"""ResNeXt-101 model from
        `"Aggregated Residual Transformation for Deep Neural Networks"
        <https://arxiv.org/pdf/1611.05431.pdf>`_.
    """
    model_name = 'resnext101'
    pretrained_config, model_config = get_configs(**kwargs)

    model = ResNeXt(
        ResNeXtBottleneck,
        [3, 4, 23, 3],
        base_width=8,
        **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model
