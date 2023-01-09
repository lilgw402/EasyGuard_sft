import torch
import torch.nn as nn
from collections import OrderedDict

from titan.utils.registry import register_model
from titan.utils.helper import (
    download_weights,
    load_pretrained_model_weights,
    get_configs
)
from titan.utils.logger import logger

__all__ = ['resnet50_chilu']

QATV2_FLAG = False


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups,
                     bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        if QATV2_FLAG:
            from feather.quant import nn as quant_nn
            from feather.quant.quantizer.quant_module import TensorQuantizer
            self.identity_quantizer = TensorQuantizer(
                quant_nn.QuantConv2d.default_quant_desc_input)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if QATV2_FLAG:
            identity = self.identity_quantizer(identity)
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if QATV2_FLAG:
            from feather.quant import nn as quant_nn
            from feather.quant.quantizer.quant_module import TensorQuantizer
            self.identity_quantizer = TensorQuantizer(
                quant_nn.QuantConv2d.default_quant_desc_input)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if QATV2_FLAG:
            identity = self.identity_quantizer(identity)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block,
            layers,
            num_classes=1000,
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            token_type=0,
            in_channels=3,
            out_indices=None,
            global_pool='avg',
            features_only=False,
            name=None):
        super(ResNet, self).__init__()
        self.name = name
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.token_type = token_type
        self.num_classes = num_classes
        assert global_pool in ['', 'avg', None], \
            f'global_pool type:{global_pool} is not supported.'
        self.global_pool = global_pool
        self.features_only = features_only
        self.out_channels = []
        self.out_indices = [i for i in range(
            4)] if out_indices is None else out_indices
        assert isinstance(self.out_indices, (tuple, list)), \
            'out_indices must be tuple or list.'

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}")
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            in_channels,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.out_channels.append(64 * block.expansion)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.out_channels.append(128 * block.expansion)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.out_channels.append(256 * block.expansion)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.out_channels.append(512 * block.expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.num_classes > 0:
            self.fc = nn.Linear(512 * block.expansion, self.num_classes)

        if self.token_type == 2:
            du_sample = {}
            channels = [64, 128, 256, 512]
            du_sample['1'] = nn.Linear(64, 256)
            for i in range(4):
                du_sample[str(i + 2)] = nn.Linear(channels[i]
                                                  * block.expansion, 256)
            self.du_sample = nn.ModuleDict(du_sample)
            self.visual_factorize = nn.Linear(256, 128)
        elif self.token_type == 3:
            self.get_tokens = nn.Sequential(*[
                nn.AdaptiveAvgPool2d((1, 1,)),
                nn.Flatten(),
                nn.Linear(512 * block.expansion, 256),
            ])
            self.visual_factorize = nn.Linear(256, 128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _multi_stage_tokens(self, stage_feats):
        stage_tokens = []
        for i, stage_feat in enumerate(stage_feats):
            stage_tokens.append(self.du_sample[str(i + 1)](
                torch.mean(stage_feat, dim=[-1, -2])
            ))
        tokens = torch.stack(stage_tokens, dim=1).mean(dim=1)
        tokens = self.visual_factorize(tokens)
        return tokens

    def forward_features(self, x):
        # See note [TorchScript super()]
        if len(x.shape) == 5:
            N, T, C, H, W = x.shape
        else:
            N, C, H, W = x.shape
            T = 1
        outs = OrderedDict()

        x = x.view(N * T, C, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        res1 = self.maxpool(x)

        res2 = self.layer1(res1)
        if 0 in self.out_indices:
            outs[0] = res2
        res3 = self.layer2(res2)
        if 1 in self.out_indices:
            outs[1] = res3
        res4 = self.layer3(res3)
        if 2 in self.out_indices:
            outs[2] = res4
        res5 = self.layer4(res4)
        if 3 in self.out_indices:
            outs[3] = res5
        if self.features_only:
            return outs

        if self.token_type == 1:
            tokens = res5.clone()
            tokens = tokens.view((N, T,) + tokens.size()[1:])
            tokens = tokens.mean(dim=1)
        elif self.token_type == 2:
            tokens = self._multi_stage_tokens([res1, res2, res3, res4, res5])
            tokens = tokens.view(N, T, -1)
        elif self.token_type == 3:
            tokens = self.get_tokens(res5)
            tokens = self.visual_factorize(tokens)
            tokens = tokens.view(N, T, -1)
        else:
            tokens = None

        if self.global_pool == 'avg':
            x = self.avgpool(res5)
            x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        if len(x.shape) == 5:
            N, T, _, _, _ = x.shape
        else:
            N, _, _, _ = x.shape
            T = 1
        x = self.forward_features(x)
        if isinstance(x, OrderedDict):
            return list(x.values())
        if self.num_classes > 0:
            x = self.fc(x)
            x = x / (x.norm(dim=1, keepdim=True) + 1e-12)
            x = x.view(N, T, -1)
            x = x.mean(dim=1)
        return x


@register_model
def resnet50_chilu(pretrained=False, **kwargs):
    r"""
    ResNet-50 model from `"Deep Residual Learning
    for Image Recognition"
    <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Weights are provided by chilu@bytedance.com.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download
            to stderr
    """
    model_name = 'resnet50_chilu'
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
