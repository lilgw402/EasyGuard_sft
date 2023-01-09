"""
Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and
Mark D. Plumbley. "Panns: Large-scale pretrained audio neural networks
for audio pattern recognition." IEEE/ACM Transactions on Audio, Speech,
and Language Processing 28 (2020): 2880-2894ICCV 2019,
https://arxiv.org/abs/1812.03982
"""

from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from titan.models.audio.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from titan.models.audio.pytorch_utils import do_mixup
from titan.utils.registry import register_model
from titan.utils.helper import (
    download_weights,
    load_pretrained_model_weights,
    get_configs
)
from titan.utils.logger import logger



__all__ = [
    'audio_cnn6',
    'audio_cnn10',
    'audio_cnn14',
    'audio_resnet22',
    'audio_resnet38',
    'audio_resnet54',
    'audio_mobilenetv1',
    'audio_mobilenetv2',
    'audio_leenet11',
    'audio_dainet19',
    'audio_res1dnet31',
    'audio_res1dnet51',
    'audio_wavegram_cnn14',
    'audio_wavegram_logmel_cnn14',
]


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock5x5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class AttBlock(nn.Module):
    def __init__(self, n_in, n_out, activation='linear', temperature=1.):
        super(AttBlock, self).__init__()

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(in_channels=n_in, out_channels=n_out,
                             kernel_size=1, stride=1, padding=0, bias=True)
        self.cla = nn.Conv1d(in_channels=n_in, out_channels=n_out,
                             kernel_size=1, stride=1, padding=0, bias=True)

        self.bn_att = nn.BatchNorm1d(n_out)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


class Cnn14(nn.Module):
    def __init__(
            self,
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            num_classes=527,
            global_pool='avg+max',
            no_fc=False,
            features_only=False,
            name=None,
            **kwargs):

        super(Cnn14, self).__init__()
        self.name = name

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.features_only = features_only
        self.global_pool = global_pool
        self.no_fc = no_fc
        self.num_classes = num_classes

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        self.out_indices = kwargs.pop('out_indices', None)
        if self.out_indices is None:
            self.out_indices = [i for i in range(6)]
        assert isinstance(self.out_indices, (tuple, list)), \
            'out_indices must be tuple or list.'
        self.out_channels = [64, 128, 256, 512, 1024, 2048]
        self.last_out_channels = 2048

        if not self.no_fc:
            self.fc1 = nn.Linear(2048, 2048, bias=True)
            if self.num_classes > 0:
                self.fc_audioset = nn.Linear(2048, num_classes, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        if not self.no_fc:
            init_layer(self.fc1)
            if self.num_classes > 0:
                init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # (batch_size, 1, time_steps, freq_bins)
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        outs = OrderedDict()
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        if 0 in self.out_indices:
            outs[0] = x
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        if 1 in self.out_indices:
            outs[1] = x
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        if 2 in self.out_indices:
            outs[2] = x
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        if 3 in self.out_indices:
            outs[3] = x
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        if 4 in self.out_indices:
            outs[4] = x
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        if 5 in self.out_indices:
            outs[5] = x

        if self.features_only:
            return outs

        if self.global_pool == 'avg+max':
            x = torch.mean(x, dim=3)
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)

            if self.no_fc:
                return x
            
            x = F.relu_(self.fc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            if self.num_classes > 0:
                clipwise_output = torch.sigmoid(self.fc_audioset(x))
                return clipwise_output
        return x


class Cnn6(nn.Module):
    def __init__(
            self,
            name,
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            num_classes=527,
            global_pool='avg+max',
            no_fc=False,
            features_only=False,
            **kwargs):

        super(Cnn6, self).__init__()
        self.name = name

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.features_only = features_only
        self.global_pool = global_pool
        self.no_fc = no_fc
        self.num_classes = num_classes

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)
        self.out_indices = kwargs.pop('out_indices', None)
        if self.out_indices is None:
            self.out_indices = [i for i in range(4)]
        assert isinstance(self.out_indices, (tuple, list)), \
            'out_indices must be tuple or list.'
        self.out_channels = [64, 128, 256, 512]
        self.last_out_channels = 512

        if not self.no_fc:
            self.fc1 = nn.Linear(512, 512, bias=True)
            if self.num_classes > 0:
                self.fc_audioset = nn.Linear(512, num_classes, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        if not self.no_fc:
            init_layer(self.fc1)
            if self.num_classes > 0:
                init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # (batch_size, 1, time_steps, freq_bins)
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        outs = OrderedDict()
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        if 0 in self.out_indices:
            outs[0] = x
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        if 1 in self.out_indices:
            outs[1] = x
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        if 2 in self.out_indices:
            outs[2] = x
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        if 3 in self.out_indices:
            outs[3] = x

        if self.features_only:
            return outs

        if self.global_pool == 'avg+max':
            x = torch.mean(x, dim=3)
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)

            if self.no_fc:
                return x
            
            x = F.relu_(self.fc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            if self.num_classes > 0:
                clipwise_output = torch.sigmoid(self.fc_audioset(x))
                return clipwise_output
        return x


class Cnn10(nn.Module):
    def __init__(
            self,
            name,
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            num_classes=527,
            global_pool='avg+max',
            no_fc=False,
            features_only=False,
            **kwargs):

        super(Cnn10, self).__init__()
        self.name = name

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.features_only = features_only
        self.global_pool = global_pool
        self.no_fc = no_fc
        self.num_classes = num_classes

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.out_indices = kwargs.pop('out_indices', None)
        if self.out_indices is None:
            self.out_indices = [i for i in range(4)]
        assert isinstance(self.out_indices, (tuple, list)), \
            'out_indices must be tuple or list.'
        self.out_channels = [64, 128, 256, 512]
        self.last_out_channels = 512

        if not self.no_fc:
            self.fc1 = nn.Linear(512, 512, bias=True)
            if self.num_classes > 0:
                self.fc_audioset = nn.Linear(512, num_classes, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        if not self.no_fc:
            init_layer(self.fc1)
            if self.num_classes > 0:
                init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # (batch_size, 1, time_steps, freq_bins)
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        outs = OrderedDict()
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        if 0 in self.out_indices:
            outs[0] = x
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        if 1 in self.out_indices:
            outs[1] = x
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        if 2 in self.out_indices:
            outs[2] = x
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        if 3 in self.out_indices:
            outs[3] = x

        if self.features_only:
            return outs

        if self.global_pool == 'avg+max':
            x = torch.mean(x, dim=3)
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)

            if self.no_fc:
                return x

            x = F.relu_(self.fc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            if self.num_classes > 0:
                clipwise_output = torch.sigmoid(self.fc_audioset(x))
                return clipwise_output
        return x


def _resnet_conv3x3(in_planes, out_planes):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)


def _resnet_conv1x1(in_planes, out_planes):
    # 1x1 convolution
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=1,
        bias=False)


class _ResnetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                '_ResnetBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in _ResnetBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when
        # stride != 1

        self.stride = stride

        self.conv1 = _resnet_conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _resnet_conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            out = F.avg_pool2d(x, kernel_size=(2, 2))
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResnetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.stride = stride
        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = _resnet_conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = _resnet_conv3x3(width, width)
        self.bn2 = norm_layer(width)
        self.conv3 = _resnet_conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        init_layer(self.conv3)
        init_bn(self.bn3)
        nn.init.constant_(self.bn3.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            x = F.avg_pool2d(x, kernel_size=(2, 2))

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None):
        super(_ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[0])
                init_bn(downsample[1])
            elif stride == 2:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2),
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[1])
                init_bn(downsample[2])

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

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResNet22(nn.Module):
    def __init__(
            self,
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            num_classes=527,
            global_pool='avg+max',
            no_fc=False,
            name=None,
            **kwargs):

        super(ResNet22, self).__init__()
        self.name = name

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.global_pool = global_pool
        self.no_fc = no_fc
        self.num_classes = num_classes

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        # self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)

        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[
                              2, 2, 2, 2], zero_init_residual=True)

        self.conv_block_after1 = ConvBlock(in_channels=512, out_channels=2048)
        self.last_out_channels = 2048

        if not self.no_fc:
            self.fc1 = nn.Linear(2048, 2048)
            if self.num_classes > 0:
                self.fc_audioset = nn.Linear(2048, num_classes, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        if not self.no_fc:
            init_layer(self.fc1)
            if self.num_classes > 0:
                init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # (batch_size, 1, time_steps, freq_bins)
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)

        if self.global_pool == 'avg+max':
            x = torch.mean(x, dim=3)
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)

            if self.no_fc:
                return x

            x = F.relu_(self.fc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            if self.num_classes > 0:
                clipwise_output = torch.sigmoid(self.fc_audioset(x))
                return clipwise_output
        return x


class ResNet38(nn.Module):
    def __init__(
            self,
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            num_classes=527,
            global_pool='avg+max',
            no_fc=False,
            name=None,
            **kwargs):

        super(ResNet38, self).__init__()
        self.name = name

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.global_pool = global_pool
        self.no_fc = no_fc
        self.num_classes = num_classes

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        # self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)

        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[
                              3, 4, 6, 3], zero_init_residual=True)

        self.conv_block_after1 = ConvBlock(in_channels=512, out_channels=2048)
        self.last_out_channels = 2048

        if not self.no_fc:
            self.fc1 = nn.Linear(2048, 2048)
            if self.num_classes > 0:
                self.fc_audioset = nn.Linear(2048, num_classes, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        if not self.no_fc:
            init_layer(self.fc1)
            if self.num_classes > 0:
                init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # (batch_size, 1, time_steps, freq_bins)
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        
        if self.global_pool == 'avg+max':
            x = torch.mean(x, dim=3)
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)
            if self.num_classes > 0:
                x = F.relu_(self.fc1(x))
                x = F.dropout(x, p=0.5, training=self.training)
                clipwise_output = torch.sigmoid(self.fc_audioset(x))
                return clipwise_output
        return x


class ResNet54(nn.Module):
    def __init__(
            self,
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            num_classes=527,
            global_pool='avg+max',
            no_fc=False,
            name=None,
            **kwargs):

        super(ResNet54, self).__init__()
        self.name = name

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.global_pool = global_pool
        self.no_fc = no_fc
        self.num_classes = num_classes

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        # self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)

        self.resnet = _ResNet(block=_ResnetBottleneck, layers=[
                              3, 4, 6, 3], zero_init_residual=True)

        self.conv_block_after1 = ConvBlock(in_channels=2048, out_channels=2048)
        self.last_out_channels = 2048

        if not self.no_fc:
            self.fc1 = nn.Linear(2048, 2048) 
            if self.num_classes > 0:
                self.fc_audioset = nn.Linear(2048, num_classes, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        if not self.no_fc:
            init_layer(self.fc1)
            if self.num_classes > 0:
                init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # (batch_size, 1, time_steps, freq_bins)
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        
        if self.global_pool == 'avg+max':
            x = torch.mean(x, dim=3)
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)

            if self.no_fc:
                return x

            x = F.relu_(self.fc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            if self.num_classes > 0:
                clipwise_output = torch.sigmoid(self.fc_audioset(x))
                return clipwise_output
        return x


class MobileNetV1(nn.Module):
    def __init__(
            self,
            name,
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            num_classes=527,
            global_pool='avg+max',
            no_fc=False,
            features_only=False,
            **kwargs):

        super(MobileNetV1, self).__init__()
        self.name = name

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.features_only = features_only
        self.global_pool = global_pool
        self.no_fc = no_fc
        self.num_classes = num_classes

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        def conv_bn(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            return _layers

        def conv_dw(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            init_layer(_layers[4])
            init_bn(_layers[5])
            return _layers

        self.features = nn.Sequential(
            conv_bn(1, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1))
        self.out_indices = kwargs.get('out_indices', None)
        if self.out_indices is None:
            self.out_indices = [i for i in range(len(self.features))]
        assert isinstance(self.out_indices, (tuple, list)), \
            'out_indices must be tuple or list.'
        self.out_channels = [32, 64, 128, 128, 256, 256, 512,
                             512, 512, 512, 512, 512, 1024, 1024]
        self.last_out_channels = 1024

        if not self.no_fc:
            self.fc1 = nn.Linear(1024, 1024, bias=True)
            if self.num_classes > 0:
                self.fc_audioset = nn.Linear(1024, num_classes, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        if not self.no_fc:
            init_layer(self.fc1)
            if self.num_classes > 0:
                init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # (batch_size, 1, time_steps, freq_bins)
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        outs = OrderedDict()
        for idx, feature_layer in enumerate(self.features):
            x = feature_layer(x)
            if idx in self.out_indices:
                outs[idx] = x
        if self.features_only:
            return outs

        if self.global_pool == 'avg+max':
            x = torch.mean(x, dim=3)
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)

            if self.no_fc:
                return x

            x = F.relu_(self.fc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            if self.num_classes > 0:
                clipwise_output = torch.sigmoid(self.fc_audioset(x))
                return clipwise_output
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            _layers = [
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            init_layer(_layers[4])
            init_bn(_layers[5])
            self.conv = _layers
        else:
            _layers = [
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[1])
            init_layer(_layers[3])
            init_bn(_layers[5])
            init_layer(_layers[7])
            init_bn(_layers[8])
            self.conv = _layers

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
            self,
            name,
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            num_classes=527,
            global_pool='avg+max',
            no_fc=False,
            features_only=False,
            out_indices=None):

        super(MobileNetV2, self).__init__()
        self.name = name

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.features_only = features_only
        self.global_pool = global_pool
        self.no_fc = no_fc
        self.num_classes = num_classes

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        width_mult = 1.
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 2],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
        ]

        def conv_bn(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            return _layers

        def conv_1x1_bn(inp, oup):
            _layers = nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
            init_layer(_layers[0])
            init_bn(_layers[1])
            return _layers

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(
            last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, 2)]
        # building inverted residual blocks
        self.out_channels = []
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(
                            input_channel,
                            output_channel,
                            s,
                            expand_ratio=t))
                else:
                    self.features.append(
                        block(
                            input_channel,
                            output_channel,
                            1,
                            expand_ratio=t))
                input_channel = output_channel
                self.out_channels.append(output_channel)
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.out_indices = [i for i in range(len(self.out_channels))] if out_indices is None else out_indices
        assert isinstance(self.out_indices, (tuple, list)), \
            'out_indices must be tuple or list.'
        self.last_out_channel = self.last_channel
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        if not self.no_fc:
            self.fc1 = nn.Linear(1280, 1024, bias=True)
            if self.num_classes > 0:
                self.fc_audioset = nn.Linear(1024, num_classes, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        if not self.no_fc:
            init_layer(self.fc1)
            if self.num_classes > 0:
                init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # (batch_size, 1, time_steps, freq_bins)
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        outs = OrderedDict()
        for idx, feature_layer in enumerate(self.features):
            x = feature_layer(x)
            if idx in self.out_indices:
                outs[idx] = x
        if self.features_only:
            return outs

        if self.global_pool == 'avg+max':
            x = torch.mean(x, dim=3)
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            # x = F.dropout(x, p=0.5, training=self.training)

            if self.no_fc:
                return x

            x = F.relu_(self.fc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            if self.num_classes > 0:
                clipwise_output = torch.sigmoid(self.fc_audioset(x))
                return clipwise_output
        return x


class LeeNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):

        super(LeeNetConvBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, x, pool_size=1):
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_size != 1:
            x = F.max_pool1d(x, kernel_size=pool_size, padding=pool_size // 2)
        return x


class LeeNet11(nn.Module):
    def __init__(
            self,
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            num_classes=527,
            global_pool='avg+max',
            no_fc=False,
            features_only=False,
            name=None,
            **kwargs):

        super(LeeNet11, self).__init__()
        self.name = name

        # window = 'hann'
        # center = True
        # pad_mode = 'reflect'
        # ref = 1.0
        # amin = 1e-10
        # top_db = None

        self.features_only = features_only
        self.global_pool = global_pool
        self.no_fc = no_fc
        self.num_classes = num_classes

        self.conv_block1 = LeeNetConvBlock(1, 64, 3, 3)
        self.conv_block2 = LeeNetConvBlock(64, 64, 3, 1)
        self.conv_block3 = LeeNetConvBlock(64, 64, 3, 1)
        self.conv_block4 = LeeNetConvBlock(64, 128, 3, 1)
        self.conv_block5 = LeeNetConvBlock(128, 128, 3, 1)
        self.conv_block6 = LeeNetConvBlock(128, 128, 3, 1)
        self.conv_block7 = LeeNetConvBlock(128, 128, 3, 1)
        self.conv_block8 = LeeNetConvBlock(128, 128, 3, 1)
        self.conv_block9 = LeeNetConvBlock(128, 256, 3, 1)
        self.out_indices = kwargs.pop('out_indices', None)
        if self.out_indices is None:
            self.out_indices = [i for i in range(9)]
        assert isinstance(self.out_indices, (tuple, list)), \
            'out_indices must be tuple or list.'
        self.out_channels = [64, 64, 64, 128, 128, 128, 128, 128, 256]
        self.last_out_channels = 256

        if not self.no_fc:
            self.fc1 = nn.Linear(256, 512, bias=True)
            if self.num_classes > 0:
                self.fc_audioset = nn.Linear(512, num_classes, bias=True)

        self.init_weight()

    def init_weight(self):
        if not self.no_fc:
            init_layer(self.fc1)
            if self.num_classes > 0:
                init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = input[:, None, :]

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        outs = OrderedDict()
        x = self.conv_block1(x)
        if 0 in self.out_indices:
            outs[0] = x
        x = self.conv_block2(x, pool_size=3)
        if 1 in self.out_indices:
            outs[1] = x
        x = self.conv_block3(x, pool_size=3)
        if 2 in self.out_indices:
            outs[2] = x        
        x = self.conv_block4(x, pool_size=3)
        if 3 in self.out_indices:
            outs[3] = x        
        x = self.conv_block5(x, pool_size=3)
        if 4 in self.out_indices:
            outs[4] = x        
        x = self.conv_block6(x, pool_size=3)
        if 5 in self.out_indices:
            outs[5] = x        
        x = self.conv_block7(x, pool_size=3)
        if 6 in self.out_indices:
            outs[6] = x        
        x = self.conv_block8(x, pool_size=3)
        if 7 in self.out_indices:
            outs[7] = x
        x = self.conv_block9(x, pool_size=3)
        if 8 in self.out_indices:
            outs[8] = x
        if self.features_only:
            return outs

        if self.global_pool == 'avg+max':
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)

            if self.no_fc:
                return x

            x = F.relu_(self.fc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            if self.num_classes > 0:
                clipwise_output = torch.sigmoid(self.fc_audioset(x))
                return clipwise_output
        return x


class LeeNetConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):

        super(LeeNetConvBlock2, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, bias=False)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x, pool_size=1):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_size != 1:
            x = F.max_pool1d(x, kernel_size=pool_size, padding=pool_size // 2)
        return x


class LeeNet24(nn.Module):
    def __init__(
            self,
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            num_classes=527,
            global_pool='avg+max',
            no_fc=False,
            features_only=False,
            name=None,
            **kwargs):

        super(LeeNet24, self).__init__()
        self.name = name

        # window = 'hann'
        # center = True
        # pad_mode = 'reflect'
        # ref = 1.0
        # amin = 1e-10
        # top_db = None

        self.features_only = features_only
        self.global_pool = global_pool
        self.no_fc = no_fc
        self.num_classes = num_classes

        self.conv_block1 = LeeNetConvBlock2(1, 64, 3, 3)
        self.conv_block2 = LeeNetConvBlock2(64, 96, 3, 1)
        self.conv_block3 = LeeNetConvBlock2(96, 128, 3, 1)
        self.conv_block4 = LeeNetConvBlock2(128, 128, 3, 1)
        self.conv_block5 = LeeNetConvBlock2(128, 256, 3, 1)
        self.conv_block6 = LeeNetConvBlock2(256, 256, 3, 1)
        self.conv_block7 = LeeNetConvBlock2(256, 512, 3, 1)
        self.conv_block8 = LeeNetConvBlock2(512, 512, 3, 1)
        self.conv_block9 = LeeNetConvBlock2(512, 1024, 3, 1)
        self.out_indices = kwargs.pop('out_indices', None)
        if self.out_indices is None:
            self.out_indices = [i for i in range(9)]
        assert isinstance(self.out_indices, (tuple, list)), \
            'out_indices must be tuple or list.'
        self.out_channels = [64, 96, 128, 128, 256, 256, 512, 512, 1024]
        self.last_out_channels = 1024

        if not self.no_fc:
            self.fc1 = nn.Linear(1024, 1024, bias=True)
            if self.num_classes > 0:
                self.fc_audioset = nn.Linear(1024, num_classes, bias=True)

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = input[:, None, :]

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        outs = OrderedDict()
        x = self.conv_block1(x)
        x = F.dropout(x, p=0.1, training=self.training)
        if 0 in self.out_indices:
            outs[0] = x
        x = self.conv_block2(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        if 1 in self.out_indices:
            outs[1] = x        
        x = self.conv_block3(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        if 2 in self.out_indices:
            outs[2] = x        
        x = self.conv_block4(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        if 3 in self.out_indices:
            outs[3] = x        
        x = self.conv_block5(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        if 4 in self.out_indices:
            outs[4] = x        
        x = self.conv_block6(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        if 5 in self.out_indices:
            outs[5] = x        
        x = self.conv_block7(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        if 6 in self.out_indices:
            outs[6] = x        
        x = self.conv_block8(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        if 7 in self.out_indices:
            outs[7] = x        
        x = self.conv_block9(x, pool_size=1)
        if 8 in self.out_indices:
            outs[8] = x
        if self.features_only:
            return outs

        if self.global_pool == 'avg+max':
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)

            if self.no_fc:
                return x

            x = F.relu_(self.fc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            if self.num_classes > 0:
                clipwise_output = torch.sigmoid(self.fc_audioset(x))
                return clipwise_output


class DaiNetResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):

        super(DaiNetResBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, bias=False)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, bias=False)

        self.conv3 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, bias=False)

        self.conv4 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, bias=False)

        self.downsample = nn.Conv1d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1, stride=1,
                                    padding=0, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.bn4 = nn.BatchNorm1d(out_channels)
        self.bn_downsample = nn.BatchNorm1d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.downsample)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)
        nn.init.constant_(self.bn4.weight, 0)
        init_bn(self.bn_downsample)

    def forward(self, input, pool_size=1):
        x = F.relu_(self.bn1(self.conv1(input)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.relu_(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        if input.shape == x.shape:
            x = F.relu_(x + input)
        else:
            x = F.relu(x + self.bn_downsample(self.downsample(input)))

        if pool_size != 1:
            x = F.max_pool1d(x, kernel_size=pool_size, padding=pool_size // 2)
        return x


class DaiNet19(nn.Module):
    def __init__(
            self,
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            num_classes=527,
            global_pool='avg+max',
            no_fc=False,
            features_only=False,
            name=None,
            **kwargs):

        super(DaiNet19, self).__init__()
        self.name = name

        # window = 'hann'
        # center = True
        # pad_mode = 'reflect'
        # ref = 1.0
        # amin = 1e-10
        # top_db = None

        self.features_only = features_only
        self.global_pool = global_pool
        self.no_fc = no_fc
        self.num_classes = num_classes

        self.conv0 = nn.Conv1d(in_channels=1, out_channels=64,
                               kernel_size=80, stride=4, padding=0, bias=False)
        self.bn0 = nn.BatchNorm1d(64)
        self.conv_block1 = DaiNetResBlock(64, 64, 3)
        self.conv_block2 = DaiNetResBlock(64, 128, 3)
        self.conv_block3 = DaiNetResBlock(128, 256, 3)
        self.conv_block4 = DaiNetResBlock(256, 512, 3)
        self.out_indices = kwargs.pop('out_indices', None)
        if self.out_indices is None:
            self.out_indices = [i for i in range(4)]
        assert isinstance(self.out_indices, (tuple, list)), \
            'out_indices must be tuple or list.'
        self.out_channels = [64, 128, 256, 512]
        self.last_out_channels = 512

        if not self.no_fc:
            self.fc1 = nn.Linear(512, 512, bias=True)
            if self.num_classes > 0:
                self.fc_audioset = nn.Linear(512, num_classes, bias=True)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv0)
        init_bn(self.bn0)
        if not self.no_fc:
            init_layer(self.fc1)
            if self.num_classes > 0:
                init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = input[:, None, :]

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        outs = OrderedDict()
        x = self.bn0(self.conv0(x))
        x = self.conv_block1(x)
        x = F.max_pool1d(x, kernel_size=4)
        if 0 in self.out_indices:
            outs[0] = x        
        x = self.conv_block2(x)
        x = F.max_pool1d(x, kernel_size=4)
        if 1 in self.out_indices:
            outs[1] = x    
        x = self.conv_block3(x)
        x = F.max_pool1d(x, kernel_size=4)
        if 2 in self.out_indices:
            outs[2] = x
        x = self.conv_block4(x)
        if 3 in self.out_indices:
            outs[3] = x
        if self.features_only:
            return outs

        if self.global_pool == 'avg+max':
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)
            
            if self.no_fc:
                return x

            x = F.relu_(self.fc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            if self.num_classes > 0:
                clipwise_output = torch.sigmoid(self.fc_audioset(x))
                return clipwise_output
        return x


def _resnet_conv3x1_wav1d(in_planes, out_planes, dilation):
    # 3x3 convolution with padding
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=dilation, groups=1, bias=False, dilation=dilation)


def _resnet_conv1x1_wav1d(in_planes, out_planes):
    # 1x1 convolution
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=1,
        bias=False)


class _ResnetBasicBlockWav1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBasicBlockWav1d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError(
                '_ResnetBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in _ResnetBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when
        # stride != 1

        self.stride = stride

        self.conv1 = _resnet_conv3x1_wav1d(inplanes, planes, dilation=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _resnet_conv3x1_wav1d(planes, planes, dilation=2)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride != 1:
            out = F.max_pool1d(x, kernel_size=self.stride)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResNetWav1d(nn.Module):
    def __init__(
            self,
            block,
            layers,
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None):
        super(_ResNetWav1d, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=4)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=4)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=4)
        self.layer5 = self._make_layer(block, 1024, layers[4], stride=4)
        self.layer6 = self._make_layer(block, 1024, layers[5], stride=4)
        self.layer7 = self._make_layer(block, 2048, layers[6], stride=4)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    _resnet_conv1x1_wav1d(
                        self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[0])
                init_bn(downsample[1])
            else:
                downsample = nn.Sequential(
                    nn.AvgPool1d(kernel_size=stride),
                    _resnet_conv1x1_wav1d(
                        self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[1])
                init_bn(downsample[2])

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

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        return x


class Res1dNet31(nn.Module):
    def __init__(
            self,
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            num_classes=527,
            global_pool='avg+max',
            no_fc=False,
            name=None,
            **kwargs):

        super(Res1dNet31, self).__init__()
        self.name = name

        # window = 'hann'
        # center = True
        # pad_mode = 'reflect'
        # ref = 1.0
        # amin = 1e-10
        # top_db = None

        self.global_pool = global_pool
        self.no_fc = no_fc
        self.num_classes = num_classes

        self.conv0 = nn.Conv1d(in_channels=1, out_channels=64,
                               kernel_size=11, stride=5, padding=5, bias=False)
        self.bn0 = nn.BatchNorm1d(64)

        self.resnet = _ResNetWav1d(
            _ResnetBasicBlockWav1d, [2, 2, 2, 2, 2, 2, 2])
        self.last_out_channels = 2048

        if not self.no_fc:
            self.fc1 = nn.Linear(2048, 2048, bias=True)
            if self.num_classes > 0:
                self.fc_audioset = nn.Linear(2048, num_classes, bias=True)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv0)
        init_bn(self.bn0)
        if not self.no_fc:
            init_layer(self.fc1)
            if self.num_classes > 0:
                init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = input[:, None, :]

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.bn0(self.conv0(x))
        x = self.resnet(x)

        if self.global_pool == 'avg+max':
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)

            if self.no_fc:
                return x

            x = F.relu_(self.fc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            if self.num_classes > 0:
                clipwise_output = torch.sigmoid(self.fc_audioset(x))
                return clipwise_output
        return x


class Res1dNet51(nn.Module):
    def __init__(
            self,
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            num_classes=527,
            global_pool='avg+max',
            no_fc=False,
            name=None,
            **kwargs):

        super(Res1dNet51, self).__init__()
        self.name = name

        # window = 'hann'
        # center = True
        # pad_mode = 'reflect'
        # ref = 1.0
        # amin = 1e-10
        # top_db = None

        self.global_pool = global_pool
        self.no_fc = no_fc
        self.num_classes = num_classes

        self.conv0 = nn.Conv1d(in_channels=1, out_channels=64,
                               kernel_size=11, stride=5, padding=5, bias=False)
        self.bn0 = nn.BatchNorm1d(64)

        self.resnet = _ResNetWav1d(
            _ResnetBasicBlockWav1d, [2, 3, 4, 6, 4, 3, 2])
        self.last_out_channels = 2048

        if not self.no_fc:
            self.fc1 = nn.Linear(2048, 2048, bias=True)
            if self.num_classes > 0:
                self.fc_audioset = nn.Linear(2048, num_classes, bias=True)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv0)
        init_bn(self.bn0)
        if not self.no_fc:
            init_layer(self.fc1)
            if self.num_classes > 0:
                init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = input[:, None, :]

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.bn0(self.conv0(x))
        x = self.resnet(x)

        if self.global_pool == 'avg+max':
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)

            if self.no_fc:
                return x

            x = F.relu_(self.fc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            if self.num_classes > 0:
                clipwise_output = torch.sigmoid(self.fc_audioset(x))
                return clipwise_output
        return x


class ConvPreWavBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvPreWavBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1, dilation=2,
                               padding=2, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=pool_size)

        return x


class Wavegram_Cnn14(nn.Module):
    def __init__(
            self,
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            num_classes=527,
            global_pool='avg+max',
            no_fc=False,
            name=None,
            **kwargs):

        super(Wavegram_Cnn14, self).__init__()
        self.name = name

        # window = 'hann'
        # center = True
        # pad_mode = 'reflect'
        # ref = 1.0
        # amin = 1e-10
        # top_db = None

        self.global_pool = global_pool
        self.no_fc = no_fc
        self.num_classes = num_classes

        self.pre_conv0 = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=11,
            stride=5,
            padding=5,
            bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 128)
        self.pre_block4 = ConvBlock(in_channels=4, out_channels=64)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        self.last_out_channels = 2048

        if not self.no_fc:
            self.fc1 = nn.Linear(2048, 2048, bias=True)
            if self.num_classes > 0:
                self.fc_audioset = nn.Linear(2048, num_classes, bias=True)

        self.init_weight()

    def init_weight(self):
        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)
        init_bn(self.bn0)
        if not self.no_fc:
            init_layer(self.fc1)
            if self.num_classes > 0:
                init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # Wavegram
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(input[:, None, :])))
        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=4)
        a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
        a1 = self.pre_block4(a1, pool_size=(2, 1))

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            a1 = do_mixup(a1, mixup_lambda)

        x = a1
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        if self.global_pool == 'avg+max':
            x = torch.mean(x, dim=3)
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)

            if self.no_fc:
                return x

            x = F.relu_(self.fc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            if self.num_classes > 0:
                clipwise_output = torch.sigmoid(self.fc_audioset(x))
                return clipwise_output
        return x


class Wavegram_Logmel_Cnn14(nn.Module):
    def __init__(
            self,
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            num_classes=527,
            global_pool='avg+max',
            no_fc=False,
            features_only=False,
            name=None,
            **kwargs):

        super(Wavegram_Logmel_Cnn14, self).__init__()
        self.name = name

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.features_only = features_only
        self.global_pool = global_pool
        self.no_fc = no_fc
        self.num_classes = num_classes

        self.pre_conv0 = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=11,
            stride=5,
            padding=5,
            bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 128)
        self.pre_block4 = ConvBlock(in_channels=4, out_channels=64)

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=128, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        self.out_channels = [64, 128, 256, 512, 1024, 2048]
        self.last_out_channels = 2048
        self.out_indices = kwargs.pop('out_indices', None)
        if self.out_indices is None:
            self.out_indices = [i for i in range(6)]
        assert isinstance(self.out_indices, (tuple, list)), \
            'out_indices must be tuple or list.'

        if not self.no_fc:
            self.fc1 = nn.Linear(2048, 2048, bias=True)
            if self.num_classes > 0:
                self.fc_audioset = nn.Linear(2048, num_classes, bias=True)

        self.init_weight()

    def init_weight(self):
        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)
        init_bn(self.bn0)
        if not self.no_fc:
            init_layer(self.fc1)
            if self.num_classes > 0:
                init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # Wavegram
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(input[:, None, :])))
        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=4)
        a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
        a1 = self.pre_block4(a1, pool_size=(2, 1))

        # Log mel spectrogram
        # (batch_size, 1, time_steps, freq_bins)
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
            a1 = do_mixup(a1, mixup_lambda)

        outs = OrderedDict()
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')

        # Concatenate Wavegram and Log mel spectrogram along the channel
        # dimension
        x = torch.cat((x, a1), dim=1)
        x = F.dropout(x, p=0.2, training=self.training)
        if 0 in self.out_indices:
            outs[0] = x
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        if 1 in self.out_indices:
            outs[1] = x        
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        if 2 in self.out_indices:
            outs[2] = x
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        if 3 in self.out_indices:
            outs[3] = x
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        if 4 in self.out_indices:
            outs[4] = x
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        if 5 in self.out_indices:
            outs[5] = x

        if self.global_pool == 'avg+max':
            x = torch.mean(x, dim=3)
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)

            if self.no_fc:
                return x

            x = F.relu_(self.fc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            if self.num_classes > 0:
                clipwise_output = torch.sigmoid(self.fc_audioset(x))
                return clipwise_output
        return x


@register_model
def audio_cnn6(pretrained=False, **kwargs):
    model_name = 'audio_cnn6'
    pretrained_config, model_config = get_configs(**kwargs)
    model = Cnn6(name=model_name, **model_config)

    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model is loaded successfully.")
    return model


@register_model
def audio_cnn10(pretrained=False, **kwargs):
    model_name = 'audio_cnn10'
    pretrained_config, model_config = get_configs(**kwargs)
    model = Cnn10(name=model_name, **model_config)

    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model is loaded successfully.")
    return model


@register_model
def audio_cnn14(pretrained=False, **kwargs):
    model_name = 'audio_cnn14'
    pretrained_config, model_config = get_configs(**kwargs)
    model = Cnn14(name=model_name, **model_config)

    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model is loaded successfully.")
    return model


@register_model
def audio_resnet22(pretrained=False, **kwargs):
    model_name = 'audio_resnet22'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config.pop('features_only', False):
        raise ValueError('Features_only is not supported for audio_resnet22.')
    model = ResNet22(name=model_name, **model_config)

    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model is loaded successfully.")
    return model


@register_model
def audio_resnet38(pretrained=False, **kwargs):
    model_name = 'audio_resnet38'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config.pop('features_only', False):
        raise ValueError('Features_only is not supported for audio_resnet38.')
    model = ResNet38(name=model_name, **model_config)

    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model is loaded successfully.")
    return model


@register_model
def audio_resnet54(pretrained=False, **kwargs):
    model_name = 'audio_resnet54'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config.pop('features_only', False):
        raise ValueError('Features_only is not supported for audio_resnet54.')
    model = ResNet54(name=model_name, **model_config)

    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model is loaded successfully.")
    return model


@register_model
def audio_mobilenetv1(pretrained=False, **kwargs):
    model_name = 'audio_mobilenetv1'
    pretrained_config, model_config = get_configs(**kwargs)
    model = MobileNetV1(name=model_name, **model_config)

    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model is loaded successfully.")
    return model


@register_model
def audio_mobilenetv2(pretrained=False, **kwargs):
    model_name = 'audio_mobilenetv2'
    pretrained_config, model_config = get_configs(**kwargs)
    model = MobileNetV2(name=model_name, **model_config)

    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model is loaded successfully.")
    return model


@register_model
def audio_leenet11(pretrained=False, **kwargs):
    model_name = 'audio_leenet11'
    pretrained_config, model_config = get_configs(**kwargs)
    model = LeeNet11(name=model_name, **model_config)

    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model is loaded successfully.")
    return model


@register_model
def audio_dainet19(pretrained=False, **kwargs):
    model_name = 'audio_dainet19'
    pretrained_config, model_config = get_configs(**kwargs)
    model = DaiNet19(name=model_name, **model_config)

    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model is loaded successfully.")
    return model


@register_model
def audio_res1dnet31(pretrained=False, **kwargs):
    model_name = 'audio_res1dnet31'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config.pop('features_only', False):
        raise ValueError(
            'Features_only is not supported for audio_res1dnet31.')
    model = Res1dNet31(name=model_name, **model_config)

    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model is loaded successfully.")
    return model


@register_model
def audio_res1dnet51(pretrained=False, **kwargs):
    model_name = 'audio_res1dnet51'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config.pop('features_only', False):
        raise ValueError(
            'Features_only is not supported for audio_res1dnet51.')
    model = Res1dNet51(model_name, **model_config)

    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model is loaded successfully.")
    return model


@register_model
def audio_wavegram_cnn14(pretrained=False, **kwargs):
    model_name = 'audio_wavegram_cnn14'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config.pop('features_only', False):
        raise ValueError(
            'Features_only is not supported for audio_wavegram_cnn14.')
    model = Wavegram_Cnn14(name=model_name, **model_config)

    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model is loaded successfully.")
    return model


@register_model
def audio_wavegram_logmel_cnn14(pretrained=False, **kwargs):
    model_name = 'audio_wavegram_logmel_cnn14'
    model = Wavegram_Logmel_Cnn14(name=model_name, **kwargs)

    if pretrained:
        logger.warning("Not available, check back later.")
    return model



if __name__ == '__main__':

    import torch 
    model = audio_wavegram_logmel_cnn14(pretrained=False, features_only=False)

    dummy_input = torch.randn(2,640000)
    out=model(dummy_input)
    print(out.shape)
