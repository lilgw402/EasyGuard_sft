import torch
from torch import nn
import torch.nn.functional as F


class MyFCHead(nn.Module):
    def __init__(self, in_channels, fc_channels):
        super(MyFCHead, self).__init__()

        if fc_channels < 0:
            self.fc = None
            self.out_channels = in_channels
        else:
            self.fc = nn.Linear(in_channels, fc_channels)
            self.out_channels = fc_channels

    def forward(self, x):
        if self.fc is None:
            return x
        else:
            return F.relu(self.fc(x))


class MyClsHead(nn.Module):

    def __init__(self, in_channels, num_classes, dp_ratio=0):
        super(MyClsHead, self).__init__()

        if isinstance(num_classes, list):
            m = nn.ModuleList()
            for c in num_classes:
                m.append(nn.Linear(in_channels, c))
        else:
            m = nn.Linear(in_channels, num_classes)

        self.m = m
        self.dp_ratio=dp_ratio
        self.dp=nn.Dropout(self.dp_ratio)

    def forward(self, x):
        if isinstance(self.m, nn.ModuleList):
            o_list = []
            for mi in self.m:
                o_list.append(mi(x))
            return o_list
        else:
            o = self.m(x)
            if self.dp_ratio>0:
                o=self.dp(o)
            return [o]