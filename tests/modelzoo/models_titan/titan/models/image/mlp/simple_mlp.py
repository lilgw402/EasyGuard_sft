from typing import List
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

from titan.models.components import SEBlock, SEBNBlock
from titan.utils.registry import register_model
from titan.utils.helper import download_weights, load_pretrained_model_weights
from titan.utils.logger import logger

__all__ = [
    'simple_mlp',
]

class MNISTModel(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, hidden_size)
        nn.init.xavier_uniform(self.layer_1.weight)
        self.layer_2 = nn.Linear(hidden_size, 10)
        nn.init.xavier_uniform(self.layer_1.weight)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        return x


@register_model
def simple_mlp(pretrained=False, **kwargs):
    model = MNISTModel()

    return model