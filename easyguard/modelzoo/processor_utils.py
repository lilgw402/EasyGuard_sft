from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import PIL
import torch

from .. import __version__
from ..utils import hexists, hopen, logging
from .hub import AutoHubClass

logger = logging.get_logger(__name__)

# image default setting
ImageInput = Union[
    "PIL.Image.Image",
    np.ndarray,
    "torch.Tensor",
    List["PIL.Image.Image"],
    List[np.ndarray],
    List["torch.Tensor"],
]  # noqa

# mean and std come from ImageNet dataset
MEAN_IMAGE = [0.485, 0.456, 0.406]
STD_IMAGE = [0.229, 0.224, 0.225]
RESIZE_IMAGE = 256
CENTER_IMAGE = 224


# for image preprocess
class ProcessorImageBase(ABC, AutoHubClass):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, image, **kwds: Any) -> Any:
        return self.preprocess(image, **kwds)

    @abstractmethod
    def preprocess(self, image, **kwds: Any):
        """for image data preprocessing"""


# for auto preprocess
class ProcessorBase(ABC, AutoHubClass):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, text=None, image=None, **kwds: Any) -> Any:
        return self.preprocess(text=text, image=image, **kwds)

    def preprocess(self, text=None, image=None, **kwds):
        """for image and text data processing"""

        if not text and not image:
            raise ValueError(f"text and image are all None, please check~")
        data_pre = OrderedDict(text_pre=None, image_pre=None)
        if hasattr(self, "text_processor"):
            data_pre["text_pre"] = (
                self.text_processor(text, **kwds) if text else None
            )
        else:
            logger.warning(f"text processor not exist, please check")

        if hasattr(self, "image_processor"):
            data_pre["image_pre"] = (
                self.image_processor(image, **kwds) if image else None
            )
        else:
            logger.warning(f"image processor not exist, please check")

        return data_pre
