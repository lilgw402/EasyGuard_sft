from dataclasses import dataclass
from typing import Any, List

import torch
from matplotlib.pyplot import isinteractive
from torchvision.transforms import transforms

from ...processor_utils import (
    CENTER_IMAGE,
    MEAN_IMAGE,
    RESIZE_IMAGE,
    STD_IMAGE,
    ImageInput,
    ProcessorImageBase,
)


@dataclass
class FashionSwinProcessor(ProcessorImageBase):
    mean: List[float]
    std: List[float]
    size: float
    center: float

    def preprocess(self, image: ImageInput, **kwds: Any):
        """preprocess image

        Parameters
        ----------
        image : ImageInput
            the target image

        Returns
        -------
        _type_
            _description_
        """
        resize = transforms.Resize(self.size)
        center_crop = transforms.CenterCrop(self.center)
        to_tensor = (
            transforms.ToTensor() if not torch.is_tensor(image) else None
        )
        normalize = transforms.Normalize(mean=self.mean, std=self.std)

        funcs = [
            resize,
            center_crop,
            to_tensor,
            normalize,
        ]

        transform_funcs = transforms.Compose([_ for _ in funcs if _])
        return transform_funcs(image)
