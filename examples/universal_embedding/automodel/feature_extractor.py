# --------------------------------------------------------
# AutoModel Infer
# Fashion Universal
# Copyright (c) 2023 EasyGuard
# Written by yangmin.priv
# --------------------------------------------------------

import sys
import os

from PIL import Image

import torch
import torchvision.transforms as transforms

try:
    import easyguard
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
from easyguard import AutoModel, AutoImageProcessor

model = AutoModel.from_pretrained("fashion-universal-vit-base-224")
# model = AutoModel.from_pretrained("fashion-universal-product-vit-base-224")
print(model)
model.eval()

dummy_input = torch.ones(1, 3, 224, 224)
dummy_output = model(dummy_input)
print(dummy_output.size())

# infer image 
image = Image.open("0.jpg").convert("RGB")
image_processor = AutoImageProcessor.from_pretrained("fashion-universal-vit-base-224")

input_tensor = image_processor(image).unsqueeze(0)
output = model(input_tensor)
print(output.size())