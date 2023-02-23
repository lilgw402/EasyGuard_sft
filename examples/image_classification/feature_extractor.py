# --------------------------------------------------------
# AutoModel Infer
# Swin-Transformer
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
    
from easyguard import AutoModel

model = AutoModel.from_pretrained("fashion-swin-base-224-fashionvtp")
print(model)
model.eval()

dummy_input = torch.ones(1, 3, 224, 224)
dummy_output = model(dummy_input)
print(dummy_output.size())

# infer image 
image = Image.open("examples/image_classification/ptms.png").convert("RGB")

transform_funcs = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
                                     ])
input = transform_funcs(image)
input_tensor = input.unsqueeze(0)
output = model(input_tensor)
print(output.size())