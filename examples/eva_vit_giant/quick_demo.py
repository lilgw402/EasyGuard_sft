import torch
from torchvision import transforms

from easyguard import AutoModel


def _convert_to_rgb(image):
    return image.convert("RGB")


def img_processer():
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
    )
    trans = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            _convert_to_rgb,
            transforms.ToTensor(),
            normalize,
        ]
    )
    return trans


vit_giant = AutoModel.from_pretrained("eva_vit_giant")

procosser = img_processer()
rand_input = torch.randn(1, 3, 224, 224)
p_image = procosser(rand_input)

out = vit_giant(p_image)
print(out)
