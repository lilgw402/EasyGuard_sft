import torch
import torch.nn as nn


#类似于 OpenAI CLIP 模型中视觉部分的实现
class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False): #`vision_tower`: 这个参数表示需要加载的预训练视觉模型的名称
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.language = args.language
        if args.language == 'chinese':
            from transformers import ChineseCLIPConfig as CLIPVisionConfig
        else:
            from transformers import CLIPVisionConfig
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        if self.language == 'chinese':
            from transformers import ChineseCLIPVisionModel as CLIPVisionModel
            from transformers import ChineseCLIPImageProcessor as CLIPImageProcessor
        else:
            from transformers import CLIPVisionModel, CLIPImageProcessor
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True #设置了一个标记 `self.is_loaded` 为 `True` 以表明模型已经被加载。

    #特征选择 (`feature_select` 方法):
    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    #使用无梯度计算（`torch.no_grad()`）来避免在前向传播过程计算不必要的梯度
    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self): #返回一个虚构的零特征向量，其大小由 `hidden_size` 属性决定
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self): #返回构建好的视觉塔的数据类型
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self): #返回视觉塔的配置
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
