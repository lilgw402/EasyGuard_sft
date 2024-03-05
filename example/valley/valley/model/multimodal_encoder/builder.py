import os
from .clip_encoder import CLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None)) #获取视觉塔的配置
    is_absolute_path_exists = os.path.exists(vision_tower) #检测配置中提供的路径是否存在
    if getattr(vision_tower_cfg, 'language', None) is None: #设定语言配置
        vision_tower_cfg.language = 'chinese' if 'chinese' in vision_tower else 'xl'
    print(f'region: {vision_tower_cfg.language}')
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("OFA-Sys") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs) #构建视觉塔实例

    raise ValueError(f'Unknown vision tower: {vision_tower}')
