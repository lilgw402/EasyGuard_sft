from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ....utils import typecheck
from ...configuration_utils import ConfigBase


class FalBertConfig(ConfigBase):
    @typecheck(object, dict, dict)
    def __init__(
        self,
        text_config: Dict[str, Any],
        vision_config: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.text_config_dict = text_config
        self.vision_config_dict = vision_config
        self.text_processor = FalBertTextConfig(**self.text_config_dict)
        self.image_processor = FalBertVisionConfig(**self.vision_config_dict)

    def config_update_for_pretrained(self, **kwargs):
        ...


@dataclass
class FalBertVisionConfig(ConfigBase):
    visual_type: str
    img_size: int
    num_classes: int
    embed_dim: int
    depths: List[int]
    num_heads: List[int]


@dataclass
class FalBertTextConfig(ConfigBase):
    model_name: str
    vocab_size: int
    embedding_size: int
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    project_embedding_first: bool
    max_position_embeddings: int
    type_vocab_size: int
    hidden_dropout_prob: float
    need_visual_ln: bool
    max_frame_num: int
    visual_front: bool
    with_pooler: bool
    initializer_range: float
    visual_dim: int
    layernorm_eps: float
    freeze_prefix: list
