from dataclasses import dataclass
from typing import List

from ...configuration_utils import ConfigBase


class FalBertConfig(ConfigBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        text_config = kwargs.pop("text_config", None)
        vision_config = kwargs.pop("vision_config", None)

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

    def __post_init__(self):
        super().__init__(**self.__dict__)

    def config_update_for_pretrained(self, **kwargs):
        ...


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

    def __post_init__(self):
        super().__init__(**self.__dict__)

    def config_update_for_pretrained(self, **kwargs):
        ...
