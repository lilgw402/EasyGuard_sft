from dataclasses import dataclass

from ...configuration_utils import ConfigBase


@dataclass
class FalBertConfig(ConfigBase):
    model_name: str

    def __post_init__(self):
        super().__init__(**self.__dict__)

    def config_update_for_pretrained(self, **kwargs):
        ...
