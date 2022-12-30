import yaml

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class YamlConfig(ABC):
    path: str
    config: Dict[str, Any]
    leafs: List[tuple]
    name: str = field(default="Models Config", repr=False)
    docstring: str = field(default="All models config settings", repr=False)

    @abstractmethod
    def check(self):
        raise NotImplementedError()

    @classmethod
    def yaml_reader(
        cls,
        path: str,
        name: str = "modelzoo config",
        docstring: str = "it is about the modelzoo setting",
        return_yamlconfig: bool = True,
    ):

        with open(path, "r") as config_file:
            config_data = yaml.full_load(config_file)

        if return_yamlconfig:
            dfs_leafs = []
            cls.dfs_decouple(config_data, dfs_leafs)
            yamlconfig_ = cls(
                path, config_data, dfs_leafs, name=name, docstring=docstring
            )
            return yamlconfig_

        return config_data

    @classmethod
    def dfs_decouple(
        cls, data: Dict[str, Any], save_data: List[tuple], prefix: Optional[str] = None
    ):
        for key_item, value in data.items():
            prefix_ = f"{prefix}.{key_item}" if prefix else f"{key_item}"
            if isinstance(value, dict):
                cls.dfs_decouple(value, save_data, prefix_)
            else:
                save_data.append((prefix_, key_item, value))

    def __getitem__(self, key: str):
        keys_ = key.split(".")
        data_ = self.config
        for index, key_item in enumerate(keys_):
            data_ = data_.get(key_item, None)
            assert (
                data_ is not None
            ), f'the target model `{".".join(keys_[:index])}` does not exist, please check the modelzoo or the config yaml~'

        return data_

    # def __repr__(self) -> str:
    #     return f'{self.docstring}'

    def __str__(self) -> str:
        return f"{self.docstring}"


if __name__ == "__main__":
    ...
