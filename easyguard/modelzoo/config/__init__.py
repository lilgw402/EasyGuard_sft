import os

from typing import Dict, Any, List
from collections import OrderedDict

from ...utils.yaml_utils import YamlConfig

MODEL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "models.yaml")
MODEZOO_NAME = "models"
YAML_DEEP = 3


class ModelZooYaml(YamlConfig):
    def check(self):
        """check modelzoo config yaml:
        1. the deepest level is 4
        2. for a specific model, each key has an unique name

        Returns
        -------

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        global MODEZOO_NAME, YAML_DEEP
        leafs = {}
        prefix = MODEZOO_NAME

        def dfs_leafs(data: Dict[str, Any], deep: int, leafs: List[str], prefix: str):

            global YAML_DEEP
            deep_ = deep + 1
            for key_item, value in data.items():
                prefix_ = f"{prefix}.{key_item}"
                if isinstance(value, dict):
                    if deep_ > YAML_DEEP:
                        raise ValueError(
                            f"the modelzoo config `{prefix_}` should not be a dict~"
                        )
                    dfs_leafs(value, deep_, leafs, f"{prefix_}")
                else:
                    leafs.append((prefix_, key_item))

        # for model_backend in self.config[MODEZOO_NAME].keys():
        for key_item, value in self.config[MODEZOO_NAME].items():
            leafs[key_item] = []
            dfs_leafs(value, 2, leafs[key_item], f"{prefix}.{key_item}")

        for key_item, value in leafs.items():
            paths, leaf_values = zip(*leafs[key_item])
            temp_dict = {}
            for index, leaf_value_item in enumerate(leaf_values):
                if leaf_value_item in temp_dict:
                    raise ValueError(
                        f"the `{paths[index]}` and `{temp_dict[leaf_value_item]}` have the same key `{leaf_value_item}`~"
                    )
                else:
                    temp_dict[leaf_value_item] = paths[index]

    def model_detail_config(self):

        model_index = 1
        self.models = {}
        for leaf_item in self.leafs:
            prefix_, key_, value_ = leaf_item
            if prefix_.startswith("models."):
                model_ = prefix_.split(".")[model_index]

                if model_ in self.models:
                    self.models[model_][key_] = (prefix_, value_)
                else:
                    self.models[model_] = {key_: (prefix_, value_)}

    def get_mapping(self, *keys: str) -> OrderedDict:
        """get mappings for huggingface models

        example:

        Returns
        -------
        OrderedDict
            a specific mapping for target keys
        """

        if not hasattr(self, "models"):
            self.model_detail_config()

        mapping = {}

        for model_, config_ in self.models.items():
            values_ = []
            for key_ in keys:
                value_ = config_.get(key_, None)
                values_.append(value_[-1] if value_ is not None else None)
            if len(values_) > 1:
                mapping[model_] = tuple(values_)
            else:
                model_value_ = values_[0]
                if model_value_ is not None:
                    mapping[model_] = model_value_

        mapping_list = [(key_item, value) for key_item, value in mapping.items()]

        return OrderedDict(mapping_list)

    def __getitem__(self, key: str):
        return super().__getitem__(key)


MODELZOO_CONFIG = ModelZooYaml.yaml_reader(MODEL_CONFIG_PATH)
MODELZOO_CONFIG.check()
