import os

from typing import Dict, Any, List, Optional
from collections import OrderedDict

MODEL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "models.yaml")
MODEL_ARCHIVE_PATH = os.path.join(os.path.dirname(__file__), "archive.yaml")
MODEL_REMOTE_ZOO = "hdfs://haruna/home/byte_ecom_govern/easyguard/models"
MODELZOO_NAME = "models"
YAML_DEEP = 3

from ...utils.yaml_utils import YamlConfig, load_yaml

"""
config: tokenizer, vocab, model全都通过models.yaml来连接, 因此, 很多操作就可以借助models.yaml来进行简化,
例如:
模型注册: 直接将自主开发的模型一次注入到models.yaml文件里即可调用, 无需在auto各个模块进行配置
模型开发: 在模型的__init__函数里只需要利用typing.TYPE_CHECKING来辅助代码提示即可,无需手动lazyimport, 可参照deberta模型进行开发
模型懒加载: 不再需要各种mapping的存在, 因为models.yaml已经把各自模型的配置归类在一起了, 所以直接借助models.yaml即可轻松完成模块按需懒加载使用

"""


class ModelZooYaml(YamlConfig):
    @classmethod
    def to_module(cls, config: tuple) -> tuple:
        """

        Parameters
        ----------
        config : tuple
            specific config
            example:
            ('models.bert.configuration_bert.config', 'BertConfig')

        Returns
        -------
        str
            module name
            example:
                'models.bert.configuration_bert.BertConfig'
        """
        module_split = config[0].split(".")
        return ".".join(module_split[:-1]), config[-1]

    def check(self):
        """check modelzoo config yaml:
        1. the deepest level is `YAML_DEEP`.
        2. for a specific model, each key has an unique name.

        Returns
        -------

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        global MODELZOO_NAME, YAML_DEEP
        leafs = {}
        prefix = MODELZOO_NAME

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

        # for model_backend in self.config[MODELZOO_NAME].keys():
        for key_item, value in self.config[MODELZOO_NAME].items():
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
        """_summary_"""
        model_index = 1
        self.models = {}
        self.models_ = {}
        for leaf_item in self.leafs:
            prefix_, key_, value_ = leaf_item
            if prefix_.startswith("models."):
                model_ = prefix_.split(".")[model_index]

                if model_ in self.models:
                    self.models[model_][key_] = (prefix_, value_)
                    self.models_[model_][key_] = value_
                else:
                    self.models[model_] = {key_: (prefix_, value_)}
                    self.models_[model_] = {key_: value_}

    def get_mapping(self, *keys) -> OrderedDict:
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

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self.models_.get(key, default)

    def __getitem__(self, key: str):
        if not hasattr(self, "models"):
            self.model_detail_config()
        if self.models.get(key, None) is not None:
            return self.models[key]
        raise KeyError(key)


MODELZOO_CONFIG = ModelZooYaml.yaml_reader(MODEL_CONFIG_PATH)
MODEL_ARCHIVE_CONFIG = load_yaml(MODEL_ARCHIVE_PATH)
MODELZOO_CONFIG.check()
