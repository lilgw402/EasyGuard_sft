# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Auto Tokenizer class."""

from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

from ...modelzoo.tokenization_utils_base import TOKENIZER_CONFIG_FILE
from ...utils import logging, sha256, lazy_model_import, cache_file, file_read
from .auto_factory import _LazyAutoMapping
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    AutoConfig,
)
from . import (
    BACKENDS,
    MODELZOO_CONFIG,
    MODEL_ARCHIVE_CONFIG,
    VOCAB_NAME,
    TOKENIZER_NAMES,
)

logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    # This significantly improves completion suggestion performance when
    # the transformers package is used with Microsoft's Pylance language server.
    TOKENIZER_MAPPING_NAMES: OrderedDict[
        str, Tuple[Optional[str], Optional[str]]
    ] = OrderedDict()
else:
    TOKENIZER_MAPPING_NAMES = MODELZOO_CONFIG.get_mapping("tokenizer", "tokenizer_fast")

TOKENIZER_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TOKENIZER_MAPPING_NAMES)

CONFIG_TO_TYPE = {v: k for k, v in CONFIG_MAPPING_NAMES.items()}


class AutoTokenizer:
    def __init__(self) -> None:
        raise EnvironmentError(
            "AutoTokenizer is designed to be instantiated "
            "using the `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_config(cls, config_path: str, *inputs, **kwargs):
        # TODO (junwei.Dong): 有待开发
        ...

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *inputs, **kwargs):
        """

        Parameters
        ----------
        pretrained_model_name_or_path : str
            _description_
        """
        if pretrained_model_name_or_path not in MODEL_ARCHIVE_CONFIG:
            # if the `model_name_or_path` is not in `MODEL_ARCHIVE_CONFIG`, what we can do
            raise KeyError(pretrained_model_name_or_path)
        else:
            model_archive = MODEL_ARCHIVE_CONFIG[pretrained_model_name_or_path]
            model_type = model_archive.get("type", None)
            model_url = model_archive.get("url_or_path", None)
            model_config = MODELZOO_CONFIG.get(model_type, None)
            assert (
                model_config is not None
            ), f"the target model `{model_type}` does not exist, please check the modelzoo or the config yaml~"

            backend = model_config.get("backend", None)
            assert backend in BACKENDS, f"backend should be one of f{BACKENDS}"
            backend_default_flag = False

            if backend == "hf":
                from .tokenization_auto_hf import HFTokenizer

                return HFTokenizer.from_pretrained(
                    pretrained_model_name_or_path, *inputs, **kwargs
                )
            elif backend == "titan":
                # TODO (junwei.Dong): 支持特殊的titan模型
                raise NotImplementedError(backend)
            elif backend == "fex":
                # TODO (junwei.Dong): 支持特殊的fex模型
                raise NotImplementedError(backend)
            else:
                backend_default_flag = True

            if backend_default_flag == True:
                # just support base tokenizer
                # lazily obtain tokenizer class
                tokenizer_name_tuple = MODELZOO_CONFIG[model_type]["tokenizer"]
                (
                    tokenizer_module_package,
                    tokenizer_module_name,
                ) = MODELZOO_CONFIG.to_module(tokenizer_name_tuple)
                tokenizer_class = lazy_model_import(
                    tokenizer_module_package, tokenizer_module_name
                )
                extra_dict = {
                    "model_type": model_type,
                    "remote_url": model_url,
                }
                # obtain vocab file path
                vocab_file_path = cache_file(
                    pretrained_model_name_or_path, VOCAB_NAME, **extra_dict
                )

                # obtain tokenizer config file path
                tokenizer_config_file_path = cache_file(
                    pretrained_model_name_or_path, TOKENIZER_NAMES, **extra_dict
                )
                tokenizer_config = file_read(tokenizer_config_file_path)
                return tokenizer_class(vocab_file_path, **tokenizer_config)
        ...