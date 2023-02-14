# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Factory function to build auto-model classes."""
import importlib
import os
from collections import OrderedDict
from distutils.command.config import config
from http import server
from typing import Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from ...modelzoo.configuration_utils import ConfigBase
from ...modelzoo.hub import AutoHubClass
from ...modelzoo.modeling_utils import ModelBase
from ...utils import (
    cache_file,
    copy_func,
    file_exist,
    file_read,
    hf_name_or_path_check,
    lazy_model_import,
    logging,
    pretrained_model_archive_parse,
)
from . import (
    BACKENDS,
    HF_PATH,
    MODEL_ARCHIVE_CONFIG,
    MODEL_CONFIG_NAMES,
    MODEL_SAVE_NAMES,
    MODELZOO_CONFIG,
)
from .configuration_auto import CONFIG_MAPPING_NAMES, AutoConfig
from .configuration_auto_hf import model_type_to_module_name

# TODO (junwei.Dong): 需要简化一下工厂函数的逻辑

logger = logging.get_logger(__name__)


def _get_model_class(config, model_mapping):
    supported_models = model_mapping[type(config)]
    if not isinstance(supported_models, (list, tuple)):
        return supported_models

    name_to_model = {model.__name__: model for model in supported_models}
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in name_to_model:
            return name_to_model[arch]
        elif f"TF{arch}" in name_to_model:
            return name_to_model[f"TF{arch}"]
        elif f"Flax{arch}" in name_to_model:
            return name_to_model[f"Flax{arch}"]

    # If not architecture is set in the config or match the supported models, the first element of the tuple is the
    # defaults.
    return supported_models[0]


class HFBaseAutoModelClass:
    # Base class for auto models.
    _model_mapping = None

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        if hasattr(config, "auto_map") and cls.__name__ in config.auto_map:
            if not trust_remote_code:
                raise ValueError(
                    "Loading this model requires you to execute the modeling file in that repo "
                    "on your local machine. Make sure you have read the code there to avoid malicious use, then set "
                    "the option `trust_remote_code=True` to remove this error."
                )
            if kwargs.get("revision", None) is None:
                logger.warning(
                    "Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure "
                    "no malicious code has been contributed in a newer revision."
                )
            class_ref = config.auto_map[cls.__name__]
            module_file, class_name = class_ref.split(".")
            model_class = get_class_from_dynamic_module(
                config.name_or_path, module_file + ".py", class_name, **kwargs
            )
            return model_class._from_config(config, **kwargs)
        elif type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class._from_config(config, **kwargs)

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *model_args, **kwargs
    ):
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        kwargs["_from_auto"] = True
        hub_kwargs_names = [
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "resume_download",
            "revision",
            "subfolder",
            "use_auth_token",
        ]
        hub_kwargs = {
            name: kwargs.pop(name)
            for name in hub_kwargs_names
            if name in kwargs
        }
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                trust_remote_code=trust_remote_code,
                **hub_kwargs,
                **kwargs,
            )
        if hasattr(config, "auto_map") and cls.__name__ in config.auto_map:
            if not trust_remote_code:
                raise ValueError(
                    f"Loading {pretrained_model_name_or_path} requires you to execute the modeling file in that repo "
                    "on your local machine. Make sure you have read the code there to avoid malicious use, then set "
                    "the option `trust_remote_code=True` to remove this error."
                )
            if hub_kwargs.get("revision", None) is None:
                logger.warning(
                    "Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure "
                    "no malicious code has been contributed in a newer revision."
                )
            class_ref = config.auto_map[cls.__name__]
            module_file, class_name = class_ref.split(".")
            model_class = get_class_from_dynamic_module(
                pretrained_model_name_or_path,
                module_file + ".py",
                class_name,
                **hub_kwargs,
                **kwargs,
            )
            return model_class.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                **hub_kwargs,
                **kwargs,
            )
        elif type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                **hub_kwargs,
                **kwargs,
            )
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

    @classmethod
    def register(cls, config_class, model_class):
        """
        Register a new model for this class.

        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            model_class ([`PreTrainedModel`]):
                The model to register.
        """
        if (
            hasattr(model_class, "config_class")
            and model_class.config_class != config_class
        ):
            raise ValueError(
                "The model class you are passing has a `config_class` attribute that is not consistent with the "
                f"config class you passed (model has {model_class.config_class} and you passed {config_class}. Fix "
                "one of those so they match!"
            )
        cls._model_mapping.register(config_class, model_class)


class _BaseAutoModelClass:
    # Base class for auto models.
    _model_mapping = None
    _model_key = None

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        ...

    @classmethod
    def from_pretrained_(
        cls,
        pretrained_model_name_or_path: str,
        region: Optional[str] = "CN",
        model_cls: Optional[str] = "model",
        *model_args,
        **kwargs,
    ):
        """instatiate a model class from a pretrained model name

        Parameters
        ----------
        pretrained_model_name_or_path : str
            the pretrained model name or local path
        region : Optional[str], optional
            avaiable region, CN (China), VA (oversea), CN/VA (both), by default "CN"
        model_cls : Optional[str], optional
            different categories of models, such as "model", "sequence_model", which can be found in unique_key of models.yaml, by default "model"

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        KeyError
            _description_
        NotImplementedError
            _description_
        NotImplementedError
            _description_
        """
        model_url = None
        server_name = None
        is_local = False
        config_path = None
        extra_dict = {}
        if pretrained_model_name_or_path not in MODEL_ARCHIVE_CONFIG:
            # if the `model_name_or_path` is not in `MODEL_ARCHIVE_CONFIG`, what we can do
            if os.path.exists(pretrained_model_name_or_path) and os.path.isdir(
                pretrained_model_name_or_path
            ):
                config_path = file_exist(
                    pretrained_model_name_or_path, MODEL_CONFIG_NAMES
                )
                assert (
                    config_path is not None
                ), f"please make sure the config file exist in f{pretrained_model_name_or_path}"
                config_dict = file_read(config_path)
                model_type = config_dict.get("model_type", None)
                assert (
                    model_type is not None
                ), f"please check the config file in f{pretrained_model_name_or_path}, make sure the `model_type` key exists"
                is_local = True
                extra_dict["config_path"] = config_path
            else:
                try:
                    from transformers import AutoModel

                    return AutoModel.from_pretrained(
                        pretrained_model_name_or_path, **kwargs
                    )
                except:
                    raise KeyError(pretrained_model_name_or_path)
            # raise ValueError(
            #     f"`{pretrained_model_name_or_path}` does not exist nor is not a directory"
            # )

        else:
            # parse model for integrating the url of targe server into model arhive config
            model_archive = pretrained_model_archive_parse(
                pretrained_model_name_or_path,
                MODEL_ARCHIVE_CONFIG[pretrained_model_name_or_path],
                region,
            )
            model_type = model_archive.get("type", None)
            model_url = model_archive.get("url_or_path", None)
            server_name = model_archive.get("server", None)

        # a model mapping for hf models, which is merely used to find the category of the target model
        cls._model_mapping = _LazyAutoMapping(
            CONFIG_MAPPING_NAMES, MODELZOO_CONFIG.get_mapping(model_cls)
        )
        # which is used to find the category of the target model for default models
        cls._model_key = model_cls
        model_config = MODELZOO_CONFIG.get(model_type, None)

        assert (
            model_config is not None
        ), f"the target model `{model_type}` does not exist, please check the modelzoo or the config yaml~"

        backend = model_config.get("backend", None)
        assert backend in BACKENDS, f"backend should be one of f{BACKENDS}"
        extra_dict.update(
            {
                "server_name": server_name,
                "archive_name": pretrained_model_name_or_path,
                "model_type": model_type,
                "remote_url": model_url,
                "backend": backend,
                "region": region,
            }
        )
        backend_default_flag = False
        if backend == "hf":
            HFBaseAutoModelClass._model_mapping = cls._model_mapping
            pretrained_model_name_or_path_ = (
                hf_name_or_path_check(
                    pretrained_model_name_or_path,
                    model_url,
                    model_type,
                )
                if not is_local
                else pretrained_model_name_or_path
            )
            hf_model = HFBaseAutoModelClass.from_pretrained(
                pretrained_model_name_or_path_, *model_args, **kwargs
            )
            setattr(hf_model, "extra_args", extra_dict)
            return hf_model
        elif backend == "titan":
            # TODO (junwei.Dong): 支持特殊的titan模型
            raise NotImplementedError(backend)
        elif backend == "fex":
            # TODO (junwei.Dong): 支持特殊的fex模型
            raise NotImplementedError(backend)
        else:
            backend_default_flag = True

        if backend_default_flag == True:
            model_name_tuple = MODELZOO_CONFIG[model_type][cls._model_key]
            (
                model_module_package,
                model_module_name,
            ) = MODELZOO_CONFIG.to_module(model_name_tuple)
            # obtain model class
            model_class = lazy_model_import(
                model_module_package, model_module_name
            )

            AutoHubClass.kwargs = extra_dict
            # obtain model config class
            kwargs["is_local"] = is_local
            kwargs["config_file"] = config_path
            model_config_class_: ConfigBase = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, **extra_dict, **kwargs
            )
            model_config_class_.config_update_for_pretrained(**kwargs)
            # obtain model weight file path
            model_weight_file_path = (
                cache_file(
                    pretrained_model_name_or_path,
                    MODEL_SAVE_NAMES,
                    **extra_dict,
                )
                if not is_local
                else file_exist(pretrained_model_name_or_path, MODEL_SAVE_NAMES)
            )
            # config merge
            model_config_class_.update(kwargs)
            config_dict = model_config_class_.asdict()
            config_dict.update({"config": model_config_class_})
            # instantiate model
            model_: ModelBase = model_class(**config_dict)
            setattr(model_, "extra_args", extra_dict)
            # load weights
            if model_weight_file_path:
                model_.load_pretrained_weights(model_weight_file_path, **kwargs)

            return model_

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        region: Optional[str] = "CN",
        model_cls: Optional[str] = "model",
        *model_args,
        **kwargs,
    ):
        """instatiate a model class from a pretrained model name

        Parameters
        ----------
        pretrained_model_name_or_path : str
            the pretrained model name or local path
        region : Optional[str], optional
            avaiable region, CN (China), VA (oversea), CN/VA (both), by default "CN"
        model_cls : Optional[str], optional
            different categories of models, such as "model", "sequence_model", which can be found in unique_key of models.yaml, by default "model"

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        KeyError
            _description_
        NotImplementedError
            _description_
        NotImplementedError
            _description_
        """

        """ >> initialize vars <<"""

        model_archive = None
        remote_url = None
        backend = None
        model_type = None
        model_config = None
        server_name = None
        region = region
        is_local = False
        model_config_path = None
        backend_default_flag = False
        model_weight_file_path = None

        extra_dict = OrderedDict()
        config_dict = OrderedDict()

        # which is used to find the category of the target model for default models
        cls._model_key = model_cls
        # a model mapping for hf models, which is merely used to find the category of the target model
        cls._model_mapping = _LazyAutoMapping(
            CONFIG_MAPPING_NAMES, MODELZOO_CONFIG.get_mapping(model_cls)
        )

        """ >> get model infomation <<"""

        if pretrained_model_name_or_path in MODEL_ARCHIVE_CONFIG:
            # parse model for integrating the url of targe server into model arhive config
            model_archive = pretrained_model_archive_parse(
                pretrained_model_name_or_path,
                MODEL_ARCHIVE_CONFIG[pretrained_model_name_or_path],
                region,
            )
            model_type = model_archive.get("type", None)
            remote_url = model_archive.get("url_or_path", None)
            server_name = model_archive.get("server", None)

            is_local = False
        elif os.path.exists(pretrained_model_name_or_path) and os.path.isdir(
            pretrained_model_name_or_path
        ):
            model_config_path = file_exist(
                pretrained_model_name_or_path, MODEL_CONFIG_NAMES
            )
            assert (
                model_config_path is not None
            ), f"please make sure the config file exist in f{pretrained_model_name_or_path}"

            config_dict_ = file_read(model_config_path)

            model_type = config_dict_.get("model_type", None)

            model_weight_file_path = file_exist(
                pretrained_model_name_or_path, MODEL_SAVE_NAMES
            )

            is_local = True
        else:
            logger.warning(
                "can not found model location, load from huggingface..."
            )
        # else:
        #     raise ValueError(
        #         f"`{pretrained_model_name_or_path}` should be a name from archive.yaml or a directory which contains some files about your model"
        #     )

        """ >> preprocessing: download files << """

        if model_type and not is_local:
            if not model_config_path:
                model_config_path = cache_file(
                    pretrained_model_name_or_path,
                    MODEL_CONFIG_NAMES,
                    remote_url,
                    model_type,
                    **kwargs,
                )

            if not model_weight_file_path:
                # obtain model weight file path
                model_weight_file_path = cache_file(
                    pretrained_model_name_or_path,
                    MODEL_SAVE_NAMES,
                    remote_url,
                    model_type,
                    **kwargs,
                )

        """ >> load model config class and model class <<"""

        model = None

        extra_dict.update(
            {
                "server_name": server_name,
                "archive_name": pretrained_model_name_or_path,
                "model_type": model_type,
                "remote_url": remote_url,
                "region": region,
            }
        )

        config_dict.update(
            {"is_local": is_local, "config_path": model_config_path}
        )

        if not model_type:
            logger.info(f"try to use transformers to load model~")
            try:
                from transformers import AutoModel

                model = AutoModel.from_pretrained(
                    pretrained_model_name_or_path, **kwargs
                )
            except:
                raise KeyError(pretrained_model_name_or_path)
        else:
            model_config = MODELZOO_CONFIG.get(model_type, None)
            assert (
                model_config is not None
            ), f"the target model `{model_type}` does not exist, please check the modelzoo or the config yaml~"
            backend = model_config.get("backend", None)
            assert backend in BACKENDS, f"backend should be one of f{BACKENDS}"

            extra_dict["backend"] = backend

            if backend == "hf":
                HFBaseAutoModelClass._model_mapping = cls._model_mapping
                pretrained_model_name_or_path_hf = (
                    hf_name_or_path_check(
                        pretrained_model_name_or_path,
                        remote_url,
                        model_type,
                    )
                    if not is_local
                    else pretrained_model_name_or_path
                )
                model = HFBaseAutoModelClass.from_pretrained(
                    pretrained_model_name_or_path_hf, *model_args, **kwargs
                )
            elif backend == "titan":
                # TODO (junwei.Dong): 支持特殊的titan模型
                raise NotImplementedError(backend)
            elif backend == "fex":
                # TODO (junwei.Dong): 支持特殊的fex模型
                raise NotImplementedError(backend)
            else:
                backend_default_flag = True

            if backend_default_flag:
                model_name_tuple = MODELZOO_CONFIG[model_type][cls._model_key]
                (
                    model_module_package,
                    model_module_name,
                ) = MODELZOO_CONFIG.to_module(model_name_tuple)
                # obtain model class
                model_class = lazy_model_import(
                    model_module_package, model_module_name
                )

                AutoHubClass.kwargs = extra_dict

                model_config_class_: ConfigBase = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    **extra_dict,
                    **config_dict,
                    **kwargs,
                )
                model_config_class_.config_update_for_pretrained(**kwargs)
                # config merge
                model_config_class_.update(kwargs)
                config_dict = model_config_class_.asdict()
                config_dict.update({"config": model_config_class_})
                # instantiate model
                model: ModelBase = model_class(**config_dict)

        """ >> model post processing <<"""

        # set extra args
        if model:
            setattr(model, "extra_args", extra_dict)

        # load weights
        if model_weight_file_path and backend != "hf":
            model.load_pretrained_weights(model_weight_file_path, **kwargs)

        return model


def auto_class_update(cls):
    # Create a new class with the right name from the base class
    # Now we need to copy and re-register `from_config` and `from_pretrained` as class methods otherwise we can't
    # have a specific docstrings for them.
    from_config = copy_func(_BaseAutoModelClass.from_config)

    cls.from_config = classmethod(from_config)

    from_pretrained = copy_func(_BaseAutoModelClass.from_pretrained)

    cls.from_pretrained = classmethod(from_pretrained)
    return cls


def get_values(model_mapping):
    result = []
    for model in model_mapping.values():
        if isinstance(model, (list, tuple)):
            result += list(model)
        else:
            result.append(model)

    return result


def getattribute_from_module(module, attr):
    if attr is None:
        return None
    if isinstance(attr, tuple):
        return tuple(getattribute_from_module(module, a) for a in attr)
    if hasattr(module, attr):
        return getattr(module, attr)
    # Some of the mappings have entries model_type -> object of another model type. In that case we try to grab the
    # object at the top level.
    transformers_module = importlib.import_module("transformers")

    if module != transformers_module:
        try:
            return getattribute_from_module(transformers_module, attr)
        except ValueError:
            raise ValueError(
                f"Could not find {attr} neither in {module} nor in {transformers_module}!"
            )
    else:
        raise ValueError(f"Could not find {attr} in {transformers_module}!")


class _LazyAutoMapping(OrderedDict):
    """
    " A mapping config to object (model or tokenizer for instance) that will load keys and values when it is accessed.

    Args:
        - config_mapping: The map model type to config class
        - model_mapping: The map model type to model (or tokenizer) class
    """

    def __init__(self, config_mapping, model_mapping):
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        self._model_mapping = model_mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key):
        if key in self._extra_content:
            return self._extra_content[key]
        model_type = self._reverse_config_mapping[key.__name__]
        if model_type in self._model_mapping:
            model_name = self._model_mapping[model_type]
            return self._load_attr_from_module(model_type, model_name)

        # Maybe there was several model types associated with this config.
        model_types = [
            k for k, v in self._config_mapping.items() if v == key.__name__
        ]
        for mtype in model_types:
            if mtype in self._model_mapping:
                model_name = self._model_mapping[mtype]
                return self._load_attr_from_module(mtype, model_name)
        raise KeyError(key)

    def _load_attr_from_module(self, model_type, attr):
        # easyguard: 为了不强制懒加载，加了try...except...
        try:
            module_name = model_type_to_module_name(model_type)
            if module_name not in self._modules:
                self._modules[module_name] = importlib.import_module(
                    f".{module_name}", HF_PATH
                )
            return getattribute_from_module(self._modules[module_name], attr)
        except:
            ...

    def keys(self):
        mapping_keys = [
            self._load_attr_from_module(key, name)
            for key, name in self._config_mapping.items()
            if key in self._model_mapping.keys()
        ]
        return mapping_keys + list(self._extra_content.keys())

    def get(self, key, default):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __bool__(self):
        return bool(self.keys())

    def values(self):
        mapping_values = [
            self._load_attr_from_module(key, name)
            for key, name in self._model_mapping.items()
            if key in self._config_mapping.keys()
        ]
        return mapping_values + list(self._extra_content.values())

    def items(self):
        mapping_items = [
            (
                self._load_attr_from_module(key, self._config_mapping[key]),
                self._load_attr_from_module(key, self._model_mapping[key]),
            )
            for key in self._model_mapping.keys()
            if key in self._config_mapping.keys()
        ]
        return mapping_items + list(self._extra_content.items())

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item):
        if item in self._extra_content:
            return True
        if (
            not hasattr(item, "__name__")
            or item.__name__ not in self._reverse_config_mapping
        ):
            return False
        model_type = self._reverse_config_mapping[item.__name__]
        return model_type in self._model_mapping

    def register(self, key, value):
        """
        Register a new model in this mapping.
        """
        if (
            hasattr(key, "__name__")
            and key.__name__ in self._reverse_config_mapping
        ):
            model_type = self._reverse_config_mapping[key.__name__]
            if model_type in self._model_mapping.keys():
                raise ValueError(
                    f"'{key}' is already used by a Transformers model."
                )

        self._extra_content[key] = value
