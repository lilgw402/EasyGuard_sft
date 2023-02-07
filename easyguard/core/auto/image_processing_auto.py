from typing import Any, Optional, Union

from ...modelzoo.hub import AutoHubClass
from ...utils import (
    cache_file,
    file_read,
    hf_name_or_path_check,
    lazy_model_import,
    pretrained_model_archive_parse,
)
from . import (
    BACKENDS,
    MODEL_ARCHIVE_CONFIG,
    MODELZOO_CONFIG,
    PROCESSOR_CONFIG_NAMES,
)

IMAGE_PROCESSOR_MAPPING_NAMES = MODELZOO_CONFIG.get_mapping("image_processor")


class AutoImageProcessor:
    def __init__(self) -> None:
        raise EnvironmentError(
            "AutoTokenizer is designed to be instantiated "
            "using the `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_config(cls, config_path: Union[str, Any], *inputs, **kwargs):
        # TODO (junwei.Dong): instantiate a processor class from local path or config instance
        ...

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        region: Optional[str] = "CN",
        *inputs,
        **kwargs,
    ):
        """instantiate a processor class based on `pretrained_model_name_or_path`

        Parameters
        ----------
        pretrained_model_name_or_path : str
            the pretrained model path

        Raises
        ------
        KeyError
            _description_
        """
        if pretrained_model_name_or_path not in MODEL_ARCHIVE_CONFIG:
            # if the `model_name_or_path` is not in `MODEL_ARCHIVE_CONFIG`, what we can do
            # TODO (junwei.Dong): instantiate a pretrained processor class from local path
            raise KeyError(pretrained_model_name_or_path)
        else:
            model_archive = pretrained_model_archive_parse(
                pretrained_model_name_or_path,
                MODEL_ARCHIVE_CONFIG[pretrained_model_name_or_path],
                region,
            )
            model_type = model_archive.get("type", None)
            model_url = model_archive.get("url_or_path", None)
            server_name = model_archive.get("server", None)
            model_config = MODELZOO_CONFIG.get(model_type, None)
            assert (
                model_config is not None
            ), f"the target model `{model_type}` does not exist, please check the modelzoo or the config yaml~"

            backend = model_config.get("backend", None)
            assert backend in BACKENDS, f"backend should be one of f{BACKENDS}"
            backend_default_flag = False

            if backend == "hf":
                from .image_processing_auto_hf import HFAutoImageProcessor

                pretrained_model_name_or_path_ = hf_name_or_path_check(
                    pretrained_model_name_or_path,
                    model_url,
                    model_type,
                )
                return HFAutoImageProcessor.from_pretrained(
                    pretrained_model_name_or_path_, *inputs, **kwargs
                )
            elif backend == "titan":
                # TODO (junwei.Dong): support titan models
                raise NotImplementedError(backend)
            elif backend == "fex":
                # TODO (junwei.Dong): support fex models
                raise NotImplementedError(backend)
            else:
                backend_default_flag = True

            if backend_default_flag:
                # just support base image_processor
                # lazily obtain image_processor class
                image_processor_name_tuple = MODELZOO_CONFIG[model_type][
                    "image_processor"
                ]
                (
                    image_processor_module_package,
                    image_processor_module_name,
                ) = MODELZOO_CONFIG.to_module(image_processor_name_tuple)
                image_processor_class = lazy_model_import(
                    image_processor_module_package, image_processor_module_name
                )
                extra_dict = {
                    "server_name": server_name,
                    "archive_name": pretrained_model_name_or_path,
                    "model_type": model_type,
                    "remote_url": model_url,
                    "region": region,
                }
                AutoHubClass.kwargs = extra_dict
                # obtain image_processor config file path
                image_processor_config_file_path = cache_file(
                    pretrained_model_name_or_path,
                    PROCESSOR_CONFIG_NAMES,
                    **extra_dict,
                )
                image_processor_config = file_read(
                    image_processor_config_file_path
                )
                return image_processor_class(
                    **image_processor_config, **extra_dict
                )
