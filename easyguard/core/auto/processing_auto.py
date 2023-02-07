from http import server
from typing import Any, List, Optional, Union

from ...modelzoo.hub import AutoHubClass
from ...utils import hf_name_or_path_check, pretrained_model_archive_parse
from . import (
    BACKENDS,
    MODEL_ARCHIVE_CONFIG,
    MODEL_CONFIG_NAMES,
    MODELZOO_CONFIG,
)

PROCESSOR_MAPPING_NAMES = MODELZOO_CONFIG.get_mapping("processor")


class AutoProcessor:
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
        """

        Parameters
        ----------
        pretrained_model_name_or_path : str
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
                from .processing_auto_hf import HFAutoProcessor

                pretrained_model_name_or_path_ = hf_name_or_path_check(
                    pretrained_model_name_or_path,
                    model_url,
                    model_type,
                )
                return HFAutoProcessor.from_pretrained(
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
                ...
                extra_dict = {
                    "server_name": server_name,
                    "archive_name": pretrained_model_name_or_path,
                    "model_type": model_type,
                    "remote_url": model_url,
                    "region": region,
                }

                AutoHubClass.kwargs = extra_dict
                # support simplified models
                # support the huggingface-like models
