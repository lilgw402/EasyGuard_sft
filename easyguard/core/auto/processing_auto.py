from typing import Any, List, Optional, Union

from ...utils import pretrained_model_archive_parse
from . import BACKENDS, MODEL_ARCHIVE_CONFIG, MODELZOO_CONFIG

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
            model_config = MODELZOO_CONFIG.get(model_type, None)
            assert (
                model_config is not None
            ), f"the target model `{model_type}` does not exist, please check the modelzoo or the config yaml~"

            backend = model_config.get("backend", None)
            assert backend in BACKENDS, f"backend should be one of f{BACKENDS}"
            backend_default_flag = False

            if backend == "hf":
                from .processing_auto_hf import HFAutoProcessor

                return HFAutoProcessor.from_pretrained(
                    pretrained_model_name_or_path, *inputs, **kwargs
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
