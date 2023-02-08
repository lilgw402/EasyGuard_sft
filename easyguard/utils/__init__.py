import os
from collections import OrderedDict
from typing import TYPE_CHECKING

from .. import __version__
from .import_utils import _LazyPackage

WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
CONFIG_NAME = "config.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
IMAGE_PROCESSOR_NAME = FEATURE_EXTRACTOR_NAME
GENERATION_CONFIG_NAME = "generation_config.json"
MODEL_CARD_NAME = "modelcard.json"

SENTENCEPIECE_UNDERLINE = "▁"
SPIECE_UNDERLINE = SENTENCEPIECE_UNDERLINE  # Kept for backward compatibility
# for local cache
EASYGUARD_CACHE = os.path.join(f"{os.environ['HOME']}/.cache", "easyguard")
EASYGUARD_MODEL_CACHE = os.path.join(EASYGUARD_CACHE, "models")
REMOTE_PATH_SEP = "/"
# for pretrained model parse
HDFS_HUB_CN = "hdfs://haruna/home/byte_ecom_govern/easyguard"
HDFS_HUB_VA = "hdfs://haruna/home/byte_ecom_govern/easyguard"
SERVER_MAPPING = OrderedDict([["hdfs", (HDFS_HUB_CN, HDFS_HUB_VA)]])
REGION_MAPPING = OrderedDict([["CN", 0], ["VA", 1]])
_import_structure = {
    "yaml_utils": [
        "yaml_check",
        "load_yaml",
        "load_json",
        "file_read",
        "json2yaml",
        "YamlConfig",
    ],
    "doc": [
        "add_code_sample_docstrings",
        "add_end_docstrings",
        "add_start_docstrings",
        "add_start_docstrings_to_model_forward",
        "copy_func",
        "replace_return_docstrings",
    ],
    "generic": [
        "ContextManagers",
        "ExplicitEnum",
        "ModelOutput",
        "PaddingStrategy",
        "TensorType",
        "cached_property",
        "can_return_loss",
        "expand_dims",
        "find_labels",
        "flatten_dict",
        "is_jax_tensor",
        "is_numpy_array",
        "is_tensor",
        "is_tf_tensor",
        "is_torch_device",
        "is_torch_dtype",
        "is_torch_tensor",
        "reshape",
        "squeeze",
        "tensor_size",
        "to_numpy",
        "to_py_obj",
        "transpose",
        "working_or_temp_dir",
    ],
    "hub": [
        "CLOUDFRONT_DISTRIB_PREFIX",
        "DISABLE_TELEMETRY",
        "HF_MODULES_CACHE",
        "HUGGINGFACE_CO_PREFIX",
        "HUGGINGFACE_CO_RESOLVE_ENDPOINT",
        "PYTORCH_PRETRAINED_BERT_CACHE",
        "PYTORCH_TRANSFORMERS_CACHE",
        "S3_BUCKET_PREFIX",
        "TRANSFORMERS_CACHE",
        "TRANSFORMERS_DYNAMIC_MODULE_NAME",
        "EntryNotFoundError",
        "PushToHubMixin",
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
        "cached_file",
        "default_cache_path",
        "define_sagemaker_information",
        "download_url",
        "extract_commit_hash",
        "get_cached_models",
        "get_file_from_repo",
        "get_full_repo_name",
        "has_file",
        "http_user_agent",
        "is_offline_mode",
        "is_remote_url",
        "move_cache",
        "send_example_telemetry",
    ],
    "import_utils": [
        "ENV_VARS_TRUE_AND_AUTO_VALUES",
        "ENV_VARS_TRUE_VALUES",
        "TORCH_FX_REQUIRED_VERSION",
        "USE_JAX",
        "USE_TF",
        "USE_TORCH",
        "DummyObject",
        "OptionalDependencyNotAvailable",
        "_LazyModule",
        "_LazyPackage",
        "ccl_version",
        "is_accelerate_available",
        "is_apex_available",
        "is_bitsandbytes_available",
        "is_bs4_available",
        "is_coloredlogs_available",
        "is_datasets_available",
        "is_decord_available",
        "is_detectron2_available",
        "is_faiss_available",
        "is_flax_available",
        "is_ftfy_available",
        "is_in_notebook",
        "is_ipex_available",
        "is_jumanpp_available",
        "is_kenlm_available",
        "is_keras_nlp_available",
        "is_librosa_available",
        "is_more_itertools_available",
        "is_natten_available",
        "is_ninja_available",
        "is_onnx_available",
        "is_pandas_available",
        "is_phonemizer_available",
        "is_protobuf_available",
        "is_psutil_available",
        "is_py3nvml_available",
        "is_pyctcdecode_available",
        "is_pytesseract_available",
        "is_pytorch_quantization_available",
        "is_rjieba_available",
        "is_sacremoses_available",
        "is_safetensors_available",
        "is_sagemaker_dp_enabled",
        "is_sagemaker_mp_enabled",
        "is_scipy_available",
        "is_sentencepiece_available",
        "is_sklearn_available",
        "is_soundfile_availble",
        "is_spacy_available",
        "is_speech_available",
        "is_sudachi_available",
        "is_tensorflow_probability_available",
        "is_tensorflow_text_available",
        "is_tf2onnx_available",
        "is_tf_available",
        "is_timm_available",
        "is_tokenizers_available",
        "is_torch_available",
        "is_torch_bf16_available",
        "is_torch_bf16_cpu_available",
        "is_torch_bf16_gpu_available",
        "is_torch_compile_available",
        "is_torch_cuda_available",
        "is_torch_fx_available",
        "is_torch_fx_proxy",
        "is_torch_onnx_dict_inputs_support_available",
        "is_torch_tensorrt_fx_available",
        "is_torch_tf32_available",
        "is_torch_tpu_available",
        "is_torchaudio_available",
        "is_torchdistx_available",
        "is_torchdynamo_available",
        "is_training_run_on_sagemaker",
        "is_vision_available",
        "requires_backends",
        "tf_required",
        "torch_only_method",
        "torch_required",
        "torch_version",
        "lazy_model_import",
    ],
    "hdfs_utils": [
        "hlist_files",
        "hopen",
        "hexists",
        "hmkdir",
        "hglob",
        "hisdir",
        "hcountline",
        "hrm",
        "hcopy",
        "hmget",
    ],
    "auxiliary_utils": [
        "sha256",
        "EASYGUARD_CACHE",
        "cache_file",
        "get_configs",
        "load_pretrained_model_weights",
        "list_pretrained_models",
        "typecheck",
        "hf_name_or_path_check",
        "pretrained_model_archive_parse",
    ],
}

# keep each module independent
if TYPE_CHECKING:
    from .auxiliary_utils import (
        EASYGUARD_CACHE,
        cache_file,
        get_configs,
        hf_name_or_path_check,
        list_pretrained_models,
        load_pretrained_model_weights,
        pretrained_model_archive_parse,
        sha256,
        typecheck,
    )
    from .doc import (
        add_code_sample_docstrings,
        add_end_docstrings,
        add_start_docstrings,
        add_start_docstrings_to_model_forward,
        copy_func,
        replace_return_docstrings,
    )
    from .generic import (
        ContextManagers,
        ExplicitEnum,
        ModelOutput,
        PaddingStrategy,
        TensorType,
        cached_property,
        can_return_loss,
        expand_dims,
        find_labels,
        flatten_dict,
        is_jax_tensor,
        is_numpy_array,
        is_tensor,
        is_tf_tensor,
        is_torch_device,
        is_torch_dtype,
        is_torch_tensor,
        reshape,
        squeeze,
        tensor_size,
        to_numpy,
        to_py_obj,
        transpose,
        working_or_temp_dir,
    )
    from .hdfs_utils import (
        hcopy,
        hcountline,
        hexists,
        hglob,
        hisdir,
        hlist_files,
        hmget,
        hmkdir,
        hopen,
        hrm,
    )
    from .hub import (
        CLOUDFRONT_DISTRIB_PREFIX,
        DISABLE_TELEMETRY,
        HF_MODULES_CACHE,
        HUGGINGFACE_CO_PREFIX,
        HUGGINGFACE_CO_RESOLVE_ENDPOINT,
        PYTORCH_PRETRAINED_BERT_CACHE,
        PYTORCH_TRANSFORMERS_CACHE,
        S3_BUCKET_PREFIX,
        TRANSFORMERS_CACHE,
        TRANSFORMERS_DYNAMIC_MODULE_NAME,
        EntryNotFoundError,
        PushToHubMixin,
        RepositoryNotFoundError,
        RevisionNotFoundError,
        cached_file,
        default_cache_path,
        define_sagemaker_information,
        download_url,
        extract_commit_hash,
        get_cached_models,
        get_file_from_repo,
        get_full_repo_name,
        has_file,
        http_user_agent,
        is_offline_mode,
        is_remote_url,
        move_cache,
        send_example_telemetry,
    )
    from .import_utils import (
        ENV_VARS_TRUE_AND_AUTO_VALUES,
        ENV_VARS_TRUE_VALUES,
        TORCH_FX_REQUIRED_VERSION,
        USE_JAX,
        USE_TF,
        USE_TORCH,
        DummyObject,
        OptionalDependencyNotAvailable,
        _LazyModule,
        _LazyPackage,
        ccl_version,
        is_accelerate_available,
        is_apex_available,
        is_bitsandbytes_available,
        is_bs4_available,
        is_coloredlogs_available,
        is_datasets_available,
        is_decord_available,
        is_detectron2_available,
        is_faiss_available,
        is_flax_available,
        is_ftfy_available,
        is_in_notebook,
        is_ipex_available,
        is_jumanpp_available,
        is_kenlm_available,
        is_keras_nlp_available,
        is_librosa_available,
        is_more_itertools_available,
        is_natten_available,
        is_ninja_available,
        is_onnx_available,
        is_pandas_available,
        is_phonemizer_available,
        is_protobuf_available,
        is_psutil_available,
        is_py3nvml_available,
        is_pyctcdecode_available,
        is_pytesseract_available,
        is_pytorch_quantization_available,
        is_rjieba_available,
        is_sacremoses_available,
        is_safetensors_available,
        is_sagemaker_dp_enabled,
        is_sagemaker_mp_enabled,
        is_scipy_available,
        is_sentencepiece_available,
        is_sklearn_available,
        is_soundfile_availble,
        is_spacy_available,
        is_speech_available,
        is_sudachi_available,
        is_tensorflow_probability_available,
        is_tensorflow_text_available,
        is_tf2onnx_available,
        is_tf_available,
        is_timm_available,
        is_tokenizers_available,
        is_torch_available,
        is_torch_bf16_available,
        is_torch_bf16_cpu_available,
        is_torch_bf16_gpu_available,
        is_torch_compile_available,
        is_torch_cuda_available,
        is_torch_fx_available,
        is_torch_fx_proxy,
        is_torch_onnx_dict_inputs_support_available,
        is_torch_tensorrt_fx_available,
        is_torch_tf32_available,
        is_torch_tpu_available,
        is_torchaudio_available,
        is_torchdistx_available,
        is_torchdynamo_available,
        is_training_run_on_sagemaker,
        is_vision_available,
        lazy_model_import,
        requires_backends,
        tf_required,
        torch_only_method,
        torch_required,
        torch_version,
    )
    from .yaml_utils import *
else:
    import sys

    globals_ = dict(globals().items())
    for key_, value_ in globals_.items():
        if isinstance(value_, (str, list, dict)):
            _import_structure[key_] = value_

    sys.modules[__name__] = _LazyPackage(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
