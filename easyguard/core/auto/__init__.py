# used for the huggingface model import
HF_PATH = "easyguard.modelzoo.models"
EASYGUARD_PATH = "easyguard.modelzoo.models"
# defalut constant variables used for model config setting
BACKENDS = set(["default", "hf", "titan", "fex"])
# easyguard will search for the special files which can be found in target directory based on these names
MODEL_CONFIG_NAMES = set(["config.yaml", "config.json"])
PYTORCH_MODLE = "pytorch_model"
MODEL_SAVE_NAMES = set(
    [f"{PYTORCH_MODLE}.ckpt", f"{PYTORCH_MODLE}.bin", f"{PYTORCH_MODLE}.pt"]
)
VOCAB_NAME = "vocab.txt"
VOCAB_JSON_NAMES = set(["vocab.yaml", "vocab.json"])
TOKENIZER_CONFIG = "tokenizer_config"
TOKENIZER_CONFIG_NAMES = set(
    [f"{TOKENIZER_CONFIG}.yaml", f"{TOKENIZER_CONFIG}.json"]
)
PROCESSOR_CONFIG = "preprocessor_config"
PROCESSOR_CONFIG_NAMES = set(
    [f"{PROCESSOR_CONFIG}.yaml", f"{PROCESSOR_CONFIG}.json"]
)
SPECIAL_TOKENS_MAP_NAMES = set(
    ["special_tokens_map.yaml", "special_tokens_map.json"]
)

from ...modelzoo import MODEL_ARCHIVE_CONFIG, MODELZOO_CONFIG
from .image_processing_auto import AutoImageProcessor
from .modeling_auto import AutoModel
from .processing_auto import AutoProcessor
from .tokenization_auto import AutoTokenizer
