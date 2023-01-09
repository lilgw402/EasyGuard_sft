# be used for the huggingface model import

HF_PATH = "easyguard.modelzoo.models"
EASYGUARD_PATH = "easyguard.modelzoo.models"
BACKENDS = set(["default", "hf", "titan", "fex"])
MODEL_CONFIG_NAMES = set(["config.yaml", "config.json"])
PYTORCH_MODLE = "pytorch_model"
MODEL_SAVE_NAMES = set([f"{PYTORCH_MODLE}.ckpt", f"{PYTORCH_MODLE}.bin"])
VOCAB_NAME = "vocab.txt"
TOKENIZER_CONFIG = "tokenizer_config"
TOKENIZER_NAMES = set([f"{TOKENIZER_CONFIG}.yaml", f"{TOKENIZER_CONFIG}.json"])


from ...modelzoo import MODEL_ARCHIVE_CONFIG, MODELZOO_CONFIG
from .modeling_auto import AutoModel
from .tokenization_auto import AutoTokenizer
