import argparse
import torch
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer
from valley.model.language_model.valley_llama import ValleyVideoLlamaForCausalLM, ValleyProductLlamaForCausalLM
import torch
import os
from valley.utils import disable_torch_init
import os
import random
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
import traceback
from torch.utils.data.distributed import DistributedSampler
from valley.util.config import DEFAULT_GANDALF_TOKEN
from valley.util.data_util import KeywordsStoppingCriteria
from peft import PeftConfig, PeftModel
from transformers import set_seed
from valley.data.dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset
from valley.util.data_util import smart_tokenizer_and_embedding_resize
from valley import conversation as conversation_lib
from valley.util.config import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_VIDEO_FRAME_TOKEN, DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN
set_seed(42)
disable_torch_init()
device = torch.device('cuda:'+str(0)
                        if torch.cuda.is_available() else 'cpu')
lora_path = '/mnt/bn/yangmin-priv-fashionmm/Data/sk/checkpoints/valley-chinese-7b-lora-product-continue-pretrain-down-pool-5epoch-v2/checkpoint-12000'
model_name = os.path.expanduser(lora_path)
# load model
if 'lora' in model_name:
    config = PeftConfig.from_pretrained(model_name)
    print('load model')
    model_old = ValleyProductLlamaForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float16)
    print('load end')
    model_old = PeftModel.from_pretrained(model_old, model_name)
    model_old = model_old.merge_and_unload().half()
    if os.path.exists(os.path.join(model_name,'non_lora_trainables.bin')):
        non_lora_state_dict = torch.load(os.path.join(model_name,'non_lora_trainables.bin'))
        new_state_dict = dict()
        for key in non_lora_state_dict.keys():
            key_new = '.'.join(key.split('.')[2:]) # base_model.model.model.xxxx
            new_state_dict[key_new] = non_lora_state_dict[key]
        model_old_state = model_old.state_dict()
        model_old_state.update(new_state_dict)
        model_old.load_state_dict(model_old_state)
    model = model_old
    tokenizer = LlamaTokenizer.from_pretrained(config.base_model_name_or_path, use_fast = False)
    # tokenizer.padding_side = 'left'
    print("load end")
else:
    print('load model')
    model = ValleyProductLlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16)
    tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast = False)
    # tokenizer.padding_side = 'left'
    print('load end')
model.save_pretrained(lora_path.replace('lora','merge'))
tokenizer.save_pretrained(lora_path.replace('lora','merge'))
### test load after merge
# model = ValleyProductLlamaForCausalLM.from_pretrained(
#         "valley_test_lora_merge", torch_dtype=torch.float16)
# print(model)