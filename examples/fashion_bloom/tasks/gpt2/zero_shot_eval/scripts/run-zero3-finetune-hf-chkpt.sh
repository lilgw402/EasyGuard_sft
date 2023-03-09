#!/bin/bash
set -x

declare -A model_dir
model_dir=(
    # ['opt_1b3']='/mnt/bn/mlsys-nas/hf_models/opt-1.3b'
    ['opt_1b3']='hdfs://haruna/home/byte_data_aml_research/user/guokun.lai/hf_models/opt-1.3b'
    ['opt_66b']='/mnt/bn/mlsys-nas/hf_models/opt-66b'
    ['bloom_175b']='/mnt/bn/mlsys-nas/hf_models/bloom-175b'
)

# chkpt_path=/mnt/bn/mlsys-nas/hf_models/opt-1.3b
# chkpt_path=hdfs://haruna/home/byte_data_aml_research/user/guokun.lai/hf_models/opt-1.3b

chkpt_path=${model_dir[$@]}

bash launch.sh tasks/gpt2/zero_shot_eval/model.py \
  --model.use_hf_ckpt=True \
  --data.from_hf_tokenizer=True \
  --data.tokenizer="$chkpt_path" \
  --data.hf_tokenizer_use_fast=False \
  --data.max_seq_len=512 \
  --model=tasks/gpt2/zero_shot_eval/1b.yaml \
  --model.partial_pretrain="$chkpt_path" \
  --model.model_config="$chkpt_path/config.json" \
  --data.train_num_workers=1 \
  --data.train_batch_size=1 \
  --data.val_num_workers=1 \
  --data.val_batch_size=1 \
  --trainer.val_check_interval=0.5 \
  --data.train_path=hdfs://haruna/home/byte_data_aml_research/user/zhengyu.chen/lambada/dev_4864/dev.parquet  \
  --data.val_path=hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/lambada/clean_test \
  --data.dataset_name=lambada \
  --data.template_name=please+next+word  \
  --data.drop_last=True \
  --model.network.use_rmpad_lmloss=false \
  --model.network.use_rmpad_lnmlp=false \
  --model.network.use_rmpad_attn=false \
  --model.network.pad_idx=2 \
  --trainer.max_epochs=20 \
  --trainer.optimizer_kwargs.optimizer.params.lr=1e-5 \
  --trainer=tasks/gpt2/zero_shot_eval/zero3-hf.yaml \

