#!/bin/bash
set -x

declare -A model_dir
model_dir=(
    # ['opt_1b3']='/mnt/bn/mlsys-nas/hf_models/opt-1.3b'
    ['opt_1b3']='hdfs://haruna/home/byte_data_aml_research/user/guokun.lai/hf_models/opt-1.3b'
    ['opt_66b']='/mnt/bn/mlsys-nas/hf_models/opt-66b'
    ['bloom_176b']='/mnt/bn/ecom-govern-maxiangqian/doushihan/hf_models/bloom176b/bloom'
    ['bloom_7b1']='/mnt/bn/ecom-govern-maxiangqian/doushihan/hf_models/bloom7b1/bloom-7b1'
)

trainer_config=(
    ['opt_1b3']='tasks/gpt2/zero_shot_eval/zero3-hf-no-offload.yaml'
    ['opt_66b']='tasks/gpt2/zero_shot_eval/zero3-hf-no-offload.yaml'
    ['bloom_7b1']='tasks/gpt2/zero_shot_eval/zero3-hf-bloom-175b-offload-param-none-optim-cpu.yaml'
    ['bloom_176b']='tasks/gpt2/zero_shot_eval/zero3-hf-bloom-175b-offload-param-none-optim-cpu.yaml'
)

chkpt_path=${model_dir[$@]}

bash launch.sh tasks/gpt2/zero_shot_eval/model.py \
  --model.use_hf_ckpt=True \
  --data.from_hf_tokenizer=True \
  --data.tokenizer="$chkpt_path" \
  --data.hf_tokenizer_use_fast=False \
  --data.max_seq_len=1024 \
  --model=tasks/gpt2/zero_shot_eval/1b.yaml \
  --model.partial_pretrain="$chkpt_path" \
  --model.model_config="$chkpt_path/config.json" \
  --data.train_num_workers=1 \
  --data.train_batch_size=6 \
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
  --model.network.pad_idx=3 \
  --trainer.max_epochs=5 \
  --trainer.optimizer_kwargs.optimizer.params.lr=1e-5 \
  --trainer.default_hdfs_dir=hdfs://haruna/home/byte_ecom_govern/user/doushihan/models/bloom/testSpeed_bsz6 \
  --trainer=tasks/gpt2/zero_shot_eval/zero3-hf-bloom-175b-offload-param-none-optim-cpu.yaml