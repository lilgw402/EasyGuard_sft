# config

```bash
launch --gpu 1 --cpu 40 --memory 128 -- doas --krb5-username doushihan bash
hdfs://haruna/home/byte_ecom_govern/user/doushihan

Model       model.network
Datamodule  data.
Trainer.    trainer
```

## save ckpt
```bash
python3 ./tasks/gpt2/merge_zero3_ckpt.py --checkpoint_dir hdfs://haruna/home/byte_ecom_govern/user/doushihan/models/gpt2/13b_0228_qa_finetune_bsz2/checkpoints/global_step_50/ --dtype=bf16
```

# 1b3_v1

## pretrain
```bash
bash launch.sh tasks/gpt2/unsup/model.py --model=tasks/gpt2/unsup/1b3_v1.yaml --trainer=tasks/gpt2/unsup/zero2-small.yaml --data.train_batch_size=2 --data.train_size=10000000 --model.network.use_ft_flash_attn=true --model.network.use_ft_linear=true --model.network.use_ft_layernorm=true --trainer.precision=bf16 --data.max_seq_len=2048 --model.network.pad_idx=2 --model.network.use_rmpad=True --data.dyn_bsz=True --data.dyn_bsz_margin=0 --data.stride=1920 --data.bsz_warmup=True --trainer.default_hdfs_dir=hdfs://haruna/home/byte_ecom_govern/user/doushihan/models/gpt2/1b3_0221_mini_data/ --trainer.optimizer_kwargs.optimizer.params.lr=2e-4 --trainer.project_name=gpt_pretrain --trainer.experiment_name=1b3_0221_mini_data
```

## finetune
```bash
bash launch.sh tasks/gpt2/unsup/model_qa.py --model=tasks/gpt2/unsup/1b3_v1.yaml --trainer=tasks/gpt2/unsup/zero2-small.yaml --data.train_batch_size=8 --model.network.use_ft_flash_attn=true --trainer.precision=bf16 --trainer.max_epochs=10 --model.network.gradient_checkpointing=false --data.max_seq_len=1024 --data.mask_prompt_loss=true --model.network.use_ft_linear=true --model.network.use_ft_layernorm=true --model.network.use_rmpad=True --model.partial_pretrain=hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/models/gpt/1b_cleandata_v2_20230212_96gpus/checkpoints/global_step_300000/mp_rank_00_model_states.pt --trainer.default_hdfs_dir=hdfs://haruna/home/byte_ecom_govern/user/doushihan/models/gpt2/1b3_0228_final_qa_finetune/
```

## inference
### 交互式
```bash
python3 tasks/gpt2/unsup/model.py --play --generate-temp 0.7 --model=tasks/gpt2/unsup/1b3_v1.yaml --model.partial_pretrain=hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/models/gpt/1b_cleandata_v2_20230212_96gpus/checkpoints/global_step_300000/mp_rank_00_model_states.pt
```

### 跑文件
```bash
bash launch.sh tasks/gpt2/unsup/model.py --model=tasks/gpt2/unsup/1b3_v1.yaml --play-file-type="qa" --play-file hdfs://haruna/home/byte_data_aml_research/user/jiankai.sun/large_model/play_files/play_file_qa.jsonl --generate-temp 0.7 --model.partial_pretrain=hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/models/gpt/1b_cleandata_v2_20230212_96gpus/checkpoints/global_step_300000/mp_rank_00_model_states.pt
```


# 13b_v2

## pretrain
```bash
bash launch.sh tasks/gpt2/unsup/model.py  --model=tasks/gpt2/unsup/13b_v120_bbpe_100k.yaml --trainer=tasks/gpt2/unsup/zero3-no-cg.yaml --data.train_batch_size=5 --model.network.use_ft_flash_attn=true --model.network.use_ft_linear=true --model.network.use_ft_layernorm=true --trainer.precision=bf16 --data.max_seq_len=2048 --model.network.pad_idx=1 --model.network.use_rmpad=True  --data.dyn_bsz=True --data.stride=1920 --data.dyn_bsz_margin=-0.1 --data.bsz_warmup=True --model.network.gradient_checkpointing_mlp=true --model.network.gradient_checkpointing_ln=true --trainer.project_name=gpt_pretrain --trainer.experiment_name=13b_gptV120_0224_300b_v300 --trainer.optimizer_kwargs.optimizer.params.lr=1.25e-4 --trainer.default_hdfs_dir=hdfs://haruna/home/byte_ecom_govern/user/doushihan/models/gpt2/13b_gptV120_0224_300b_v300 --data.train_size=10000000 --data.tokenizer_type=bbpe --data.tokenizer=hdfs://haruna/home/byte_data_aml_research/user/qian.xian/tokenizer/fast_tokenizer_v5

--data.train_size=450_000_000_000
```

## finetune
```bash
bash launch.sh tasks/gpt2/unsup/model_qa.py --model=tasks/gpt2/unsup/13b_v2.yaml --trainer=tasks/gpt2/unsup/zero3-no-cg.yaml --data.train_batch_size=2 --model.network.use_ft_flash_attn=true  --trainer.precision=bf16 --trainer.max_epochs=10 --model.network.gradient_checkpointing=false --data.max_seq_len=2048 --model.network.pad_idx=2 --data.mask_prompt_loss=true --model.network.use_ft_linear=true --model.network.use_ft_layernorm=true --model.network.use_rmpad=True --model.partial_pretrain=hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/models/gpt/13b_compactdata_v220_20220210/checkpoints/global_step_70000/zero3_merge_states.pt --trainer.default_hdfs_dir=hdfs://haruna/home/byte_ecom_govern/user/doushihan/models/gpt2/13b_0228_qa_finetune_bsz2/
```

## inference
### 交互式
```bash
python3 tasks/gpt2/unsup/model.py --model=tasks/gpt2/unsup/13b_v2.yaml --play --generate-temp 0.7 --model.partial_pretrain=hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/models/gpt/13b_gpt_0222_300b_v300_train_v111/checkpoints/global_step_80000/zero3_merge_states.pt
```

#### tokenizer
--data.train_size=300_000_000_000 --data.tokenizer_type=bbpe --data.tokenizer=hdfs://haruna/home/byte_data_aml_research/user/qian.xian/tokenizer/fast_tokenizer_v5 

### 跑文件
```bash
bash launch.sh tasks/gpt2/unsup/model.py --model=tasks/gpt2/unsup/13b_v2.yaml --play-file-type="qa" --play-file hdfs://haruna/home/byte_data_aml_research/user/jiankai.sun/large_model/play_files/play_file_qa.jsonl --generate-temp 0.7 --model.partial_pretrain=hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/models/gpt/13b_gpt_0222_300b_v300_train_v111/checkpoints/global_step_80000/zero3_merge_states.pt
```

