# Debug on Dev Machine
## 1gpu
### 开启rmpad
mlx worker launch --gpu 1 --cpu 40 --memory 512 --type "a100-80g" -- python3 tasks/gpt2/unsup/model.py  --model=tasks/gpt2/unsup/1b3_v1.yaml --trainer=tasks/gpt2/unsup/zero2-small.yaml --data.train_batch_size=2 --model.network.use_ft_flash_attn=true --model.network.use_ft_linear=true --model.network.use_ft_layernorm=true  --data.max_seq_len=2048  --model.network.use_rmpad=True --data.dyn_bsz=True  --data.stride=1920 --data.bsz_warmup=True

* 测试Resume功能:
%--trainer.default_hdfs_dir=hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/inbox
%--trainer.resume_ckpt_path=hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/inbox/checkpoints

%hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/models/gpt/1b_cleandata_v1_20220113/checkpoints/global_step_690000/mp_rank_00_model_states.pt
% --trainer.precision=bf16 (Default)
% --model.network.pad_idx=2
% --data.dyn_bsz_margin=0

### 关闭rmpad
mlx worker launch --gpu 1 --cpu 40 --memory 512 --type "a100-80g" -- python3 tasks/gpt2/unsup/model.py  --model=tasks/gpt2/unsup/1b3_v1.yaml --trainer=tasks/gpt2/unsup/zero2-small.yaml  --data.train_batch_size=2 --model.network.use_ft_flash_attn=true --model.network.use_ft_linear=true --model.network.use_ft_layernorm=true --data.max_seq_len=2048 --data.stride=-1

### Resume/Load Model
* Load ckpt
--model.partial_pretrain=hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/inbox/checkpoints/global_step_10/mp_rank_00_model_states.pt

* Resume from xx step
--trainer.resume_ckpt_path=hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/models/gpt/1b_cleandata_v2_20220201/checkpoints/global_step_120000
只需要指定到xxstep目录

## 4gpu
 
mlx worker launch --gpu 2 --cpu 50 --memory 1000 --type "a100-80g" -- bash launch.sh tasks/gpt2/unsup/model.py  --model=tasks/gpt2/unsup/1b3_v1.yaml  --data.train_batch_size=2 --trainer=tasks/gpt2/unsup/zero3-no-cg.yaml --model.network.use_ft_flash_attn=true --model.network.use_ft_linear=true --model.network.use_ft_layernorm=true --trainer.precision=bf16 --data.max_seq_len=2048 --model.network.use_rmpad=True --data.dyn_bsz=True --data.stride=1920 --data.bsz_warmup=True --trainer.optimizer_kwargs.optimizer.params.lr=2.34e-4 --model.network.gradient_checkpointing_ln=true  
<!-- --trainer.resume_ckpt_path=hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/models/gpt/1b_cleandata_v2_20220201/checkpoints/global_step_120000 -->



## tools
% --data.train_num_workers=0
% -m pdb -c continue

# Job
## 1.3b
bash launch.sh tasks/gpt2/unsup/model.py  --model=tasks/gpt2/unsup/1b3_v1.yaml --trainer=tasks/gpt2/unsup/zero2-small.yaml --data.train_batch_size=6 --model.network.use_ft_flash_attn=true --model.network.use_ft_linear=true --model.network.use_ft_layernorm=true --trainer.precision=bf16 --data.max_seq_len=2048 --model.network.pad_idx=2 --model.network.use_rmpad=True  --data.dyn_bsz=True --data.dyn_bsz_margin=0 --data.stride=1920 --data.bsz_warmup=True --trainer.default_hdfs_dir=hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/models/gpt/1b_cleandata_v2_2023020x --model.network.gradient_checkpointing_ln=true --trainer.optimizer_kwargs.optimizer.params.lr=2e-4 




## 13b
bash launch.sh tasks/gpt2/unsup/model.py  --model=tasks/gpt2/unsup/13b_v1.yaml --trainer=tasks/gpt2/unsup/zero3.yaml --data.train_batch_size=10 --model.network.use_ft_flash_attn=true --model.network.use_ft_linear=true --model.network.use_ft_layernorm=true --trainer.precision=bf16 --data.max_seq_len=1024 --model.network.pad_idx=2 --model.network.use_rmpad=True  --data.dyn_bsz=True --data.stride=-1 --data.dyn_bsz_margin=1.0 --model.network.gradient_checkpointing_mlp=true --model.network.gradient_checkpointing_ln=true

# Play Mode
bash launch.sh tasks/gpt2/unsup/model.py --play --model=tasks/gpt2/unsup/13b_v1.yaml --model.partial_pretrain=hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/models/gpt/1b_cleandata_v2_20220202_e2/checkpoints/global_step_60000/mp_rank_00_model_states.pt --data.max_seq_len=2048

# Notes
* 如果是eval一个使用了rmpad训练得到的模型, 命令中需要包含如下参数
--model.network.use_rmpad=True --data.dyn_bsz=False --data.dyn_bsz_margin=0 --data.stride=-1 --model.network.pad_output=True --model.network.use_ft_flash_attn=true --model.network.use_ft_linear=true --model.network.use_ft_layernorm=true

* 如果是进行play的话，可以不开rmpad.



# doushihan

launch --gpu 1 --cpu 40 --memory 128 -- doas --krb5-username doushihan bash
hdfs://haruna/home/byte_ecom_govern/user/doushihan

Model       model.network
Datamodule  data.
Trainer.    trainer

## save ckpt
python3 ./tasks/gpt2/merge_zero3_ckpt.py --checkpoint_dir hdfs://haruna/home/byte_ecom_govern/user/doushihan/models/gpt2/13b_0228_qa_finetune_bsz2/checkpoints/global_step_50/ --dtype=bf16


# 1b3_v1

## pretrain
bash launch.sh tasks/gpt2/unsup/model.py --model=tasks/gpt2/unsup/1b3_v1.yaml --trainer=tasks/gpt2/unsup/zero2-small.yaml --data.train_batch_size=2 --data.train_size=10000000 --model.network.use_ft_flash_attn=true --model.network.use_ft_linear=true --model.network.use_ft_layernorm=true --trainer.precision=bf16 --data.max_seq_len=2048 --model.network.pad_idx=2 --model.network.use_rmpad=True --data.dyn_bsz=True --data.dyn_bsz_margin=0 --data.stride=1920 --data.bsz_warmup=True --trainer.default_hdfs_dir=hdfs://haruna/home/byte_ecom_govern/user/doushihan/models/gpt2/1b3_0221_mini_data/ --trainer.optimizer_kwargs.optimizer.params.lr=2e-4 --trainer.project_name=gpt_pretrain --trainer.experiment_name=1b3_0221_mini_data

## instruction tuning finetune
bash launch.sh tasks/gpt2/unsup/model_qa.py --model=tasks/gpt2/unsup/1b3_v1.yaml --trainer=tasks/gpt2/unsup/zero2-small.yaml --data.train_batch_size=8 --model.network.use_ft_flash_attn=true --trainer.precision=bf16 --trainer.max_epochs=10 --model.network.gradient_checkpointing=false --data.max_seq_len=1024 --data.mask_prompt_loss=true --model.network.use_ft_linear=true --model.network.use_ft_layernorm=true --model.network.use_rmpad=True --model.partial_pretrain=hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/models/gpt/1b_cleandata_v2_20230212_96gpus/checkpoints/global_step_300000/mp_rank_00_model_states.pt --trainer.default_hdfs_dir=hdfs://haruna/home/byte_ecom_govern/user/doushihan/models/gpt2/1b3_0228_final_qa_finetune/

## inference
### 交互式
python3 tasks/gpt2/unsup/model.py --play --generate-temp 0.7 --model=tasks/gpt2/unsup/1b3_v1.yaml --model.partial_pretrain=hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/models/gpt/1b_cleandata_v2_20230212_96gpus/checkpoints/global_step_300000/mp_rank_00_model_states.pt

### 跑文件
bash launch.sh tasks/gpt2/unsup/model.py --model=tasks/gpt2/unsup/13b_v120_bbpe_100k.yaml --play-file-type="qa" --play-file hdfs://haruna/home/byte_data_aml_research/user/jiankai.sun/large_model/play_files/play_file_qa.jsonl --generate-temp 0.7 --model.partial_pretrain=hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/models/gpt/1b_cleandata_v2_20230212_96gpus/checkpoints/global_step_300000/mp_rank_00_model_states.pt


# 13b_v2

## pretrain
bash launch.sh tasks/gpt2/unsup/model.py  --model=tasks/gpt2/unsup/13b_v120_bbpe_100k.yaml --trainer=tasks/gpt2/unsup/zero3-no-cg.yaml --data.train_batch_size=5 --model.network.use_ft_flash_attn=true --model.network.use_ft_linear=true --model.network.use_ft_layernorm=true --trainer.precision=bf16 --data.max_seq_len=2048 --model.network.pad_idx=1 --model.network.use_rmpad=True  --data.dyn_bsz=True --data.stride=1920 --data.dyn_bsz_margin=-0.1 --data.bsz_warmup=True --model.network.gradient_checkpointing_mlp=true --model.network.gradient_checkpointing_ln=true --trainer.project_name=gpt_pretrain --trainer.experiment_name=13b_gptV120_0224_300b_v300 --trainer.optimizer_kwargs.optimizer.params.lr=1.25e-4 --trainer.default_hdfs_dir=hdfs://haruna/home/byte_ecom_govern/user/doushihan/models/gpt2/13b_gptV120_0224_300b_v300 --data.train_size=450_000_000_000 --data.tokenizer_type=bbpe --data.tokenizer=hdfs://haruna/home/byte_data_aml_research/user/qian.xian/tokenizer/fast_tokenizer_v5

## instruction tuning finetune
bash launch.sh tasks/gpt2/unsup/model_qa.py --model=tasks/gpt2/unsup/13b_v2.yaml --trainer=tasks/gpt2/unsup/zero3-no-cg.yaml --data.train_batch_size=2 --model.network.use_ft_flash_attn=true  --trainer.precision=bf16 --trainer.max_epochs=10 --model.network.gradient_checkpointing=false --data.max_seq_len=2048 --model.network.pad_idx=2 --data.mask_prompt_loss=true --model.network.use_ft_linear=true --model.network.use_ft_layernorm=true --model.network.use_rmpad=True --model.partial_pretrain=hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/models/gpt/13b_compactdata_v220_20220210/checkpoints/global_step_70000/zero3_merge_states.pt --trainer.default_hdfs_dir=hdfs://haruna/home/byte_ecom_govern/user/doushihan/models/gpt2/13b_0228_qa_finetune_bsz2/

## inference
### 交互式
python3 tasks/gpt2/unsup/model.py --model=tasks/gpt2/unsup/13b_v2.yaml --play --generate-temp 0.7 --model.partial_pretrain=hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/models/gpt/13b_gpt_0222_300b_v300_train_v111/checkpoints/global_step_80000/zero3_merge_states.pt

#### tokenizer
--data.train_size=300_000_000_000 --data.tokenizer_type=bbpe --data.tokenizer=hdfs://haruna/home/byte_data_aml_research/user/qian.xian/tokenizer/fast_tokenizer_v5 

### 跑文件
bash launch.sh tasks/gpt2/unsup/model.py --model=tasks/gpt2/unsup/13b_v2.yaml --play-file-type="qa" --play-file hdfs://haruna/home/byte_data_aml_research/user/jiankai.sun/large_model/play_files/play_file_qa.jsonl --generate-temp 0.7 --model.partial_pretrain=hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/models/gpt/13b_gpt_0222_300b_v300_train_v111/checkpoints/global_step_80000/zero3_merge_states.pt





<!-- ## Tnews classification finetune
bash launch.sh tasks/gpt2/finetune/tnews/model.py  --model=tasks/gpt2/finetune/tnews/1b.yaml --model.partial_pretrain=hdfs://haruna/home/byte_arnold_lq/data/reckon/mlxlab_aml_test/tasks/3320480/trials/10942777/output/checkpoints/global_step_290000

bash launch.sh tasks/gpt2/finetune/tnews/model.py --model=tasks/gpt2/finetune/tnews/1b3_v1.yaml --trainer=tasks/gpt2/unsup/zero2-small.yaml --data.train_batch_size=4 --data.val_batch_size=4 --model.network.use_ft_flash_attn=true  --trainer.precision=bf16 --trainer.max_epochs=5 --model.network.gradient_checkpointing=false --data.max_seq_len=1024 --model.network.use_ft_linear=true --model.network.use_ft_layernorm=true --model.partial_pretrain=hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/models/gpt/1b_cleandata_v2_20230212_96gpus/checkpoints/global_step_300000/mp_rank_00_model_states.pt --trainer.default_hdfs_dir=hdfs://haruna/home/byte_ecom_govern/user/doushihan/models/gpt2/1b3_0221_mini_data_tnews_finetune_with_trainer_yaml/ -->


<!-- ## train v100 0222-ft16
bash launch.sh tasks/gpt2/unsup/model.py --model=tasks/gpt2/unsup/1b3_v1.yaml --trainer=tasks/gpt2/unsup/zero3-fp16.yaml --data.train_batch_size=2 --data.train_size=1000000 --model.network.use_ft_flash_attn=False --model.network.use_ft_linear=False --model.network.use_ft_layernorm=False --trainer.precision=fp16 --data.max_seq_len=1024 --model.network.pad_idx=2 --model.network.use_rmpad=False --data.dyn_bsz=True --data.dyn_bsz_margin=0 --data.stride=192 --data.bsz_warmup=True --trainer.default_hdfs_dir=hdfs://haruna/home/byte_ecom_govern/user/doushihan/models/gpt2/1b3_0222_fp16_mini_data_use_ft_false/ --trainer.optimizer_kwargs.optimizer.params.lr=2e-4 --trainer.project_name=gpt_pretrain --trainer.experiment_name=1b3_0222_fp16_mini_data -->
