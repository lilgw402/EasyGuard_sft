# doushihan

launch --gpu 2 --cpu 40 --memory 300 -- doas --krb5-username doushihan bash
hdfs://haruna/home/byte_ecom_govern/user/doushihan

hdfs://haruna/home/byte_ecom_govern/user/doushihan/models/gpt2/1b3_0221_mini_data/

Model       model.network
Datamodule  data.
Trainer.    trainer

export BYTED_TORCH_AUTO_UPDATE=off
export BYTED_TORCH_BYTECCL=O0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64


# bloom 7b1

## instruction tuning finetune
./tasks/gpt2/zero_shot_eval/scripts/run-finetune-hf-ckpt.sh bloom_7b1

## inference
### 交互式
./tasks/gpt2/zero_shot_eval/scripts/run-play.sh bloom_7b1

### 跑文件
./tasks/gpt2/zero_shot_eval/scripts/run-play-file.sh bloom_7b1
