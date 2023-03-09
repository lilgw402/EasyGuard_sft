# some config
```bash
# launch GPUs
launch --gpu 4 --cpu 40 --memory 300 -- doas --krb5-username doushihan bash
```
```bash
# my hdfs

hdfs://haruna/home/byte_ecom_govern/user/doushihan
hdfs://haruna/home/byte_ecom_govern/user/doushihan/models/gpt2/1b3_0221_mini_data/

cd /opt/tiger/EasyGuard/examples/fashion_bloom/
```

```bash
# image config
export BYTED_TORCH_AUTO_UPDATE=off
export BYTED_TORCH_BYTECCL=O0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
```

# bloom 7b1

## instruction tuning finetune
```bash
./tasks/gpt2/zero_shot_eval/scripts/run-finetune-hf-ckpt.sh bloom_7b1
```

## inference
### 交互式(命令行)
```bash
./tasks/gpt2/zero_shot_eval/scripts/run-play-load-from-hf.sh bloom_560m
./tasks/gpt2/zero_shot_eval/scripts/run-play-load-from-hf.sh bloom_7b1
```

### 跑文件
文件 key = ['question', 'answer']

```bash
./tasks/gpt2/zero_shot_eval/scripts/run-play-file-load-from-hf.sh bloom_7b1
./tasks/gpt2/zero_shot_eval/scripts/run-play-file-load-from-hf.sh bloom_560m

./tasks/gpt2/zero_shot_eval/scripts/run-play-file-load-from-cruise.sh bloom_7b1_finetune
```