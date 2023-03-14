task_id=$(date +%Y%m%d%H)
bash launch.sh tasks/gpt2/unsup/model_qa.py \
    --model=tasks/gpt2/unsup/1b3_v1.yaml \
    --trainer=tasks/gpt2/unsup/zero2-small.yaml \
    --generate-steps=1 \
    --data.tokenizer=hdfs://haruna/home/byte_ecom_govern/user/doushihan/tokenizer/tokenizer/zh_0620_newcut_caster_145665_lowercase \
    --data.train_path=hdfs://haruna/home/byte_ecom_govern/user/wanli.0815/experiments/gpt2/data/ccr_train.parquet \
    --data.train_batch_size=32 \
    --model.network.use_ft_flash_attn=true \
    --trainer.precision=bf16 \
    --trainer.max_epochs=3 \
    --model.network.gradient_checkpointing=false \
    --data.max_seq_len=256 \
    --data.mask_prompt_loss=true \
    --model.network.use_ft_linear=true \
    --model.network.use_ft_layernorm=true \
    --model.network.use_rmpad=True \
    --model.partial_pretrain=/mnt/bn/wanli/resources/pretrained_lm/gpt2/mp_rank_00_model_states.pt \
    --trainer.default_hdfs_dir=/mnt/bn/wanli/experiments/instruct_tunning/outputs/${task_id}