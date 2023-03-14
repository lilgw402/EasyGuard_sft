task_id=$(date +%Y%m%d%H)

bash launch.sh tasks/gpt2/unsup/model.py \
    --model=tasks/gpt2/unsup/1b3_v1.yaml \
    --play-file-type="qa" \
    --generate-temp=0.7 \
    --generate-trial-num=1 \
    --generate-steps=1 \
    --data.tokenizer=hdfs://haruna/home/byte_ecom_govern/user/doushihan/tokenizer/tokenizer/zh_0620_newcut_caster_145665_lowercase \
    --output_file_path=/mnt/bn/wanli/experiments/instruct_tunning/outputs/inference_result_${task_id}.txt \
    --play-file=/mlx_devbox/users/wanli.0815/repo/EasyGuard/examples/fashion_gpt2/valid.jsonl \
    --model.partial_pretrain=/mnt/bn/wanli/experiments/instruct_tunning/outputs/20230313-1404/checkpoints/global_step_18000/mp_rank_00_model_states.pt