bash launch.sh tasks/gpt2/unsup/model.py \
    --model=tasks/gpt2/unsup/1b3_v1.yaml \
    --play-file-type="qa" \
    --play-file=hdfs://haruna/home/byte_ecom_govern/user/wanli.0815/experiments/gpt2/data/ccr_valid.parquet \
    --generate-temp=0.7 \
    --model.partial_pretrain=hdfs://haruna/home/byte_ecom_govern/user/doushihan/models/gpt2/global_step_300000/mp_rank_00_model_states.pt