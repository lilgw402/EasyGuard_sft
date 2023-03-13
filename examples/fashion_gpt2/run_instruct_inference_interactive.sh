python3 tasks/gpt2/unsup/model.py \
      --play \
      --generate-temp=0.7 \
      --model=tasks/gpt2/unsup/1b3_v1.yaml \
      --generate-steps=1 \
      --generate-trial-num=1 \
      --model.partial_pretrain=/mnt/bn/wanli/resources/pretrained_lm/gpt2/mp_rank_00_model_states.pt \
      --data.tokenizer=hdfs://haruna/home/byte_ecom_govern/user/doushihan/tokenizer/tokenizer/zh_0620_newcut_caster_145665_lowercase