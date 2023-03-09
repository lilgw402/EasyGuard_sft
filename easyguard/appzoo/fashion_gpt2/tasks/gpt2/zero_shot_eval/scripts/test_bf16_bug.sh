declare -A model_dir
model_dir=(
    ['bloom_3b']='/mnt/bn/aml-gpt-dev-nas/projects/hf_models/bloom-3b'
    ['bloom_175b']='/mnt/bn/aml-gpt-dev-nas/projects/hf_models/bloom-175b'
)

val_path_array=(
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/clue/chid/clean_dev
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/ptb/clean_test
)

dataset_name_array=(
    clue
    ptb
)

subset_name_array=(
    chid
    ""
)

template_name_array=(
    fill_the_blank
    ""
)

for i in "${!val_path_array[@]}"
do
    # printf "${model_dir[$@]}\t${model_dir[$@]}/config.json\t${dataset_name_array[$i]}\t${subset_name_array[$i]}\t${template_name_array[$i]}\n"
    bash launch.sh tasks/gpt2/zero_shot_eval/model.py \
        --model.use_hf_ckpt=True \
        --data.from_hf_tokenizer=True \
        --data.tokenizer="${model_dir[$@]}" \
        --data.hf_tokenizer_use_fast=False \
        --data.max_seq_len=512 \
        --model.partial_pretrain="${model_dir[$@]}" \
        --model.model_config="${model_dir[$@]}/config.json" \
        --data.val_num_workers=1 \
        --data.val_batch_size=1 \
        --trainer.val_check_interval=1.0 \
        --data.val_path="${val_path_array[$i]}" \
        --data.dataset_name="${dataset_name_array[$i]}" \
        --data.subset_name="${subset_name_array[$i]}" \
        --data.template_name="${template_name_array[$i]}"  \
        --data.drop_last=True \
        --trainer=tasks/gpt2/zero_shot_eval/zero3_hf.yaml \
        --trainer.logger=console \
        --val-only
done