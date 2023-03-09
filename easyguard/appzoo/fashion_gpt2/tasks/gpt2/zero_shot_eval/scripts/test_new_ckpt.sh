declare -A model_config
declare -A model_ckpt
model_config=(
    ['aml_1b3_30k_clean_data_0201']='tasks/gpt2/zero_shot_eval/1b.yaml'
    ['aml_1b3_180k_clean_data_0201']='tasks/gpt2/zero_shot_eval/1b.yaml'
    ['aml_1b3_240k_clean_data_0201']='tasks/gpt2/zero_shot_eval/1b.yaml'
    ['aml_1b3_300k_clean_data_0201']='tasks/gpt2/zero_shot_eval/1b.yaml'
    ['aml_1b3_200k_0201']='tasks/gpt2/zero_shot_eval/1b.yaml'
    ['aml_1b3_250k_0201']='tasks/gpt2/zero_shot_eval/1b.yaml'
    ['aml_1b3_300k_0201']='tasks/gpt2/zero_shot_eval/1b.yaml'
    ['aml_1b3_360k_0201']='tasks/gpt2/zero_shot_eval/1b.yaml'
    ['aml_1b3_400k_0201']='tasks/gpt2/zero_shot_eval/1b.yaml'
    ['aml_1b3_480k_0201']='tasks/gpt2/zero_shot_eval/1b.yaml'
)
model_ckpt=(
    ['aml_1b3_30k_clean_data_0201']='hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/models/gpt/1b_cleandata_v2_20220202_e2/checkpoints/global_step_30000/mp_rank_00_model_states.pt'
    ['aml_1b3_180k_clean_data_0201']='hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/models/gpt/1b_cleandata_v2_20220202_e2/checkpoints/global_step_180000/mp_rank_00_model_states.pt'
    ['aml_1b3_240k_clean_data_0201']='hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/models/gpt/1b_cleandata_v2_20220202_e2/checkpoints/global_step_240000/mp_rank_00_model_states.pt'
    ['aml_1b3_300k_clean_data_0201']='hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/models/gpt/1b_cleandata_v2_20220202_e2/checkpoints/global_step_300000/mp_rank_00_model_states.pt'
    ['aml_1b3_200k_0201']='hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/models/gpt/1b_cleandata_v2_20220202_128gpus/checkpoints/global_step_200000/mp_rank_00_model_states.pt'
    ['aml_1b3_250k_0201']='hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/models/gpt/1b_cleandata_v2_20220202_128gpus/checkpoints/global_step_250000/mp_rank_00_model_states.pt'
    ['aml_1b3_300k_0201']='hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/models/gpt/1b_cleandata_v2_20220202_128gpus/checkpoints/global_step_300000/mp_rank_00_model_states.pt'
    ['aml_1b3_360k_0201']='hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/models/gpt/1b_cleandata_v2_20220202_128gpus/checkpoints/global_step_360000/mp_rank_00_model_states.pt'
    ['aml_1b3_400k_0201']='hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/models/gpt/1b_cleandata_v2_20220202_128gpus/checkpoints/global_step_400000/mp_rank_00_model_states.pt'
    ['aml_1b3_480k_0201']='hdfs://haruna/home/byte_data_aml_research/user/yanshipeng/models/gpt/1b_cleandata_v2_20220202_128gpus/checkpoints/global_step_480000/mp_rank_00_model_states.pt'
)

val_path_array=(
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/ptb/clean_test
)

dataset_name_array=(
    ptb
)

subset_name_array=(
    ""
)

template_name_array=(
    ""
)

for i in "${!val_path_array[@]}"
do
    # printf "${model_config[$@]}\t${model_ckpt[$@]}\t${dataset_name_array[$i]}\t${subset_name_array[$i]}\t${template_name_array[$i]}\n"
    bash launch.sh tasks/gpt2/zero_shot_eval/model.py \
        --model="${model_config[$@]}" \
        --model.partial_pretrain="${model_ckpt[$@]}" \
        --data.val_num_workers=1 \
        --data.val_batch_size=1 \
        --trainer.val_check_interval=1.0 \
        --data.val_path="${val_path_array[$i]}" \
        --data.dataset_name="${dataset_name_array[$i]}" \
        --data.subset_name="${subset_name_array[$i]}" \
        --data.template_name="${template_name_array[$i]}" \
        --val-only \
        --model.network.pad_idx=2 \
        --data.max_seq_len=256 \
        --model.network.use_ft_flash_attn=true \
        --model.network.use_ft_linear=true \
        --model.network.use_ft_layernorm=true \
        --model.network.use_rmpad=True \
        --model.network.pad_output=True
done
