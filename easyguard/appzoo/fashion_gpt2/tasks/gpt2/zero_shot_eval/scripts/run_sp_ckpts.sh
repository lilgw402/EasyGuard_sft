declare -A model_config
declare -A model_ckpt
model_config=(
    ['sp_200k']='tasks/gpt2/zero_shot_eval/1b3_v1_sp_mariana.yaml'
    ['sp_300k']='tasks/gpt2/zero_shot_eval/1b3_v1_sp_mariana.yaml'
    ['sp_400k']='tasks/gpt2/zero_shot_eval/1b3_v1_sp_mariana.yaml'
)
model_ckpt=(
    ['sp_200k']='hdfs://haruna/home/byte_data_aml_research/user/qian.xian/models/gpt/sp27W/checkpoints/global_step_400000/mp_rank_00_model_states.pt'
    ['sp_300k']='hdfs://haruna/home/byte_data_aml_research/user/qian.xian/models/gpt/sp27W/checkpoints/global_step_400000/mp_rank_00_model_states.pt'
    ['sp_400k']='hdfs://haruna/home/byte_data_aml_research/user/qian.xian/models/gpt/sp27W/checkpoints/global_step_400000/mp_rank_00_model_states.pt'
)

val_path_array=(
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/story_cloze/2016/clean_test
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/lambada/clean_test
    hdfs://haruna/home/byte_data_aml_research/user/taiqing.wang/corpus/lambda_zh_parquet
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/hellaswag/clean_dev
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/clue/chid/clean_dev_slim
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/ai2_arc/ARC-Challenge/clean_test
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/ai2_arc/ARC-Easy/clean_test
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/super_glue/rte/clean_dev
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/clue/ocnli/clean_dev
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/super_glue/wsc/clean_dev
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/ptb/clean_test
)

dataset_name_array=(
    story_cloze
    lambada
    lambada_zh
    hellaswag
    clue
    ai2_arc
    ai2_arc
    super_glue
    clue
    super_glue
    ptb
)

subset_name_array=(
    2016
    ""
    ""
    ""
    chid
    ARC-Challenge
    ARC-Easy
    rte
    ocnli
    wsc.fixed
    ""
)

template_name_array=(
    Generate+Ending
    please+next+word
    please_next_word
    Open-ended+completion
    fill_the_blank
    heres_a_problem
    pick_the_most_correct_option
    MNLI+crowdsource
    OCNLI+crowdsource
    does+the+pronoun+refer+to
    ""
)

for i in "${!val_path_array[@]}"
do
    # printf "${model_config[$@]}\t${model_ckpt[$@]}\t${dataset_name_array[$i]}\t${subset_name_array[$i]}\t${template_name_array[$i]}\n"
    bash launch.sh tasks/gpt2/zero_shot_eval/model.py --model="${model_config[$@]}" --model.partial_pretrain="${model_ckpt[$@]}" --data.val_num_workers=1 --data.val_batch_size=1 --trainer.val_check_interval=1.0 --data.val_path="${val_path_array[$i]}" --data.dataset_name="${dataset_name_array[$i]}" --data.subset_name="${subset_name_array[$i]}" --data.template_name="${template_name_array[$i]}" --data.tokenizer=hdfs://haruna/home/byte_data_aml_research/user/qian.xian/tokenizer/sentencepiece_mariana --val-only --model.network.pad_idx=2 --data.max_seq_len=256 --model.network.use_ft_flash_attn=true --model.network.use_ft_linear=true --model.network.use_ft_layernorm=true
done

