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
    ['aml_13b_48k_0216']='tasks/gpt2/zero_shot_eval/13b_v2.yaml'
    ['aml_13b_78k_0216']='tasks/gpt2/zero_shot_eval/13b_v2.yaml'
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
    ['aml_13b_48k_0216']='hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/models/gpt/13b_compactdata_v220_20220210/checkpoints/global_step_48000/zero3_merge_states.pt'
    ['aml_13b_78k_0216']='hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/models/gpt/13b_compactdata_v220_20220210/checkpoints/global_step_78000/zero3_merge_states.pt '
)

val_path_array=(
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/logiQA_en/clean_dev
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/PIQA/clean_dev
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/openbook_qa/clean_dev
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/logiQA_zh/clean_dev
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/PIQA_zh/clean_dev
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/openbook_qa_zh/clean_dev
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/ai2_arc_zh/arc_challenge/clean_dev
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/ai2_arc_zh/arc_easy/clean_dev
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/story_cloze/2016/clean_test
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/lambada/clean_test_fix_v2
    hdfs://haruna/home/byte_data_aml_research/user/taiqing.wang/corpus/lambda_zh_parquet
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/hellaswag/clean_dev
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/clue/chid/clean_dev_slim
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/ai2_arc/ARC-Challenge/clean_test
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/ai2_arc/ARC-Easy/clean_test
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/super_glue/rte/clean_dev
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/clue/ocnli/clean_dev
    hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/super_glue/wsc/clean_dev
)

dataset_name_array=(
    logi_qa
    piqa
    openbookqa
    logi_qa_zh
    piqa_zh
    openbookqa_zh
    ai2_arc
    ai2_arc
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
)

subset_name_array=(
    ""
    ""
    main
    ""
    ""
    main
    ARC-Challenge-zh
    ARC-Easy-zh
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
)

template_name_array=(
    pick_best_answer_en
    pick_best_answer_en
    pick_best_answer_en
    pick_best_answer_zh
    pick_best_answer_zh
    pick_best_answer_zh
    pick_best_answer_zh
    pick_best_answer_zh
    Generate+Ending
    please+next+word
    please_next_word
    Open-ended+completion
    fill_the_blank
    pick_best_answer_en
    pick_best_answer_en
    MNLI+crowdsource
    OCNLI+crowdsource
    does+the+pronoun+refer+to
)

if [ ! -n "$2" ]; then 
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
            --data.max_seq_len=1024 \
            --model.network.use_ft_flash_attn=true \
            --model.network.use_ft_linear=true \
            --model.network.use_ft_layernorm=true \
            --model.network.use_rmpad=True \
            --model.network.pad_output=True
    done
else
    # $1 = model_config
    # $2 = model_ckpt
    # $3 = dataset_index
    # printf "$1\t$2\t${dataset_name_array[$3]}\t${subset_name_array[$3]}\t${template_name_array[$3]}\n"
    cd /opt/tiger/mariana 
    bash launch.sh tasks/gpt2/zero_shot_eval/model.py \
        --model="$1" \
        --model.partial_pretrain="$2" \
        --data.val_num_workers=1 \
        --data.val_batch_size=1 \
        --trainer.val_check_interval=1.0 \
        --data.val_path="${val_path_array[$3]}" \
        --data.dataset_name="${dataset_name_array[$3]}" \
        --data.subset_name="${subset_name_array[$3]}" \
        --data.template_name="${template_name_array[$3]}" \
        --val-only \
        --model.network.pad_idx=2 \
        --data.max_seq_len=1024 \
        --model.network.use_ft_flash_attn=true \
        --model.network.use_ft_linear=true \
        --model.network.use_ft_layernorm=true \
        --model.network.use_rmpad=True \
        --model.network.pad_output=True
fi
