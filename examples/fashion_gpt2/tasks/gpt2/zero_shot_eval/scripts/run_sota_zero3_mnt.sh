declare -A model_dir
model_dir=(
    ['bloom_3b']='/mnt/bn/aml-gpt-dev-nas/projects/hf_models/bloom-3b'
    ['bloom_175b']='/mnt/bn/aml-gpt-dev-nas/projects/hf_models/bloom-175b'
    ['bloom_175b_fuse']='/mnt/bn/aml-gpt-dev-nas-fuse/hf_models/bloom-175b'
    ['opt_66b']='/mnt/bn/mlsys-nas/hf_models/opt-66b'
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
        # printf "${model_dir[$@]}\t${model_dir[$@]}/config.json\t${dataset_name_array[$i]}\t${subset_name_array[$i]}\t${template_name_array[$i]}\n"
        bash launch.sh tasks/gpt2/zero_shot_eval/model.py \
            --model.use_hf_ckpt=True \
            --data.from_hf_tokenizer=True \
            --data.tokenizer="${model_dir[$@]}" \
            --data.hf_tokenizer_use_fast=False \
            --data.max_seq_len=1024 \
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
else
    # $1 = model_config
    # $2 = model_ckpt
    # $3 = dataset_index
    # printf "$1\t$2\t${dataset_name_array[$3]}\t${subset_name_array[$3]}\t${template_name_array[$3]}\n"
    cd /opt/tiger/mariana 
    bash launch.sh tasks/gpt2/zero_shot_eval/model.py \
        --model.use_hf_ckpt=True \
        --data.from_hf_tokenizer=True \
        --data.tokenizer="$2" \
        --data.hf_tokenizer_use_fast=False \
        --data.max_seq_len=1024 \
        --model.partial_pretrain="$2" \
        --model.model_config="$2/config.json" \
        --data.val_num_workers=1 \
        --data.val_batch_size=1 \
        --trainer.val_check_interval=1.0 \
        --data.val_path="${val_path_array[$3]}" \
        --data.dataset_name="${dataset_name_array[$3]}" \
        --data.subset_name="${subset_name_array[$3]}" \
        --data.template_name="${template_name_array[$3]}"  \
        --data.drop_last=True \
        --trainer=tasks/gpt2/zero_shot_eval/zero3_hf.yaml \
        --trainer.logger=console \
        --val-only
fi
