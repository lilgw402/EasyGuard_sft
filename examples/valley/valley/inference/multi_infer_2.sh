

for num in {30000..30000..5000}
do  
    torchrun --nproc_per_node $ARNOLD_WORKER_GPU --nnodes $ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr $ARNOLD_WORKER_0_HOST \
    --master_port 12701 valley/inference/inference_valley_jinshou.py --model-class valley-product \
    --model-name /mnt/bn/yangmin-priv-fashionmm/Data/wuji/data_process/new_process/product_checkpoints/data-mutli-$1-valley-7b-jinshou-class-lora-multi-class/checkpoint-$num \
    --data_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/new_wupian/wupian_test_data_4w_new_v18_valley_product.json \
    --image_folder /mnt/bn/yangmin-priv-fashionmm/Data/yangshuang/jinshou_benchmark_image_data \
    --out_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/output/data-wj-$1-valley-product-7b-jinshou-class-lora-multi-class-test-0109-$num.txt \
    --DDP --prompt_version jinshou_cot
done

# for num in {15000..20000..5000}
# do  
#     torchrun --nproc_per_node $ARNOLD_WORKER_GPU --nnodes $ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr $ARNOLD_WORKER_0_HOST \
#     --master_port 12701 valley/inference/inference_valley_jinshou.py --model-class valley-product \
#     --model-name /mnt/bn/yangmin-priv-fashionmm/Data/wuji/data_process/new_process/product_checkpoints/data-mutli-$1-valley-7b-jinshou-class-lora-multi-class/checkpoint-$num \
#     --data_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/new_wupian/wupian_test_data_4w_new_v19_valley_product.json \
#     --image_folder /mnt/bn/yangmin-priv-fashionmm/Data/yangshuang/jinshou_benchmark_image_data \
#     --out_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/output/data-wj-$1-valley-product-7b-jinshou-class-lora-multi-class-test-v19-$num.txt \
#     --DDP --prompt_version jinshou_cot
# done

# for num in {5000..25000..5000}
# do  
#     torchrun --nproc_per_node $ARNOLD_WORKER_GPU --nnodes $ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr $ARNOLD_WORKER_0_HOST \
#     --master_port 12701 valley/inference/inference_valley_jinshou.py --model-class valley-product \
#     --model-name /mnt/bn/yangmin-priv-fashionmm/Data/wuji/data_process/new_process/product_checkpoints/data-mutli-$1-valley-7b-jinshou-class-lora-multi-class/checkpoint-$num \
#     --data_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/new_wupian/wupian_test_data_4w_new_valley_product.json \
#     --image_folder /mnt/bn/yangmin-priv-fashionmm/Data/yangshuang/jinshou_benchmark_image_data \
#     --out_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/output/data-wj-$1-valley-product-7b-jinshou-class-lora-multi-class-test-ocr512-$num.txt \
#     --DDP --prompt_version jinshou_cot
# done

# for num in {5000..20000..5000}
# do  
#     torchrun --nproc_per_node $ARNOLD_WORKER_GPU --nnodes $ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr $ARNOLD_WORKER_0_HOST \
#     --master_port 12701 valley/inference/inference_valley_jinshou.py --model-class valley-product \
#     --model-name /mnt/bn/yangmin-priv-fashionmm/Data/wuji/data_process/new_process/product_checkpoints/data-mutli-$1-valley-7b-jinshou-class-lora-multi-class/checkpoint-$num \
#     --data_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/new_wupian/wupian_test_data_4w_new_add_keyword_valley_product.json \
#     --image_folder /mnt/bn/yangmin-priv-fashionmm/Data/yangshuang/jinshou_benchmark_image_data \
#     --out_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/output/data-wj-$1-valley-product-7b-jinshou-class-lora-multi-class-test-ocr512-add_keyword-$num.txt \
#     --DDP --prompt_version jinshou_cot
# done


# for num in {5000..20000..5000}
# do  
#     torchrun --nproc_per_node $ARNOLD_WORKER_GPU --nnodes $ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr $ARNOLD_WORKER_0_HOST \
#     --master_port 12701 valley/inference/inference_valley_jinshou.py --model-class valley-product \
#     --model-name /mnt/bn/yangmin-priv-fashionmm/Data/wuji/data_process/new_process/product_checkpoints/data-mutli-$1-valley-7b-jinshou-class-lora-multi-class/checkpoint-$num \
#     --data_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/new_wupian/wupian_test_data_4w_new_valley_product.json \
#     --image_folder /mnt/bn/yangmin-priv-fashionmm/Data/yangshuang/jinshou_benchmark_image_data \
#     --out_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/output/data-wj-$1-valley-product-7b-jinshou-class-lora-multi-class-test-ocr512-$num.txt \
#     --DDP --prompt_version jinshou_cot
# done


# for num in {25000..40000..5000}
# do  
#     torchrun --nproc_per_node $ARNOLD_WORKER_GPU --nnodes $ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr $ARNOLD_WORKER_0_HOST \
#     --master_port 12701 valley/inference/inference_valley_jinshou.py --model-class valley-product \
#     --model-name /mnt/bn/yangmin-priv-fashionmm/Data/wuji/data_process/new_process/product_checkpoints/data-mutli-$1-valley-7b-jinshou-class-lora-multi-class/checkpoint-$num \
#     --data_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/new_wupian/wupian_test_data_4w_ocr512_n_multi_ans_small_model_valley_product.json \
#     --image_folder /mnt/bn/yangmin-priv-fashionmm/Data/yangshuang/jinshou_benchmark_image_data \
#     --out_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/output/data-wj-$1-valley-product-7b-jinshou-class-lora-multi-class-test-ocr512-$num.txt \
#     --DDP --prompt_version jinshou_cot
# done

# for num in {15000..30000..5000}
# do  
#     torchrun --nproc_per_node $ARNOLD_WORKER_GPU --nnodes $ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr $ARNOLD_WORKER_0_HOST \
#     --master_port 12701 valley/inference/inference_valley_jinshou.py --model-class valley-product \
#     --model-name /mnt/bn/yangmin-priv-fashionmm/Data/wuji/data_process/new_process/product_checkpoints/data-ys-$1-valley-product-7b-jinshou-class-lora-multi-class/checkpoint-$num \
#     --data_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/new_wupian/wupian_test_data_4w_ocr512_n_valley_product.json \
#     --image_folder /mnt/bn/yangmin-priv-fashionmm/Data/yangshuang/jinshou_benchmark_image_data \
#     --DDP --prompt_version v0
# done
