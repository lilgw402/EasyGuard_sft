for num in {15000..30000..5000}
do  
    torchrun --nproc_per_node $ARNOLD_WORKER_GPU --nnodes $ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr $ARNOLD_WORKER_0_HOST \
    --master_port 12701 valley/inference/inference_valley_jinshou.py --model-class valley-product \
    --model-name /mnt/bn/yangmin-priv-fashionmm/Data/wuji/data_process/new_process/product_checkpoints/data-ys-$1-valley-product-7b-jinshou-class-lora-multi-class/checkpoint-$num \
    --data_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/new_wupian/wupian_test_data_4w_ocr512_n_valley_product.json \
    --image_folder /mnt/bn/yangmin-priv-fashionmm/Data/yangshuang/jinshou_benchmark_image_data \
    --out_path /mnt/bn/yangmin-priv-fashionmm/Data/yangshuang/jinshou_mllm_output/data-ys-$1-valley-product-7b-jinshou-class-lora-multi-class-test-ocr512-$num.txt \
    --DDP --prompt_version v0
done

# for num in {15000..30000..5000}
# do  
#     torchrun --nproc_per_node $ARNOLD_WORKER_GPU --nnodes $ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr $ARNOLD_WORKER_0_HOST \
#     --master_port 12701 valley/inference/inference_valley_jinshou.py --model-class valley-product \
#     --model-name /mnt/bn/yangmin-priv-fashionmm/Data/wuji/data_process/new_process/product_checkpoints/data-ys-$1-valley-product-7b-jinshou-class-lora-multi-class/checkpoint-$num \
#     --data_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/new_wupian/wupian_test_data_4w_ocr512_n_valley_product.json \
#     --image_folder /mnt/bn/yangmin-priv-fashionmm/Data/yangshuang/jinshou_benchmark_image_data \
#     --DDP --prompt_version v0
# done