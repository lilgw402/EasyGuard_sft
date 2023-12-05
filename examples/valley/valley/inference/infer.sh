torchrun --nproc_per_node $ARNOLD_WORKER_GPU --nnodes $ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr $ARNOLD_WORKER_0_HOST \
--master_port 12701 valley/inference/inference_valley_jinshou.py --model-class valley-product \
--model-name /mnt/bn/yangmin-priv-fashionmm/Data/wuji/data_process/new_process/product_checkpoints/data-ys-v1-valley-product-7b-jinshou-class-lora-multi-class/checkpoint-15000 \
--data_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/new_wupian/wupian_test_data_4w_ocr512_n_valley_product.json \
--image_folder /mnt/bn/yangmin-priv-fashionmm/Data/wuji/big_model_wupian_image_data \
--out_path /mnt/bn/yangmin-priv-fashionmm/Data/yangshuang/jinshou_mllm_output/data-ys-v1-valley-product-7b-jinshou-class-lora-multi-class-test-35000.txt \
--DDP --prompt_version v0

# python3 valley/inference/inference_valley.py --model-class valley-product \
# --model-name /mnt/bn/yangmin-priv-fashionmm/Data/wuji/data_process/new_process/product_checkpoints/data-ys-v1-valley-product-7b-jinshou-class-lora-multi-class/checkpoint-20000 \
# --data_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/new_wupian/wupian_test_data_4w_ocr512_n_valley_product.json \
# --image_folder /mnt/bn/yangmin-priv-fashionmm/Data/wuji/big_model_wupian_image_data \
# --out_path /mnt/bn/yangmin-priv-fashionmm/Data/yangshuang/jinshou_mllm_output/data-ys-v1-valley-product-7b-jinshou-class-lora-multi-class-test-35000.txt \
# --DDP --DDP_port 12580 --world_size 1 --prompt_version v0

# python3 valley/inference/inference_valley_multi_image.py \
# --model-name /mnt/bn/yangmin-priv-fashionmm/Data/wuji/data_process/new_process/product_checkpoints/data-ys-v1-valley-product-7b-jinshou-class-lora-multi-class/checkpoint-15000 \
# --inference_json_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/new_wupian/wupian_test_data_4w_ocr512_n_valley_product.json \
# --image_folder /mnt/bn/yangmin-priv-fashionmm/Data/wuji/big_model_wupian_image_data \
# --out_path /mnt/bn/yangmin-priv-fashionmm/Data/yangshuang/jinshou_mllm_output/data-ys-v1-valley-product-7b-jinshou-class-lora-multi-class-test-35000-debug.txt \
# --world_size 1

# gpu_num=2
# for part_id in {0..1..1} # 0 1 2 3 4 5 6 7
# do
#     export CUDA_VISIBLE_DEVICES=${part_id}
#     {
#         python3 valley/inference/inference_valley_multi_image.py \
#         --model-name /mnt/bn/yangmin-priv-fashionmm/Data/wuji/data_process/new_process/product_checkpoints/data-ys-v1-valley-product-7b-jinshou-class-lora-multi-class/checkpoint-30000 \
#         --inference_json_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/new_wupian/wupian_test_data_4w_ocr512_n_valley_product.json  \
#         --out_path /mnt/bn/yangmin-priv-fashionmm/Data/yangshuang/jinshou_mllm_output/data-ys-v1-valley-product-7b-jinshou-class-lora-multi-class-test-35000-debug-${part_id}.txt \
#         --image_folder /mnt/bn/yangmin-priv-fashionmm/Data/yangshuang/jinshou_benchmark_image_data \
#         --world_size $gpu_num \
#         --part_no ${part_id}
#     } &
#     sleep 0.5
# done
# wait