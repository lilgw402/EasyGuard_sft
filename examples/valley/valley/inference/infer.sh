# torchrun --nproc_per_node $ARNOLD_WORKER_GPU --nnodes $ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr $ARNOLD_WORKER_0_HOST \
# --master_port 12701 valley/inference/inference_valley.py --model-class valley-product \
# --model-name /mnt/bn/yangmin-priv-fashionmm/Data/wuji/data_process/new_process/product_checkpoints/data-ys-v1-valley-product-7b-jinshou-class-lora-multi-class/checkpoint-15000 \
# --data_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/new_wupian/wupian_test_data_4w_ocr512_n_valley_product.json \
# --image_folder /mnt/bn/yangmin-priv-fashionmm/Data/wuji/big_model_wupian_image_data \
# --out_path /mnt/bn/yangmin-priv-fashionmm/Data/yangshuang/jinshou_mllm_output/data-ys-v1-valley-product-7b-jinshou-class-lora-multi-class-test-35000.txt \
# --DDP --world_size 1

python3 valley/inference/inference_valley.py --model-class valley-product \
--model-name /mnt/bn/yangmin-priv-fashionmm/Data/wuji/data_process/new_process/product_checkpoints/data-ys-v1-valley-product-7b-jinshou-class-lora-multi-class/checkpoint-20000 \
--data_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/data_process/new_process/train_data/train_data_v12_dup_neg_and_pos_valley_product.json \
--image_folder /mnt/bn/yangmin-priv-fashionmm/Data/wuji/big_model_wupian_image_data \
--out_path /mnt/bn/yangmin-priv-fashionmm/Data/yangshuang/jinshou_mllm_output/data-ys-v1-valley-product-7b-jinshou-class-lora-multi-class-test-35000.txt \
--DDP --DDP_port 12580 --world_size 1 --prompt_version v0

# python3 valley/inference/inference_single.py \
# --model-name /mnt/bn/yangmin-priv-fashionmm/Data/wuji/data_process/new_process/product_checkpoints/data-ys-v1-valley-product-7b-jinshou-class-lora-multi-class/checkpoint-15000 \
# --inference_json_path /mnt/bn/yangmin-priv-fashionmm/Data/wuji/wupian_process/new_wupian/wupian_test_data_4w_ocr512_n_valley_product.json \
# --image_folder /mnt/bn/yangmin-priv-fashionmm/Data/wuji/big_model_wupian_image_data \
# --out_path /mnt/bn/yangmin-priv-fashionmm/Data/yangshuang/jinshou_mllm_output/data-ys-v1-valley-product-7b-jinshou-class-lora-multi-class-test-35000.txt \
# --world_size 1