# ViT Albert Two Tower Model

## MLX Project Page
MLX project: https://reckon.bytedance.net/mlxlab/project/repo/182/detail

## Have a Try

Running a 6-layer model with zero2:
```
bash launch.sh tasks/clip/dy_cover/vit_albert_6l/model.py \
    --data.train_batch_size=20 --trainer.max_epochs=2  \
    --data.val_benchmarks=['douyin_recall_v2'] \
    --trainer=tasks/clip/dy_cover/vit_albert_6l/zero2_warmup_0.2_lr_5e-5.yaml \
    --data.val_kwargs="{'query_batch_size':8,'video_batch_size':8}" \
    --trainer.val_check_interval=1000 --trainer.logger=['console'] \
    --trainer.precision=32 \
    --trainer.optimizer_kwargs.optimizer.params.lr=1e-4 \
    --trainer.optimizer_kwargs.scheduler.params.warmup_step_rate=0.005
```

## End-to-end Giant Model Trials

[ViT 13B example](https://reckon.bytedance.net/mlxlab/project/job/detail?job_id=48pprmsrof6352e7b5&bid=14)

[MLX tracking link](https://reckon.bytedance.net/mlxlab/tracking/project_20220923_6c07ac19?tab=overview&mine=true&bid=14&keyword=&current=1&pageSize=50&selected=run_20220924_86b3b4e0%2Crun_20221009_b6dd5a51%2Crun_20221009_d94c49fa%2Crun_20220925_80fb2b33%2Crun_20221021_5179d8bd)

```
bash launch.sh tasks/clip/dy_cover/vit_albert_6l/model.py \
    --data.train_batch_size=26 --trainer.max_epochs=5  \
    --data.val_benchmarks=['douyin_recall_v2'] \
    --trainer=tasks/clip/dy_cover/vit_albert_6l/zero3_warmup_0.2_lr_5e-5.yaml \
    --data.val_kwargs="{'query_batch_size':8,'video_batch_size':8}" \
    --trainer.val_check_interval=5.0 \
    --model=tasks/clip/dy_cover/vit_albert_6l/13b_drop_0.yaml
```

The following config enables zero3:
```
--trainer=tasks/clip/dy_cover/vit_albert_6l/zero3_warmup_0.2_lr_5e-5.yaml
```
To switch to zero2, replace the trainer config above with
```
--trainer=tasks/clip/dy_cover/vit_albert_6l/zero2_warmup_0.2_lr_5e-5.yaml
```

Define model dim in `tasks/clip/dy_cover/vit_albert_6l/13b_drop_0.yaml`.
Define learning rate in trainer yaml files such as `tasks/clip/dy_cover/vit_albert_6l/zero3_warmup_0.2_lr_5e-5.yaml`.

ViT 3B Reference Trials
- cruise: https://arnold.byted.org/trial/9106165 930 samples/s
- fex: https://arnold.byted.org/trial/9105880 576 samples/s