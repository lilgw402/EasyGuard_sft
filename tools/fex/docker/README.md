# Dockerfile and Easyrun
## Dockerfile
基于lab.pytorch镜像预装fex的依赖，由于fex的版本更新更频繁，需要手动安装至`/opt/tiger/fex`
在arnold上类似MPIRUN的使用方式，通过以下命令开始训练任务：
```shell
FEXRUN train.py --config_path ... 
```

## 使用easyrun运行训练
fex训练需要依赖apex和DALI等第三方库，如果对安装方法有疑惑，可以直接使用easyrun进行训练，无需担心环境问题。

1. 安装easyrun相关环境(lark搜索：EasyRun用户群，里面有详尽的安装使用文档)

2. 进入FEX项目根目录，运行：
```sh
easyrun -H docker/easyrun_fex_train_torch1.7_dali_apex.yaml
```
待镜像构建完成(仅第一次构建时间久，之后秒进)后进入docker环境

3. Run a sample：在imagenet数据集上运行resnet分类任务。
```python
CUDA_VISIBLE_DEVICES=3 python3 -m tasks.image_classify.train --config_path config/imagenet/resnet_json_default.yaml --output_path "test_resnet_train
```
