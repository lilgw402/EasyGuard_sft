<img src="ci/test_data/fex_logo.png" alt="Logo of the project" align="right" width="110" height="60">

# FEX: Fusion of languagE and X &middot; 

(factor X can be Vision, Audio etc.)

[Documentation]() |
[Contributors](https://code.byted.org/nlp/fex/graphs/master) |
[Release Notes]()

[![pipeline status](https://code.byted.org/nlp/fex/badges/master/pipeline.svg)](https://code.byted.org/nlp/fex/commits/master)
[![coverage report](https://code.byted.org/nlp/fex/badges/master/coverage.svg)](https://code.byted.org/nlp/fex/commits/master)

[Fex](https://ttna43rawldhnpdv2k.web.bytedance.net/) 是一个多模态训练代码库，旨在提供语言+视觉的多模态落地的解决方案。

Fex 提供了多模态预训练模型，旨在帮助业务快速应用最前沿的多模态技术。我们支持跨模态检索、多模态相关性、图像生成文本、多模态分类等场景。

Fex 提供了简洁的API，支持基于预训练模型进行finetune。支持分布式训练、大规模hdfs数据训练、transformer kernel 训练加速。


<p align="center">
    <br>
    <img src="ci/test_data/fex_group.png" alt="Logo group" height="260">
    <br>
<p>


<br>

更详细的介绍可以看这个doc：[**Fex All in One**](https://bytedance.feishu.cn/docs/doccnk0nOpzvERZSHPCuGJKiZCM)

<!-- ## 支持模型

| 模型        | 使用     |   训练    |   部署    |
| ---------- | :------: | :------: | :------: |
| CLIP       | ✅       |   ✅      |  ✅      |
| CLIP       | ✅       |   ✅      |  ✅      | -->


***

## Install


Fex可以作为package install来使用。

```shell
git clone git@code.byted.org:nlp/fex.git
cd fex
pip install --editable .
```

更详细的使用方法请参考这个文档：[**Fex from Scratch**](https://bytedance.feishu.cn/docx/doxcn6L5Uml7OgK2oA9ftsMzo6d)


<br>



## FAQ：
[FAQ 汇总的地方](https://bytedance.feishu.cn/docx/doxcn6L5Uml7OgK2oA9ftsMzo6d#doxcn0wQ0YMqOa8EA1APh2FLqNN)


<br>

## 手动安装其他依赖

如果没有选择使用easyrun，使用的过程中可能会遇到一些包需要安装的情况，基本都列在下面了。

- Nvida Apex (optional，只有在使用混合精度的情况下需要用它)
```sh
git clone https://github.com/NVIDIA/apex ~/.apex
# master版本有问题，编译会报错，先切换到一个可用的版本
# issue: https://github.com/NVIDIA/apex/issues/1091
git reset --hard a651e2c24ecf97cbf367fd3f330df36760e1c597
export CUDA_HOME=path_to_your_cuda (such as: /opt/tiger/cuda_10_1)
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ~/.apex
```
- Arnold Data Loader (optional，只有在使用arnold dataset的时候需要）
```sh
pip install -U pip
pip install byted-dataloader==0.2.6 -i "https://bytedpypi.byted.org/simple"
```

- Matx （optional，matx/byted_vision/nlp_tokenizer，只有在需要matx做预处理的时候用到。2022年4月7日）
推荐使用matx 来做预处理：[matx 安装](https://bytedance.feishu.cn/wiki/wikcnS7vYo5ZwvCImghxdLUYlag)


- Byted-Optimizer （optional，data/optimizer，只有在需要AdaBelief/LAMB作为optimizer进行训练时用到，参考文档：[Fex接入统一优化器](https://bytedance.feishu.cn/docx/doxcnYPeVGsSDAm54PvdwbwWRTe)，[Fex-CLIP-AdaBelief加速训练](https://bytedance.feishu.cn/docs/doccnCzyAcNJshJMtqMd92ns8Gf)。2022年2月23日）
```sh
# data/optimizer
pip3 install https://d.scm.byted.org/api/v2/download/data.aml.optimizer_1.0.0.2.tar.gz --no-cache-dir -i https://bytedpypi.byted.org/simple/

```

- DALI (optional，只有在需要dali做预处理的时候用到）
```sh
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100
```


***


## Develop Setup

十分欢迎给 Fex 提交issue 或代码，让 Fex 变得更好。下面是提交代码前，关于commit 和 code format的检查方法。


1. 设置pre-commit hook，提前检查code style问题：
```sh
bash ci/install-pre-commit.sh
```
3. 提交代码前需要code-format， 保证code style没有问题：
```sh
bash ci/code-format.sh -i
```

## 更新日志

参考[CHANGELOG](CHANGELOG.md)。

## CopyRight
2021 ByteDance Inc. All Rights Reserved.

