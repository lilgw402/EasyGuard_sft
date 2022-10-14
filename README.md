# EasyGuard

(Pre-Trained Models and Applications for E-com. Govern.)

[Documentation]() |
[Contributors](https://code.byted.org/ecom_govern/EasyGuard/) |
[Release Notes]()

EasyGuard是基于AML的Cruise框架扩展的业务算法的代码库，旨在给平台治理提供NLP，CV，和多模态的基础算法和解决方案。
接口设计说明详见：[EasyGuard的框架设计](https://bytedance.feishu.cn/docx/doxcnjT9CWaIH1PNDMg5TLUz7UN)

[Warning:代码处于快速迭代期，核心代码需要UT保护，不然可能会被破坏!]()

## Job
* GCP：https://arnold-i18n.byted.org/job/3807
* 中国区：https://arnold.byted.org/job/27576

## 依赖

EasyGuard依赖公司内外的NLP，CV，和多模态的框架来构建基础业务算法

* [Cruise框架](https://codebase.byted.org/repo/data/cruise)
* [Huggingface/Transformers框架](https://github.com/huggingface/transformers)
* 后续考虑新增ptx

# FashionModels（待注册）

| Model | Parameters | Note |
| --- | --- | --- |
| **BERT** |
| bert-small-uncased | L=6,H=768,A=12 |  |
| bert-base-uncased | L=12,H=768,A=12 |  |
| fashiondeberta-base-zh | L=12,H=768,A=12 | Pretrain w/ ASR & CCR datasets |
| **Chinese BERT** |
| hfl/chinese-roberta-wwm-ext | L=12,H=768,A=12 |  |
| hfl/chinese-roberta-wwm-ext-large | L=24,H=768,A=12 |  |
| langboat/mengzi-bert-base | L=12,H=768,A=12| Pretrain w/ Chinese datasets|
| **Multilingual BERT**  |
| fashionxlm-base-xl | L=12,H=768,A=12 | Pretrain w/ TTS datasets|
| **Vision-Language BERT** |
| fashionbert-base-zh | L=12,H=768,A=12 | Pretrain w/ Product datasets|
| **Video-Text BERT** |
| videoclip | L=12,H=768,A=12 |  |
| framealbert | L=12,H=x,A=x |  |


# 代码使用方式

```python
app = SequenceClassification(pretrained_model_name_or_path='bert-small-uncased')
```
or:
```python
model = AutoModel.from_pretrained('bert-small-uncased')
```



## CopyRight
2021 ByteDance Inc. All Rights Reserved.
