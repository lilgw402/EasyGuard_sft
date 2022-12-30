Ptx has many dependencies(matx, libcut, tokenizer...), which makes users spend a lot of time on environment configuration. We handle the installation of ptx dependencies in titan contrib.
Users can import a ptx chinese model and tokenizer processor in this way:
```python
from titan.contrib.PTX.create_chinese_model import create_ptx_chinese_model, create_ptx_chinese_tokenizer

model = create_ptx_chinese_model('zh_electra_base_6l_share_qa_site_web_l1_64gpus_all_fp16_20200526')
pipeline = create_ptx_chinese_tokenizer('zh_electra_base_6l_share_qa_site_web_l1_64gpus_all_fp16_20200526')

print(model)
print(pipeline)
``` 

Notice there are 4 pretrained chinese model in ptx: zh_albert_base_l6_mix_oldcut_20200315_20200315, zh_albert_base_l6_qa_site_web_l1_64gpus_20200520, zh_electra_base_6l_share_qa_site_web_l1_64gpus_all_fp16_20200526, zh_deberta_base_l6_emd_20210720. Users can get more details from this doc: https://bytedance.feishu.cn/wiki/wikcn62fnFsSgv5hVNlZKKKxoDe?useNewLarklet=1


For distributed training, users must specify the `local_rank` argument:
```python
from titan.contrib.PTX.create_chinese_model import create_ptx_chinese_model, create_ptx_chinese_tokenizer

model = create_ptx_chinese_model('zh_deberta_moe', local_rank=MY_LOCAL_RANK)
tokenizer = create_ptx_chinese_tokenizer('zh_deberta_moe', local_rank=MY_LOCAL_RANK)

text_input = '这里是PTX的一个中文模型的输入样例'
token_id = tokenizer.process_text(text_input, max_len=100)
``` 