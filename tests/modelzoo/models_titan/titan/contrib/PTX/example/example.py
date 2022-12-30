from titan.contrib.PTX.create_chinese_model import create_ptx_chinese_model, create_ptx_chinese_tokenizer

model = create_ptx_chinese_model('zh_deberta_moe')
tokenizer = create_ptx_chinese_tokenizer('zh_deberta_moe')

text_input = '这里是PTX的一个中文模型的输入样例'
token_id = tokenizer.process_text(text_input, max_len=100)