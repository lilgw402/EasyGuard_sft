import torch
import torch.nn as nn
from titan import create_model
from titan.models.lang.tokenization_deberta import BertTokenizer


class YourModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 模型名称
        ## deberta_base_6l：搜索deberta为backbone，统一使用一个model_name，不同模型直接通过pretrained_model_path区分
        ## deberta_base_12l: 搜索deberta为backbone，12层模型
        ## deberta_base_12l_erlangshen：huggingface二郎神deberta
        model_name = "deberta_base_6l"

        # 本地/hdfs文件路径，参考第一部分模型与资源中的模型路径
        pretrained_model_path = "hdfs://haruna/home/byte_ecom_govern/user/yangzheming/gold/asr_v2_wordpiece_add_cl_step_7w_128dim.ckpt"
        # pretrained_model_path = "hdfs://haruna/home/byte_ecom_govern/user/yangzheming/chinese/common_model/trails/ccr_v2_multi_cls/model_outputs3/checkpoints/epoch=4-step=310000-total_ccr_micro_f1=0.914.ckpt"

        self.deberta = create_model(
            model_name=model_name,
            pretrained=True,
            pretrained_uri=pretrained_model_path,
            dim_shrink=128,
        )

    def forward(self, input_ids, input_masks, input_segment_ids):
        """
        search-deberta usage
        USE output['pooled_output'] OR output['sequence_output']
        eg:
        pooled_output = self.deberta(input_ids, attention_mask, segment_ids, pinyin_ids)['pooled_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        """
        output = self.deberta(
            input_ids=input_ids,
            segment_ids=input_segment_ids,
            attention_mask=input_masks,
            output_pooled=True,
        )
        return output
        # huggingface usage
        # output = self.deberta(input_ids, input_masks, input_segment_ids)
        # last_hidden_states = output.hidden_states[-1]


def test():
    my_model = YourModel()
    my_model.eval()
    vocab_file = "hdfs://haruna/home/byte_ecom_govern/user/yangzheming/asr_model/zh_deberta_base_l6_emd_20210720/vocab.txt"
    # vocab = build_vocab(vocab_file)
    my_tokenizer = BertTokenizer(
        vocab_file,
        do_lower_case=True,
        tokenize_emoji=False,
        greedy_sharp=True,
    )
    max_length = 512
    text = "我的手机"
    text_token = my_tokenizer.tokenize(text)
    text_token = ["[CLS]"] + text_token + ["[SEP]"]
    token_ids = my_tokenizer.convert_tokens_to_ids(text_token)
    input_ids = token_ids[:max_length] + my_tokenizer.convert_tokens_to_ids(
        ["[PAD]"]
    ) * (max_length - len(token_ids))
    input_mask = [1] * len(token_ids[:max_length]) + [0] * (max_length - len(token_ids))
    input_segment_ids = [0] * max_length
    print(f"input_ids: {input_ids}")
    print(f"input_mask: {input_mask}")
    print(f"input_segment_ids: {input_segment_ids}")
    input_ids = torch.tensor([input_ids])
    input_mask = torch.tensor([input_mask])
    input_segment_ids = torch.tensor([input_segment_ids])
    with torch.no_grad():
        result = my_model(input_ids, input_mask, input_segment_ids)
    print(result)


test()
