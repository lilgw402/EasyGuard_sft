import torch
import torch.nn as nn
from titan import create_model
from titan.models.lang.tokenization_deberta import BertTokenizer


class Deberta(nn.Module):
    def __init__(self):
        super().__init__()

        # 模型名称
        ## deberta_base_6l：搜索deberta为backbone，统一使用一个model_name，不同模型直接通过pretrained_model_path区分
        ## deberta_base_12l: 搜索deberta为backbone，12层模型
        ## deberta_base_12l_erlangshen：huggingface二郎神deberta
        model_name = "deberta_base_6l"

        # 本地/hdfs文件路径，参考第一部分模型与资源中的模型路径
        pretrained_model_path = "hdfs://haruna/home/byte_ecom_govern/user/yangzheming/chinese/common_model/trails/ccr_v2_multi_cls/model_outputs3/checkpoints/epoch=4-step=310000-total_ccr_micro_f1=0.914.ckpt"

        self.deberta = create_model(
            model_name=model_name,
            pretrained=True,
            pretrained_uri=pretrained_model_path,
            # dim_shrink=128, # only for sentence embeddings model
        )

    def forward(self, input_ids, input_masks, input_segment_ids):
        """
        search-deberta usage
        USE output['pooled_output'] OR output['sequence_output'] OR output['shrinked_output']
        output['shrinked_output']: only for sentence embeddings model
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
        pooled_output = output["pooled_output"]
        sequence_output = output["sequence_output"]
        shrinked_output = output["shrinked_output"]

        #
        # huggingface usage
        output = self.deberta(input_ids, input_masks, input_segment_ids)
        last_hidden_states = output.hidden_states[-1]


if __name__ == "__main__":
    input_str = "hello world"
    path = "hdfs://haruna/home/byte_ecom_govern/user/yangzheming/asr_model/zh_deberta_base_l6_emd_20210720/vocab.txt"
    deberta_tokenizer = BertTokenizer(path)
    tokens = deberta_tokenizer.tokenize(input_str)
    token_ids = torch.tensor(deberta_tokenizer.convert_tokens_to_ids(tokens))
    input_masks = torch.tensor([1 for _ in token_ids])
    input_segment_ids = torch.tensor([0 for _ in token_ids])
    deberta_model = Deberta()
    ...
