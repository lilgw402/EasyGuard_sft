from easyguard.core import AutoModel, AutoTokenizer


def main():
    import torch

    archive = "fashion-deberta"
    my_tokenizer = AutoTokenizer.from_pretrained(archive)
    my_model = AutoModel.from_pretrained(
        archive, dim_shrink=128, rm_deberta_prefix=True
    )
    my_model.eval()
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
        result = my_model(
            input_ids=input_ids,
            attention_mask=input_mask,
            segment_ids=input_segment_ids,
            output_pooled=True,
        )
    print(result)


if __name__ == "__main__":
    # auto_model_test()
    main()
