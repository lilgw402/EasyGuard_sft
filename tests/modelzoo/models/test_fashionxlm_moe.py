from easyguard import AutoModel, AutoTokenizer


def test_fashionxlm_moe():
    text = "good good study, day day up!"
    archive = "fashionxlm-moe-base"
    tokenizer = AutoTokenizer.from_pretrained(archive)
    inputs = tokenizer(text, return_tensors="pt", max_length=84)
    model = AutoModel.from_pretrained(archive, model_cls="sequence_model")
    ouputs = model(**inputs, language=["GB"])
    print(ouputs)


if __name__ == "__main__":
    test_fashionxlm_moe()
