from easyguard import AutoModel, AutoTokenizer


def test_fashionxlm():
    text = "good good study, day day up!"

    archive = "fashionxlm-base"
    tokenizer = AutoTokenizer.from_pretrained(archive)
    inputs = tokenizer(text, return_tensors="pt", max_length=84)
    model = AutoModel.from_pretrained(archive)
    ouputs = model(**inputs)
    print(ouputs)


if __name__ == "__main__":
    test_fashionxlm()
