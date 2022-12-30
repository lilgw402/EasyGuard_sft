from easyguard.core import AutoModel, AutoTokenizer

archive = "bert-base-uncased"
# archive = "facebook/bart-large"


def auto_model_test():
    tokenizer = AutoTokenizer.from_pretrained(archive)
    model = AutoModel.from_pretrained(archive)
    inputs = tokenizer("Hello world!", return_tensors="pt")
    print(inputs)
    ouputs = model(**inputs)
    print(ouputs)


if __name__ == "__main__":
    auto_model_test()
