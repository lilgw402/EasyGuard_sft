from easyguard.utils.auxiliary_utils import *


def test_sha256(data: str) -> str:
    result = sha256(data)
    return result


def test_cache_file():
    return cache_file(
        "test",
        set(["vocab.txt", "vocab1.txt", "vocab2.txt"]),
        model_type="deberta",
        remote_url="hdfs://haruna/home/byte_ecom_govern/user/yangzheming/asr_model/zh_deberta_base_l6_emd_20210720/",
    )


def test_list_pretrained_models():
    list_pretrained_models()


if __name__ == "__main__":
    data = "hello world~"
    test_sha256(data)
    test_cache_file()
    test_list_pretrained_models()
