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


def test_hf_name_or_path_check():
    name_or_path = "fashion-deberta-ccr-order"
    model_url = "hdfs://haruna/home/byte_ecom_govern/easyguard/models/fashion_deberta_ccr_order"
    file_name = "vocab.txt"
    model_type = "debert"
    print(hf_name_or_path_check(name_or_path, model_url, file_name, model_type))


if __name__ == "__main__":
    data = "hello world~"
    test_sha256(data)
    # test_cache_file()
    # test_list_pretrained_models()
    test_hf_name_or_path_check()
