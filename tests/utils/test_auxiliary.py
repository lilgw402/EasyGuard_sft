from easyguard.utils import sha256, cache_file


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


if __name__ == "__main__":
    data = "hello world~"
    test_sha256(data)
    test_cache_file()
    ...
