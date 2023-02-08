from easyguard.utils.hdfs_utils import *


def test_hopen():
    hdfs_file = "hdfs://haruna/home/byte_ecom_govern/user/yangzheming/asr_model/zh_deberta_base_l6_emd_20210720/vocab.txt"
    vocab_set = set()
    with hopen(hdfs_file) as fr:
        for item in fr:
            item = str(item, encoding="utf-8")
            item = item.strip()
            vocab_set.add(item)
    for item in vocab_set:
        print(item)
        break

    is_bool = "中亚" in vocab_set
    print(is_bool)


if __name__ == "__main__":
    test_hopen()
