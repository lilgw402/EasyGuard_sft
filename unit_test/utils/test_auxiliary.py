import unittest

from unit_test import TEST_FLAGS

TEST_FLAGS = ["cache_file"]


class TestAuxiliary(unittest.TestCase):
    @unittest.skipUnless(
        "all" in TEST_FLAGS or "sha256" in TEST_FLAGS, "just do it"
    )
    def test_sha256(self) -> str:
        from easyguard.utils.auxiliary_utils import sha256

        data = "bert"
        result = sha256(data)
        print(result)

    @unittest.skipUnless(
        "all" in TEST_FLAGS or "cache_file" in TEST_FLAGS, "just do it"
    )
    def test_cache_file(self):
        from easyguard.utils.auxiliary_utils import cache_file

        print(
            cache_file(
                "test",
                set(
                    [
                        "pytorch_model.bin",
                        "pytorch_model.ckpt",
                        "pytorch_model.pt",
                        "pytorch_model.th",
                    ]
                ),
                model_type="deberta",
                remote_url="hdfs://haruna/home/byte_ecom_govern/easyguard/models/bert_base_uncased",
            )
        )

    @unittest.skipUnless(
        "all" in TEST_FLAGS or "list_pretrained_models" in TEST_FLAGS,
        "just do it",
    )
    def test_list_pretrained_models(self):
        from easyguard.utils.auxiliary_utils import list_pretrained_models

        list_pretrained_models()

    @unittest.skipUnless(
        "all" in TEST_FLAGS or "hf_name_or_path_check" in TEST_FLAGS,
        "just do it",
    )
    def test_hf_name_or_path_check(self):
        from easyguard.utils.auxiliary_utils import hf_name_or_path_check

        name_or_path = "fashion-deberta-ccr-order"
        model_url = "hdfs://haruna/home/byte_ecom_govern/easyguard/models/fashion_deberta_ccr_order"
        file_name = "vocab.txt"
        model_type = "debert"
        print(hf_name_or_path_check(name_or_path, model_url, model_type))

    @unittest.skipUnless(
        "all" in TEST_FLAGS or "convert_model_weight" in TEST_FLAGS,
        "just do it",
    )
    def test_convert_model_weight(self):
        from easyguard.utils.auxiliary_utils import convert_model_weights

        path = "/mnt/bn/ecom-govern-maxiangqian/dongjunwei/cache/epoch=9-step=970000-val_loss=0.749.ckpt"
        convert_model_weights(path, "backbone.", remove_old=False)


if __name__ == "__main__":
    unittest.main()
