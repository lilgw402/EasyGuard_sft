import unittest

# test module

TEST_FLAGS = ["get"]

from easyguard.utils.tos_utils import TOS
from easyguard.utils.logging import logging


logger = logging.getLogger(__name__)
tos = TOS()


class TestTOS(unittest.TestCase):
    @unittest.skipUnless(
        "all" in TEST_FLAGS or "exist" in TEST_FLAGS, "just do it"
    )
    def test_exist(self):
        # directory
        logger.info(tos.exist("fashionxlm_moe"))
        # file
        logger.info(tos.exist("config.json"))

    @unittest.skipUnless(
        "all" in TEST_FLAGS or "ls" in TEST_FLAGS, "just do it"
    )
    def test_ls(self):
        tos.ls("fashionxlm_moe")

    @unittest.skipUnless(
        "all" in TEST_FLAGS or "get" in TEST_FLAGS, "just do it"
    )
    def test_get(self):
        # directory
        tos.get("fashion_deberta_ner/config.json", "test_download/test_v1")
        # file
        tos.get("config.json", "test_tos")

    @unittest.skipUnless(
        "all" in TEST_FLAGS or "put" in TEST_FLAGS, "just do it"
    )
    def test_put(self):
        # directory
        tos.put(
            "/mnt/bn/ecom-govern-maxiangqian/dongjunwei/EasyGuard/test_download/fashionxlm_moe",
            "test_download/just_test",
        )
        # file
        tos.put("/mnt/bn/ecom-govern-maxiangqian/dongjunwei/test.txt")

    @unittest.skipUnless(
        "all" in TEST_FLAGS or "rm" in TEST_FLAGS, "just do it"
    )
    def test_rm(self):
        # directory
        logger.info(tos.rm("test_download/just_test"))
        # file
        logger.info(tos.rm("test.txt"))


if __name__ == "__main__":
    unittest.main()
