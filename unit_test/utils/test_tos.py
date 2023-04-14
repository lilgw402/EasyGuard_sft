import unittest

# test module

TEST_FLAGS = ["all"]


class TestTOS(unittest.TestCase):
    @unittest.skipUnless(
        "all" in TEST_FLAGS or "put" in TEST_FLAGS, "just do it"
    )
    def test_put(self):
        from easyguard.utils.tos_utils import TOS

        tos = TOS()
        tos.get("fashionxlm_moe", "test_download/abab6")


if __name__ == "__main__":
    unittest.main()
