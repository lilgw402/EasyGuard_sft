import unittest

from unit_test import TEST_FLAGS

TEST_FLAGS = ["all"]


class TestLogging(unittest.TestCase):
    @unittest.skipUnless(
        "all" in TEST_FLAGS or "logging" in TEST_FLAGS, "just do it"
    )
    def test_logging(self):
        from easyguard.utils.logging import get_logger

        logger = get_logger(__name__)
        logger.info(f"hello, easyguard")


# from logging import getLogger
if __name__ == "__main__":
    unittest.main()
