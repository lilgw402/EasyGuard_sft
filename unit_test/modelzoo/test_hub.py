import unittest

from unit_test import TEST_FLAGS

TEST_FLAGS = ["all"]


class TestHub(unittest.TestCase):
    def test_BaseAutoHubClass(self):
        from easyguard.modelzoo.hub import AutoHubClass

        class a(AutoHubClass):
            def __init__(
                self,
                server_name: str,
                archive_name: str,
                model_type: str,
                *args,
                **kwargs,
            ) -> None:
                super().__init__(
                    server_name,
                    archive_name,
                    model_type,
                    *args,
                    **kwargs,
                )

        x = a("hdfs", "fashionxlm-moe-base", "mdeberta_v2_moe", region="CN")
        x.get_file("config.json")


if __name__ == "__main__":
    unittest.main()
