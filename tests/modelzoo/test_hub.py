from easyguard.modelzoo.hub import *


def test_BaseAutoHubClass():
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
    test_BaseAutoHubClass()
