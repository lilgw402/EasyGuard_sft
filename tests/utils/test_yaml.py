from easyguard.utils.yaml_utils import *
from easyguard.modelzoo import MODEL_CONFIG_PATH


class ModelZooYaml(YamlConfig):
    def check(self):
        ...

    def test(self):
        print("123")


if __name__ == "__main__":
    # config_yaml = ModelZooYaml.yaml_reader(MODEL_CONFIG_PATH)
    data = {"name": "djw", "indicators": [1, 2, 3, 4]}
    path = (
        r"/mnt/bn/ecom-govern-maxiangqian/dongjunwei/EasyGuard/tests/utils/output.yaml"
    )
    json2yaml(path, data)
    ...
