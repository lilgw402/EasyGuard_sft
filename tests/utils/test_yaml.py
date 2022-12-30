from EasyGuard.easyguard.utils.yaml_utils import *
from easyguard.modelzoo import MODEL_CONFIG_PATH


class ModelZooYaml(YamlConfig):
    def check(self):
        ...

    def test(self):
        print("123")


if __name__ == "__main__":
    config_yaml = ModelZooYaml.yaml_reader(MODEL_CONFIG_PATH)
    
    ...
