from easyguard.utils.import_utils import lazy_model_import


def test_lazy_model_import():
    module_name = "models.bert"
    lazy_model_import(module_name)


if __name__ == "__main__":
    test_lazy_model_import()
    ...
