# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-03-02 14:29:11
# Modified: 2023-03-02 14:29:11
# coding=utf-8
# Email: panziqi@bytedance.com
# Create: 2021/3/24 2:42 下午
from torch.nn import DataParallel, Sequential
from torch.nn.parallel import DistributedDataParallel
from examples.live_gandalf.utils.registry import Registry
from examples.live_gandalf.utils.driver import get_logger

MODELS = Registry("model")
DATASETS = Registry('dataset')
FEATURE_PROVIDERS = Registry("feature_provider")
TRAINERS = Registry("trainer")

def get_model(model_type):
    model = MODELS.get(model_type)
    return model

def get_feature_provider(feature_provider_type):
    feature_provider = FEATURE_PROVIDERS.get(feature_provider_type)
    return feature_provider

def get_module(root_module, module_path):
    module_names = module_path.split('.')
    module = root_module
    for module_name in module_names:
        if not hasattr(module, module_name):
            if isinstance(module, (DataParallel, DistributedDataParallel)):
                module = module.module
                if not hasattr(module, module_name):
                    if isinstance(module, Sequential) and module_name.isnumeric():
                        module = module[int(module_name)]
                    else:
                        get_logger().info('`{}` of `{}` could not be reached in `{}`'.format(module_name, module_path,
                                                                                       type(root_module).__name__))
                else:
                    module = getattr(module, module_name)
            elif isinstance(module, Sequential) and module_name.isnumeric():
                module = module[int(module_name)]
            else:
                get_logger().info('`{}` of `{}` could not be reached in `{}`'.format(module_name, module_path,
                                                                               type(root_module).__name__))
                return None
        else:
            module = getattr(module, module_name)
    return module
