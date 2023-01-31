#!/usr/bin/env python
# TODO (junwei.Dong): 新开一个分支: 除了无法调整的auto机制以外【transformer实现的懒加载中的import module是写死的，无法通过修改全局变量来统一修改，所以auto机制好像还是要迁移过来】，除此之外，凡是要调用到transformers的地方全部精简为调包形式，这样既能精简代码让框架干净，又能实时对接transformers的最新改动, 做些映射来无缝对接hf模型，例如用自己的logging来接管tf的logging
import os

__version__ = "0.0.2"

from .core import AutoImageProcessor, AutoModel, AutoProcessor, AutoTokenizer
from .modelzoo.config import MODELZOO_CONFIG
from .utils import EASYGUARD_CACHE

# set the easyguard cache directory
# os.environ["EASYGUARD_CACHE"] = EASYGUARD_CACHE


# This line will be programatically read/write by setup.py.
# Leave them at the bottom of this file and don't touch them.
