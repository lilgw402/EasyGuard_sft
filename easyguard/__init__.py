#!/usr/bin/env python
import os

__version__ = "0.0.2"

from .utils import EASYGUARD_CACHE

# set the easyguard cache directory
os.environ["EASYGUARD_CACHE"] = EASYGUARD_CACHE

# This line will be programatically read/write by setup.py.
# Leave them at the bottom of this file and don't touch them.
