#!/usr/bin/env python
import time
import sys
import os
import logging as python_logging

__version__ = "0.0.2"

from .modelzoo.tokenization_utils import PreTrainedTokenizer

_logger = python_logging.getLogger("fex")
_logger.setLevel(python_logging.INFO)
_logger.propagate = False

head = "%(asctime)-15s %(filename)s:%(lineno)d [%(levelname)s] %(message)s"
formatter = python_logging.Formatter(head, datefmt="%Y-%m-%d %H:%M:%S")
try:
    from colorlog import ColoredFormatter

    color_head = "%(log_color)s %(asctime)-15s %(filename)s:%(lineno)d [%(levelname)s] %(message)s %(reset)s"
    color_formatter = ColoredFormatter(color_head, datefmt="%Y-%m-%d %H:%M:%S")
except:
    color_formatter = formatter
    _logger.warn(
        "[NOTICE] colorlog python package is not installed, will display log in normal mode"
    )

base_path = "/tmp/"
log_file = "{}_{}.log".format("fex_training", time.strftime("%Y-%m-%d-%H-%M-%S"))
log_file = os.path.join(base_path, log_file)
fh = python_logging.FileHandler(filename=log_file)
fh.setFormatter(formatter)
_logger.addHandler(fh)

consoleHandler = python_logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(color_formatter)
_logger.addHandler(consoleHandler)

# This line will be programatically read/write by setup.py.
# Leave them at the bottom of this file and don't touch them.
