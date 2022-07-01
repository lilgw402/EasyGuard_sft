#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Huang Wenguan (huangwenguan@bytedance.com)
Date: 2020-11-10 14:03:33
LastEditTime: 2020-11-10 15:15:13
LastEditors: Huang Wenguan
Description: setup script for fex
'''

import re
import os
import io
import setuptools

CUR_DIR = os.path.dirname(__file__)


def read(*names, **kwargs):
    with io.open(
        os.path.join(CUR_DIR, *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    return "0.1.0"


def load_requirements(file_path):
    ret = []
    try:
        with open(os.path.join(CUR_DIR, file_path), "r") as f:
            for line in f.readlines():
                if not line.startswith("#"):
                    ret.append(line.strip())
    except Exception as e:
        print(
            "Warning: Loading [{}] failed!\n".format(file_path),
            "You can install them later.\n"
            "Exception:", e
        )
        return []
    return ret


with open("README.md", "r") as fh:
    long_description = fh.read()


VERSION = find_version('fex', '__init__.py')

REQUIREMENTS = load_requirements("requirements.txt")

setuptools.setup(
    name="fex",  # Replace with your own username
    version=VERSION,
    author="huangwenguan",
    author_email="search_nlp@bytedance.com",
    description="Cross Modal training library for Bytedance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://code.byted.org/nlp/fex",
    packages=setuptools.find_packages(exclude=['docs.*', 'docs', 'tools', 'example']),
    package_data={"": ["*.so.*", "*.so"]},
    include_package_data=True,
    entry_points={'console_scripts': [
        'FEXRUN=fex_cli.fexrun:cli'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=REQUIREMENTS
)
