import os

from easyguard import __version__
from setuptools import find_packages, setup

path = os.path.abspath(os.path.dirname(__file__))
package_name = "byted-easyguard"
version = __version__
author = ["xiaochen.qiu", "dongjunwei"]
author_email = ["xiaochen.qiu@bytedance.com", "dongjunwei@bytedance.com"]
maintainer = "jwdong"
maintainer_email = "dongjuwnei@bytedance.com"
description = "EasyGuard is a cruise-based framework, a simple and powerful toolkit to make it easy to train and deploy nlp, cv, and multimodal models for e-commerce."
home_page = r"https://code.byted.org/ecom_govern/EasyGuard"

author = ", ".join(author)
author_email = ", ".join(author_email)

install_reuqires = [
    "byted-dataloader>=0.3.1,<=0.3.1",
    "jsonargparse>=4.8.0,<4.15.0",
    "fsspec>=2021.10,<2022.10",
    "dill>=0.3.6,<=0.3.6",
    "tqdm>=4.0,<6.0",
    "pyyaml>=6.0,<8.0",
    "protobuf>=3.17,<=3.20",
    "tensorflow-cpu>=2.2,<=2.8.1",
    "py-spy>=0.3.14,<=0.3.14",
    "tensorflow-io>=0.25.0, <=0.25.0",
    "pyarrow>=4.0.1,<7.0",
    "pandas>=1.3.5,<2.0",
    "timm>=0.6.7,<=0.6.7",
    "transformers>=4.21.0,<=4.26.1",
    "deepspeed>=0.7.3,<=0.7.3",
    "bytedtcc>=1.4.2,<=1.4.2",
    "sentencepiece>=0.1.97,<=0.1.97",
    "opencv-python>=4.7.0.72,<=4.7.0.72",
    "yacs>=0.1.8,<=0.1.8",
    "absl-py>=0.15.0,<=0.15.0",
    "bytedabase>=0.5.1,<=0.5.1",
    "prettytable>=3.6.0,<=3.6.0",
    "huggingface-hub>=0.10.0,<=0.10.0",
    "byted-cruise>=0.3.4,<=0.4.9",
    "matplotlib>=3.5.3,<=3.5.3",
]

setup(
    name=package_name,
    version=version,
    author=author,
    author_email=author_email,
    maintainer=maintainer,
    maintainer_email=maintainer_email,
    description=description,
    url=home_page,
    packages=find_packages(),
    include_package_data=True,
    # package_dir={"": "easyguard"},
    python_requires=">=3.7",
    install_requires=install_reuqires,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
    ],
)
