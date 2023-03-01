from setuptools import find_packages, setup

install_reuqires = [
    "byted-dataloader==0.3.1",
    "jsonargparse[signatures]==4.14",
    "fsspec",
    "dill",
    "tqdm",
    "pyyaml>=5.4.1",
    "protobuf==3.20",
    "tensorflow-cpu==2.8",
    "py-spy",
    "tensorflow-io==0.25.0",
    "pyarrow==4.0.1",
    "pandas==1.3.5",
    "timm==0.6.7",
    "transformers>=4.21.0",
    "deepspeed==0.7.3",
    "bytedtcc==1.4.2",
    "sentencepiece==0.1.97",
    "opencv-python",
    "yacs==0.1.8",
    "absl-py",
    "bytedabase",
    "prettytable",
    "huggingface-hub>=0.10.0",
    "byted-cruise==0.3.4",
    "matplotlib",
]
setup(
    name="easyguard",
    version="1.0.0",
    author="xiaochen.qiu",
    author_email="xiaochen.qiu@bytedance.com",
    maintainer="jwdong",
    maintainer_email="dongjuwnei@bytedance.com",
    description="EasyGuard is a cruise-based framework, it is primarily used for model training and storage",
    url="https://code.byted.org/ecom_govern/EasyGuard",
    packages=find_packages(),
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
