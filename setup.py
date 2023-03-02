import os

from easyguard import __version__
from setuptools import find_packages, setup

# meta information
path = os.path.abspath(os.path.dirname(__file__))
package_name = "byted-easyguard"
version = __version__
author = ["xiaochen.qiu", "dongjunwei"]
author_email = ["xiaochen.qiu@bytedance.com", "dongjunwei@bytedance.com"]
maintainer = "jwdong"
maintainer_email = "dongjuwnei@bytedance.com"
description = "EasyGuard is a cruise-based framework, a simple and powerful toolkit to make it easy to train and deploy nlp, cv, and multimodal models for e-commerce."
home_page = r"https://code.byted.org/ecom_govern/EasyGuard"
# preprocess
author = ", ".join(author)
author_email = ", ".join(author_email)
#  dependencies
install_requires = open("requirements.txt").read().split("\n")


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
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
    ],
)
