import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


def get_version():
    version_file = 'titan/version.py'
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


install_requires = [
    'timm>=0.4.12,<1.0.0',
    'numpy>=1.19.0,<2.0.0',
    'ptflops>=0.5.1,<1.0.0',
    'torchlibrosa>=0.0.9,<1.0.0',
    'einops>=0.4.0,<1.0.0',
    'transformers>=4.6.0,<5.0.0',
    'matplotlib>=3.5.0,<4.0.0',
    'bytedtos>=0.1.0,<1.0.0',
    'bytedfeather>=0.1.5,<1.0.0',
    'byted-torch>=1.10.0.post13,<2.0',
    'byted-janus>=0.1.3,<1.0'
]


setuptools.setup(
    name="bytedtitan",
    version=get_version(),
    author="Xin Ji, Xinyu Li, Junda Zhang",
    author_email="xin.ji@bytedance.com, lixinyu.arthur@bytedance.com, junda.zhang@bytedance.com",
    description="Titan, a collective deep learning model zoo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="ByteDance, Model Zoo, PyTorch",
    url="https://code.byted.org/lab/titan",
    packages=setuptools.find_packages(exclude=['rh2_jobs', 'rh2_jobs/*']),
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3',
    install_requires=install_requires,
    dependency_links=[
        'https://pypi.byted.org/simple',
        'https://bytedpypi.byted.org/simple'
    ],
    zip_safe=False,
)
