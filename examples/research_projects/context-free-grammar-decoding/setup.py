from setuptools import setup, find_packages

setup(
    name="transformers_cfg",
    version="0.1.0",
    author="EPFL-dlab",
    author_email="saibo.geng@epfl.ch",
    description="Extension of Transformers library for Context-Free Grammar Constrained Decoding",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/epfl-dlab/transformers-CFG",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=open('requirements.txt').read().splitlines(),
    package_data={
        "transformers_cfg": ["examples/grammars/*.ebnf"],
    },
    include_package_data=True,
    # Add any additional package configuration here
)
