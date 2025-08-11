#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

from setuptools import find_packages, setup

# Get the version from sam3/__init__.py
with open(os.path.join(os.path.dirname(__file__), "sam3", "__init__.py"), "r") as f:
    content = f.read()
    version_match = re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', content, re.M)
    if version_match:
        version = version_match.group(1)
    else:
        version = "0.1.0"  # Default version if not found

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sam3",
    version=version,
    author="Meta AI Research",
    author_email="kalyanv@meta.com",  # Replace with appropriate email
    description="SAM3 (Segment Anything Model 3) implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/sam3",  # Replace with appropriate URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "timm>=1.0.17",
        "numpy<2",
        "tqdm",
        "ftfy==6.1.1",
        "regex",
        "iopath>=0.1.10",
        "typing_extensions",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black==24.2.0",
            "ufmt==2.8.0",
            "ruff-api==0.1.0",
            "usort==1.0.2",
            "gitpython==3.1.31",
        ],
        "examples": [
            "matplotlib",
            "jupyter",
            "notebook",
            "ipywidgets",
        ],
    },
)
