#!/usr/bin/python

from setuptools import setup


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name='longformer',
    version='0.1',
    packages=['longformer', 'longformer.lib', 'tvm', 'tvm._ffi', 'tvm._ffi._ctypes', 'tvm.contrib'],
    package_data={'tvm': ['*.so'], 'longformer': ['lib/*.so']},
    entry_points='',
    install_requires=required,
)

