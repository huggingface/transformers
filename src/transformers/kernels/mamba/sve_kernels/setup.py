# coding=utf-8
# Copyright 2025 FUJITSU LIMITED
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


extensions = [
    Extension(
        "seq_sve",
        sources=["seq_sve.pyx", "helper.cpp"],  # Add your C++ source here
        include_dirs=[np.get_include()],  # Include the path to seq_sve.h
        language="c++",
        extra_compile_args=["-fopenmp", "-O3", "-march=armv8-a+sve", "-ftree-vectorize"],  # Enable OpenMP
        extra_link_args=["-fopenmp", "-O3", "-march=armv8-a+sve", "-ftree-vectorize"],
    )
]

setup(
    name="seq_sve",
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
)
