# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import importlib
from pathlib import Path


# Test all the extensions added in the setup
FILES_TO_FIND = [
    "kernels/rwkv/wkv_cuda.cu",
    "kernels/rwkv/wkv_op.cpp",
    "kernels/deformable_detr/ms_deform_attn.h",
    "kernels/deformable_detr/cuda/ms_deform_im2col_cuda.cuh",
    "kernels/falcon_mamba/selective_scan_with_ln_interface.py",
    "kernels/falcon_mamba/__init__.py",
    "kernels/__init__.py",
    "models/graphormer/algos_graphormer.pyx",
]


def test_custom_files_are_present(transformers_path):
    # Test all the extensions added in the setup
    for file in FILES_TO_FIND:
        if not (transformers_path / file).exists():
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_lib", action="store_true", help="Whether to check the build or the actual package.")
    args = parser.parse_args()
    if args.check_lib:
        transformers_module = importlib.import_module("transformers")
        transformers_path = Path(transformers_module.__file__).parent
    else:
        transformers_path = Path.cwd() / "build/lib/transformers"
    if not test_custom_files_are_present(transformers_path):
        raise ValueError("The built release does not contain the custom files. Fix this before going further!")
