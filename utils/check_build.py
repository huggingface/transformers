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

from pathlib import Path

def test_custom_files_are_present():
    transformers_path = Path.cwd() / "build/lib/transformers"
    # Test all the extensions added in the setup
    if not (transformers_path / "kernels/rwkv/wkv_cuda.cu").exists():
        return False
    if not (transformers_path / "kernels/rwkv/wkv_op.cpp").exists():
        return False
    if not (transformers_path / "models/deformable_detr/custom_kernel/ms_deform_attn.h").exists():
        return False
    if not (transformers_path / "models/deformable_detr/custom_kernel/cuda/ms_deform_im2col_cuda.cuh").exists():
        return False
    if not (transformers_path / "models/graphormer/algos_graphormer.pyx").exists():
        return False
    return True


if __name__ == "__main__":
    if not test_custom_files_are_present():
        raise ValueError(
            "The built release does not contain the custom files. Fix this before going further!"
        )