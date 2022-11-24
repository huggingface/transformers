# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Loading of Deformable DETR's CUDA kernels"""

import os


def load_cuda_kernels():
    from torch.utils.cpp_extension import load

    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "custom_kernel")
    src_files = [
        os.path.join(root, filename)
        for filename in [
            "vision.cpp",
            os.path.join("cpu", "ms_deform_attn_cpu.cpp"),
            os.path.join("cuda", "ms_deform_attn_cuda.cu"),
        ]
    ]

    load(
        "MultiScaleDeformableAttention",
        src_files,
        # verbose=True,
        with_cuda=True,
        extra_include_paths=[root],
        # build_directory=os.path.dirname(os.path.realpath(__file__)),
        extra_cflags=["-DWITH_CUDA=1"],
        extra_cuda_cflags=[
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ],
    )

    import MultiScaleDeformableAttention as MSDA

    return MSDA
