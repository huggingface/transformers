# flake8: noqa
# coding=utf-8
# Copyright 2020, Microsoft and the HuggingFace Inc. team.
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

"""
Logging util @Author: penhe@microsoft.com
"""

""" Utils for torch jit tracing customer operators/functions"""
import os

import torch


def traceable(cls):
    class _Function(object):
        @staticmethod
        def apply(*args):
            if torch.onnx.is_in_onnx_export():
                return cls.forward(_Function, *args)
            else:
                return cls.apply(*args)

        @staticmethod
        def save_for_backward(*args):
            pass

    return _Function


class TraceMode:
    """Trace context used when tracing modules contains customer operators/Functions"""

    def __enter__(self):
        os.environ["JIT_TRACE"] = "True"
        return self

    def __exit__(self, exp_value, exp_type, trace):
        del os.environ["JIT_TRACE"]
