# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Wrapping functions to bridge frameworks with DLPack support to TVM"""
from .. import ndarray

def convert_func(tvm_func, tensor_type, to_dlpack_func):
    """Convert a tvm function into one that accepts a tensor from another
       framework, provided the other framework supports DLPACK

    Parameters
    ----------
    tvm_func: Function
        Built tvm function operating on arrays

    tensor_type: Type
        Type of the tensors of the target framework

    to_dlpack_func: Function
        Function to convert the source tensors to DLPACK
    """
    assert callable(tvm_func)

    def _wrapper(*args):
        args = tuple(ndarray.from_dlpack(to_dlpack_func(arg))\
            if isinstance(arg, tensor_type) else arg for arg in args)
        return tvm_func(*args)

    return _wrapper

def to_pytorch_func(tvm_func):
    """Convert a tvm function into one that accepts PyTorch tensors

    Parameters
    ----------
    tvm_func: Function
        Built tvm function operating on arrays

    Returns
    -------
    wrapped_func: Function
        Wrapped tvm function that operates on PyTorch tensors
    """
    import torch
    import torch.utils.dlpack
    return convert_func(tvm_func, torch.Tensor, torch.utils.dlpack.to_dlpack)
