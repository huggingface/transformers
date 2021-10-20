# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""This code is copied fron NVIDIA apex:
      https://github.com/NVIDIA/apex
   with some changes. """

import numbers
import torch
from torch.nn.parameter import Parameter
from torch.nn import init
import importlib
import torch.nn.functional as F

@torch.jit.script
def gelu_megatron_fwd(x):
    return  x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

@torch.jit.script
def gelu_megatron_bwd(g, x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff*g

class GeLUMegatronFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return gelu_megatron_fwd(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        tmp = gelu_megatron_bwd(grad_output, input)
        return tmp, tmp


class UpcastLayerNorm(torch.nn.LayerNorm):

  def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
      super(UpcastLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)


  def forward(self, input):
    if input.dtype == torch.half:
      return F.layer_norm(input.float(), 
              self.normalized_shape, self.weight.float(), self.bias.float(), self.eps).half()

    return F.layer_norm(input, 
            self.normalized_shape, self.weight, self.bias, self.eps)

