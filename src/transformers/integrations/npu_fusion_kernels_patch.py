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

import torch
import torch_npu

from ..models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm
from ..models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm
from ..models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2MLP as Qwen25VLMLP
from ..models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm as Qwen25VLRMSNorm
from ..models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP, Qwen2MoeRMSNorm
from ..models.qwen2_vl.modeling_qwen2_vl import Qwen2MLP as Qwen2VLMLP
from ..models.qwen2_vl.modeling_qwen2_vl import Qwen2RMSNorm as Qwen2VLRMSNorm
from ..models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm
from ..models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP, Qwen3MoeRMSNorm


def rms_norm_forward(self, x):
    """
    Refer to https://www.hiascend.com/document/detail/zh/Pytorch/
        700/ptmoddevg/trainingmigrguide/performance_tuning_0031.html

    Using the API: torch_npu.npu_rms_norm(x, gamma, epsilon)
    """

    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]


def silu_forward(self, hidden_state):
    """
    Refer to https://www.hiascend.com/document/detail/zh/Pytorch/
        700/ptmoddevg/trainingmigrguide/performance_tuning_0035.html

    Using the API: torch_npu.npu_swiglu(x, dim)
    """

    return self.down_proj(
        torch_npu.npu_swiglu(torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1), dim=-1)
    )


def npu_fusion_patch_rms_norm_forward():
    """
    patch model list:
    {
        llama,
        qwen2,
        qwen2_5_vl,
        qwen2_moe,
        qwen2_vl.
        qwen3,
        qwen3_moe
    }
    """

    LlamaRMSNorm.forward = rms_norm_forward
    Qwen2RMSNorm.forward = rms_norm_forward
    Qwen25VLRMSNorm.forward = rms_norm_forward
    Qwen2MoeRMSNorm.forward = rms_norm_forward
    Qwen2VLRMSNorm.forward = rms_norm_forward
    Qwen3RMSNorm.forward = rms_norm_forward
    Qwen3MoeRMSNorm.forward = rms_norm_forward
    print("Monkey patch RMSNorm.forward for rms_norm_forward on npu.")


def npu_fusion_patch_silu_forward():
    """
    patch model list:
    {
        llama,
        qwen2,
        qwen2_5_vl,
        qwen2_moe,
        qwen2_vl.
        qwen3,
        qwen3_moe
    }
    """

    LlamaMLP.forward = silu_forward
    Qwen2MLP.forward = silu_forward
    Qwen25VLMLP.forward = silu_forward
    Qwen2MoeMLP.forward = silu_forward
    Qwen2VLMLP.forward = silu_forward
    Qwen3MLP.forward = silu_forward
    Qwen3MoeMLP.forward = silu_forward
    print("Monkey patch MLP.forward for silu_forward on npu.")


npu_fusion_patch_rms_norm_forward()
npu_fusion_patch_silu_forward()
