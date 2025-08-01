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
from ..models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm, apply_rotary_pos_emb
from ..models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2MLP as Qwen25VLMLP
from ..models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm as Qwen25VLRMSNorm
from ..models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP, Qwen2MoeRMSNorm
from ..models.qwen2_vl.modeling_qwen2_vl import Qwen2MLP as Qwen2VLMLP
from ..models.qwen2_vl.modeling_qwen2_vl import Qwen2RMSNorm as Qwen2VLRMSNorm
from ..models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm
from ..models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP, Qwen3MoeRMSNorm


def npu_rms_norm(self, x):
    """
    Monkey patch for RMSNorm.forward to use torch_npu.npu_rms_norm()

    Refer to https://www.hiascend.com/document/detail/zh/Pytorch/
        700/ptmoddevg/trainingmigrguide/performance_tuning_0031.html
    """

    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]


def npu_silu(self, hidden_state):
    """
    Monkey patch for MLP.forward to use torch_npu.npu_swiglu()

    Refer to https://www.hiascend.com/document/detail/zh/Pytorch/
        700/ptmoddevg/trainingmigrguide/performance_tuning_0035.html
    """

    return self.down_proj(
        torch_npu.npu_swiglu(torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1), dim=-1)
    )


def npu_apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Monkey patch for apply_rotary_pos_emb to use torch_npu.npu_rotary_mul()

    Refer to https://www.hiascend.com/document/detail/zh/Pytorch/
        710/ptmoddevg/trainingmigrguide/performance_tuning_0030.html
    """

    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    cos = cos.repeat(1, 2)
    sin = sin.repeat(1, 2)
    q_embed = torch_npu.npu_rotary_mul(
        q.float(), cos.unsqueeze(0).unsqueeze(2).float(), sin.unsqueeze(0).unsqueeze(2).float()
    ).type_as(q)
    k_embed = torch_npu.npu_rotary_mul(
        k.float(), cos.unsqueeze(0).unsqueeze(2).float(), sin.unsqueeze(0).unsqueeze(2).float()
    ).type_as(k)
    return q_embed, k_embed


def npu_rms_norm_patch():
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

    LlamaRMSNorm.forward = npu_rms_norm
    Qwen2RMSNorm.forward = npu_rms_norm
    Qwen25VLRMSNorm.forward = npu_rms_norm
    Qwen2MoeRMSNorm.forward = npu_rms_norm
    Qwen2VLRMSNorm.forward = npu_rms_norm
    Qwen3RMSNorm.forward = npu_rms_norm
    Qwen3MoeRMSNorm.forward = npu_rms_norm
    print("Monkey patch RMSNorm.forward for npu_rms_norm on npu.")


def npu_silu_patch():
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

    LlamaMLP.forward = npu_silu
    Qwen2MLP.forward = npu_silu
    Qwen25VLMLP.forward = npu_silu
    Qwen2MoeMLP.forward = npu_silu
    Qwen2VLMLP.forward = npu_silu
    Qwen3MLP.forward = npu_silu
    Qwen3MoeMLP.forward = npu_silu
    print("Monkey patch MLP.forward for npu_silu on npu.")


def npu_apply_rotary_pos_emb_patch():
    """
    patch model list:
    {
        qwen2
    }
    """

    apply_rotary_pos_emb = npu_apply_rotary_pos_emb
    print("Monkey patch apply_rotary_pos_emb for npu_apply_rotary_pos_emb on npu.")


# Apply the patches
npu_rms_norm_patch()
npu_silu_patch()
npu_apply_rotary_pos_emb_patch()
