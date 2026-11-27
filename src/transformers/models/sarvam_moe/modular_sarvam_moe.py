# Copyright 2026 Sarvam AI and the HuggingFace Inc. team. All rights reserved.
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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..glm4_moe.configuration_glm4_moe import Glm4MoeConfig


@auto_docstring(checkpoint="sarvamai/sarvam-30b")
@strict(accept_kwargs=True)
class SarvamMoEConfig(Glm4MoeConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.
    first_k_dense_replace (`int`, *optional*, defaults to 1):
        Number of dense layers in shallow layers(embed->dense->moe->moe...->lm_head).
                                                        \--k dense layers--/
    num_mtp_layers (`int`, *optional*, defaults to 0):
        Number of Multi-Token Prediction (MTP) modules available to append after the base transformer model. The
        released checkpoints do not include MTP weights.

    Example:

    ```python
    >>> from transformers import AutoModelForCausalLM, SarvamMoEConfig

    >>> # Initializing a Sarvam-30B style configuration
    >>> configuration = SarvamMoEConfig()

    >>> # Initializing a model (which uses the GLM-4-MoE implementation) from that configuration
    >>> model = AutoModelForCausalLM.from_config(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "sarvam_moe"
    attribute_map = {
        "num_local_experts": "n_routed_experts",
        "num_mtp_layers": "num_nextn_predict_layers",
        # Names used by the original `sarvamai/sarvam-30b` config
        "num_experts": "n_routed_experts",
        "num_shared_experts": "n_shared_experts",
        "use_qkv_bias": "attention_bias",
    }
    default_theta = 8_000_000.0

    vocab_size: int = 262144
    intermediate_size: int = 8192
    num_hidden_layers: int = 19
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 64
    initializer_range: float = 0.006
    rms_norm_eps: float = 1e-6
    moe_intermediate_size: int = 1024
    num_experts_per_tok: int = 6
    routed_scaling_factor: float = 2.5
    use_qk_norm: bool = True
    num_mtp_layers: int = 0
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1

    def __post_init__(self, **kwargs):
        # Unlike GLM-4-MoE, rotary embeddings cover the full head
        kwargs.setdefault("partial_rotary_factor", 1.0)
        PreTrainedConfig.__post_init__(self, **kwargs)


__all__ = ["SarvamMoEConfig"]
