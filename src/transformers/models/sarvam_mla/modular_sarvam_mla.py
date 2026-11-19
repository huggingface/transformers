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

from ...utils import auto_docstring, logging
from ..deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config


logger = logging.get_logger(__name__)

# Keys used by the original `sarvamai/sarvam-105b` config, renamed to their DeepSeek-V3 equivalents
LEGACY_KEY_RENAMES = {"num_experts": "n_routed_experts", "num_shared_experts": "n_shared_experts"}


@auto_docstring(checkpoint="sarvamai/sarvam-105b")
@strict(accept_kwargs=True)
class SarvamMLAConfig(DeepseekV3Config):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts. The released checkpoints route over all experts at once.
    topk_group (`int`, *optional*, defaults to 1):
        Number of selected groups for each token.
    first_k_dense_replace (`int`, *optional*, defaults to 1):
        Number of dense layers in shallow layers(embed->dense->moe->moe...->lm_head).
                                                        \--k dense layers--/
    rope_interleave (`bool`, *optional*, defaults to `True`):
        Whether to interleave the rotary position embeddings.
    num_mtp_layers (`int`, *optional*, defaults to 0):
        Number of Multi-Token Prediction (MTP) modules available to append after the base transformer model. The
        released checkpoints do not include MTP weights.

    Example:

    ```python
    >>> from transformers import SarvamMLAConfig

    >>> # Initializing a SarvamMLA style configuration
    >>> configuration = SarvamMLAConfig()

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "sarvam_mla"

    vocab_size: int = 262144
    hidden_size: int = 4096
    intermediate_size: int = 16384
    num_hidden_layers: int = 32
    num_attention_heads: int = 64
    num_key_value_heads: int | None = None
    n_routed_experts: int = 128
    q_lora_rank: int | None = None
    n_group: int | None = 1
    topk_group: int | None = 1
    first_k_dense_replace: int | None = 1
    initializer_range: float = 0.006
    num_mtp_layers: int = 0

    def __post_init__(self, **kwargs):
        # Translate the original Sarvam config format (written for the remote code) to the DeepSeek-V3 format
        legacy_keys = [key for key in (*LEGACY_KEY_RENAMES, "q_head_dim") if key in kwargs]
        for old_key, new_key in LEGACY_KEY_RENAMES.items():
            if old_key in kwargs:
                setattr(self, new_key, kwargs.pop(old_key))
        kwargs.pop("q_head_dim", None)
        # `head_dim` is derived; the original config stored `kv_lora_rank + qk_rope_head_dim` for vLLM's MLA
        head_dim = kwargs.pop("head_dim", None)
        if head_dim is not None and head_dim != self.qk_rope_head_dim:
            legacy_keys.append("head_dim")
        # `deepseek_yarn` is the same as standard YaRN
        rope_scaling = kwargs.get("rope_scaling") or {}
        if "deepseek_yarn" in (rope_scaling.get("type"), rope_scaling.get("rope_type")):
            kwargs["rope_scaling"] = {k: v for k, v in rope_scaling.items() if k != "type"} | {"rope_type": "yarn"}
            legacy_keys.append("rope_scaling.type='deepseek_yarn'")
        if legacy_keys:
            logger.warning_once(
                f"Loading a SarvamMLA config in the deprecated format (found {', '.join(legacy_keys)}). It has been "
                "converted to the DeepSeek-V3 format; re-save it with `config.save_pretrained()` to update it."
            )
        super().__post_init__(**kwargs)


__all__ = ["SarvamMLAConfig"]
