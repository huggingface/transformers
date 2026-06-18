# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

"""PyTorch Phi-MoE model."""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/Phi-3.5-MoE-instruct")
@strict
class PhimoeConfig(PreTrainedConfig):
    r"""
    num_local_experts (`int`, *optional*, defaults to 16):
        Number of experts per Sparse MLP layer.
    input_jitter_noise (`float`, *optional*, defaults to 0.0):
        Input jitter noise
    lm_head_bias (`bool`, *optional*, defaults to `False`):
        LM head bias

    Example:

    ```python
    >>> from transformers import PhimoeModel, PhimoeConfig
    >>> # Initializing a Phi-3 style configuration
    >>> configuration = PhimoeConfig.from_pretrained("microsoft/Phi-3.5-MoE-instruct")
    >>> # Initializing a model from the configuration
    >>> model = PhimoeModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "phimoe"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1000000.0

    vocab_size: int = 32064
    hidden_size: int = 4096
    intermediate_size: int = 6400
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096 * 32
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    sliding_window: int | None = None
    attention_dropout: float | int = 0.0
    num_experts_per_tok: int = 2
    num_local_experts: int = 16
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.01
    input_jitter_noise: float = 0.0
    attention_bias: bool = False
    lm_head_bias: bool = False

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)

    def validate_rope(self):
        """
        Validate the `rope_parameters` configuration.
        """
        super().validate_rope()

        # Run model-specific rope validation
        if self.rope_parameters["rope_type"] != "default":
            if "original_max_position_embeddings" in self.rope_parameters:
                self.original_max_position_embeddings = self.rope_parameters["original_max_position_embeddings"]
            rope_parameters_short_mscale = self.rope_parameters.get("short_mscale", None)
            rope_parameters_long_mscale = self.rope_parameters.get("long_mscale", None)
            if not isinstance(rope_parameters_short_mscale, (int, float)):
                raise TypeError(
                    f"`rope_parameters`'s short_mscale field must be a number, got {rope_parameters_short_mscale}"
                )
            if not isinstance(rope_parameters_long_mscale, (int, float)):
                raise TypeError(
                    f"`rope_parameters`'s long_mscale field must be a number, got {rope_parameters_long_mscale}"
                )


__all__ = ["PhimoeConfig"]
