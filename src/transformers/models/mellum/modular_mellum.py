# Copyright 2026 JetBrains and the HuggingFace Inc. team. All rights reserved.
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

import torch
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import (
    ROPE_INIT_FUNCTIONS,
    RopeParameters,
)
from ...utils import auto_docstring, logging
from ..laguna.modeling_laguna import LagunaModel, LagunaRotaryEmbedding
from ..qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from ..qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoePreTrainedModel,
    Qwen3MoeRMSNorm,
    Qwen3MoeSparseMoeBlock,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="JetBrains/Mellum2-12B-A2.5B-Base")
@strict
class MellumConfig(Qwen3MoeConfig):
    r"""
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer MLP type — `"dense"` or `"sparse"`. Length must equal
        `num_hidden_layers`. Defaults to all sparse.

    ```python
    >>> from transformers import MellumModel, MellumConfig

    >>> configuration = MellumConfig()
    >>> model = MellumModel(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "mellum"

    vocab_size: int = 98304
    hidden_size: int = 2304
    intermediate_size: int = 7168
    num_hidden_layers: int = 28
    head_dim: int = 128
    max_position_embeddings: int = 131072
    sliding_window: int | None = 1024
    num_experts: int = 64
    moe_intermediate_size: int = 896
    norm_topk_prob: bool = True
    layer_types: list[str] | None = None
    mlp_layer_types: list[str] | None = None
    rope_parameters: dict | RopeParameters | None = None

    use_sliding_window = AttributeError()
    decoder_sparse_step = AttributeError()
    mlp_only_layers = AttributeError()

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["sparse"] * self.num_hidden_layers

        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {"rope_type": "default", "rope_theta": 500000.0},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            }

        PreTrainedConfig.__post_init__(
            self,
            **kwargs,
            ignore_keys_at_rope_validation={"sliding_attention", "full_attention"},
        )

    def convert_rope_params_to_dict(self, **kwargs):
        # No need to handle BC for new models, because they have no old-format `rope_scaling`
        return kwargs

    def validate_architecture(self):
        """Part of `@strict`-powered validation."""
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types length ({len(self.layer_types)}) "
                f"must equal num_hidden_layers ({self.num_hidden_layers})."
            )
        if len(self.mlp_layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"mlp_layer_types length ({len(self.mlp_layer_types)}) "
                f"must equal num_hidden_layers ({self.num_hidden_layers})."
            )


class MellumRotaryEmbedding(LagunaRotaryEmbedding):
    pass


class MellumAttention(Qwen3MoeAttention):
    def __init__(self, config: MellumConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None


class MellumMLP(Qwen3MoeMLP):
    pass


class MellumSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    pass


class MellumRMSNorm(Qwen3MoeRMSNorm):
    pass


class MellumDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config: MellumConfig, layer_idx: int):
        torch.nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = MellumAttention(config, layer_idx)
        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = MellumSparseMoeBlock(config)
        else:
            self.mlp = MellumMLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = MellumRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MellumRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class MellumPreTrainedModel(Qwen3MoePreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, MellumRotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


class MellumModel(LagunaModel):
    pass


class MellumForCausalLM(Qwen3MoeForCausalLM):
    pass


__all__ = [
    "MellumConfig",
    "MellumForCausalLM",
    "MellumModel",
    "MellumPreTrainedModel",
]
