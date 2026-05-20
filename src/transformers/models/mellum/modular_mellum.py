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
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_rope_utils import (
    ROPE_INIT_FUNCTIONS,
    RopeParameters,
)
from ...processing_utils import Unpack
from ...utils import auto_docstring, logging
from ...utils.generic import TransformersKwargs
from ..gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
from ..qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from ..qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoePreTrainedModel,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="JetBrains/Mellum2-12B-A2.5B-Base")
@strict
class MellumConfig(Qwen3MoeConfig):
    r"""
    mlp_only_layers (`list[int]`, *optional*, defaults to `[]`):
        Layers that use a dense MLP instead of a sparse MoE block.
    rope_theta (`float`, *optional*, defaults to 500000.0):
        Base period of the RoPE embeddings.
    layer_types (`list[str]`, *optional*):
        Per-layer attention type — `"full_attention"` or `"sliding_attention"`.
        Length must equal `num_hidden_layers`. Defaults to all `"full_attention"`.

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
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    max_position_embeddings: int = 131072
    use_sliding_window: bool = True
    sliding_window: int | None = 1024
    max_window_layers: int = 0
    num_experts: int = 64
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 896
    norm_topk_prob: bool = True
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    rope_theta: float = 500000.0

    layer_types: list[str] | None = None
    rope_parameters: dict | RopeParameters | None = None

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        self.mlp_only_layers = [] if self.mlp_only_layers is None else self.mlp_only_layers

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
        # Same no-op as Laguna/Gemma3: the base implementation assumes
        # a flat `rope_parameters` dict and would add a top-level
        # `rope_theta` key, breaking the per-layer-type format.
        return kwargs


class MellumRotaryEmbedding(Gemma3RotaryEmbedding):
    def __init__(self, config: MellumConfig):
        super().__init__(config)

    @staticmethod
    def compute_default_rope_parameters(
        config: MellumConfig | None = None,
        device: "torch.device | None" = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple["torch.Tensor", float]:
        # Logic is identical to Gemma3 but must be inlined due to codegen
        # limitations (no cross-imports; `pass` pulls in the base method
        # with the wrong type annotation).
        base = config.rope_parameters[layer_type]["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor


class MellumAttention(Qwen3MoeAttention):
    def __init__(self, config: MellumConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_type = config.layer_types[layer_idx]
        if config.use_sliding_window and self.layer_type == "sliding_attention":
            self.sliding_window = config.sliding_window
        else:
            self.sliding_window = None


class MellumDecoderLayer(Qwen3MoeDecoderLayer):
    # Stub required as the converter uses this to rename Qwen3MoeDecoderLayer
    # references to MellumDecoderLayer in the generated file.
    pass


class MellumPreTrainedModel(Qwen3MoePreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # Per-layer-type RoPE stores buffers under dynamic names like
        # `full_attention_inv_freq` that the base _init_weights doesn't know
        # about. Therefore, they need to be initialized here.
        if isinstance(module, MellumRotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


class MellumModel(Qwen3MoeModel):
    def __init__(self, config: MellumConfig):
        super().__init__(config)
        self.rotary_emb = MellumRotaryEmbedding(config=config)
        self.has_sliding_layers = "sliding_attention" in config.layer_types

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in set(self.config.layer_types):
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class MellumForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = MellumModel(config)


__all__ = [
    "MellumConfig",
    "MellumForCausalLM",
    "MellumModel",
    "MellumPreTrainedModel",
]
