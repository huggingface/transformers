# Copyright 2025
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

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import layer_type_validation
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
)
from ..qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RotaryEmbedding


logger = logging.get_logger(__name__)


class CwmConfig(LlamaConfig):
    """
    Configuration for Code World Model (CWM).
    This is an inherited Llama3-compatible configuration with layer-interleaved
    sliding-window attention. Configures a `CwmModel`. Designed to yield a configuration mirroring the model in the
    [facebook/cwm](https://huggingface.co/facebook/cwm) architecture by default. Other models include:
    - [facebook/cwm-sft](https://huggingface.co/facebook/cwm-sft)
    - [facebook/cwm-pretrain](https://huggingface.co/facebook/cwm-pretrain)

    Args:
        vocab_size (`int`, *optional*, defaults to 128256):
            Vocabulary size of the CWM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`CwmModel`]
        hidden_size (`int`, *optional*, defaults to 6144):
            Dimension of the hidden representations
        intermediate_size (`int`, *optional*, defaults to 21504):
            Dimension of the MLP representations
        num_hidden_layers (`int`, *optional*, defaults to 64):
            Number of hidden layers in the Transformer decoder
        num_attention_heads (`int`, *optional*, defaults to 48):
            Number of attention heads for each attention layer in the Transformer decoder
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention (GQA).
            If it is not specified, will default to `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with. CWM's attention allows sequence
            lengths up to 131072 tokens.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        eos_token_id (`int` or `list[int]`, *optional*, defaults to `[128001, 128008, 128009]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        bos_token_id (`int`, *optional*, defaults to 128000):
            The id of the *beginning-of-sequence* token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Tensor parallelism degree used during pretraining. See [this
            document](https://huggingface.co/docs/transformers/parallelism) and [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        sliding_window (`int`, *optional*, defaults to 8192):
            Sliding window attention window size.
        layer_types (`List[str]`, *optional*):
            List of layer types for each layer. Each element should be either "full_attention" or "sliding_attention".
            If not specified, will default to alternating pattern based on the provided window pattern.
    """

    model_type = "cwm"
    default_theta = 1_000_000.0

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 6144,
        intermediate_size: int = 21504,
        num_hidden_layers: int = 64,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        eos_token_id=[128001, 128008, 128009],
        bos_token_id: int = 128000,
        tie_word_embeddings: bool = False,
        attention_dropout: float = 0.0,
        pretraining_tp: int = 1,
        mlp_bias: bool = False,
        rope_parameters: dict | None = None,
        # CWM interleaved sliding window fields
        sliding_window: int = 8192,
        layer_types: list[str] | None = None,  # ["full_attention"|"sliding_attention"] per layer
        **kwargs,
    ):
        if rope_parameters is None:
            rope_parameters = {
                "rope_theta": 1_000_000.0,
                "factor": 16.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            }

        if layer_types is None:
            # Default pattern: every 4th layer uses full attention, others use sliding attention
            window_pattern = 4
            layer_types = [
                ("full_attention" if (i % window_pattern == 0) else "sliding_attention")
                for i in range(num_hidden_layers)
            ]
        else:
            layer_type_validation(layer_types, num_hidden_layers)

        self.sliding_window = int(sliding_window) if sliding_window else None
        self.layer_types = list(layer_types)

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            eos_token_id=list(eos_token_id),
            bos_token_id=bos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            attention_bias=False,
            attention_dropout=attention_dropout,
            rope_parameters=rope_parameters,
            pretraining_tp=pretraining_tp,
            mlp_bias=mlp_bias,
            **kwargs,
        )

        # CWM models don't use attention bias, remove it from config
        del self.attention_bias


class CwmRotaryEmbedding(Qwen2RotaryEmbedding):
    pass


class CwmAttention(Qwen2Attention):
    def __init__(self, config: CwmConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.q_proj = torch.nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)


class CwmDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: CwmConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = CwmAttention(config=config, layer_idx=layer_idx)


class CwmPreTrainedModel(LlamaPreTrainedModel):
    pass


class CwmModelOutputWithPast(BaseModelOutputWithPast):
    pass


class CwmModel(LlamaModel):
    config_class = CwmConfig

    def __init__(self, config: CwmConfig):
        super().__init__(config)
        self.layers = torch.nn.ModuleList(
            [CwmDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CwmModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            sliding_mask_kwargs = mask_kwargs.copy()

            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return CwmModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class CwmForCausalLM(LlamaForCausalLM):
    pass


__all__ = [
    "CwmConfig",
    "CwmPreTrainedModel",
    "CwmModel",
    "CwmForCausalLM",
]
