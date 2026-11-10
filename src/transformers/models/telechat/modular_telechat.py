# Copyright 2026 China Telecom Artificial Intelligence Technology Co., Ltd. and the HuggingFace Inc. team. All rights reserved.
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
from torch import nn

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRotaryEmbedding,
)


@auto_docstring(checkpoint="Tele-AI/TeleChat2-3B")
@strict
class TeleChatConfig(LlamaConfig):
    r"""
    training_seqlen (`int`, *optional*, defaults to 8192):
        Sequence length the model was trained with. Beyond it, RoPE is rescaled with stepwise NTK scaling.
    base_seqlen (`int`, *optional*, defaults to 4096):
        Sequence length the stepwise NTK scaling factor is computed relative to.

    ```python
    >>> from transformers import TeleChatModel, TeleChatConfig

    >>> # Initializing a TeleChat2-3B style configuration
    >>> configuration = TeleChatConfig()

    >>> # Initializing a model from the TeleChat2-3B style configuration
    >>> model = TeleChatModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "telechat"
    attribute_map = {
        "n_layer": "num_hidden_layers",
        "n_head": "num_attention_heads",
        "ffn_hidden_size": "intermediate_size",
        "layer_norm_epsilon": "rms_norm_eps",
    }

    vocab_size: int = 131072
    hidden_size: int = 3072
    intermediate_size: int = 6144
    num_hidden_layers: int = 24
    num_attention_heads: int = 24
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int | None = None
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 3
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    # Checkpoints which predate `tie_word_embeddings` relied on the old default of tying them
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    attention_dropout: float | int = 0.0
    head_dim: int | None = None
    training_seqlen: int = 8192
    base_seqlen: int = 4096

    attention_bias = AttributeError()
    mlp_bias = AttributeError()
    pretraining_tp = AttributeError()

    def __post_init__(self, **kwargs):
        # Resolve the aliases now, otherwise `head_dim` and `num_key_value_heads` get derived from the defaults
        for alias, attribute in self.attribute_map.items():
            if alias in kwargs:
                setattr(self, attribute, kwargs.pop(alias))
        if self.max_position_embeddings is None:
            self.max_position_embeddings = kwargs.get("seq_length") or self.training_seqlen
        super().__post_init__(**kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        # The original code ignores `rope_scaling` and always applies its own stepwise NTK scaling instead
        kwargs.pop("rope_scaling", None)
        return PreTrainedConfig.convert_rope_params_to_dict(self, **kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        super().validate_architecture()
        if getattr(self, "n_embd", None) is not None:
            raise ValueError(
                "This config uses the TeleChat-52B architecture, which is not supported by `TeleChatConfig`. "
                "Load it with `trust_remote_code=True` instead."
            )
        # The original code supports these options, but no released checkpoint enables them
        for option in ("embed_layernorm", "apply_residual_connection_post_layernorm"):
            if getattr(self, option, False):
                raise ValueError(f"TeleChat does not support `{option}`.")


class TeleChatRotaryEmbedding(LlamaRotaryEmbedding):
    @staticmethod
    def compute_ntk_parameters(config: TeleChatConfig, seq_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the inverse frequencies and attention scaling for the stepwise NTK scaling of the original code,
        given the current total sequence length `seq_len`.
        """
        base = config.rope_parameters["rope_theta"]
        dim = config.head_dim
        seq_len = seq_len.clamp(min=config.training_seqlen).float()
        # Equal to `2 ** ceil(log2(seq_len / base_seqlen) + 1) - 1` from the original code, but exact for powers of 2
        doublings = torch.ceil(torch.log2(torch.ceil(seq_len / config.base_seqlen)))
        ntk_alpha = (2 ** (doublings + 1) - 1).clamp(min=1)
        base = base * ntk_alpha ** (dim / (dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float, device=seq_len.device) / dim))
        attention_scaling = 0.1 * torch.log(seq_len / config.training_seqlen) + 1.0
        return inv_freq, attention_scaling

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq, attention_scaling = self.compute_ntk_parameters(self.config, position_ids.max() + 1)
        freqs = position_ids[..., None].float() * inv_freq.to(x.device)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class TeleChatMLP(LlamaMLP):
    def __init__(self, config: TeleChatConfig):
        super().__init__(config)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)


class TeleChatAttention(LlamaAttention):
    def __init__(self, config: TeleChatConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)


class TeleChatDecoderLayer(LlamaDecoderLayer):
    pass


class TeleChatPreTrainedModel(LlamaPreTrainedModel):
    pass


class TeleChatModel(LlamaModel):
    pass


@auto_docstring
class TeleChatForCausalLM(LlamaForCausalLM):
    pass


__all__ = [
    "TeleChatConfig",
    "TeleChatForCausalLM",
    "TeleChatModel",
    "TeleChatPreTrainedModel",
]
