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
from ...utils import auto_docstring, is_torchdynamo_compiling
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
)


@auto_docstring(checkpoint="Tele-AI/TeleChat2-3B")
@strict
class TeleChatConfig(LlamaConfig):
    r"""
    TeleChat applies its own stepwise NTK scaling of RoPE, described by `rope_parameters` with `"rope_type": "telechat_ntk"`:

    - `original_max_position_embeddings` (`training_seqlen` in the original configs, defaults to 8192): the sequence
      length the model was trained with. Beyond it, the RoPE base and the cos/sin are rescaled.
    - `base_seqlen` (defaults to 4096): the sequence length the NTK scaling factor of the base is computed relative to.

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

    attention_bias = AttributeError()
    mlp_bias = AttributeError()
    pretraining_tp = AttributeError()

    def __post_init__(self, **kwargs):
        # The original configs use different names for these. Resolve them now, otherwise `head_dim` and
        # `num_key_value_heads` get derived from the defaults
        if "n_layer" in kwargs:
            self.num_hidden_layers = kwargs.pop("n_layer")
        if "n_head" in kwargs:
            self.num_attention_heads = kwargs.pop("n_head")
        if "ffn_hidden_size" in kwargs:
            self.intermediate_size = kwargs.pop("ffn_hidden_size")
        if "layer_norm_epsilon" in kwargs:
            self.rms_norm_eps = kwargs.pop("layer_norm_epsilon")
        if "seq_length" in kwargs:
            self.max_position_embeddings = kwargs.pop("seq_length")
        if self.max_position_embeddings is None:
            self.max_position_embeddings = kwargs.get("training_seqlen", 8192)
        super().__post_init__(**kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        # The original code ignores `rope_scaling` and always applies its own stepwise NTK scaling instead
        kwargs.pop("rope_scaling", None)
        rope_parameters = self.rope_parameters or {}
        self.rope_parameters = {
            "rope_type": "telechat_ntk",
            "rope_theta": rope_parameters.get("rope_theta", kwargs.pop("rope_theta", self.default_theta)),
            "original_max_position_embeddings": rope_parameters.get(
                "original_max_position_embeddings", kwargs.pop("training_seqlen", 8192)
            ),
            "base_seqlen": rope_parameters.get("base_seqlen", kwargs.pop("base_seqlen", 4096)),
        }
        kwargs.pop("training_seqlen", None)
        kwargs.pop("base_seqlen", None)
        return PreTrainedConfig.convert_rope_params_to_dict(self, **kwargs)

    def _validate_telechat_ntk_rope_parameters(self, rope_parameters: dict, ignore_keys: set[str] | None = None):
        required_keys = {"rope_type", "rope_theta", "original_max_position_embeddings", "base_seqlen"}
        self._check_received_keys("telechat_ntk", set(rope_parameters), required_keys, ignore_keys=ignore_keys)

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


class TeleChatRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: TeleChatConfig):
        super().__init__()
        self.config = config
        # Within the training length the frequencies are fixed, beyond it they depend on the sequence length
        self.original_max_seq_len = config.rope_parameters["original_max_position_embeddings"]
        self.max_seq_len_cached = self.original_max_seq_len
        # `_init_weights` recomputes the buffers with `compute_default_rope_parameters`
        self.rope_type = "default"
        inv_freq, _ = self.compute_default_rope_parameters(config)
        # The cos/sin are only scaled beyond the training length
        self.attention_scaling = 1.0

        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: TeleChatConfig, device: torch.device | None = None, seq_len: int | torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the inverse frequencies and attention scaling for the stepwise NTK scaling of the original code,
        given the current total sequence length `seq_len` (the training length if `None`).
        """
        rope_parameters = config.rope_parameters
        training_seqlen = rope_parameters["original_max_position_embeddings"]
        dim = config.head_dim
        seq_len = training_seqlen if seq_len is None else seq_len
        seq_len = torch.as_tensor(seq_len, device=device).clamp(min=training_seqlen).float()
        # Equal to `2 ** ceil(log2(seq_len / base_seqlen) + 1) - 1` from the original code, but exact for powers of 2
        doublings = torch.ceil(torch.log2(torch.ceil(seq_len / rope_parameters["base_seqlen"])))
        ntk_alpha = (2 ** (doublings + 1) - 1).clamp(min=1)
        base = rope_parameters["rope_theta"] * ntk_alpha ** (dim / (dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
        attention_scaling = 0.1 * torch.log(seq_len / training_seqlen) + 1.0
        return inv_freq, attention_scaling

    def ntk_frequency_update(self, position_ids: torch.Tensor, device: torch.device):
        """
        Like `dynamic_rope_update`, but the original code rescales for the exact sequence length rather than the
        longest one seen so far, so the frequencies are recomputed whenever it changes beyond the training length.
        """
        seq_len = max(int(position_ids.max()) + 1, self.original_max_seq_len)
        if seq_len == self.max_seq_len_cached:
            return
        if seq_len == self.original_max_seq_len:
            # The buffer is moved with the model, but not the original copy
            inv_freq, self.attention_scaling = self.original_inv_freq.to(device), 1.0
        else:
            inv_freq, self.attention_scaling = self.compute_default_rope_parameters(self.config, device, seq_len)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if is_torchdynamo_compiling():
            # The cache lookup depends on the data, which can't be traced, so recompute the parameters instead
            inv_freq, attention_scaling = self.compute_default_rope_parameters(
                self.config, x.device, position_ids.max() + 1
            )
        else:
            self.ntk_frequency_update(position_ids, x.device)
            inv_freq, attention_scaling = self.inv_freq, self.attention_scaling
        freqs = position_ids[..., None].float() * inv_freq.to(device=x.device, dtype=torch.float)
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
