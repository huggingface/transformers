# Copyright 2025 Arc Institute and The HuggingFace Inc. team. All rights reserved.
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
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg
from ..lfm2.modeling_lfm2 import (
    Lfm2DecoderLayer,
    Lfm2ForCausalLM,
    Lfm2Model,
    Lfm2PreTrainedModel,
    apply_mask_to_padding_states,
)
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


@auto_docstring(checkpoint="arcinstitute/evo2_1b_base")
@strict
class Evo2Config(PreTrainedConfig):
    r"""
    mixer_types (`list[str]`, *optional*):
        Mixer of each layer: `"attention"`, `"hcs"` (short explicit filter), `"hcm"` (medium explicit filter) or
        `"hcl"` (long implicit filter). Defaults to the repeating `hcs, hcm, hcl, attention, hcs, hcm, hcl` stripe.
    short_conv_kernel_size (`int`, *optional*, defaults to 3):
        Kernel size of the depthwise convolution applied to the hyena input projections.
    hcs_filter_length (`int`, *optional*, defaults to 7):
        Length of the explicit filter of `"hcs"` layers.
    hcm_filter_length (`int`, *optional*, defaults to 128):
        Length of the explicit filter of `"hcm"` layers.
    filter_groups (`int`, *optional*, defaults to 128):
        Number of channel groups sharing one explicit filter in `"hcs"` / `"hcm"` layers.
    state_size (`int`, *optional*, defaults to 16):
        Number of modes (poles/residues) per channel of the `"hcl"` long filter.
    """

    model_type = "evo2"
    keys_to_ignore_at_inference = ["cache"]
    attribute_map = {"num_key_value_heads": "num_attention_heads"}
    # q/k/v have no bias in every released checkpoint, o_proj always has one (set in `Evo2Attention`)
    attention_bias = False
    attention_dropout = 0.0
    mlp_bias = False
    # hyena layers cache two states: the short conv window, and the explicit filter window or the modal state
    number_of_conv_states = 2

    vocab_size: int = 512
    hidden_size: int = 1920
    intermediate_size: int = 5120
    num_hidden_layers: int = 25
    num_attention_heads: int = 15
    hidden_act: str = "gelu"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = 0
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    mixer_types: list[str] | None = None
    short_conv_kernel_size: int = 3
    hcs_filter_length: int = 7
    hcm_filter_length: int = 128
    filter_groups: int = 128
    state_size: int = 16

    def __post_init__(self, **kwargs):
        if self.mixer_types is None:
            stripe = ["hcs", "hcm", "hcl", "attention", "hcs", "hcm", "hcl"]
            self.mixer_types = [stripe[i % len(stripe)] for i in range(self.num_hidden_layers)]
        super().__post_init__(**kwargs)

    @property
    def layer_types(self):
        return ["full_attention" if mixer == "attention" else "conv" for mixer in self.mixer_types]


class Evo2RMSNorm(LlamaRMSNorm):
    pass


class Evo2RotaryEmbedding(LlamaRotaryEmbedding):
    pass


class Evo2MLP(LlamaMLP):
    def __init__(self, config: Evo2Config, layer_idx: int):
        super().__init__(config)
        # Evo 2 only applies the activation in the first block, every other MLP is a plain bilinear gate
        if layer_idx > 0:
            self.act_fn = nn.Identity()


class Evo2Attention(LlamaAttention):
    def __init__(self, config: Evo2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)


def causal_conv(hidden_states, weight, cache, layer_idx, state_idx):
    """Depthwise causal conv (cross-correlation, like `nn.Conv1d`) whose left window is kept in a conv cache slot."""
    seq_len, kernel_size = hidden_states.shape[-1], weight.shape[-1]
    if cache is None or not cache.has_previous_state(layer_idx, state_idx):
        hidden_states = F.pad(hidden_states, (kernel_size - 1, 0))
    if cache is not None:
        hidden_states = cache.update_conv_state(hidden_states, layer_idx, state_idx, conv_kernel_size=kernel_size - 1)
    out = F.conv1d(hidden_states.float(), weight.float(), groups=hidden_states.shape[1])
    return out[..., -seq_len:].to(hidden_states.dtype)


class Evo2ShortConv(nn.Module):
    """Hyena operator `x2 * filter(x1 * v)`: the gated short conv of LFM2, with a featurizer conv in front and either an
    explicit filter (`hcs`, `hcm`) or a modal long filter (`hcl`, a diagonal linear recurrence)."""

    def __init__(self, config: Evo2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.mixer_type = config.mixer_types[layer_idx]
        hidden_size = config.hidden_size
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.short_conv = nn.Parameter(torch.zeros(3 * hidden_size, 1, config.short_conv_kernel_size))
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        if self.mixer_type == "hcl":
            self.log_poles = nn.Parameter(torch.zeros(hidden_size, config.state_size, 1))
            self.residues = nn.Parameter(torch.zeros(hidden_size, config.state_size))
        else:
            length = config.hcs_filter_length if self.mixer_type == "hcs" else config.hcm_filter_length
            self.filter = nn.Parameter(torch.zeros(config.filter_groups, 1, length))
        self.D = nn.Parameter(torch.zeros(hidden_size)) if self.mixer_type != "hcs" else None

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        has_state = cache is not None and cache.has_previous_state(self.layer_idx, 0)
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        projected = self.in_proj(hidden_states).transpose(1, 2)
        projected = causal_conv(projected, self.short_conv, cache, self.layer_idx, 0)
        x2, x1, v = projected.unflatten(1, (-1, 3)).unbind(2)  # checkpoint channels are interleaved
        x1v = x1 * v

        if self.mixer_type == "hcl":
            y = self.long_filter(x1v, cache, has_state)
        else:
            weight = self.filter.repeat_interleave(x1v.shape[1] // self.filter.shape[0], dim=0)
            # `hcm` filters are applied as a true convolution (FFT) in the reference, `hcs` as a cross-correlation
            weight = weight.flip(-1) if self.mixer_type == "hcm" else weight
            y = causal_conv(x1v, weight, cache, self.layer_idx, 1)
        if self.D is not None:
            y = y + x1v * self.D[:, None]
        return self.out_proj((x2 * y).transpose(1, 2))

    def long_filter(self, x1v, cache, has_state):
        """`y_t = sum_s residues_s * state_t,s` with `state_t = exp(log_poles) * state_{t-1} + x1v_t`, FFT for the prefill."""
        seq_len = x1v.shape[-1]
        log_poles, residues = self.log_poles.float(), self.residues.float()
        decay = torch.exp(log_poles * torch.arange(seq_len, device=x1v.device))  # [hidden, state, seq]
        filter_ = (residues[..., None] * decay).sum(1)
        fft_size = 2 * seq_len
        y = torch.fft.irfft(torch.fft.rfft(x1v.float(), n=fft_size) * torch.fft.rfft(filter_, n=fft_size), n=fft_size)[
            ..., :seq_len
        ]
        if cache is None:
            return y.to(x1v.dtype)
        state = (x1v.float()[:, :, None] @ decay.flip(-1).transpose(1, 2)).squeeze(2)  # [batch, hidden, state]
        if has_state:
            previous = cache.layers[self.layer_idx].recurrent_states[1]
            y = y + (previous[:, :, None] @ (residues[..., None] * decay * log_poles.exp())).squeeze(2)
            state = state + torch.exp(log_poles[..., 0] * seq_len) * previous
        cache.update_recurrent_state(state, self.layer_idx, 1)
        return y.to(x1v.dtype)


class Evo2DecoderLayer(Lfm2DecoderLayer):
    def __init__(self, config: Evo2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.feed_forward = Evo2MLP(config, layer_idx)


class Evo2PreTrainedModel(Lfm2PreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Evo2ShortConv):
            for param in module.parameters(recurse=False):
                init.normal_(param, std=self.config.initializer_range)


class Evo2Model(Lfm2Model):
    pass


class Evo2ForCausalLM(Lfm2ForCausalLM):
    pass


__all__ = ["Evo2Config", "Evo2PreTrainedModel", "Evo2Model", "Evo2ForCausalLM"]
