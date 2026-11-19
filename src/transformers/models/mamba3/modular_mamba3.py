# Copyright 2026 state-spaces/mamba org and HuggingFace Inc. team.
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
import math

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...integrations import use_kernel_func_from_hub_with_fallback
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ..glm.modeling_glm import rotate_half
from ..mamba2.configuration_mamba2 import Mamba2Config
from ..mamba2.modeling_mamba2 import (
    Mamba2CausalLMOutput,
    Mamba2ForCausalLM,
    Mamba2Model,
    Mamba2Output,
    Mamba2RMSNorm,
    apply_mask_to_padding_states,
    segment_sum,
)
from ..mistral.modeling_mistral import MistralMLP


@auto_docstring(checkpoint="state-spaces/mamba3-siso-187m")
@strict
class Mamba3Config(Mamba2Config):
    r"""
    layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
        The epsilon to use in the layer normalization layers.
    expand (`int`, *optional*, defaults to 2):
        Expanding factor used to determine the size of the mixer's inner dimension.
    n_groups (`int`, *optional*, defaults to 1):
        Number of B/C heads, shared across the value heads (grouped-query style).
    residual_in_fp32 (`bool`, *optional*, defaults to `True`):
        Whether or not residuals should be in `float32`.
    rope_fraction (`float`, *optional*, defaults to 0.5):
        Fraction of `state_size` rotated by the data-dependent rotary embedding on B and C.
    a_floor (`float`, *optional*, defaults to 1e-4):
        The data-dependent decay `A` is clamped to be at most `-a_floor`.
    chunk_size (`int`, *optional*, defaults to 64):
        Size of the chunks used by the chunked scan.

    Example:

    ```python
    >>> from transformers import Mamba3Config, Mamba3Model

    >>> configuration = Mamba3Config()
    >>> model = Mamba3Model(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "mamba3"

    num_heads: int = 24
    vocab_size: int = 128256
    hidden_size: int = 768
    intermediate_size: int = 1536
    num_hidden_layers: int = 12
    pad_token_id: int | None = None
    bos_token_id: int | None = 128000
    eos_token_id: int | list[int] | None = 128001
    n_groups: int = 1
    initializer_range: float = 0.02
    rope_fraction: float = 0.5
    a_floor: float = 1e-4
    chunk_size: int = 64
    tie_word_embeddings: bool = True
    # Cached recurrent states per layer: angle, ssm, k and v
    number_of_conv_states = 4

    conv_kernel = AttributeError()
    use_bias = AttributeError()
    use_conv_bias = AttributeError()
    time_step_rank = AttributeError()
    time_step_limit = AttributeError()
    rescale_prenorm_residual = AttributeError()

    def __post_init__(self, **kwargs):
        PreTrainedConfig.__post_init__(self, **kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        super().validate_architecture()
        if self.rope_fraction not in (0.5, 1.0):
            raise ValueError(f"`rope_fraction` must be 0.5 or 1.0, got {self.rope_fraction}.")


def apply_rotary_pairs(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved partial rope: rotates pairs `(x[2i], x[2i+1])` for `i < cos.shape[-1]`, the rest passes through."""
    cos, sin = cos.repeat_interleave(2, dim=-1), sin.repeat_interleave(2, dim=-1)
    rotary_dim = cos.shape[-1]
    rotated = tensor[..., :rotary_dim] * cos + rotate_half(tensor[..., :rotary_dim]) * sin
    return torch.cat([rotated, tensor[..., rotary_dim:]], dim=-1)


@use_kernel_func_from_hub_with_fallback("mamba3_siso_combined", "mamba_ssm")
def mamba3_siso_combined(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Trap: torch.Tensor,
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    D: torch.Tensor | None = None,
    Z: torch.Tensor | None = None,
    Input_States: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    chunk_size: int = 64,
    return_final_states: bool = False,
    **kwargs,
):
    """
    Torch fallback of `mamba_ssm`'s Mamba-3 SISO scan (same signature). The trapezoidal recurrence

        h_t = exp(ADT_t) h_{t-1} + (1 - trap_t) DT_t exp(ADT_t) k_{t-1} v_{t-1}^T + trap_t DT_t k_t v_t^T

    is computed chunk by chunk in its quadratic form, carrying `(angle, ssm, k, v)` states across chunks.
    Shapes: Q/K `(b, l, g, n)`, V/Z `(b, l, h, p)`, ADT/DT/Trap `(b, h, l)`, Angles `(b, l, h, a)`.
    """
    batch_size, seq_len, num_heads, head_dim = V.shape
    group_size = num_heads // Q.shape[2]
    Q = Q.float().repeat_interleave(group_size, dim=2) + Q_bias.float()
    K = K.float().repeat_interleave(group_size, dim=2) + K_bias.float()
    values = V.float()
    ADT, DT, trap = ADT.float(), DT.float(), torch.sigmoid(Trap.float())

    if Input_States is None:
        angle_state = Q.new_zeros(batch_size, num_heads, Angles.shape[-1])
        ssm_state = Q.new_zeros(batch_size, num_heads, head_dim, Q.shape[-1])
        k_state = Q.new_zeros(batch_size, num_heads, Q.shape[-1])
        v_state = Q.new_zeros(batch_size, num_heads, head_dim)
    else:
        angle_state, ssm_state, k_state, v_state = (state.float() for state in Input_States)

    # Data-dependent rotary: angles accumulate `tanh(angle) * pi * dt` over time
    angles = torch.tanh(Angles.float()) * math.pi * DT.transpose(1, 2)[..., None]
    angles = torch.remainder(angles.cumsum(dim=1) + angle_state[:, None], 2 * math.pi)
    Q = apply_rotary_pairs(Q, angles.cos(), angles.sin())
    K = apply_rotary_pairs(K, angles.cos(), angles.sin())

    outputs = []
    for start in range(0, seq_len, chunk_size):
        chunk = slice(start, start + chunk_size)
        q, k, v = Q[:, chunk], K[:, chunk], values[:, chunk]
        adt, dt, tr = ADT[..., chunk], DT[..., chunk], trap[..., chunk]

        # Each k_s v_s enters the state with `trap_s dt_s` at step s and `(1 - trap_{s+1}) dt_{s+1}` at step s+1.
        # The latter term of the chunk's last token is deferred to the next chunk through the (k, v) states.
        current = tr * dt
        deferred = F.pad(((1 - tr) * dt)[..., 1:], (0, 1))
        strictly_causal = torch.ones(q.shape[1], q.shape[1], device=q.device, dtype=torch.bool).tril(-1)
        decay = torch.exp(segment_sum(adt))
        weights = decay * (current[..., None, :] + deferred[..., None, :] * strictly_causal)

        scores = (q.transpose(1, 2) @ k.permute(0, 2, 3, 1)) * weights
        out = (scores @ v.transpose(1, 2)).transpose(1, 2)

        # Contribution of the carried state, including the deferred term of the previous chunk's last token
        ssm_state = (
            ssm_state + v_state[..., None] * k_state[..., None, :] * ((1 - tr[..., 0]) * dt[..., 0])[..., None, None]
        )
        adt_cumsum = adt.cumsum(dim=-1)
        state_out = (q.transpose(1, 2) @ ssm_state.transpose(-1, -2)) * adt_cumsum.exp()[..., None]
        out = out + state_out.transpose(1, 2)
        outputs.append(out)

        decay_to_end = torch.exp(adt_cumsum[..., -1:] - adt_cumsum) * (current + deferred)
        weighted_values = v.transpose(1, 2) * decay_to_end[..., None]
        ssm_state = ssm_state * adt_cumsum[..., -1, None, None].exp() + weighted_values.transpose(
            -1, -2
        ) @ k.transpose(1, 2)
        k_state, v_state = k[:, -1], v[:, -1]

    out = torch.cat(outputs, dim=1)
    if D is not None:
        out = out + D.float()[:, None] * values
    if Z is not None:
        out = out * F.silu(Z.float())
    out = out.to(V.dtype)

    if return_final_states:
        return out, angles[:, -1], ssm_state, k_state, v_state
    return out


class Mamba3RMSNorm(Mamba2RMSNorm):
    pass


class Mamba3MLP(MistralMLP):
    pass


class Mamba3Mixer(nn.Module):
    """
    Mamba-3 SISO mixer: a selective SSM with data-dependent decay, trapezoidal discretization and a data-dependent
    rotary embedding on B and C. Unlike Mamba-2 there is no convolution: the recurrent state is
    `(angle, ssm, k, v)`, cached as 4 recurrent states.
    """

    def __init__(self, config: Mamba3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.n_groups = config.n_groups
        self.ssm_state_size = config.state_size
        self.intermediate_size = self.num_heads * self.head_dim
        self.chunk_size = config.chunk_size
        self.a_floor = config.a_floor
        # The reference implementation rounds down to an even number of rotated dimensions
        self.num_rope_angles = int(self.ssm_state_size * config.rope_fraction) // 2

        projection_size = (
            2 * self.intermediate_size
            + 2 * self.n_groups * self.ssm_state_size
            + 3 * self.num_heads
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(self.num_heads))
        # (num_heads, mimo_rank=1, state_size), the original layout
        self.B_bias = nn.Parameter(torch.ones(self.num_heads, 1, self.ssm_state_size))
        self.C_bias = nn.Parameter(torch.ones(self.num_heads, 1, self.ssm_state_size))
        self.B_norm = Mamba3RMSNorm(self.ssm_state_size, eps=config.layer_norm_epsilon)
        self.C_norm = Mamba3RMSNorm(self.ssm_state_size, eps=config.layer_norm_epsilon)
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        gate, values, B, C, dt, A, trap, angles = self.in_proj(hidden_states).split(
            [
                self.intermediate_size,
                self.intermediate_size,
                self.n_groups * self.ssm_state_size,
                self.n_groups * self.ssm_state_size,
                self.num_heads,
                self.num_heads,
                self.num_heads,
                self.num_rope_angles,
            ],
            dim=-1,
        )
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)
        gate = gate.view(batch_size, seq_len, self.num_heads, self.head_dim)
        B = self.B_norm(B.view(batch_size, seq_len, self.n_groups, self.ssm_state_size))
        C = self.C_norm(C.view(batch_size, seq_len, self.n_groups, self.ssm_state_size))

        # Data-dependent decay, with a heavy-tailed activation: A = -(1 + x) if x >= 0 else -1 / (1 - x)
        A = A.float()
        A = -(A.clamp_min(0) + torch.reciprocal(1 - A.clamp_max(0)))
        A = A.clamp(max=-self.a_floor)
        dt = F.softplus(dt + self.dt_bias)
        angles = angles[:, :, None, :].expand(-1, -1, self.num_heads, -1).float()

        states = None
        if cache is not None:
            states = tuple(cache.get_recurrent_state(self.layer_idx, state_idx=idx) for idx in range(4))
            states = None if states[0] is None else states

        scan_output = mamba3_siso_combined(
            Q=C,
            K=B,
            V=values,
            ADT=(A * dt).transpose(1, 2),
            DT=dt.transpose(1, 2),
            Trap=trap.transpose(1, 2),
            Q_bias=self.C_bias.squeeze(1),
            K_bias=self.B_bias.squeeze(1),
            Angles=angles,
            D=self.D,
            Z=gate,
            Input_States=states,
            chunk_size=self.chunk_size,
            return_final_states=cache is not None,
        )
        if cache is not None:
            scan_output, *final_states = scan_output
            for idx, state in enumerate(final_states[:4]):
                cache.update_recurrent_state(state, self.layer_idx, state_idx=idx)

        return self.out_proj(scan_output.reshape(batch_size, seq_len, -1).to(hidden_states.dtype))


class Mamba3Block(GradientCheckpointingLayer):
    def __init__(self, config: Mamba3Config, layer_idx: int):
        super().__init__()
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = Mamba3RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = Mamba3Mixer(config, layer_idx=layer_idx)
        self.mlp_norm = Mamba3RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = Mamba3MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states.float() if self.residual_in_fp32 else hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        hidden_states = self.mixer(hidden_states, cache=cache, attention_mask=attention_mask, **kwargs)
        residual = residual + hidden_states

        hidden_states = self.mlp_norm(residual.to(dtype=self.mlp_norm.weight.dtype))
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


@auto_docstring
class Mamba3PreTrainedModel(PreTrainedModel):
    config: Mamba3Config
    base_model_prefix = "backbone"
    _no_split_modules = ["Mamba3Block"]
    supports_gradient_checkpointing = True
    _can_compile_fullgraph = True
    _is_stateful = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Mamba3Mixer):
            config = self.config
            dt = torch.exp(
                torch.rand(module.num_heads, device=module.dt_bias.device, dtype=torch.float32)
                * (math.log(config.time_step_max) - math.log(config.time_step_min))
                + math.log(config.time_step_min)
            ).clamp(min=config.time_step_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            init.copy_(module.dt_bias, dt + torch.log(-torch.expm1(-dt)))
            init.ones_(module.B_bias)
            init.ones_(module.C_bias)
            init.ones_(module.D)


class Mamba3Output(Mamba2Output):
    pass


class Mamba3CausalLMOutput(Mamba2CausalLMOutput):
    pass


class Mamba3Model(Mamba2Model):
    pass


class Mamba3ForCausalLM(Mamba2ForCausalLM):
    pass


__all__ = ["Mamba3Config", "Mamba3ForCausalLM", "Mamba3Model", "Mamba3PreTrainedModel"]
