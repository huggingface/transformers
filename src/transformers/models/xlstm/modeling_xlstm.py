# Copyright 2025 NXAI GmbH. All rights reserved.
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
"""PyTorch xLSTM Model."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...generation import GenerationMixin
from ...integrations import use_kernel_func_from_hub
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from .configuration_xlstm import round_up_to_next_multiple_of, xLSTMConfig


def soft_cap(values: torch.Tensor, cap_value: float | None = None) -> torch.Tensor:
    """Smoothly bounds `values` to `(-cap_value, cap_value)` with a scaled tanh, as Gemma2 does for its logits."""
    if cap_value is None:
        return values
    return cap_value * torch.tanh(values / cap_value)


def segment_sum(values: torch.Tensor) -> torch.Tensor:
    """
    `[..., t, j] = values[..., j + 1] + ... + values[..., t]` for `j <= t`, `-inf` above the diagonal. Summed directly
    rather than as a difference of cumsums, which loses precision next to large values. Same as Mamba2's `segment_sum`.
    """
    length = values.shape[-1]
    causal = torch.ones(length, length, dtype=torch.bool, device=values.device).tril()
    values = values[..., None].expand(*values.shape, length).masked_fill(causal.mT, 0.0)  # [s, j] = values[s] if s > j
    return values.cumsum(-2).masked_fill(~causal, -float("inf"))


@use_kernel_func_from_hub("mlstm")
def eager_mlstm_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    input_gate: torch.Tensor,
    forget_gate: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    chunk_size: int = 64,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    The [mLSTM](https://huggingface.co/papers/2405.04517): linear attention with a matrix memory `C`, a
    normalizer `n`, an exponential input gate `i` and a sigmoid forget gate `f`. The gates act as exponentials, so
    their logs are kept relative to a running maximum `m`. One step of the recurrence is:

        m_t = max(log f_t + m_{t-1}, i_t)                      with log f_t = logsigmoid(f_t)
        C_t = exp(log f_t + m_{t-1} - m_t) C_{t-1} + exp(i_t - m_t) k_t v_t^T    (n_t: the same with k_t)
        h_t = q_t C_t / (max(|q_t n_t|, exp(-m_t)) + eps)      with q_t scaled by qk_head_dim ** -0.5

    It runs chunk by chunk. Unrolled inside a chunk, key `j` reaches query `t >= j` with the log-weight
    `D[t, j] = i_j + log f_{j+1} + ... + log f_t`, and the state entering the chunk reaches it with
    `m_prev + log f_1 + ... + log f_t`. A chunk is therefore causal attention weighted by `exp(D - m_t)` plus a read
    of the carried state. Seen from the next chunk, the whole chunk is one step of the recurrence above. Decoding is a
    chunk of 1. The last chunk is padded with steps that neither write (`i = -inf`) nor forget (`log f = 0`), so the
    returned state is exact.

    Args:
        query, key (`torch.Tensor` of shape `(batch_size, num_heads, seq_len, qk_head_dim)`)
        value (`torch.Tensor` of shape `(batch_size, num_heads, seq_len, v_head_dim)`)
        input_gate, forget_gate (`torch.Tensor` of shape `(batch_size, num_heads, seq_len)`):
            Gate pre-activations `i` and `f`.
        state (`tuple` of `(C, n, m)`, *optional*):
            Shapes `(batch_size, num_heads, qk_head_dim, v_head_dim)`, `(batch_size, num_heads, qk_head_dim)` and
            `(batch_size, num_heads, 1)`. Zeros if `None`.
        chunk_size (`int`, *optional*, defaults to 64):
            Length of the chunks processed in parallel.
        eps (`float`, *optional*, defaults to 1e-6):
            Added to the denominator.

    Returns:
        The output `h`, shaped and typed like `value`, and the last `(C, n, m)` state in float32.
    """
    batch_size, num_heads, seq_len, qk_head_dim = query.shape
    output_dtype = value.dtype
    chunk_size = min(chunk_size, seq_len)
    padding = -seq_len % chunk_size

    # (batch, heads, seq_len, ...) -> (batch, heads, num_chunks, chunk_size, ...) in fp32, last chunk padded
    query, key, value = (
        F.pad(x.float(), (0, 0, 0, padding)).unflatten(2, (-1, chunk_size)) for x in (query, key, value)
    )
    input_gate = F.pad(input_gate.float(), (0, padding), value=-float("inf")).unflatten(2, (-1, chunk_size))
    log_forget = F.pad(F.logsigmoid(forget_gate.float()), (0, padding)).unflatten(2, (-1, chunk_size))
    query = query * qk_head_dim**-0.5

    # D[t, j]; a segment sum, so a -100_000 reset gate does not round away the decays after it
    log_weights = segment_sum(log_forget) + input_gate[..., None, :]
    state_decay = log_forget.cumsum(-1)  # log f_1 + ... + log f_t

    # A chunk as one step: it forgets by its total decay and writes its keys weighted as seen from its last position
    chunk_forget = state_decay[..., -1:]
    chunk_input = log_weights[..., -1, :].max(-1, keepdim=True).values
    key_weights = torch.exp(log_weights[..., -1, :, None] - chunk_input[..., None]) * key
    chunk_memory, chunk_normalizer = key_weights.mT @ value, key_weights.sum(-2)

    if state is None:
        memory = query.new_zeros(batch_size, num_heads, qk_head_dim, value.shape[-1])
        normalizer = query.new_zeros(batch_size, num_heads, qk_head_dim)
        max_log_gate = query.new_zeros(batch_size, num_heads, 1)
    else:
        memory, normalizer, max_log_gate = (state_tensor.float() for state_tensor in state)

    chunk_start_states = []
    for chunk in range(query.shape[2]):
        chunk_start_states.append((memory, normalizer, max_log_gate))
        new_max_log_gate = torch.maximum(chunk_forget[:, :, chunk] + max_log_gate, chunk_input[:, :, chunk])
        forget_weight = torch.exp(chunk_forget[:, :, chunk] + max_log_gate - new_max_log_gate)
        input_weight = torch.exp(chunk_input[:, :, chunk] - new_max_log_gate)
        memory = forget_weight[..., None] * memory + input_weight[..., None] * chunk_memory[:, :, chunk]
        normalizer = forget_weight * normalizer + input_weight * chunk_normalizer[:, :, chunk]
        max_log_gate = new_max_log_gate
    memories, normalizers, max_log_gates = (torch.stack(states, dim=2) for states in zip(*chunk_start_states))

    # Each query reads the state entering its chunk and the keys before it in the chunk, all relative to m_t
    state_log_weights = state_decay + max_log_gates
    max_log_weights = torch.maximum(state_log_weights, log_weights.max(-1).values)  # m_t
    state_weights = torch.exp(state_log_weights - max_log_weights)[..., None]
    scores = (query @ key.mT) * torch.exp(log_weights - max_log_weights[..., None])
    numerator = state_weights * (query @ memories) + scores @ value
    denominator = state_weights * (query @ normalizers[..., None]) + scores.sum(-1, keepdim=True)
    output = numerator / (torch.maximum(denominator.abs(), torch.exp(-max_log_weights)[..., None]) + eps)

    output = output.flatten(2, 3)[:, :, :seq_len]
    return output.to(output_dtype), (memory, normalizer, max_log_gate)


mLSTMLayerStateType = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
mLSTMStateType = dict[int, mLSTMLayerStateType]


class xLSTMRMSNorm(nn.Module):
    """Root mean square normalization layer implementation similar
    to https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html.

    It normalizes the input tensor by the root mean square of the last dimension.

    Args:
        num_features: The number of features in the input tensor.
        eps: A small value to avoid division by zero.
        use_weight: Whether to use a learnable weight.
        use_bias: Whether to use a learnable bias.
        force_float32_reductions: Whether to force float32 reductions.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-6,
        use_weight: bool = True,
        use_bias: bool = False,
        force_float32_reductions: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.force_float32_reductions = force_float32_reductions

        if use_weight:
            self.weight = nn.Parameter(torch.ones(num_features))
        else:
            self.weight = None

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.bias = None

    def _apply_weight_bias(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x

    def _rms_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # apply rms norm over the last dimension, i.e. HD dimension
        in_dtype = x.dtype
        if self.force_float32_reductions:
            x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x.to(in_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._rms_normalize(x)
        x = self._apply_weight_bias(x)
        return x


class xLSTMMultiHeadLayerNorm(nn.Module):
    """Multi-head version of the LayerNorm layer.

    It normalizes the last dimension of the input tensor.

    The input is assumed to have the shape (batch_size, sequence_length, nh, DH), where:
    batch_size: batch size
    sequence_length: sequence length
    nh: number of heads
    DH: head dimension

    The normalization is applied over the last dimension (DH) of the input tensor.

    Args:
        num_heads: The number of heads.
        head_dim: The head dimension.
        eps: A small value to avoid division by zero.
        use_weight: Whether to use a learnable weight.
        use_bias: Whether to use a learnable bias.
        force_float32_reductions: Whether to force float32 reductions

    Returns:
        The normalized tensor with the shape (batch_size, sequence_length, nh * DH).
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        use_weight: bool = True,
        use_bias: bool = False,
        force_float32_reductions: bool = True,
    ):
        super().__init__()
        self.num_features = num_heads * head_dim
        self.eps = eps
        self.force_float32_reductions = force_float32_reductions

        if use_weight:
            self.weight = nn.Parameter(torch.ones(self.num_features))
        else:
            self.weight = None

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.bias = None
        self.num_heads = num_heads
        self.head_dim = head_dim

    def _apply_weight_bias(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x

    def _layer_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # apply layer norm over the last dimension, i.e. HD dimension
        in_dtype = x.dtype
        if self.force_float32_reductions:
            x = x.float()
        x_centered = x - x.mean(dim=-1, keepdim=True)
        y = x_centered * torch.rsqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        return y.to(in_dtype)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, nh, DH = x.shape
        if nh != self.num_heads:
            raise ValueError(f"Expected {self.num_heads} heads, got {nh}, input shape: {x.shape}")
        if self.head_dim != DH:
            raise ValueError(f"Expected {self.head_dim} head dimension, got {DH}, input shape: {x.shape}")

        x = self._layer_normalize(x)
        x = x.reshape(batch_size, sequence_length, -1)
        x = self._apply_weight_bias(x)
        return x


class xLSTMFeedForward(nn.Module):
    def __init__(self, config: xLSTMConfig):
        super().__init__()
        self.config = config

        self.up_proj_dim = round_up_to_next_multiple_of(
            config.hidden_size * config.ffn_proj_factor,
            config.ffn_round_up_to_multiple_of,
        )

        if self.config.weight_mode == "single":
            self.proj_up_gate = nn.Linear(
                in_features=config.hidden_size,
                out_features=self.up_proj_dim,
                bias=self.config.use_bias,
            )
            self.proj_up = nn.Linear(
                in_features=config.hidden_size,
                out_features=self.up_proj_dim,
                bias=self.config.use_bias,
            )
        elif self.config.weight_mode == "fused":
            self.proj_up_gate_z = nn.Linear(
                in_features=config.hidden_size,
                out_features=2 * self.up_proj_dim,
                bias=self.config.use_bias,
            )

        self.proj_down = nn.Linear(
            in_features=self.up_proj_dim,
            out_features=config.hidden_size,
            bias=self.config.use_bias,
        )

        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.weight_mode == "single":
            x = self.act_fn(self.proj_up_gate(x)) * self.proj_up(x)
        elif self.config.weight_mode == "fused":
            x = self.proj_up_gate_z(x)
            gate, z = torch.tensor_split(x, (self.up_proj_dim,), dim=-1)
            x = self.act_fn(gate) * z

        y = self.proj_down(x)
        return y


class xLSTMLayer(nn.Module):
    def __init__(self, config: xLSTMConfig):
        super().__init__()
        self.config = config

        self.v_dim = int(config.hidden_size * config.v_dim_factor)
        self.qk_dim = int(config.hidden_size * config.qk_dim_factor)

        if self.config.weight_mode == "single":
            self.q = nn.Linear(
                in_features=self.config.hidden_size,
                out_features=self.qk_dim,
                bias=self.config.use_bias,
            )
            self.k = nn.Linear(
                in_features=self.config.hidden_size,
                out_features=self.qk_dim,
                bias=self.config.use_bias,
            )
            self.v = nn.Linear(
                in_features=self.config.hidden_size,
                out_features=self.v_dim,
                bias=self.config.use_bias,
            )

            self.ogate_preact = nn.Linear(
                in_features=self.config.hidden_size,
                out_features=self.v_dim,
                bias=self.config.use_bias,
            )
            self.igate_preact = nn.Linear(
                in_features=self.config.hidden_size,
                out_features=self.config.num_heads,
                bias=True,
            )
            self.fgate_preact = nn.Linear(
                in_features=self.config.hidden_size,
                out_features=self.config.num_heads,
                bias=True,
            )
        elif self.config.weight_mode == "fused":
            self.qkv_opreact = nn.Linear(
                in_features=self.config.hidden_size,
                out_features=2 * self.qk_dim + 2 * self.v_dim,
                bias=self.config.use_bias,
            )
            self.ifgate_preact = nn.Linear(
                in_features=self.config.hidden_size,
                out_features=2 * self.config.num_heads,
                bias=True,
            )

        self.ogate_act_fn = nn.Sigmoid()

        self.multihead_norm = xLSTMMultiHeadLayerNorm(
            num_heads=self.config.num_heads,
            head_dim=self.v_dim // self.config.num_heads,
            eps=self.config.norm_eps,
            use_weight=True,
            use_bias=self.config.use_bias,
            force_float32_reductions=self.config.norm_reduction_force_float32,
        )
        self.out_proj = nn.Linear(
            in_features=self.v_dim,
            out_features=self.config.hidden_size,
            bias=self.config.use_bias,
        )

    def forward(
        self, x: torch.Tensor, state: mLSTMLayerStateType | None = None
    ) -> tuple[torch.Tensor, mLSTMLayerStateType | None]:
        if x.ndim != 3:
            raise ValueError(f"Input must have shape [batch_size, sequence_length, HD], got {x.shape}")
        batch_size, sequence_length, _ = x.shape
        if self.config.weight_mode == "single":
            query = self.q(x)
            key = self.k(x)
            value = self.v(x)
            o_preact = self.ogate_preact(x)
            i_preact = soft_cap(self.igate_preact(x), cap_value=self.config.gate_soft_cap)
            f_preact = soft_cap(self.fgate_preact(x), cap_value=self.config.gate_soft_cap)

        elif self.config.weight_mode == "fused":
            qkv_opreact = self.qkv_opreact(x)
            query, key, value, o_preact = torch.tensor_split(
                qkv_opreact,
                (
                    self.qk_dim,
                    2 * self.qk_dim,
                    2 * self.qk_dim + self.v_dim,
                ),
                dim=-1,
            )

            if_preact = soft_cap(self.ifgate_preact(x), cap_value=self.config.gate_soft_cap)
            i_preact, f_preact = torch.tensor_split(if_preact, (self.config.num_heads,), dim=-1)

        query = query.reshape(batch_size, sequence_length, self.config.num_heads, -1).transpose(1, 2)
        key = key.reshape(batch_size, sequence_length, self.config.num_heads, -1).transpose(1, 2)
        value = value.reshape(batch_size, sequence_length, self.config.num_heads, -1).transpose(1, 2)
        i_preact = i_preact.transpose(1, 2)
        f_preact = f_preact.transpose(1, 2)
        h, state = eager_mlstm_forward(
            query,
            key,
            value,
            input_gate=i_preact,
            forget_gate=f_preact,
            state=state,
            chunk_size=self.config.chunk_size,
            eps=self.config.eps,
        )
        expected_h_shape = (
            batch_size,
            self.config.num_heads,
            sequence_length,
            self.v_dim // self.config.num_heads,
        )
        if h.shape != expected_h_shape:
            raise ValueError(f"Got {h.shape}, expected {expected_h_shape}")

        h = h.transpose(1, 2)
        h_norm = self.multihead_norm(h)
        h_norm = h_norm.reshape(batch_size, sequence_length, -1)

        h_out = self.ogate_act_fn(o_preact) * h_norm

        y = self.out_proj(h_out)
        return y, state


class xLSTMBlock(GradientCheckpointingLayer):
    def __init__(self, config: xLSTMConfig):
        super().__init__()
        self.config = config
        self.norm_mlstm = xLSTMRMSNorm(
            num_features=config.hidden_size,
            eps=config.norm_eps,
            use_weight=True,
            use_bias=config.use_bias,
            force_float32_reductions=config.norm_reduction_force_float32,
        )
        self.mlstm_layer = xLSTMLayer(config)
        self.norm_ffn = xLSTMRMSNorm(
            num_features=config.hidden_size,
            eps=config.norm_eps,
            use_weight=True,
            use_bias=config.use_bias,
            force_float32_reductions=config.norm_reduction_force_float32,
        )
        self.ffn = xLSTMFeedForward(config)

    def forward(self, x: torch.Tensor, state: mLSTMStateType | None = None) -> tuple[torch.Tensor, mLSTMStateType]:
        x_mlstm = self.norm_mlstm(x)
        x_mlstm, state = self.mlstm_layer(x_mlstm, state)
        x = x + x_mlstm

        x_ffn = self.norm_ffn(x)
        x_ffn = self.ffn(x_ffn)
        x = x + x_ffn

        return x, state


def small_init_method(dim):
    """
    Adapted from: https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py
    Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution."""
    std = (2 / (5 * dim)) ** (1 / 2)

    def init_(tensor):
        return init.normal_(tensor, mean=0.0, std=std)

    return init_


def wang_init_method(n_layers, dim):
    """
    Adapted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py
    """
    std = 2 / n_layers / dim ** (1 / 2)

    def init_(tensor):
        return init.normal_(tensor, mean=0.0, std=std)

    return init_


class xLSTMPreTrainedModel(PreTrainedModel):
    """
    An abstract class for an interface to loading a pre-trained xLSTM model.
    """

    config: xLSTMConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["xLSTMBlock"]
    supports_gradient_checkpointing = True
    _is_stateful = True
    _can_record_outputs = {
        "hidden_states": xLSTMBlock,
    }

    def _module_name_map(self, module):
        for name, mod in self.named_modules():
            if mod is module:
                return name
        return ""

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, nn.Embedding):
            small_init_method(self.config.hidden_size)(self.embeddings.weight)
        elif isinstance(module, nn.Linear):
            if module.bias is not None:
                init.zeros_(module.bias)
            if self.config.weight_mode == "single" and "gate" in self._module_name_map(module):
                init.zeros_(module.weight)

                if "igate" in self._module_name_map(module):
                    init.copy_(module.bias, -10.0 * torch.ones_like(module.bias))
                elif "fgate" in self._module_name_map(module):
                    init.copy_(
                        module.bias,
                        torch.linspace(
                            3.0,
                            6.0,
                            module.bias.shape[-1],
                        ).to(
                            device=module.bias.device,
                            dtype=module.bias.dtype,
                        ),
                    )
            elif self.config.weight_mode == "fused" and "gate" in self._module_name_map(module):
                init.zeros_(module.weight)

                init.copy_(
                    module.bias[: self.config.num_heads],
                    module.bias[: self.config.num_heads]
                    - module.bias[: self.config.num_heads]
                    - 10.0 * torch.ones_like(module.bias),
                )
                init.copy_(
                    module.bias[: self.config.num_heads],
                    module.bias[: self.config.num_heads]
                    - module.bias[self.config.num_heads :]
                    + torch.linspace(
                        3.0,
                        6.0,
                        module.bias.shape[-1],
                    ).to(
                        device=module.bias.device,
                        dtype=module.bias.dtype,
                    ),
                )
            elif "proj_down" in self._module_name_map(module):
                wang_init_method(dim=module.weight.shape[1], n_layers=self.config.num_hidden_layers)(module.weight)
            elif "out_proj" in self._module_name_map(module):
                wang_init_method(dim=self.config.hidden_size, n_layers=self.config.num_hidden_layers)(module.weight)
            elif module.weight is not None:
                small_init_method(self.config.hidden_size)(module.weight)


class xLSTMCache:
    """
    Cache for xLSTM model which does not have attention mechanism and key value states.

    Arguments:
        config (`PreTrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The batch size with which the model will be used.
        dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
            The default `dtype` to use when initializing the layer.
        device (`torch.device` or `str`, *optional*):
            The device on which the cache should be initialized. Should be the same as the layer.

    Attributes:
        seqlen_offset: torch.Tensor
        dtype: torch.dtype

    Example:

        ```python
        >>> from transformers import AutoTokenizer, xLSTMForCausalLM, xLSTMCache

        >>> model = xLSTMForCausalLM.from_pretrained("NX-AI/xLSTM-7b")
        >>> tokenizer = xLSTMTokenizer.from_pretrained("NX-AI/xLSTM-7b")

        >>> inputs = tokenizer(text="I am an xLSTM", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> cache_params = xLSTMCache(config=model.config, max_batch_size=1, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, cache_params=cache_params, use_cache=True)
        >>> outputs.cache_params
        xLSTMCache()
    """

    def __init__(
        self,
        config: xLSTMConfig,
        max_batch_size: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str | None = None,
        **kwargs,
    ):
        # Follows the static-cache convention (`StaticLayer.cumulative_length` in cache_utils.py):
        # compile/export-friendly caches store the position counter as a tensor, not a Python int,
        # so it stays a tensor leaf in the cache pytree across `+= shape[i]` operations rather
        # than getting promoted to a SymInt during dynamic-shape tracing.
        self.seqlen_offset = torch.tensor([0], dtype=int, device=device)
        self.dtype = dtype
        self.config = config
        self.rnn_state = {
            layer: (
                torch.zeros(
                    [max_batch_size, config.num_heads, config.qk_head_dim, config.v_head_dim],
                    dtype=dtype,
                    device=device,
                ),
                torch.zeros([max_batch_size, config.num_heads, config.qk_head_dim], dtype=dtype, device=device),
                torch.zeros([max_batch_size, config.num_heads, 1], dtype=dtype, device=device),
            )
            for layer in range(config.num_hidden_layers)
        }

    def reset(self):
        self.rnn_state = {
            layer: (
                torch.zeros_like(self.rnn_state[layer][0]),
                torch.zeros_like(self.rnn_state[layer][1]),
                torch.zeros_like(self.rnn_state[layer][2]),
            )
            for layer in self.rnn_state
        }


@auto_docstring
@dataclass
class xLSTMOutput(ModelOutput):
    r"""
    cache_params (`xLSTMCache`):
        The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
        avoid providing the old `input_ids`.
    """

    last_hidden_state: torch.FloatTensor | None
    cache_params: xLSTMCache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None


@auto_docstring
class xLSTMModel(xLSTMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # use embedding_dim and num_blocks once here to make use of them
        self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.blocks = nn.ModuleList([xLSTMBlock(config) for _ in range(config.num_blocks)])
        self.out_norm = xLSTMRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embedding):
        self.embeddings = new_embedding

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.LongTensor | None = None,
        cache_params: xLSTMCache | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | xLSTMOutput:
        r"""
        cache_params (`xLSTMCache`, *optional*):
            The xLSTMCache that carries the RNN states.
        """
        # Resolved here (not just by @capture_outputs) because the chunked inference path below
        # is incompatible with hidden state collection and we need the value to pick the right branch.
        output_hidden_states = kwargs.get("output_hidden_states")
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if use_cache and cache_params is None:
            cache_params = xLSTMCache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        hidden_states = inputs_embeds

        if (
            not self.training
            and self.config.max_inference_chunksize < hidden_states.shape[1]
            and not output_hidden_states
        ):
            offset = 0
            with torch.no_grad():
                if cache_params is None:
                    cache_params = xLSTMCache(config=self.config, max_batch_size=hidden_states.shape[0])
                final_state = torch.zeros_like(hidden_states)
                while offset < hidden_states.shape[1]:
                    hidden_states_chunk = hidden_states[
                        :, offset : min(offset + self.config.max_inference_chunksize, hidden_states.shape[1])
                    ]
                    for layer_idx, xlstm_block in enumerate(self.blocks):
                        hidden_states_chunk, rnn_state = xlstm_block(
                            hidden_states_chunk,
                            state=cache_params.rnn_state[layer_idx],
                        )
                        for state_idx in range(len(cache_params.rnn_state[layer_idx])):
                            local_rnn_state = rnn_state[state_idx]
                            cache_params.rnn_state[layer_idx][state_idx].copy_(local_rnn_state)
                        cache_params.rnn_state_initial = False
                    final_state[
                        :, offset : min(offset + self.config.max_inference_chunksize, hidden_states.shape[1])
                    ] = hidden_states_chunk
                    offset += self.config.max_inference_chunksize
                hidden_states = final_state
        else:
            for layer_idx, xlstm_block in enumerate(self.blocks):
                hidden_states, rnn_state = xlstm_block(
                    hidden_states,
                    cache_params.rnn_state[layer_idx] if cache_params is not None else None,
                )

                if cache_params:
                    for state_idx in range(len(cache_params.rnn_state[layer_idx])):
                        local_rnn_state = rnn_state[state_idx]
                        cache_params.rnn_state[layer_idx][state_idx].copy_(local_rnn_state)
                    cache_params.rnn_state_initial = False

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.out_norm(hidden_states)

        return xLSTMOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params,
        )


@auto_docstring
@dataclass
class xLSTMCausalLMOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    cache_params (`xLSTMCache`, *optional*, carrying the RNN states):
        The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
        avoid providing the old `input_ids`.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    cache_params: xLSTMCache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None


@auto_docstring
class xLSTMForCausalLM(xLSTMPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = xLSTMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_params: xLSTMCache | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | xLSTMCausalLMOutput:
        r"""
        cache_params (`xLSTMCache`, *optional*):
            The xLSTMCache that carries the RNN states.
        """
        xlstm_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = xlstm_outputs[0]

        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        if not self.training and self.config.max_inference_chunksize < logits.shape[1]:
            offset = 0
            with torch.no_grad():
                while offset < logits.shape[1]:
                    logits[:, offset : min(offset + self.config.max_inference_chunksize, logits.shape[1])] = soft_cap(
                        logits[:, offset : min(offset + self.config.max_inference_chunksize, logits.shape[1])],
                        self.config.output_logit_soft_cap,
                    )
                    offset += self.config.max_inference_chunksize
        else:
            logits = soft_cap(logits, self.config.output_logit_soft_cap)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return xLSTMCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=xlstm_outputs.cache_params,
            hidden_states=xlstm_outputs.hidden_states,
        )


__all__ = [
    "xLSTMForCausalLM",
    "xLSTMModel",
    "xLSTMPreTrainedModel",
]
