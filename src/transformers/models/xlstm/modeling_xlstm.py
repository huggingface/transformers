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
from ...integrations import use_kernel_forward_from_hub
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from .configuration_xlstm import round_up_to_next_multiple_of, xLSTMConfig


def soft_cap(values: torch.Tensor, cap_value: float) -> torch.Tensor:
    """Smoothly bounds `values` to `(-cap_value, cap_value)` with a scaled tanh, as Gemma2 does for its logits."""
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


@use_kernel_forward_from_hub("mlstm")
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


@use_kernel_forward_from_hub("RMSNorm")
class xLSTMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class xLSTMMultiHeadLayerNorm(nn.Module):
    """Layer norm of each head of the mLSTM output, in fp32, then one weight over all heads."""

    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = F.layer_norm(hidden_states.float(), hidden_states.shape[-1:], eps=self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype).flatten(2)


class xLSTMFeedForward(nn.Module):
    def __init__(self, config: xLSTMConfig):
        super().__init__()
        intermediate_size = round_up_to_next_multiple_of(
            config.hidden_size * config.ffn_proj_factor, config.ffn_round_up_to_multiple_of
        )
        self.proj_up_gate = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.proj_up = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.proj_down = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj_down(self.act_fn(self.proj_up_gate(hidden_states)) * self.proj_up(hidden_states))


class xLSTMLayer(nn.Module):
    def __init__(self, config: xLSTMConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.gate_soft_cap = config.gate_soft_cap
        self.q = nn.Linear(config.hidden_size, config.qk_dim, bias=False)
        self.k = nn.Linear(config.hidden_size, config.qk_dim, bias=False)
        self.v = nn.Linear(config.hidden_size, config.v_dim, bias=False)
        self.ogate_preact = nn.Linear(config.hidden_size, config.v_dim, bias=False)
        self.igate_preact = nn.Linear(config.hidden_size, config.num_heads)
        self.fgate_preact = nn.Linear(config.hidden_size, config.num_heads)
        self.multihead_norm = xLSTMMultiHeadLayerNorm(config.v_dim, eps=config.norm_eps)
        self.out_proj = nn.Linear(config.v_dim, config.hidden_size, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, state: mLSTMLayerStateType | None = None
    ) -> tuple[torch.Tensor, mLSTMLayerStateType]:
        head_shape = (*hidden_states.shape[:-1], self.num_heads, -1)
        query = self.q(hidden_states).view(head_shape).transpose(1, 2)
        key = self.k(hidden_states).view(head_shape).transpose(1, 2)
        value = self.v(hidden_states).view(head_shape).transpose(1, 2)
        input_gate = soft_cap(self.igate_preact(hidden_states), self.gate_soft_cap).transpose(1, 2)
        forget_gate = soft_cap(self.fgate_preact(hidden_states), self.gate_soft_cap).transpose(1, 2)

        mlstm_output, state = eager_mlstm_forward(query, key, value, input_gate, forget_gate, state=state)
        mlstm_output = self.multihead_norm(mlstm_output.transpose(1, 2))
        output = self.out_proj(torch.sigmoid(self.ogate_preact(hidden_states)) * mlstm_output)
        return output, state


class xLSTMBlock(GradientCheckpointingLayer):
    def __init__(self, config: xLSTMConfig):
        super().__init__()
        self.norm_mlstm = xLSTMRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlstm_layer = xLSTMLayer(config)
        self.norm_ffn = xLSTMRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn = xLSTMFeedForward(config)

    def forward(
        self, hidden_states: torch.Tensor, state: mLSTMLayerStateType | None = None
    ) -> tuple[torch.Tensor, mLSTMLayerStateType]:
        mlstm_output, state = self.mlstm_layer(self.norm_mlstm(hidden_states), state)
        hidden_states = hidden_states + mlstm_output
        hidden_states = hidden_states + self.ffn(self.norm_ffn(hidden_states))
        return hidden_states, state


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

    @torch.no_grad()
    def _init_weights(self, module):
        # "small init" and "wang init" of GPT-NeoX (https://github.com/EleutherAI/gpt-neox)
        super()._init_weights(module)
        num_layers, hidden_size = self.config.num_hidden_layers, self.config.hidden_size
        if isinstance(module, (nn.Embedding, nn.Linear)):
            init.normal_(module.weight, std=(2 / (5 * hidden_size)) ** 0.5)
        if isinstance(module, xLSTMLayer):
            for gate in (module.igate_preact, module.fgate_preact, module.ogate_preact):
                init.zeros_(gate.weight)
            init.constant_(module.igate_preact.bias, -10.0)
            init.copy_(module.fgate_preact.bias, torch.linspace(3.0, 6.0, self.config.num_heads))
            init.normal_(module.out_proj.weight, std=2 / num_layers / hidden_size**0.5)
        elif isinstance(module, xLSTMFeedForward):
            init.zeros_(module.proj_up_gate.weight)
            init.normal_(module.proj_down.weight, std=2 / num_layers / module.proj_down.in_features**0.5)


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
        >>> tokenizer = AutoTokenizer.from_pretrained("NX-AI/xLSTM-7b")

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
    def __init__(self, config: xLSTMConfig):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([xLSTMBlock(config) for _ in range(config.num_hidden_layers)])
        self.out_norm = xLSTMRMSNorm(config.hidden_size, eps=config.norm_eps)
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
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if use_cache and cache_params is None:
            cache_params = xLSTMCache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        hidden_states = inputs_embeds
        for layer_idx, block in enumerate(self.blocks):
            state = cache_params.rnn_state[layer_idx] if cache_params is not None else None
            hidden_states, new_state = block(hidden_states, state)
            if cache_params is not None:
                for cached, new in zip(state, new_state):
                    cached.copy_(new)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        return xLSTMOutput(last_hidden_state=self.out_norm(hidden_states), cache_params=cache_params)


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
    def __init__(self, config: xLSTMConfig):
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
        logits = soft_cap(self.lm_head(xlstm_outputs.last_hidden_state).float(), self.config.output_logit_soft_cap)

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
