# This is an automatically generated file by the modularization script. Please do not make any manual changes
# to this file as they will be overwritten.

import math
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_flash_attention_utils import flash_attn_supports_top_left_mask, is_flash_attn_available
from ...modeling_layers import (
    GenericForSequenceClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    can_return_tuple,
    is_causal_conv1d_available,
    is_mamba_ssm_available,
    logging,
)
from ...utils.deprecation import deprecate_kwarg
from ..llama.modeling_llama import LlamaRMSNorm, repeat_kv, LlamaMLP
from .configuration_jamba import JambaConfig
from ..mixtral.modeling_mixtral import MixtralForCausalLM, MixtralPreTrainedModel, MixtralDecoderLayer, load_balancing_loss_func, MixtralExperts

if is_flash_attn_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward


if is_mamba_ssm_available():
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
else:
    selective_state_update, selective_scan_fn, mamba_inner_fn = None, None, None

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

is_fast_path_available = all(
    (selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)
)


logger = logging.get_logger(__name__)



class JambaRMSNorm(LlamaRMSNorm):
    pass


class JambaPreTrainedModel(PreTrainedModel):
    config_class = JambaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["JambaAttentionDecoderLayer", "JambaMambaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    # Note: only supports HybridMambaAttentionDynamicCache
    _is_stateful = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, JambaRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, JambaMambaMixer):
            A = torch.arange(1, module.ssm_state_size + 1)[None, :]
            A = A.expand(module.intermediate_size, -1).contiguous()
            module.A_log.data.copy_(torch.log(A))
            module.D.data.fill_(1.0)


class HybridMambaAttentionDynamicCache:
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the mamba cache
    (which has a constant shape regardless of seq_len).

    This cache has two sets of lists of tensors: `key_cache` and `value_cache` for attention cache and `conv_states`
    and `ssm_states` for mamba cache. Each of these lists has `num_layers` tensors. The expected shape for each tensor
    For attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_heads, seq_len, head_dim)`,
    while `conv_states` and `ssm_states` have a shape of `(batch_size, 0)` (empty tensors).
    For mamba layers, `key_cache` and `value_cache` have a shape of `(batch_size, 0)` (empty tensors),
    while `conv_states` represents the convolution state and has a shape of `(batch_size, d_inner, d_conv)`,
    and `ssm_states` represents the ssm state and has a shape of `(batch_size, d_inner, d_state)`.
    """

    is_compileable = False

    def __init__(self, config, batch_size, dtype=torch.float16, device=None):
        self.dtype = dtype
        self.layers_block_type = config.layers_block_type
        self.has_previous_state = False  # only used by mamba
        intermediate_size = config.mamba_expand * config.hidden_size
        ssm_state_size = config.mamba_d_state
        conv_kernel_size = config.mamba_d_conv
        self.conv_states = []
        self.ssm_states = []
        self.transformer_layers = []
        for i in range(config.num_hidden_layers):
            if self.layers_block_type[i] == "mamba":
                self.conv_states += [
                    torch.zeros(batch_size, intermediate_size, conv_kernel_size, device=device, dtype=dtype)
                ]
                self.ssm_states += [
                    torch.zeros(batch_size, intermediate_size, ssm_state_size, device=device, dtype=dtype)
                ]
            else:
                self.conv_states += [torch.tensor([[]] * batch_size, device=device)]
                self.ssm_states += [torch.tensor([[]] * batch_size, device=device)]
                self.transformer_layers.append(i)

        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Update the cache
        if self.key_cache[layer_idx].shape[-1] == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

            device = self.conv_states[layer_idx].device
            self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx.to(device))
            device = self.ssm_states[layer_idx].device
            self.ssm_states[layer_idx] = self.ssm_states[layer_idx].index_select(0, beam_idx.to(device))

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if layer_idx is None or len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class JambaAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: JambaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[HybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[HybridMambaAttentionDynamicCache]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_position=cache_position)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class JambaMambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: JambaConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = config.mamba_expand * config.hidden_size
        self.time_step_rank = (
            math.ceil(self.hidden_size / 16) if config.mamba_dt_rank == "auto" else config.mamba_dt_rank
        )
        self.use_conv_bias = config.mamba_conv_bias
        self.use_bias = config.mamba_proj_bias
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=self.conv_kernel_size - 1,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        self.use_fast_kernels = config.use_mamba_kernels

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=self.use_bias)
        # selective projection used to make dt, B and C input dependent
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.ssm_state_size + 1)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.use_bias)

        self.dt_layernorm = JambaRMSNorm(self.time_step_rank, eps=config.rms_norm_eps)
        self.b_layernorm = JambaRMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
        self.c_layernorm = JambaRMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
        self.b_layernorm = JambaRMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
        self.c_layernorm = JambaRMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)

        if not is_fast_path_available:
            logger.warning(
                "The fast path is not available because on of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`"
                " is None. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d. If you want to use the naive implementation, set `use_mamba_kernels=False` in the model config"
            )

    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        if cache_params is None:
            use_precomputed_states = False
        else:
            use_precomputed_states = (
                cache_params.has_previous_state
                and seq_len == 1
                and cache_params.conv_states[self.layer_idx].shape[0]
                == cache_params.ssm_states[self.layer_idx].shape[0]
                == batch_size
            )
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        # We can't use `mamba_inner_fn` even if in training and without cache params because we have the
        # inner layernorms which isn't supported by this fused kernel
        hidden_states, gate = projected_states.chunk(2, dim=1)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if use_precomputed_states:
            if causal_conv1d_update is None:
                raise ImportError("causal_conv1d_update is not available")
            hidden_states = causal_conv1d_update(
                hidden_states.squeeze(-1),
                cache_params.conv_states[self.layer_idx],
                conv_weights,
                self.conv1d.bias,
                self.activation,
            )
            hidden_states = hidden_states.unsqueeze(-1)
        else:
            if cache_params is not None:
                conv_states = nn.functional.pad(hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0))
                cache_params.conv_states[self.layer_idx].copy_(conv_states)
            if causal_conv1d_fn is None:
                raise ImportError("causal_conv1d_fn is not available")
            hidden_states = causal_conv1d_fn(hidden_states, conv_weights, self.conv1d.bias, activation=self.activation)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 3. State Space Model sequence transformation
        # 3.a. input varying initialization of time_step, B and C
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )

        time_step = self.dt_layernorm(time_step)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)

        # Here we need to apply dt_proj without the bias, as the bias is added in the selective scan kernel.
        # This is a hack to apply dt_proj while still using the forward pass of `torch.nn.Linear`, which is needed
        # in order to make quantization work. Quantization code replaces `torch.nn.Linear` layers with quantized
        # linear layers, and requires to call the forward pass directly.
        # Quantized model can't work with the original code:
        # ```discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)```
        time_proj_bias = self.dt_proj.bias.data
        with torch.no_grad():
            self.dt_proj.bias.data = torch.zeros_like(self.dt_proj.bias.data)
        discrete_time_step = self.dt_proj(time_step).transpose(1, 2)
        with torch.no_grad():
            self.dt_proj.bias.data = time_proj_bias

        A = -torch.exp(self.A_log.float())
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        time_proj_bias = time_proj_bias.float() if time_proj_bias is not None else None
        if use_precomputed_states:
            if selective_state_update is None:
                raise ImportError("selective_state_update is not available")
            scan_outputs = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states[..., 0],
                discrete_time_step[..., 0],
                A,
                B[:, 0],
                C[:, 0],
                self.D,
                gate[..., 0],
                time_proj_bias,
                dt_softplus=True,
            ).unsqueeze(-1)
        else:
            if selective_scan_fn is None:
                raise ImportError("selective_scan_fn is not available")
            scan_outputs, ssm_state = selective_scan_fn(
                hidden_states,
                discrete_time_step,
                A,
                B.transpose(1, 2),
                C.transpose(1, 2),
                self.D.float(),
                gate,
                time_proj_bias,
                delta_softplus=True,
                return_last_state=True,
            )
            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))

        return contextualized_states

    # fmt: off
    def slow_forward(self, input_states, cache_params: Optional[HybridMambaAttentionDynamicCache] = None, attention_mask: Optional[torch.LongTensor] = None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(input_states).transpose(1, 2)                   # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, dim=1)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        use_cache = isinstance(cache_params, HybridMambaAttentionDynamicCache)
        # 2. Convolution sequence transformation
        if use_cache and cache_params is not None and cache_params.ssm_states[self.layer_idx].shape[0] == batch_size:
            if self.training:
                # In training mode, we don't want to perform in-place operations on ssm_state so we can compute the backwards pass
                ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            else:
                ssm_state = cache_params.ssm_states[self.layer_idx]

            ssm_state = ssm_state.to(hidden_states.device)

            if cache_params.has_previous_state and seq_len == 1 and \
                    cache_params.conv_states[self.layer_idx].shape[0] == batch_size:
                conv_state = cache_params.conv_states[self.layer_idx]                   # [batch, intermediate_size, conv_kernel_size]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache_params.conv_states[self.layer_idx] = conv_state
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)         # [batch, intermediate_size, 1] : decoding
            else:
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx] = conv_state
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])     # [batch, intermediate_size, seq_len]
        else:
            ssm_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                device=hidden_states.device, dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])         # [batch, intermediate_size, seq_len]

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )

        time_step = self.dt_layernorm(time_step)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)

        discrete_time_step = self.dt_proj(time_step)                                    # [batch, seq_len, intermediate_size]
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]

        # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        A = -torch.exp(self.A_log.float())                                              # [intermediate_size, ssm_state_size]
        discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()       # [batch, intermediate_size, seq_len, ssm_state_size]
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        scan_outputs = []
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediate_size, ssm_state]
            scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediate_size, 1]
            scan_outputs.append(scan_output[:, :, 0])
        scan_output = torch.stack(scan_outputs, dim=-1)                                # [batch, intermediate_size, seq_len]
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = (scan_output * self.act(gate))

        if use_cache:
            cache_params.ssm_states[self.layer_idx] = ssm_state

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.transpose(1, 2))  # [batch, seq_len, hidden_size]
        return contextualized_states
    # fmt: on

    def forward(
        self,
        hidden_states,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        if self.use_fast_kernels:
            if not is_fast_path_available or "cuda" not in self.x_proj.weight.device.type:
                raise ValueError(
                    "Fast Mamba kernels are not available. Make sure to they are installed and that the mamba module is on a CUDA device"
                )
            return self.cuda_kernels_forward(hidden_states, cache_params, attention_mask)
        return self.slow_forward(hidden_states, cache_params, attention_mask)


class JambaMLP(LlamaMLP):
    pass


class JambaExperts(MixtralExperts):
    def route_tokens_to_experts(self, hidden_states, router_logits):
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
        top_k_weights, top_k_index = torch.topk(routing_weights, self.top_k, dim=-1)
        return top_k_index, top_k_weights



class JambaSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accommodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config: JambaConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.router = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = JambaExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits = self.router(hidden_states)
        hidden_states = self.experts(hidden_states, router_logits)
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states, router_logits


class JambaAttentionDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: JambaConfig, layer_idx: int):
        super().__init__()
        num_experts = config.layers_num_experts[layer_idx] if config.layers_num_experts else 1
        self.self_attn = JambaAttention(config, layer_idx)

        ffn_layer_class = JambaSparseMoeBlock if num_experts > 1 else JambaMLP
        self.feed_forward = ffn_layer_class(config)
        self.input_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridMambaAttentionDynamicCache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor, Optional[torch.Tensor], Optional[HybridMambaAttentionDynamicCache]]:
        hidden_states = self.input_layernorm(hidden_states)
        residual = hidden_states
        hidden_states, _= self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        ff_outputs = self.feed_forward(hidden_states)
        if isinstance(ff_outputs, tuple):
            hidden_states, _ = ff_outputs
        hidden_states = residual + hidden_states
        return hidden_states


class JambaMambaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: JambaConfig, layer_idx: int):
        super().__init__()
        num_experts = config.layers_num_experts[layer_idx] if config.layers_num_experts else 1
        self.mamba = JambaMambaMixer(config=config, layer_idx=layer_idx)
        ffn_layer_class = JambaSparseMoeBlock if num_experts > 1 else JambaMLP
        self.feed_forward = ffn_layer_class(config)
        self.input_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridMambaAttentionDynamicCache] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor, Optional[torch.Tensor], Optional[HybridMambaAttentionDynamicCache]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        ff_outputs = self.feed_forward(hidden_states)
        if isinstance(ff_outputs, tuple):
            hidden_states, _ = ff_outputs
        return hidden_states


ALL_DECODER_LAYER_TYPES = {"attention": JambaAttentionDecoderLayer, "mamba": JambaMambaDecoderLayer}


class JambaModel(JambaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`JambaDecoderLayer`]

    Args:
        config: JambaConfig
    """

    def __init__(self, config: JambaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        decoder_layers = []
        for i in range(config.num_hidden_layers):
            layer_class = ALL_DECODER_LAYER_TYPES[config.layers_block_type[i]]
            decoder_layers.append(layer_class(config, layer_idx=i))
        self.layers = nn.ModuleList(decoder_layers)

        self._attn_implementation = config._attn_implementation
        self.final_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridMambaAttentionDynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            causal_mask_mapping["full_attention"] = create_causal_mask(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            ) if self.config.use_mamba_kernels else attention_mask


            causal_mask_mapping["mamba"] = self._update_mamba_mask(attention_mask, cache_position)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            # Depending on the layer type we opt for 2D base attention mask (Mamba) or 4D causal mask (Attention)
            layer_mask = causal_mask_mapping["mamba"] if isinstance(decoder_layer, JambaMambaDecoderLayer) else causal_mask_mapping["full_attention"]

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        hidden_states = self.final_layernorm(hidden_states)

        if past_key_values and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
        )

    def _update_mamba_mask(self, attention_mask, cache_position):
        """
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        mamba_mask = attention_mask
        if (cache_position is not None and cache_position[0] > 0) or (
            attention_mask is not None and torch.all(attention_mask == 1)
        ):
            mamba_mask = None
        return mamba_mask


class JambaForCausalLM(MixtralForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: JambaConfig):
        super().__init__(config)
        self.model = JambaModel(config)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        # Overwritten -- has a unique cache type, `HybridMambaAttentionDynamicCache`
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values.get_seq_length()

        # create cache if necessary
        if past_key_values is None:
            past_key_values = HybridMambaAttentionDynamicCache(
                self.config, input_ids.shape[0], self.dtype, device=self.device
            )

        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_ids.shape[1], device=input_ids.device)
        else:
            # past_length in this case is close to the actual seq_len - 1
            # and we want to add the new token to the cache
            cache_position = torch.tensor([past_length], device=input_ids.device)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values.get_seq_length() == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "cache_position": cache_position,
            }
        )
        return model_inputs


class JambaForSequenceClassification(GenericForSequenceClassification, JambaPreTrainedModel):
    pass
