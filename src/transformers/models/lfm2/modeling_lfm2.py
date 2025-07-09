"""LFM2 model."""
from typing import Any, Callable, ClassVar, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import DynamicCache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.import_utils import is_causal_conv1d_available
from ...utils.generic import check_model_inputs
from .configuration_lfm2 import LFM2Config

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_fn, causal_conv1d_update = None, None


kernel_modules = (causal_conv1d_fn, causal_conv1d_update)
is_fast_path_available = all(kernel_modules)

logger = logging.get_logger(__name__)


@use_kernel_forward_from_hub("RMSNorm")
class LFM2RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        return output.type_as(x) * self.weight


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LFM2RotaryEmbedding(nn.Module):
    def __init__(self, config: LFM2Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    num_key_value_groups = query.shape[1] // key.shape[1]
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    else:
        seq_len = key_states.shape[-2]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=attn_weights.device),
            diagonal=1,
        )
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LFM2MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_dim: int,
        multiple_of: int,
        auto_adjust_ff_dim: bool,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        if auto_adjust_ff_dim:
            ff_dim = int(2 * ff_dim / 3)
            # custom dim factor multiplier
            if ffn_dim_multiplier is not None:
                ff_dim = int(ffn_dim_multiplier * ff_dim)
            ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, ff_dim, bias=False)
        self.w3 = nn.Linear(dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LFM2Cache(DynamicCache):
    """
    Attention and conv cache for LFM2.

    It stores the Key and Value states as a list of tensors, one for each layer.
    Attention layer cache shape: `[batch_size, num_heads, seq_len, head_dim]`.
    Conv layer cache shape: `[batch_size, conv_dim, L_cache-1]`.
    """

    def __init__(
        self,
        config: LFM2Config,
        max_batch_size: int,
        dtype: torch.dtype = torch.float32,
        device: Union[torch.device, str, None] = None,
    ):
        super().__init__()  # initialize key and value cache
        self.max_batch_size = max_batch_size
        self.full_attn_idxs = config.full_attn_idxs
        self.conv_L_cache = config.conv_L_cache
        self._dtype = dtype

        self.conv_cache: list[torch.Tensor] = []
        device = torch.device(device) if device is not None else None

        for _ in range(config.num_hidden_layers):
            conv_state = torch.zeros(
                self.max_batch_size,
                config.conv_dim,
                self.conv_L_cache,
                dtype=self._dtype,
                device=device,
            )
            torch._dynamo.mark_static_address(conv_state)
            self.conv_cache.append(conv_state)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == self.full_attn_idxs[0]:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif (
                not self.key_cache[layer_idx].numel()  # prefers not t.numel() to len(t) == 0 to export the model
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

            device = self.conv_cache[layer_idx].device
            self.conv_cache[layer_idx] = self.conv_cache[layer_idx].index_select(0, beam_idx.to(device))

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor
        layer_idx = self.full_attn_idxs[0] if layer_idx not in self.full_attn_idxs else layer_idx
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx].numel() == 0:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def reset(self):
        for layer_idx in range(len(self.conv_cache)):
            # In-place ops prevent breaking the static address
            self.conv_cache[layer_idx].zero_()


class LFM2Attention(nn.Module):
    def __init__(self, config: LFM2Config, layer_idx: Optional[int] = None, **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and "
                "will lead to errors during the forward call if caching is used. Please make sure to provide a "
                "`layer_idx` when creating this class."
            )
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True

        self.q_layernorm = LFM2RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = LFM2RMSNorm(self.head_dim, eps=config.norm_eps)

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[LFM2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_layernorm(self.q_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
        k = self.k_layernorm(self.k_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(*hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(
                key_states=k, value_states=v, layer_idx=self.layer_idx, cache_kwargs=cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        output = self.out_proj(attn_output)
        return output, attn_weights


class LFM2ShortConv(nn.Module):
    def __init__(
        self,
        config: LFM2Config,
        dim: int,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.L_cache = config.conv_L_cache
        self.bias = config.conv_bias

        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=self.L_cache,
            groups=dim,
            bias=self.bias,
            padding=self.L_cache - 1,
        )
        self.in_proj = nn.Linear(dim, 3 * dim, bias=self.bias)
        self.out_proj = nn.Linear(dim, dim, bias=self.bias)

    def cuda_kernels_forward(
        self,
        x: torch.Tensor,
        cache_params: Optional[LFM2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        BCx = self.in_proj(x).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)

        Bx = B * x

        conv_weights = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))
        if cache_params is not None and cache_position[0] > 0:
            conv_out = causal_conv1d_update(
                Bx.squeeze(-1),
                cache_params.conv_cache[self.layer_idx],
                conv_weights,
                self.conv.bias,
                None,
            )
            conv_out = conv_out.unsqueeze(-1)
        else:
            if cache_params is not None:
                conv_state = nn.functional.pad(Bx, (self.L_cache - Bx.shape[-1], 0))
                cache_params.conv_cache[self.layer_idx].copy_(conv_state)

            conv_out = causal_conv1d_fn(Bx, conv_weights, self.conv.bias, activation=None)

        y = C * conv_out
        y = self.out_proj(y.transpose(-1, -2).contiguous())
        return y

    def slow_forward(
        self,
        x: torch.Tensor,
        cache_params: Optional[LFM2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        seqlen = x.shape[1]
        BCx = self.in_proj(x).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)

        Bx = B * x

        if cache_params is not None and cache_position[0] > 0:
            conv_state = cache_params.conv_cache[self.layer_idx]
            cache_position = cache_position.clamp(0, self.L_cache - 1)
            conv_state = conv_state.roll(shifts=-1, dims=-1)
            conv_state[:, :, cache_position] = Bx.to(device=conv_state.device, dtype=conv_state.dtype)
            cache_params.conv_cache[self.layer_idx].copy_(conv_state)
            conv_out = torch.sum(conv_state.to(Bx.device) * self.conv.weight[:, 0, :], dim=-1)
            if self.bias:
                conv_out += self.conv.bias

            conv_out = conv_out.unsqueeze(-1)
        else:
            if cache_params is not None:
                conv_state = nn.functional.pad(Bx, (self.L_cache - Bx.shape[-1], 0))
                cache_params.conv_cache[self.layer_idx].copy_(conv_state)

            conv_out = self.conv(Bx)[..., :seqlen]

        y = C * conv_out
        y = y.transpose(-1, -2).contiguous()
        y = self.out_proj(y)
        return y

    def forward(
        self,
        x: torch.Tensor,
        cache_params: Optional[LFM2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if is_fast_path_available and "cuda" in x.device.type and not torch._dynamo.is_compiling():
            return self.cuda_kernels_forward(x, cache_params, cache_position, attention_mask)
        return self.slow_forward(x, cache_params, cache_position, attention_mask)


class LFM2AttentionDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LFM2Config, layer_idx: int):
        super().__init__()
        self.self_attn = LFM2Attention(config, layer_idx)
        self.feed_forward = LFM2MLP(
            dim=config.block_dim,
            ff_dim=config.block_ff_dim,
            multiple_of=config.block_multiple_of,
            auto_adjust_ff_dim=config.block_auto_adjust_ff_dim,
            ffn_dim_multiplier=config.block_ffn_dim_multiplier,
        )
        self.operator_norm = LFM2RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = LFM2RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        h, self_attn_weights = self.self_attn(
            hidden_states=self.operator_norm(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
        h += hidden_states
        out = h + self.feed_forward.forward(self.ffn_norm(h))

        outputs = (out,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class LFM2ShortConvDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LFM2Config, layer_idx: int):
        super().__init__()
        self.conv = LFM2ShortConv(
            config=config,
            dim=config.conv_dim,
            layer_idx=layer_idx,
        )
        self.feed_forward = LFM2MLP(
            dim=config.block_dim,
            ff_dim=config.block_ff_dim,
            multiple_of=config.block_multiple_of,
            auto_adjust_ff_dim=config.block_auto_adjust_ff_dim,
            ffn_dim_multiplier=config.block_ffn_dim_multiplier,
        )
        self.operator_norm = LFM2RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = LFM2RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[LFM2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        h = self.conv(
            self.operator_norm(hidden_states),
            cache_params=past_key_value,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        self_attn_weights = None

        h += hidden_states
        out = h + self.feed_forward.forward(self.ffn_norm(h))

        outputs = (out,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


@auto_docstring
class LFM2PretrainedModel(PreTrainedModel):
    config_class = LFM2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules: ClassVar = ["LFM2AttentionDecoderLayer", "LFM2ShortConvDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

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
        elif isinstance(module, LFM2RMSNorm):
            module.weight.data.fill_(1.0)


@auto_docstring
class LFM2Model(LFM2PretrainedModel):
    def __init__(self, config: LFM2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.pos_emb = LFM2RotaryEmbedding(config)

        decoder_layers = []
        for i in range(config.num_hidden_layers):
            if i in config.full_attn_idxs:
                decoder_layers.append(LFM2AttentionDecoderLayer(config, layer_idx=i))
            else:
                decoder_layers.append(LFM2ShortConvDecoderLayer(config, layer_idx=i))
        self.layers = nn.ModuleList(decoder_layers)

        self.embedding_norm = LFM2RMSNorm(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # @check_model_inputs
    # @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[LFM2Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            batch_size = inputs_embeds.shape[0]
            past_key_values = LFM2Cache(
                config=self.config, max_batch_size=batch_size, dtype=self.dtype, device=self.device
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        hidden_states = inputs_embeds

        position_embeddings = self.pos_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.embedding_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


@auto_docstring
class LFM2ForCausalLM(LFM2PretrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LFM2Config):
        super().__init__(config)
        self.model = LFM2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    # @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[LFM2Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # Overwritten -- Support custom LFM2Cache.

        empty_past_kv = past_key_values is None or (
            isinstance(past_key_values, DynamicCache) and past_key_values._seen_tokens == 0
        )

        # Omit tokens covered by past_key_values.
        if not empty_past_kv:
            # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
            # Exception 1: when passing input_embeds, input_ids may be missing entries
            # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
            # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
            #              (we can't check exception 3 while compiling)
            if (
                inputs_embeds is not None  # Exception 1
                or cache_position[-1] >= input_ids.shape[1]  # Exception 3
            ):
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
        else:
            past_key_values = LFM2Cache(self.config, input_ids.shape[0], dtype=self.dtype, device=self.device)

        # if attention_mask is not None and position_ids is None:
        #     # create position_ids on the fly for batch generation
        #     position_ids = attention_mask.long().cumsum(-1) - 1
        #     position_ids.masked_fill_(attention_mask == 0, 1)
        #     if not empty_past_kv:
        #         position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and empty_past_kv:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                # "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
            }
        )
        return model_inputs


__all__ = ["LFM2ForCausalLM", "LFM2Model", "LFM2PretrainedModel"]
