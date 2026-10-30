# Copyright 2026 Cartesia AI and The HuggingFace Inc. team. All rights reserved.
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
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask, create_sliding_window_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import no_inherit_decorator
from ..bamba.modeling_bamba import BambaRotaryEmbedding, apply_rotary_pos_emb
from ..lfm2.modeling_lfm2 import Lfm2Attention
from ..llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaMLP,
    LlamaRMSNorm,
    eager_attention_forward,
)
from ..zamba2.modeling_zamba2 import Zamba2MambaMixer, mamba2_chunk_scan


@auto_docstring(checkpoint="cartesia-ai/hnet_1stage_L")
@strict
class HNetConfig(PreTrainedConfig):
    r"""
    n_mamba_heads (`int`, *optional*):
        Number of heads of the Mamba2 layers, defaults to `mamba_expand * hidden_size // mamba_headdim`.
    encoder_layer_types (`list[list[str]]`, *optional*):
        Layer types (`"mamba"` or `"attention"`) of the encoder of each outer stage, outermost (bytes) first.
    main_layer_types (`list[str]`, *optional*):
        Layer types of the innermost (main) network.
    decoder_layer_types (`list[list[str]]`, *optional*):
        Layer types of the decoder of each outer stage.
    mamba_d_state (`int`, *optional*, defaults to 128):
        State size of the Mamba2 layers.
    mamba_d_conv (`int`, *optional*, defaults to 4):
        Convolution kernel size of the Mamba2 layers.
    mamba_expand (`int`, *optional*, defaults to 2):
        Expansion factor of the Mamba2 layers.
    mamba_headdim (`int`, *optional*, defaults to 64):
        Head dimension of the Mamba2 layers.
    mamba_ngroups (`int`, *optional*, defaults to 1):
        Number of groups of the Mamba2 B/C projections.
    chunk_size (`int`, *optional*, defaults to 256):
        Chunk size of the Mamba2 scan.
    time_step_min (`float`, *optional*, defaults to 0.0):
        Lower bound of the Mamba2 time step.
    add_bias_linear (`bool`, *optional*, defaults to `False`):
        Whether the Mamba2 projections have a bias.
    use_conv_bias (`bool`, *optional*, defaults to `True`):
        Whether the Mamba2 convolution has a bias.
    use_mamba_kernels (`bool`, *optional*, defaults to `True`):
        Deprecated and has no effect. The fast Mamba kernels are used automatically when available.

    Each stage is wider than the previous one: `hidden_size` and the sizes derived from it are the outermost stage's,
    and `per_layer_config` holds the inner stages'. Instead of `per_layer_config`, the stages can be given as lists,
    outermost first: `hidden_sizes`, `intermediate_sizes`, `num_heads_per_stage` and `sliding_windows` (`None` for
    global attention). The defaults are `cartesia-ai/hnet_1stage_L`.

    ```python
    >>> from transformers import HNetConfig, HNetForCausalLM

    >>> configuration = HNetConfig(hidden_sizes=[1024, 1536], sliding_windows=[1024, None])
    >>> model = HNetForCausalLM(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "hnet"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 10000.0

    vocab_size: int = 256
    hidden_size: int = 1024
    intermediate_size: int = 0
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    n_mamba_heads: int | None = None
    sliding_window: int | None = 1024
    encoder_layer_types: list[list[str]] | None = None
    main_layer_types: list[str] | None = None
    decoder_layer_types: list[list[str]] | None = None
    layer_types: list[str] | None = None
    num_hidden_layers: int | None = None
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_headdim: int = 64
    mamba_ngroups: int = 1
    chunk_size: int = 256
    time_step_min: float = 0.0
    add_bias_linear: bool = False
    use_conv_bias: bool = True
    use_mamba_kernels: bool = True
    hidden_act: str = "silu"
    mlp_bias: bool = False
    max_position_embeddings: int = 1_048_576
    rope_parameters: RopeParameters | dict | None = None
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 254
    eos_token_id: int | list[int] | None = 255
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        # `hidden_sizes`, `intermediate_sizes`, `num_heads_per_stage` and `sliding_windows` are builder-only kwargs:
        # they shape `layer_types` and `per_layer_config` and are not kept on the config
        hidden_sizes = kwargs.pop("hidden_sizes", [self.hidden_size, 1536])
        intermediate_sizes = kwargs.pop("intermediate_sizes", [self.intermediate_size, 4096])
        num_heads_per_stage = kwargs.pop("num_heads_per_stage", [self.num_attention_heads] * len(hidden_sizes))
        sliding_windows = kwargs.pop("sliding_windows", [1024] + [None] * (len(hidden_sizes) - 1))
        self.encoder_layer_types = self.encoder_layer_types or [["mamba"] * 4]
        self.main_layer_types = self.main_layer_types or ["attention"] * 22
        self.decoder_layer_types = self.decoder_layer_types or [["mamba"] * 4]
        if self.rope_parameters is None:
            self.rope_parameters = {
                "rope_type": "default",
                "rope_theta": self.default_theta,
                "partial_rotary_factor": 0.5,
            }

        # Flat view of every cached layer in forward order: Mamba layers and the routing / dechunk states are
        # recurrent, attention layers keep keys and values
        layer_stages = [stage_idx for _, stage_idx, layer_types in self.stack_layout for _ in layer_types]
        if self.layer_types is None:
            if len({window for window in sliding_windows if window is not None}) > 1:
                raise ValueError(f"All sliding stages must share one window, got `sliding_windows={sliding_windows}`.")
            self.sliding_window = next((window for window in sliding_windows if window is not None), None)
            self.layer_types = [
                "linear_attention"
                if layer_type in ("mamba", "state")
                else ("sliding_attention" if sliding_windows[stage_idx] else "full_attention")
                for _, stage_idx, layer_types in self.stack_layout
                for layer_type in layer_types
            ]
        self.num_hidden_layers = len(self.layer_types)

        if "per_layer_config" not in kwargs:
            stage_configs = [
                {
                    "hidden_size": hidden_size,
                    "intermediate_size": intermediate_size,
                    "num_attention_heads": num_heads,
                    "num_key_value_heads": num_heads,
                    "head_dim": hidden_size // num_heads,
                    "n_mamba_heads": self.mamba_expand * hidden_size // self.mamba_headdim,
                }
                for hidden_size, intermediate_size, num_heads in zip(
                    hidden_sizes, intermediate_sizes, num_heads_per_stage
                )
            ]
            for attribute, value in stage_configs[0].items():
                setattr(self, attribute, value)
            kwargs["per_layer_config"] = {
                layer_idx: stage_configs[stage_idx]
                for layer_idx, stage_idx in enumerate(layer_stages)
                if stage_idx > 0
            }
        super().__post_init__(**kwargs)

    @property
    def stack_layout(self) -> list[tuple[str, int, list[str]]]:
        """Every unit that keeps a cache, in forward order: `(kind, stage_idx, layer_types)`."""
        num_outer_stages = len(self.encoder_layer_types)
        layout = []
        for stage_idx in range(num_outer_stages):
            layout += [("encoder", stage_idx, self.encoder_layer_types[stage_idx]), ("routing", stage_idx, ["state"])]
        layout.append(("main", num_outer_stages, self.main_layer_types))
        for stage_idx in reversed(range(num_outer_stages)):
            layout += [("dechunk", stage_idx, ["state"]), ("decoder", stage_idx, self.decoder_layer_types[stage_idx])]
        return layout

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if len(self.decoder_layer_types) != len(self.encoder_layer_types):
            raise ValueError(
                f"Got {len(self.encoder_layer_types)} encoders and {len(self.decoder_layer_types)} decoders: expected "
                "one encoder and one decoder per outer stage."
            )


class HNetRMSNorm(LlamaRMSNorm):
    pass


class HNetRotaryEmbedding(BambaRotaryEmbedding):
    pass


class HNetMLP(LlamaMLP):
    pass


class HNetMambaMixer(Zamba2MambaMixer):
    pass


@no_inherit_decorator  # Lfm2's kernelized rotary is not partial
class HNetAttention(Lfm2Attention):
    def __init__(self, config: HNetConfig, layer_idx: int):
        nn.Module.__init__(self)
        # Sizes come from the layer's config, but `_attn_implementation` must follow the live model config
        self.config = config
        self.layer_idx = layer_idx
        layer_config = config.per_layer_config[layer_idx]
        self.head_dim = layer_config.head_dim
        self.num_key_value_groups = 1
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        # CODEPATH: hnet_2stage_* slide over 1024 chunks at stage 1, every main network is global
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        projection_size = layer_config.num_attention_heads * self.head_dim
        self.q_proj = nn.Linear(layer_config.hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(layer_config.hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(layer_config.hidden_size, projection_size, bias=False)
        self.out_proj = nn.Linear(projection_size, layer_config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.out_proj(attn_output), attn_weights


class HNetDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: HNetConfig, layer_idx: int):
        super().__init__()
        layer_config = config.per_layer_config[layer_idx]
        self.is_mamba = config.layer_types[layer_idx] == "linear_attention"
        self.input_layernorm = HNetRMSNorm(layer_config.hidden_size, eps=config.rms_norm_eps)
        if self.is_mamba:
            self.mixer = HNetMambaMixer(layer_config, layer_idx=layer_idx)
        else:
            # Released checkpoints pair every attention layer with an MLP, and no Mamba layer with one
            self.mixer = HNetAttention(config, layer_idx=layer_idx)
            self.post_attention_layernorm = HNetRMSNorm(layer_config.hidden_size, eps=config.rms_norm_eps)
            self.mlp = HNetMLP(layer_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.is_mamba:
            return residual + self.mixer(hidden_states, cache_params=past_key_values, attention_mask=attention_mask)

        hidden_states, _ = self.mixer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class HNetStack(nn.Module):
    """
    Mamba2 / attention layers running at the width of one stage: an encoder, decoder or main network. `layer_ids`
    index `config.layer_types`, shared by all stacks so that they write to distinct cache layers.
    """

    def __init__(self, config: HNetConfig, layer_ids: range):
        super().__init__()
        self.config = config
        stage_config = config.per_layer_config[layer_ids[0]]
        self.layers = nn.ModuleList([HNetDecoderLayer(config, idx) for idx in layer_ids])
        self.norm_f = HNetRMSNorm(stage_config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HNetRotaryEmbedding(stage_config)
        self.first_attention_idx = next(
            (idx for idx in layer_ids if config.layer_types[idx] != "linear_attention"), None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        causal_mask = position_embeddings = None
        if self.first_attention_idx is not None:
            past_length = (
                past_key_values.get_seq_length(self.first_attention_idx) if past_key_values is not None else 0
            )
            position_ids = torch.arange(past_length, past_length + hidden_states.shape[1], device=hidden_states.device)
            position_embeddings = self.rotary_emb(hidden_states, position_ids[None])
            is_sliding = self.config.layer_types[self.first_attention_idx] == "sliding_attention"
            create_mask = create_sliding_window_causal_mask if is_sliding else create_causal_mask
            causal_mask = create_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids[None],
                layer_idx=self.first_attention_idx,
            )
        mamba_mask = create_recurrent_attention_mask(self.config, hidden_states, attention_mask)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=mamba_mask if layer.is_mamba else causal_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                **kwargs,
            )
        return self.norm_f(hidden_states)


class HNetRoutingModule(nn.Module):
    """Predicts a chunk boundary wherever two consecutive positions are dissimilar (cosine similarity)."""

    def __init__(self, config: HNetConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        hidden_size = config.per_layer_config[layer_idx].hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, past_key_values: Cache | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # When decoding, the first position is compared to the last one of the previous step
        previous = past_key_values.get_recurrent_state(self.layer_idx) if past_key_values is not None else None
        states = hidden_states if previous is None else torch.cat([previous[:, None], hidden_states], dim=1)
        queries = F.normalize(self.q_proj(states[:, :-1]), dim=-1)
        keys = F.normalize(self.k_proj(states[:, 1:]), dim=-1)
        boundary_prob = ((1 - (queries * keys).sum(dim=-1)) / 2).clamp(min=0.0, max=1.0)
        if previous is None:
            # The first position (the first non-padding one with left padding) always starts a chunk
            boundary_prob = F.pad(boundary_prob, (1, 0), value=1.0)
            is_first = attention_mask & ~F.pad(attention_mask[:, :-1], (1, 0), value=False)
            boundary_prob = boundary_prob.masked_fill(is_first, 1.0)
        if past_key_values is not None:
            past_key_values.update_recurrent_state(hidden_states[:, -1], self.layer_idx)
        boundary_mask = (boundary_prob > 0.5) & attention_mask
        return boundary_prob, boundary_mask


def boundaries_first(boundary_mask: torch.Tensor) -> torch.Tensor:
    """Indices that move the boundary positions to the front, in order: `[b, n, b, n, b] -> [0, 2, 4, 1, 3]`."""
    seq_len = boundary_mask.shape[1]
    positions = torch.arange(seq_len, device=boundary_mask.device) + (~boundary_mask).long() * seq_len
    return positions.argsort(dim=1)


def chunk(hidden_states: torch.Tensor, boundary_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Keeps the boundary positions only, right-padded: `(B, L, D) -> (B, num_chunks, D)` and its mask."""
    num_chunks = boundary_mask.sum(dim=1)
    max_chunks = int(num_chunks.max())
    order = boundaries_first(boundary_mask)[:, :max_chunks]
    chunk_states = hidden_states.gather(1, order[..., None].expand(-1, -1, hidden_states.shape[-1]))
    chunk_mask = torch.arange(max_chunks, device=boundary_mask.device)[None] < num_chunks[:, None]
    return chunk_states, chunk_mask


def dechunk(
    chunk_states: torch.Tensor,
    boundary_prob: torch.Tensor,
    boundary_mask: torch.Tensor,
    config: HNetConfig,
    previous: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Smooths the chunks with an EMA weighted by their boundary probability `p`, then repeats each chunk over the
    positions it covers: `(B, num_chunks, D) -> (B, L, D)`. `previous` is the last EMA value of the previous step.

    `out_t = p_t * h_t + (1 - p_t) * out_{t-1}` is a Mamba2 scan with `dt = -log(1 - p)`, `A = -1`, `B = p`, `C = 1`
    and inputs `h / dt`, as in the original implementation.
    """
    batch_size, num_chunks, hidden_size = chunk_states.shape
    num_heads = hidden_size // config.mamba_headdim
    smoothed = previous[:, None].float() if previous is not None else chunk_states.float()
    if num_chunks > 0:
        prob = boundary_prob.float().clamp(min=1e-4, max=1 - 1e-4)
        prob = prob.gather(1, boundaries_first(boundary_mask)[:, :num_chunks])
        dt = -torch.log1p(-prob)
        initial_states = None if previous is None else previous.float().view(batch_size, num_heads, -1, 1)
        smoothed = mamba2_chunk_scan(
            (chunk_states.float() / dt[..., None]).view(batch_size, num_chunks, num_heads, -1),
            dt[..., None].expand(-1, -1, num_heads),
            -torch.ones(num_heads, device=dt.device),
            prob[..., None, None],
            torch.ones_like(prob)[..., None, None],
            config.chunk_size,
            initial_states=initial_states,
        ).view(batch_size, num_chunks, hidden_size)
        if previous is not None:
            smoothed = torch.cat([previous[:, None].float(), smoothed], dim=1)
    if previous is None:
        chunk_index = (boundary_mask.cumsum(dim=1) - 1).clamp(min=0)
    else:
        # Positions before the first new boundary still belong to the previous chunk, at index 0
        chunk_index = boundary_mask.cumsum(dim=1)
    smoothed = smoothed.gather(1, chunk_index[..., None].expand(-1, -1, hidden_size))
    return smoothed.to(chunk_states.dtype)


@auto_docstring
class HNetPreTrainedModel(PreTrainedModel):
    config: HNetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HNetDecoderLayer"]
    _keep_in_fp32_modules = ["residual_projs"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _is_stateful = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, HNetMambaMixer):
            init.copy_(module.A_log, torch.log(torch.arange(1, module.num_heads + 1, dtype=torch.float32)))
            init.ones_(module.D)
            init.ones_(module.dt_bias)
        elif isinstance(module, HNetRoutingModule):
            init.copy_(module.q_proj.weight, torch.eye(module.q_proj.in_features))
            init.copy_(module.k_proj.weight, torch.eye(module.k_proj.in_features))
        elif isinstance(module, HNetModel):
            for residual_proj in module.residual_projs:
                init.zeros_(residual_proj.weight)
                init.zeros_(residual_proj.bias)
            for pad_dimension in module.pad_dimensions:
                init.zeros_(pad_dimension)


@auto_docstring
class HNetModel(HNetPreTrainedModel):
    def __init__(self, config: HNetConfig):
        super().__init__(config)
        num_outer_stages = len(config.encoder_layer_types)
        # Where each unit starts in `config.layer_types`, i.e. in the cache and `per_layer_config`
        offsets, start = {}, 0
        for kind, stage_idx, layer_types in config.stack_layout:
            offsets[kind, stage_idx] = range(start, start + len(layer_types))
            start += len(layer_types)
        self.routing_ids = [offsets["routing", idx][0] for idx in range(num_outer_stages)]
        self.dechunk_ids = [offsets["dechunk", idx][0] for idx in range(num_outer_stages)]
        widths = [config.per_layer_config[idx].hidden_size for idx in self.routing_ids]
        widths.append(config.per_layer_config[offsets["main", num_outer_stages][0]].hidden_size)
        self.stage_widths = widths

        self.embed_tokens = nn.Embedding(config.vocab_size, widths[0])
        self.encoders = nn.ModuleList([HNetStack(config, offsets["encoder", idx]) for idx in range(num_outer_stages)])
        self.routing_modules = nn.ModuleList([HNetRoutingModule(config, idx) for idx in self.routing_ids])
        self.residual_projs = nn.ModuleList([nn.Linear(widths[idx], widths[idx]) for idx in range(num_outer_stages)])
        self.main_network = HNetStack(config, offsets["main", num_outer_stages])
        self.decoders = nn.ModuleList([HNetStack(config, offsets["decoder", idx]) for idx in range(num_outer_stages)])
        # Learned padding that widens the input of stage `idx + 1` from `widths[idx]` to `widths[idx + 1]`
        self.pad_dimensions = nn.ParameterList(
            [nn.Parameter(torch.zeros(widths[idx + 1] - widths[idx])) for idx in range(num_outer_stages)]
        )
        self.post_init()

    def widen(self, hidden_states: torch.Tensor, stage_idx: int) -> torch.Tensor:
        pad = self.pad_dimensions[stage_idx - 1].to(hidden_states.dtype)
        return torch.cat([hidden_states, pad.expand(*hidden_states.shape[:-1], -1)], dim=-1)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        # Positions are counted per stage, in chunks
        kwargs.pop("position_ids", None)

        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if past_key_values is not None and hidden_states.shape[0] > 1:
            raise ValueError(
                "H-Net only supports a cache with batch size 1: each sequence closes chunks at different steps."
            )
        byte_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        # The 2D mask spans the past bytes when decoding, routing and chunking only see the new ones
        mask = attention_mask[:, -hidden_states.shape[1] :].bool()

        # Down: encode, then keep only the chunk boundaries for the next stage
        stage_masks, residuals, boundaries = [], [], []
        for stage_idx, (encoder, routing_module, residual_proj) in enumerate(
            zip(self.encoders, self.routing_modules, self.residual_projs)
        ):
            if stage_idx > 0:
                hidden_states = self.widen(hidden_states, stage_idx)
            # Inner stages are right-padded, which a causal model never sees: only the bytes need a mask
            stage_mask = byte_mask if stage_idx == 0 else None
            hidden_states = encoder(hidden_states, stage_mask, past_key_values, **kwargs)
            boundary_prob, boundary_mask = routing_module(hidden_states, mask, past_key_values)
            stage_masks.append(stage_mask)
            residuals.append(residual_proj(hidden_states.to(residual_proj.weight.dtype)))
            boundaries.append((boundary_prob, boundary_mask))
            hidden_states, mask = chunk(hidden_states, boundary_mask)
            # Decoding a byte that closes no chunk: the inner stages have nothing to process
            if hidden_states.shape[1] == 0:
                break
        else:
            hidden_states = self.widen(hidden_states, len(self.encoders))
            hidden_states = self.main_network(hidden_states, None, past_key_values, **kwargs)

        # Up: spread each chunk back over its positions, add the encoder residual, decode
        for stage_idx in reversed(range(len(boundaries))):
            hidden_states = hidden_states[..., : self.stage_widths[stage_idx]]
            boundary_prob, boundary_mask = boundaries[stage_idx]
            dechunk_idx = self.dechunk_ids[stage_idx]
            hidden_states = dechunk(
                hidden_states,
                boundary_prob,
                boundary_mask,
                self.config,
                past_key_values.get_recurrent_state(dechunk_idx) if past_key_values is not None else None,
            )
            if past_key_values is not None:
                past_key_values.update_recurrent_state(hidden_states[:, -1].float(), dechunk_idx)
            # Straight-through: the confidence of the routing decision is 1 in the forward, but gets the gradient
            confidence = torch.maximum(boundary_prob, 1 - boundary_prob)[..., None].float()
            confidence = 1 + confidence - confidence.detach()
            residual = residuals[stage_idx]
            hidden_states = (hidden_states.to(residual.dtype) * confidence + residual).to(hidden_states.dtype)
            hidden_states = self.decoders[stage_idx](hidden_states, stage_masks[stage_idx], past_key_values, **kwargs)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class HNetForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        HNetPreTrainedModel.__init__(self, config)
        self.model = HNetModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(self.model.stage_widths[0], config.vocab_size, bias=False)
        self.post_init()

    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
    ):
        # Each sequence closes chunks at different steps, so batched generation recomputes the full sequence
        if batch_size > 1:
            generation_config.use_cache = False
        return GenerationMixin._prepare_cache_for_generation(
            self, generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
        )


__all__ = ["HNetConfig", "HNetForCausalLM", "HNetModel", "HNetPreTrainedModel"]
