"""PyTorch VideoLLaMA3 model."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn import LayerNorm

from ..auto import AutoModel
from ...modeling_outputs import BaseModelOutput, ModelOutput
from ...activations import ACT2FN
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...modeling_layers import GradientCheckpointingLayer
from ...processing_utils import Unpack
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torchdynamo_compiling,
    logging,
)
from .configuration_videollama3 import Videollama3VisionConfig, Videollama3Config


logger = logging.get_logger(__name__)


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for VideoLLaMA3 outputs, with hidden states and attentions.
    """
)
class Videollama3ModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    compression_mask (`torch.BoolTensor` of shape `(batch_size, seq_len)`, *optional*):
        The mask indicating which tokens are kept when token compression is enabled.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    compression_mask: Optional[torch.BoolTensor] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for VideoLLaMA3 causal language model (or autoregressive) outputs.
    """
)
class Videollama3CausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    compression_mask (`torch.BoolTensor` of shape `(batch_size, seq_len)`, *optional*):
        The mask indicating which tokens are kept when token compression is enabled.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    compression_mask: Optional[torch.BoolTensor] = None


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
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


@auto_docstring
class Videollama3PreTrainedModel(PreTrainedModel):
    config: Videollama3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer", "Videollama3VisionEncoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = False
    _supports_attention_backend = True


class Videollama3VisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Videollama3VisionEmbeddings(nn.Module):
    def __init__(self, config: Videollama3VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1, self.config.num_channels, self.patch_size, self.patch_size
        )
        patch_embeds = self.patch_embedding(hidden_states)
        embeddings = patch_embeds.view(-1, self.embed_dim)
        return embeddings


class Videollama3VisionMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Videollama3VisionAttention(nn.Module):
    def __init__(self, config: Videollama3VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states = self.q_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2":
            # Flash Attention 2: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs, attn_weights = [], []
            for q, k, v in zip(*splits):
                attn_output, attn_weight = attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )
                attn_outputs.append(attn_output)
                attn_weights.append(attn_weight)

            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class Videollama3VisionEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Videollama3VisionConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Videollama3VisionAttention(config=config)
        self.layer_norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Videollama3VisionMLP(config=config)
        self.layer_norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        output_attentions: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, attn_weights = self.self_attn(
            self.layer_norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Videollama3VisionTransformerEncoder(nn.Module):
    def __init__(self, config: Videollama3VisionConfig) -> None:
        super().__init__()
        self.config = config
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_pos_emb = Videollama3VisionRotaryEmbedding(head_dim // 2)
        self.layers = nn.ModuleList(
            [Videollama3VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_sizes, merge_sizes):
        pos_ids = []
        for (t, h, w), merge_size in zip(grid_sizes, merge_sizes):
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // merge_size,
                merge_size,
                w // merge_size,
                merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // merge_size,
                merge_size,
                w // merge_size,
                merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_sizes[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)

        return rotary_pos_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_sizes: torch.Tensor,
        merge_sizes: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        rotary_pos_emb = self.rot_pos_emb(grid_sizes, merge_sizes)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_sizes[:, 1] * grid_sizes[:, 2], grid_sizes[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_sizes.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class Videollama3VisionModel(Videollama3PreTrainedModel):
    config_class = Videollama3VisionConfig
    _no_split_modules = ["Videollama3VisionEncoderLayer"]

    def __init__(self, config: Videollama3VisionConfig):
        super().__init__(config=config)
        embed_dim = config.hidden_size

        self.embeddings = Videollama3VisionEmbeddings(config)
        self.encoder = Videollama3VisionTransformerEncoder(config)
        self.post_layernorm = LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.post_init()

    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_sizes: torch.Tensor,
        merge_sizes: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Args:
            grid_sizes (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image or video before vision encoder.
            merge_sizes (`torch.Tensor` of shape `(num_images_or_videos,)`):
                The spatial downsampling ratio of each image or video feature.
        """
        hidden_states = self.embeddings(pixel_values.type(self.dtype))
        encoder_outputs = self.encoder(
            hidden_states,
            grid_sizes,
            merge_sizes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        hidden_states_chunks = last_hidden_state.split(grid_sizes.prod(dim=1).tolist(), dim=0)
        outputs = []

        for hidden_states, grid_size, merge_size in zip(hidden_states_chunks, grid_sizes, merge_sizes):
            c = hidden_states.shape[-1]
            hidden_states = hidden_states.view(
                grid_size[0], grid_size[1] // merge_size, grid_size[2] // merge_size, merge_size, merge_size,  c
            ).permute(0, 1, 3, 2, 4, 5)
            hidden_states = hidden_states.reshape(
                grid_size[0], grid_size[1], grid_size[2], c
            ).permute(0, 3, 1, 2)
            hidden_states = torch.nn.functional.interpolate(
                hidden_states,
                size=(grid_size[1] // merge_size, grid_size[2] // merge_size),
                mode='bilinear'
            )
            hidden_states = hidden_states.permute(0, 2, 3, 1).view(-1, c)
            outputs.append(hidden_states)

        last_hidden_state = torch.cat(outputs, dim=0)

        if not return_dict:
            return (last_hidden_state,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Videollama3Projector(nn.Module):
    def __init__(self, config: Videollama3Config) -> None:
        super().__init__()
        in_hidden_size = config.vision_config.hidden_size
        out_hidden_size = config.text_config.hidden_size
        self.readout = nn.Sequential(
            nn.Linear(in_hidden_size, out_hidden_size),
            nn.GELU(),
            nn.Linear(out_hidden_size, out_hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.readout(x)
        return x


class Videollama3Model(Videollama3PreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {
        "^model": "language_model",
    }

    def __init__(self, config: Videollama3Config):
        super().__init__(config)
        self.vision_model = AutoModel.from_config(config.vision_config)
        self.projector = Videollama3Projector(config)
        self.language_model = AutoModel.from_config(config.text_config)

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
    ):
        if input_ids is None:
            image_token_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_token_mask = image_token_mask.all(-1)
            video_token_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            video_token_mask = video_token_mask.all(-1)
        else:
            image_token_mask = input_ids == self.config.image_token_id
            video_token_mask = input_ids == self.config.video_token_id

        image_token_mask = image_token_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        video_token_mask = video_token_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

        return image_token_mask, video_token_mask

    def get_multimodal_embeddings(
        self,
        pixel_values: torch.Tensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
        pixel_values_videos: torch.FloatTensor,
        grid_sizes_videos: torch.LongTensor,
        merge_sizes_videos: torch.LongTensor,
    ):
        if pixel_values is None and pixel_values_videos is None:
            image_embeds = torch.empty((0, self.config.text_config.hidden_size), device=self.device, dtype=self.dtype)
            video_embeds = image_embeds.clone()
            return image_embeds, video_embeds

        if pixel_values is None:
            pixel_values = pixel_values_videos.new_empty((0, pixel_values_videos.size(1)))
            grid_sizes = pixel_values_videos.new_zeros((0, 3), dtype=torch.long)
            merge_sizes = pixel_values_videos.new_ones((0,), dtype=torch.long)

        if pixel_values_videos is None:
            pixel_values_videos = pixel_values.new_empty((0, pixel_values.size(1)))
            grid_sizes_videos = pixel_values.new_zeros((0, 3), dtype=torch.long)
            merge_sizes_videos = pixel_values.new_ones((0,), dtype=torch.long)

        num_image_features = torch.sum(grid_sizes.prod(dim=1) // (merge_sizes ** 2)).item()
        num_video_features = torch.sum(grid_sizes_videos.prod(dim=1).sum() // (merge_sizes_videos ** 2)).item()

        pixel_values = torch.cat([pixel_values, pixel_values_videos], dim=0)
        grid_sizes = torch.cat([grid_sizes, grid_sizes_videos], dim=0)
        merge_sizes = torch.cat([merge_sizes, merge_sizes_videos], dim=0)
        mm_features = self.vision_model(
            pixel_values=pixel_values,
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
            return_dict=True,
        ).last_hidden_state
        mm_features = self.projector(mm_features)

        image_embeds, video_embeds = mm_features.split([num_image_features, num_video_features], dim=0)

        return image_embeds, video_embeds

    def _get_compression_mask(
        self,
        pixel_values_videos: torch.FloatTensor,
        grid_sizes_videos: torch.LongTensor,
        merge_sizes_videos: torch.LongTensor,
        threshold: float = 0.1,
        min_tokens: int = 1,
    ) -> torch.BoolTensor:
        videos = pixel_values_videos.split(grid_sizes_videos.prod(dim=1).tolist(), dim=0)
        compression_masks = []

        for images, grid_size, merge_size in zip(videos, grid_sizes_videos, merge_sizes_videos):
            t, h, w = grid_size
            if t == 1:
                num_tokens = images.size(0) // (merge_size ** 2)
                compression_masks.append(torch.ones((num_tokens,), dtype=torch.bool, device=images.device))
            else:
                # NOTE: video token compressor
                images = images.view(t, (h // merge_size) * (w // merge_size), -1)

                pixel_diff = images[1:] - images[:-1]
                pixel_diff = torch.abs(pixel_diff).mean(dim=-1) * 255
                pixel_diff = torch.cat([torch.full_like(pixel_diff[0:1], threshold + 1), pixel_diff], dim=0)
                mask = pixel_diff > threshold
                padding_ids = torch.nonzero(mask.sum(dim=1) < min_tokens)[:, 0]
                mask[padding_ids, :min_tokens] = 1
                compression_masks.append(mask.flatten())

        compression_mask = torch.cat(compression_masks)
        return compression_mask

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        grid_sizes_videos: Optional[torch.LongTensor] = None,
        merge_sizes_videos: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Videollama3ModelOutputWithPast]:
        """
        Args:
            grid_sizes (`torch.Tensor` of shape `(num_images, 3)`):
                The temporal, height and width of feature shape of each image before vision encoder.
            merge_sizes (`torch.Tensor` of shape `(num_images,)`):
                The spatial downsampling ratio of each image feature.
            grid_sizes_videos (`torch.Tensor` of shape `(num_videos, 3)`):
                The temporal, height and width of feature shape of each video before vision encoder.
            merge_sizes_videos (`torch.Tensor` of shape `(num_videos,)`):
                The spatial downsampling ratio of each video feature.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_token_mask, video_token_mask = self.get_placeholder_mask(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )
        image_embeds, video_embeds = self.get_multimodal_embeddings(
            pixel_values=pixel_values,
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
            pixel_values_videos=pixel_values_videos,
            grid_sizes_videos=grid_sizes_videos,
            merge_sizes_videos=merge_sizes_videos,
        )

        if pixel_values is not None:
            if image_embeds.numel() != image_token_mask.sum():
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {image_token_mask.sum()}, features {image_embeds.shape}"
                )
            inputs_embeds = inputs_embeds.masked_scatter(image_token_mask, image_embeds)

        compression_mask = None
        if pixel_values_videos is not None:
            if video_embeds.numel() != video_token_mask.sum():
                raise ValueError(
                    f"Videos features and video tokens do not match: tokens: {video_token_mask.sum()}, features {video_embeds.shape}"
                )
            inputs_embeds = inputs_embeds.masked_scatter(video_token_mask, video_embeds)

            if self.config.use_token_compression and inputs_embeds.size(0) == 1:
                video_token_mask = video_token_mask[..., 0]
                video_compression_mask = self._get_compression_mask(
                    pixel_values_videos=pixel_values_videos,
                    grid_sizes_videos=grid_sizes_videos,
                    merge_sizes_videos=merge_sizes_videos,
                )
                compression_mask = torch.logical_not(video_token_mask)
                compression_mask[video_token_mask] = video_compression_mask
                inputs_embeds = inputs_embeds[compression_mask].unsqueeze(0)

                if attention_mask is not None:
                    attention_mask = attention_mask[compression_mask].unsqueeze(0)
                if position_ids is not None:
                    position_ids = position_ids[compression_mask].unsqueeze(0)
                    print(position_ids.shape)
                if cache_position is not None:
                    cache_position = cache_position[compression_mask[0]]
                    print(cache_position.shape)

            elif inputs_embeds.size(0) != 1:
                logger.info("Token compression is automatically disabled since the input batch size is not equal to 1.")

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = Videollama3ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            compression_mask=compression_mask,
        )
        return output if return_dict else output.to_tuple()


class Videollama3ForConditionalGeneration(Videollama3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Videollama3Config):
        super().__init__(config)
        self.model = Videollama3Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    # Make modules available throught conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_model(self):
        return self.model.vision_model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        grid_sizes_videos: Optional[torch.LongTensor] = None,
        merge_sizes_videos: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Videollama3CausalLMOutputWithPast]:
        """
        Args:
            grid_sizes (`torch.Tensor` of shape `(num_images, 3)`):
                The temporal, height and width of feature shape of each image before vision encoder.
            merge_sizes (`torch.Tensor` of shape `(num_images,)`):
                The spatial downsampling ratio of each image feature.
            grid_sizes_videos (`torch.Tensor` of shape `(num_videos, 3)`):
                The temporal, height and width of feature shape of each video before vision encoder.
            merge_sizes_videos (`torch.Tensor` of shape `(num_videos,)`):
                The spatial downsampling ratio of each video feature.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
            pixel_values_videos=pixel_values_videos,
            grid_sizes_videos=grid_sizes_videos,
            merge_sizes_videos=merge_sizes_videos,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if outputs.compression_mask is not None:
                labels = labels[outputs.compression_mask].unsqueeze(0)
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return Videollama3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            compression_mask=outputs.compression_mask,
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
        pixel_values: Optional[torch.Tensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        grid_sizes_videos: Optional[torch.LongTensor] = None,
        merge_sizes_videos: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
            pixel_values_videos=pixel_values_videos,
            grid_sizes_videos=grid_sizes_videos,
            merge_sizes_videos=merge_sizes_videos,
            use_cache=use_cache,
            **kwargs,
        )

        if model_inputs["cache_position"][0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs


__all__ = ["Videollama3ForConditionalGeneration", "Videollama3Model", "Videollama3PreTrainedModel", "Videollama3VisionModel"]
