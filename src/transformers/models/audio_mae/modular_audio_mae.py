from copy import deepcopy
from dataclasses import dataclass

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...utils import ModelOutput, auto_docstring, logging
from ..vit_mae.configuration_vit_mae import ViTMAEConfig
from ..vit_mae.modeling_vit_mae import (
    ViTMAEAttention,
    ViTMAEDecoderOutput,
    ViTMAEEmbeddings,
    ViTMAEForPreTrainingOutput,
    ViTMAELayer,
    ViTMAEMLP,
    build_2d_sinusoidal_position_embedding,
)

logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="justinluong/audiomae-base-patch16-audioset-2m")
@strict
class AudioMAEConfig(ViTMAEConfig):
    r"""
    decoder_num_hidden_layers (`int`, *optional*, defaults to 16):
        Number of hidden layers in the decoder.
    mask_ratio (`float`, *optional*, defaults to 0.8):
        The ratio of the number of masked tokens in the input sequence. Set it to `0.0` to disable masking, e.g. to
        extract embeddings or fine-tune for classification.
    norm_pix_loss (`bool`, *optional*, defaults to `True`):
        Whether or not to train with per-patch normalized spectrogram values as reconstruction targets.
    max_length (`int`, *optional*, defaults to 1024):
        Temporal dimension (number of frames) of the input spectrograms. Must be a multiple of `patch_size`.
    num_mel_bins (`int`, *optional*, defaults to 128):
        Number of mel-frequency bins of the input spectrograms. Must be a multiple of `patch_size`.
    use_mean_pooling (`bool`, *optional*, defaults to `False`):
        Whether to mean-pool the patch tokens (followed by a layer norm) rather than using the final hidden state of
        the `[CLS]` token to compute the pooled output. The self-supervised AudioSet-2M checkpoint uses the `[CLS]`
        token, while the checkpoints fine-tuned for classification use mean pooling.
    decoder_window_size (`int`, *optional*, defaults to 4):
        Size (in patches, along both time and frequency) of the local attention windows used by the decoder. Every
        other decoder layer shifts the windows by half a window along the time axis.
    decoder_position_bias_hidden_size (`int`, *optional*, defaults to 384):
        Hidden size of the small MLP that produces the continuous relative position bias of the decoder attention.

    Example:

    ```python
    >>> from transformers import AudioMAEConfig, AudioMAEModel

    >>> # Initializing an Audio-MAE justinluong/audiomae-base-patch16-audioset-2m style configuration
    >>> configuration = AudioMAEConfig()

    >>> # Initializing a model (with random weights) from the justinluong/audiomae-base-patch16-audioset-2m style configuration
    >>> model = AudioMAEModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "audio_mae"

    layer_norm_eps: float = 1e-6
    image_size = AttributeError()
    num_channels = AttributeError()
    max_length: int = 1024
    num_mel_bins: int = 128
    use_mean_pooling: bool = False
    decoder_num_hidden_layers: int = 16
    decoder_window_size: int = 4
    decoder_position_bias_hidden_size: int = 384
    mask_ratio: float = 0.8
    norm_pix_loss: bool = True

    def __post_init__(self, **kwargs):
        if self.max_length % self.patch_size != 0 or self.num_mel_bins % self.patch_size != 0:
            raise ValueError(
                f"`max_length` ({self.max_length}) and `num_mel_bins` ({self.num_mel_bins}) must both be multiples of "
                f"`patch_size` ({self.patch_size})."
            )
        num_time_patches, num_frequency_patches = (
            self.max_length // self.patch_size,
            self.num_mel_bins // self.patch_size,
        )
        if num_time_patches % self.decoder_window_size != 0 or num_frequency_patches % self.decoder_window_size != 0:
            raise ValueError(
                f"The patch grid ({num_time_patches}x{num_frequency_patches}) must be divisible by "
                f"`decoder_window_size` ({self.decoder_window_size})."
            )
        PreTrainedConfig.__post_init__(self, **kwargs)


@auto_docstring(
    custom_intro="""
    Class for AudioMAEModel's outputs, with potential hidden states and attentions.
    """
)
@dataclass
class AudioMAEModelOutput(ModelOutput):
    r"""
    pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
        Final hidden state of the `[CLS]` token, or the layer-normed mean of the patch tokens when
        `config.use_mean_pooling` is `True`.
    mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
        Tensor indicating which patches are masked (1) and which are not (0).
    ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
        Tensor containing the original index of the (shuffled) masked patches.
    """

    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    mask: torch.FloatTensor | None = None
    ids_restore: torch.LongTensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


@auto_docstring(
    custom_intro="""
    Class for AudioMAEDecoder's outputs, with potential hidden states and attentions.
    """
)
@dataclass
class AudioMAEDecoderOutput(ViTMAEDecoderOutput):
    r"""
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2)`):
        Spectrogram patch reconstruction logits.
    """


@auto_docstring(
    custom_intro="""
    Class for AudioMAEForPreTraining's outputs, with potential hidden states and attentions.
    """
)
@dataclass
class AudioMAEForPreTrainingOutput(ViTMAEForPreTrainingOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`):
        Spectrogram patch reconstruction loss.
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2)`):
        Spectrogram patch reconstruction logits.
    mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
        Tensor indicating which patches are masked (1) and which are not (0).
    ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
        Tensor containing the original index of the (shuffled) masked patches.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    mask: torch.FloatTensor | None = None
    ids_restore: torch.LongTensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


class AudioMAEPatchEmbeddings(nn.Module):
    """
    This class turns `input_values` (mel spectrograms) of shape `(batch_size, max_length, num_mel_bins)` into the
    initial `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer. Patches are ordered time-major (all frequency patches of the first time step come first), matching
    the original implementation.
    """

    def __init__(self, config: AudioMAEConfig):
        super().__init__()
        self.max_length = config.max_length
        self.num_mel_bins = config.num_mel_bins
        self.patch_size = config.patch_size
        # (time patches, frequency patches)
        self.grid_size = (config.max_length // config.patch_size, config.num_mel_bins // config.patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.projection = nn.Conv2d(1, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        _, max_length, num_mel_bins = input_values.shape
        if max_length != self.max_length or num_mel_bins != self.num_mel_bins:
            raise ValueError(
                f"Input spectrogram size ({max_length}*{num_mel_bins}) doesn't match model"
                f" ({self.max_length}*{self.num_mel_bins})."
            )
        # (batch_size, time, frequency) -> (batch_size, 1, time, frequency)
        input_values = input_values.unsqueeze(1)
        embeddings = self.projection(input_values).flatten(2).transpose(1, 2)
        return embeddings


def build_audio_mae_2d_sinusoidal_position_embedding(grid_size: tuple[int, int], embed_dim: int) -> torch.Tensor:
    """2D sinusoidal position embeddings for a `(time, frequency)` patch grid, laid out as in the original Audio-MAE.

    The original implementation differs from the library helper in two ways. It builds the table for the transposed
    grid, and it keeps ViT-MAE's `[sin_w|cos_w|sin_h|cos_h]` layout. The transposed grid is a bug in the original
    implementation (its `patch_hw` is `(8, 64)` while the patches are flattened time-major) that the released weights
    were trained with. The pretrained weights expect that table, so this function swaps the grid and rotates the two
    halves to reproduce it.

    Returns:
        Tensor of shape ``(1, num_patches + 1, embed_dim)``, with a zero row for the `[CLS]` token.
    """
    num_time_patches, num_frequency_patches = grid_size
    pos_embed = build_2d_sinusoidal_position_embedding(
        height=num_frequency_patches, width=num_time_patches, embed_dim=embed_dim, cls_token=True
    )
    half = embed_dim // 2
    pos_embed = torch.cat([pos_embed[..., half:], pos_embed[..., :half]], dim=-1)
    return pos_embed.unsqueeze(0)


class AudioMAEEmbeddings(ViTMAEEmbeddings):
    """
    Construct the CLS token, position and patch embeddings for audio spectrograms, and randomly mask patches.
    """

    def __init__(self, config: AudioMAEConfig):
        nn.Module.__init__(self)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = AudioMAEPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.hidden_size), requires_grad=False
        )
        self.config = config

    def interpolate_pos_encoding(self, embeddings, height, width):
        raise AttributeError(
            "Audio-MAE operates on a fixed spectrogram size, position embeddings are not interpolated"
        )

    def initialize_weights(self):
        pos_embed = build_audio_mae_2d_sinusoidal_position_embedding(
            self.patch_embeddings.grid_size, self.position_embeddings.shape[-1]
        )
        init.copy_(self.position_embeddings, pos_embed)

        if getattr(self.patch_embeddings.projection, "_is_hf_initialized", False):
            return
        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embeddings.projection.weight
        init.xavier_uniform_(w.view([w.shape[0], -1]))
        init.normal_(self.cls_token, std=self.config.initializer_range)

    def forward(
        self, input_values: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = self.patch_embeddings(input_values)

        embeddings = embeddings + self.position_embeddings[:, 1:, :]

        # CODEPATH: mask_ratio=0.8 for the pre-training checkpoint, 0.0 for the fine-tuned classification checkpoints.
        if self.config.mask_ratio > 0:
            embeddings, mask, ids_restore = self.random_masking(embeddings, noise)
        else:
            batch_size, seq_length = embeddings.shape[:2]
            mask = torch.zeros(batch_size, seq_length, device=embeddings.device)
            ids_restore = torch.arange(seq_length, device=embeddings.device).expand(batch_size, -1)

        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore


class AudioMAEAttention(ViTMAEAttention):
    pass


class AudioMAEMLP(ViTMAEMLP):
    pass


class AudioMAELayer(ViTMAELayer):
    pass


def window_partition(hidden_states: torch.Tensor, window_size: int) -> torch.Tensor:
    """Splits a `(batch_size, height, width, channels)` grid into `(batch_size * num_windows, window_size**2, channels)`."""
    batch_size, height, width, num_channels = hidden_states.shape
    hidden_states = hidden_states.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    windows = hidden_states.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, num_channels)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int) -> torch.Tensor:
    """Inverse of `window_partition`."""
    num_channels = windows.shape[-1]
    batch_size = windows.shape[0] // ((height // window_size) * (width // window_size))
    hidden_states = windows.view(
        batch_size, height // window_size, width // window_size, window_size, window_size, num_channels
    )
    hidden_states = hidden_states.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, height, width, num_channels)
    return hidden_states


class AudioMAEDecoderPositionBiasMLP(nn.Module):
    """Produces one relative position bias per attention head from log-spaced relative coordinates."""

    def __init__(self, config: AudioMAEConfig):
        super().__init__()
        self.fc1 = nn.Linear(2, config.decoder_position_bias_hidden_size, bias=True)
        self.activation_fn = nn.ReLU()
        self.fc2 = nn.Linear(config.decoder_position_bias_hidden_size, config.decoder_num_attention_heads, bias=True)

    def forward(self, relative_coordinates_log: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(relative_coordinates_log)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class AudioMAEDecoderAttention(nn.Module):
    """Window attention with scaled cosine similarity and a continuous relative position bias (Swin Transformer V2).

    Adapted from:
    https://github.com/facebookresearch/AudioMAE/blob/bd60e29651285f80d32a6405082835ad26e6f19f/timm_patch/swin_transformer.py#L75-L212
    """

    def __init__(self, config: AudioMAEConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.decoder_num_attention_heads
        self.head_dim = config.decoder_hidden_size // config.decoder_num_attention_heads
        self.window_size = config.decoder_window_size

        self.q_proj = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size, bias=True)
        self.k_proj = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size, bias=True)
        self.v_proj = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size, bias=True)
        self.o_proj = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size, bias=True)
        self.temperature = nn.Parameter(torch.ones(config.decoder_num_attention_heads))
        self.continuous_position_bias_mlp = AudioMAEDecoderPositionBiasMLP(config)

        self.relative_coordinates_log = nn.Buffer(self.create_relative_coordinates_log(), persistent=False)

    def create_relative_coordinates_log(self) -> torch.Tensor:
        """Log-spaced relative `(time, frequency)` offsets between every pair of positions in a window."""
        coordinates = torch.stack(
            torch.meshgrid(torch.arange(self.window_size), torch.arange(self.window_size), indexing="ij")
        ).flatten(1)
        relative_coordinates = (coordinates[:, :, None] - coordinates[:, None, :]).permute(1, 2, 0).reshape(-1, 2)
        relative_coordinates = relative_coordinates.float()
        return torch.sign(relative_coordinates) * torch.log1p(relative_coordinates.abs())

    def relative_position_bias(self) -> torch.Tensor:
        window_area = self.window_size * self.window_size
        relative_coordinates_log = self.relative_coordinates_log.to(self.continuous_position_bias_mlp.fc1.weight.dtype)
        bias = self.continuous_position_bias_mlp(relative_coordinates_log)
        bias = bias.transpose(1, 0).reshape(self.num_attention_heads, window_area, window_area)
        return bias.unsqueeze(0)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_windows, window_area, _ = hidden_states.shape
        hidden_shape = (batch_windows, window_area, self.num_attention_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
        denominator = torch.matmul(
            query_states.norm(dim=-1, keepdim=True), key_states.norm(dim=-1, keepdim=True).transpose(-2, -1)
        )
        attn_weights = attn_weights / denominator.clamp(min=1e-6)
        attn_weights = attn_weights / self.temperature.clamp(min=0.01).reshape(1, self.num_attention_heads, 1, 1)
        attn_weights = attn_weights + self.relative_position_bias()

        if attention_mask is not None:
            num_windows = attention_mask.shape[0]
            attn_weights = attn_weights.view(
                batch_windows // num_windows, num_windows, self.num_attention_heads, window_area, window_area
            )
            attn_weights = attn_weights + attention_mask.unsqueeze(1).unsqueeze(0).to(attn_weights.dtype)
            attn_weights = attn_weights.view(batch_windows, self.num_attention_heads, window_area, window_area)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.config.attention_probs_dropout_prob, training=self.training
        )

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(batch_windows, window_area, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output


class AudioMAEDecoderLayer(GradientCheckpointingLayer):
    """Post-norm Swin Transformer V2 layer with shifted local attention windows over the `(time, frequency)` grid.

    Adapted from:
    https://github.com/facebookresearch/AudioMAE/blob/bd60e29651285f80d32a6405082835ad26e6f19f/timm_patch/swin_transformer.py#L214-L374
    """

    def __init__(self, config: AudioMAEConfig, grid_size: tuple[int, int], shift: bool):
        super().__init__()
        self.grid_size = grid_size
        self.window_size = config.decoder_window_size
        # the original shifts the windows along the time axis only
        self.shift_size = (config.decoder_window_size // 2, 0) if shift else (0, 0)

        self.attention = AudioMAEDecoderAttention(config)
        self.layernorm_after_attention = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.mlp = AudioMAEMLP(config)
        self.layernorm_after_mlp = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)

        attention_mask = self.build_attention_mask()
        self.attention_mask = nn.Buffer(attention_mask, persistent=False) if attention_mask is not None else None

    def build_attention_mask(self) -> torch.Tensor | None:
        """Mask preventing tokens that only became neighbours through the cyclic shift from attending to each other."""
        if not any(self.shift_size):
            return None
        height, width = self.grid_size
        window_size = self.window_size
        region_mask = torch.zeros((1, height, width, 1))
        region_index = 0
        for height_slice in (
            slice(0, -window_size),
            slice(-window_size, -self.shift_size[0]),
            slice(-self.shift_size[0], None),
        ):
            for width_slice in (
                slice(0, -window_size),
                slice(-window_size, -self.shift_size[1]),
                slice(-self.shift_size[1], None),
            ):
                region_mask[:, height_slice, width_slice, :] = region_index
                region_index += 1
        mask_windows = window_partition(region_mask, window_size).squeeze(-1)
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(attention_mask != 0, -100.0).masked_fill(attention_mask == 0, 0.0)
        return attention_mask

    def shifted_window_attention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        height, width = self.grid_size
        batch_size, seq_length, num_channels = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, height, width, num_channels)

        shift_height, shift_width = self.shift_size
        if any(self.shift_size):
            hidden_states = torch.roll(hidden_states, shifts=(-shift_height, -shift_width), dims=(1, 2))

        windows = window_partition(hidden_states, self.window_size)
        windows = self.attention(windows, attention_mask=self.attention_mask)
        hidden_states = window_reverse(windows, self.window_size, height, width)

        if any(self.shift_size):
            hidden_states = torch.roll(hidden_states, shifts=(shift_height, shift_width), dims=(1, 2))

        return hidden_states.reshape(batch_size, seq_length, num_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.layernorm_after_attention(self.shifted_window_attention(hidden_states))
        hidden_states = hidden_states + self.layernorm_after_mlp(self.mlp(hidden_states))
        return hidden_states


class AudioMAEDecoder(nn.Module):
    def __init__(self, config: AudioMAEConfig, grid_size: tuple[int, int]):
        super().__init__()
        num_patches = grid_size[0] * grid_size[1]
        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size),
            requires_grad=False,
        )

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [
                AudioMAEDecoderLayer(decoder_config, grid_size, shift=layer_idx % 2 == 1)
                for layer_idx in range(config.decoder_num_hidden_layers)
            ]
        )

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.decoder_pred = nn.Linear(config.decoder_hidden_size, config.patch_size**2, bias=True)
        self.gradient_checkpointing = False
        self.grid_size = grid_size
        self.config = decoder_config

    def forward(self, hidden_states: torch.Tensor, ids_restore: torch.Tensor) -> AudioMAEDecoderOutput:
        hidden_states = self.decoder_embed(hidden_states)

        mask_tokens = self.mask_token.repeat(
            hidden_states.shape[0], ids_restore.shape[1] + 1 - hidden_states.shape[1], 1
        )
        unmasked_tokens = torch.cat([hidden_states[:, 1:, :], mask_tokens], dim=1)

        unmasked_tokens = torch.gather(
            unmasked_tokens,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2]).to(unmasked_tokens.device),
        )
        hidden_states = torch.cat([hidden_states[:, :1, :], unmasked_tokens], dim=1)

        hidden_states = hidden_states + self.decoder_pos_embed
        # the windowed decoder works on the patch grid, so the [CLS] token is dropped
        hidden_states = hidden_states[:, 1:, :]

        for layer_module in self.decoder_layers:
            hidden_states = layer_module(hidden_states)

        hidden_states = self.decoder_norm(hidden_states)

        logits = self.decoder_pred(hidden_states)

        return AudioMAEDecoderOutput(logits=logits)


__all__ = ["AudioMAEConfig"]
