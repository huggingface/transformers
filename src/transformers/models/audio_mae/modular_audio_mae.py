import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..vit_mae.configuration_vit_mae import ViTMAEConfig
from ..vit_mae.modeling_vit_mae import ViTMAEEmbeddings, build_2d_sinusoidal_position_embedding


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


__all__ = ["AudioMAEConfig"]
