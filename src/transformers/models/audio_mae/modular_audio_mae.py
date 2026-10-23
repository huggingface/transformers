from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..vit_mae.configuration_vit_mae import ViTMAEConfig


@auto_docstring(checkpoint="justinluong/audiomae-base-patch16-audioset-2m")
@strict
class AudioMAEConfig(ViTMAEConfig):
    r"""
    max_length (`int`, *optional*, defaults to 1024):
        Temporal dimension (number of frames) of the input spectrograms. Must be a multiple of `patch_size`.
    num_mel_bins (`int`, *optional*, defaults to 128):
        Number of mel-frequency bins of the input spectrograms. Must be a multiple of `patch_size`.
    use_mean_pooling (`bool`, *optional*, defaults to `False`):
        Whether to mean-pool the patch tokens (followed by a layer norm) rather than using the final hidden state of
        the `[CLS]` token to compute the pooled output. The self-supervised AudioSet-2M checkpoint uses the `[CLS]`
        token, while the checkpoints fine-tuned for classification use mean pooling.
    decoder_num_hidden_layers (`int`, *optional*, defaults to 16):
        Number of hidden layers in the decoder.
    decoder_window_size (`int`, *optional*, defaults to 4):
        Size (in patches, along both time and frequency) of the local attention windows used by the decoder. Every
        other decoder layer shifts the windows by half a window along the time axis.
    decoder_position_bias_hidden_size (`int`, *optional*, defaults to 384):
        Hidden size of the small MLP that produces the continuous relative position bias of the decoder attention.
    mask_ratio (`float`, *optional*, defaults to 0.8):
        The ratio of the number of masked tokens in the input sequence. Set it to `0.0` to disable masking, e.g. to
        extract embeddings or fine-tune for classification.
    norm_pix_loss (`bool`, *optional*, defaults to `True`):
        Whether or not to train with per-patch normalized spectrogram values as reconstruction targets.

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
        PreTrainedConfig.__post_init__(self, **kwargs)


__all__ = ["AudioMAEConfig"]
