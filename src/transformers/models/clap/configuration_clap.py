# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""CLAP model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="laion/clap-htsat-fused")
@strict
class ClapTextConfig(PreTrainedConfig):
    r"""
    Examples:

    ```python
    >>> from transformers import ClapTextConfig, ClapTextModel

    >>> # Initializing a CLAP text configuration
    >>> configuration = ClapTextConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = ClapTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "clap_text_model"
    base_config_key = "text_config"

    vocab_size: int = 50265
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 514
    type_vocab_size: int = 1
    initializer_factor: float = 1.0
    layer_norm_eps: float = 1e-12
    projection_dim: int = 512
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    projection_hidden_act: str = "relu"


@auto_docstring(checkpoint="laion/clap-htsat-fused")
@strict
class ClapAudioConfig(PreTrainedConfig):
    r"""
    window_size (`int`, *optional*, defaults to 8):
        Image size of the spectrogram
    spec_size (`int`, *optional*, defaults to 256):
        Desired input size of the spectrogram that the model supports. It can be different from the output of the
        `ClapFeatureExtractor`, in which case the input features will be resized. Corresponds to the `image_size`
        of the audio models.
    patch_stride (`list`, *optional*, defaults to `[4, 4]`):
        Patch stride for the audio spectrogram
    num_classes (`int`, *optional*, defaults to 527):
        Number of classes used for the head training
    enable_fusion (`bool`, *optional*, defaults to `False`):
        Whether or not to enable patch fusion. This is the main contribution of the authors, and should give the
        best results.
    fusion_type (`[type]`, *optional*):
        Fusion type used for the patch fusion.
    patch_embed_input_channels (`int`, *optional*, defaults to 1):
        Number of channels used for the input spectrogram
    flatten_patch_embeds (`bool`, *optional*, defaults to `True`):
        Whether or not to flatten the patch embeddings
    patch_embeds_hidden_size (`int`, *optional*, defaults to 96):
        Hidden size of the patch embeddings. It is used as the number of output channels.
    enable_patch_layer_norm (`bool`, *optional*, defaults to `True`):
        Whether or not to enable layer normalization for the patch embeddings
    aff_block_r (`int`, *optional*, defaults to 4):
        downsize_ratio used in the AudioFF block

    Example:

    ```python
    >>> from transformers import ClapAudioConfig, ClapAudioModel

    >>> # Initializing a ClapAudioConfig with laion/clap-htsat-fused style configuration
    >>> configuration = ClapAudioConfig()

    >>> # Initializing a ClapAudioModel (with random weights) from the laion/clap-htsat-fused style configuration
    >>> model = ClapAudioModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "clap_audio_model"
    base_config_key = "audio_config"

    window_size: int = 8
    num_mel_bins: int = 64
    spec_size: int = 256
    hidden_act: str = "gelu"
    patch_size: int | list[int] | tuple[int, int] = 4
    patch_stride: int | list[int] | tuple[int, ...] = (4, 4)
    num_classes: int = 527
    hidden_size: int = 768
    projection_dim: int = 512
    depths: list[int] | tuple[int, ...] = (2, 2, 6, 2)
    num_attention_heads: list[int] | tuple[int, ...] = (4, 8, 16, 32)
    enable_fusion: bool = False
    hidden_dropout_prob: float = 0.1
    fusion_type: str | None = None
    patch_embed_input_channels: int = 1
    flatten_patch_embeds: bool = True
    patch_embeds_hidden_size: int = 96
    enable_patch_layer_norm: bool = True
    drop_path_rate: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    qkv_bias: bool = True
    mlp_ratio: float = 4.0
    aff_block_r: int = 4
    num_hidden_layers: int = 4
    projection_hidden_act: str = "relu"
    layer_norm_eps: float = 1e-5
    initializer_factor: float = 1.0


@auto_docstring(checkpoint="laion/clap-htsat-fused")
@strict
class ClapConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import ClapConfig, ClapModel

    >>> # Initializing a ClapConfig with laion-ai/base style configuration
    >>> configuration = ClapConfig()

    >>> # Initializing a ClapModel (with random weights) from the laion-ai/base style configuration
    >>> model = ClapModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a ClapConfig from a ClapTextConfig and a ClapAudioConfig
    >>> from transformers import ClapTextConfig, ClapAudioConfig

    >>> # Initializing a ClapText and ClapAudioConfig configuration
    >>> config_text = ClapTextConfig()
    >>> config_audio = ClapAudioConfig()

    >>> config = ClapConfig(text_config=config_text, audio_config=config_audio)
    ```"""

    model_type = "clap"
    sub_configs = {"text_config": ClapTextConfig, "audio_config": ClapAudioConfig}

    text_config: dict | PreTrainedConfig | None = None
    audio_config: dict | PreTrainedConfig | None = None
    logit_scale_init_value: float = 1 / 0.07
    projection_dim: int = 512
    projection_hidden_act: str = "relu"
    initializer_factor: float = 1.0

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = ClapTextConfig()
            logger.info("`text_config` is `None`. initializing the `ClapTextConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config = ClapTextConfig(**self.text_config)

        if self.audio_config is None:
            self.audio_config = ClapAudioConfig()
            logger.info("`audio_config` is `None`. initializing the `ClapAudioConfig` with default values.")
        elif isinstance(self.audio_config, dict):
            self.audio_config = ClapAudioConfig(**self.audio_config)

        self.text_config.projection_dim = self.projection_dim
        self.audio_config.projection_dim = self.projection_dim

        self.text_config.projection_hidden_act = self.projection_hidden_act
        self.audio_config.projection_hidden_act = self.projection_hidden_act
        self.hidden_size = self.text_config.hidden_size
        self.num_hidden_layers = self.text_config.num_hidden_layers + len(self.audio_config.depths)
        super().__post_init__(**kwargs)


__all__ = ["ClapAudioConfig", "ClapConfig", "ClapTextConfig"]
