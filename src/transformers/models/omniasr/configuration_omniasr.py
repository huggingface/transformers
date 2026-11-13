# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="bezzam/omniasr-ctc-300m-v2")
@strict
class OmniASRAudioConfig(PreTrainedConfig):
    r"""
    conv_dim (`tuple[int]` or `list[int]`, *optional*, defaults to `(512, 512, 512, 512, 512, 512, 512)`):
        A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the
        feature encoder. The length of *conv_dim* defines the number of 1D convolutional layers.
    conv_kernel (`tuple[int]` or `list[int]`, *optional*, defaults to `(10, 3, 3, 3, 3, 2, 2)`):
        A tuple of integers defining the kernel size of each 1D convolutional layer in the feature encoder. The
        length of *conv_kernel* defines the number of convolutional layers and has to match the length of
        *conv_dim*.
    conv_stride (`tuple[int]` or `list[int]`, *optional*, defaults to `(5, 2, 2, 2, 2, 2, 2)`):
        A tuple of integers defining the stride of each 1D convolutional layer in the feature encoder. The length
        of *conv_stride* defines the number of convolutional layers and has to match the length of *conv_dim*.
    conv_bias (`bool`, *optional*, defaults to `True`):
        Whether the 1D convolutional layers have a bias.
    num_conv_pos_embeddings (`int`, *optional*, defaults to 128):
        Number of convolutional positional embeddings. Defines the kernel size of the 1D convolutional positional
        embeddings layer.
    num_conv_pos_embedding_groups (`int`, *optional*, defaults to 16):
        Number of groups of the 1D convolutional positional embeddings layer.

    Example:

    ```python
    >>> from transformers import OmniASRAudioConfig, OmniASRAudioModel

    >>> # Initializing an OmniASR encoder configuration
    >>> configuration = OmniASRAudioConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = OmniASRAudioModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "omniasr_audio"
    base_config_key = "audio_config"

    hidden_size: int = 1024
    conv_dim: list[int] | tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    conv_kernel: list[int] | tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_stride: list[int] | tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    conv_bias: bool = True
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16
    intermediate_size: int = 4096
    attention_dropout: float | int = 0.0
    hidden_dropout: float | int = 0.1
    layerdrop: float | int = 0.1
    activation_dropout: float | int = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    hidden_act: str = "gelu"

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        num_conv_layers = len(self.conv_dim)
        if (len(self.conv_stride) != num_conv_layers) or (len(self.conv_kernel) != num_conv_layers):
            raise ValueError(
                "Configuration for convolutional layers is incorrect. It is required that `len(config.conv_dim)` =="
                " `len(config.conv_stride)` == `len(config.conv_kernel)`, but is `len(config.conv_dim) ="
                f" {len(self.conv_dim)}`, `len(config.conv_stride) = {len(self.conv_stride)}`,"
                f" `len(config.conv_kernel) = {len(self.conv_kernel)}`."
            )


@auto_docstring(checkpoint="bezzam/omniasr-llm-300m-v2")
@strict
class OmniASRConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import OmniASRForConditionalGeneration, OmniASRConfig

    >>> # Initializing an OmniASR-LLM configuration
    >>> configuration = OmniASRConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = OmniASRForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "omniasr"
    sub_configs_defaults = {
        "audio_config": SubConfigSpec(config_class=AutoConfig, model_type="omniasr_audio"),
        "text_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="llama",
            init_kwargs={
                "vocab_size": 11984,
                "hidden_size": 4096,
                "intermediate_size": 2816,
                "max_position_embeddings": 8192,
                "num_hidden_layers": 12,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-05,
            },
        ),
    }

    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 10289
    initializer_range: float = 0.02
    bos_token_id: int | None = 0
    pad_token_id: int | None = 1
    eos_token_id: int | None = 2

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.initializer_range != self.audio_config.initializer_range:
            raise ValueError(
                f"`initializer_range` ({self.initializer_range}) must match `audio_config.initializer_range` "
                f"({self.audio_config.initializer_range})."
            )


__all__ = ["OmniASRConfig", "OmniASRAudioConfig"]
