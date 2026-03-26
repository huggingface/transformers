# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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
"""Mllama model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="meta-llama/Llama-3.2-11B-Vision")
@strict
class MllamaVisionConfig(PreTrainedConfig):
    r"""
    num_global_layers (`int`, *optional*, defaults to 8):
        Number of global layers in the Transformer encoder. Vision model has a second transformer encoder, called global.
    vision_output_dim (`int`, *optional*, defaults to 7680):
        Dimensionality of the vision model output. Includes output of transformer
        encoder with intermediate layers and global transformer encoder.
    max_num_tiles (`int`, *optional*, defaults to 4):
        Maximum number of tiles for image splitting.
    intermediate_layers_indices (`list[int]`, *optional*, defaults to [3, 7, 15, 23, 30]):
        Indices of intermediate layers of transformer encoder from which to extract and output features.
        These output features are concatenated with final hidden state of transformer encoder.
    supported_aspect_ratios (`list[list[int]]`, *optional*):
        List of supported aspect ratios for image splitting. If not specified, the default supported aspect ratios
        are [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]] for `max_num_tiles=4`.

    Example:

    ```python
    >>> from transformers import MllamaVisionConfig, MllamaVisionModel

    >>> # Initializing a Llama config
    >>> config = MllamaVisionConfig()

    >>> # Initializing a vision model from the mllama-11b style configuration
    >>> model = MllamaVisionModel(config)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mllama_vision_model"
    base_config_key = "vision_config"
    attribute_map = {"num_attention_heads": "attention_heads"}

    hidden_size: int = 1280
    hidden_act: str = "gelu"
    num_hidden_layers: int = 32
    num_global_layers: int = 8
    attention_heads: int = 16
    num_channels: int = 3
    intermediate_size: int = 5120
    vision_output_dim: int = 7680
    image_size: int | list[int] | tuple[int, int] = 448
    patch_size: int | list[int] | tuple[int, int] = 14
    norm_eps: float = 1e-5
    max_num_tiles: int = 4
    intermediate_layers_indices: list[int] | None = None
    supported_aspect_ratios: list[list[int]] | None = None
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.supported_aspect_ratios is None:
            self.supported_aspect_ratios = [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]]

        if self.intermediate_layers_indices is None:
            self.intermediate_layers_indices = [3, 7, 15, 23, 30]
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if (
            self.supported_aspect_ratios == [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]]
            and self.max_num_tiles != 4
        ):
            raise ValueError("max_num_tiles must be 4 for default supported aspect ratios")

    @property
    def max_aspect_ratio_id(self) -> int:
        return len(self.supported_aspect_ratios)


@auto_docstring(checkpoint="meta-llama/Llama-3.2-11B-Vision")
@strict
class MllamaTextConfig(PreTrainedConfig):
    r"""
    cross_attention_layers (`list[int]`, *optional*):
        Indices of the cross attention layers. If not specified, will default to [3, 8, 13, 18, 23, 28, 33, 38].

    Example:

    ```python
    >>> from transformers import MllamaTextModel, MllamaTextConfig

    >>> # Initializing a Mllama text config
    >>> config = MllamaTextConfig()

    >>> # Initializing a model from the Mllama text configuration
    >>> model = MllamaTextModel(config)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mllama_text_model"
    base_config_key = "text_config"
    default_theta = 500000.0

    vocab_size: int = 128256
    hidden_size: int = 4096
    hidden_act: str = "silu"
    num_hidden_layers: int = 40
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 14_336
    rope_parameters: dict | None = None
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 131_072
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = False
    cross_attention_layers: list[int] | None = None
    dropout: float | int = 0.0
    bos_token_id: int = 128000
    eos_token_id: int | list[int] | None = 128001
    pad_token_id: int | None = 128004

    def __post_init__(self, **kwargs):
        if self.cross_attention_layers is None:
            self.cross_attention_layers = [3, 8, 13, 18, 23, 28, 33, 38]
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="meta-llama/Llama-3.2-11B-Vision")
@strict
class MllamaConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import MllamaForConditionalGeneration, MllamaConfig, MllamaVisionConfig, MllamaTextConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = MllamaVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = MllamaTextConfig()

    >>> # Initializing a mllama-11b style configuration
    >>> configuration = MllamaConfig(vision_config, text_config)

    >>> # Initializing a model from the mllama-11b style configuration
    >>> model = MllamaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mllama"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {"text_config": MllamaTextConfig, "vision_config": MllamaVisionConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_index: int = 128256

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = MllamaVisionConfig()
            logger.info("vision_config is None, using default mllama vision config")
        elif isinstance(self.vision_config, dict):
            self.vision_config = MllamaVisionConfig(**self.vision_config)

        if self.text_config is None:
            self.text_config = MllamaTextConfig()
            logger.info("text_config is None, using default mllama text config")
        elif isinstance(self.text_config, dict):
            self.text_config = MllamaTextConfig(**self.text_config)

        super().__post_init__(**kwargs)


__all__ = ["MllamaConfig"]
