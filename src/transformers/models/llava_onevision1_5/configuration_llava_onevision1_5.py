# Copyright 2025 The LLaVA-OneVision team and The HuggingFace Inc. team. All rights reserved.
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
"""LLaVA-OneVision-1.5 model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="lmms-lab/LLaVA-OneVision-1.5-4B-Instruct")
@strict
class LlavaOnevision1_5VisionConfig(PreTrainedConfig):
    r"""
    depth (`int`, *optional*, defaults to 24):
        Number of transformer blocks in the vision encoder.
    hidden_size (`int`, *optional*, defaults to 1024):
        Dimensionality of the vision encoder hidden states.
    hidden_act (`str`, *optional*, defaults to `"gelu"`):
        The non-linear activation function used in the vision encoder MLP.
    intermediate_size (`int`, *optional*, defaults to 4096):
        Dimensionality of the vision encoder MLP intermediate representations.
    num_heads (`int`, *optional*, defaults to 16):
        Number of attention heads in the vision encoder.
    in_channels (`int`, *optional*, defaults to 3):
        Number of input image channels.
    patch_size (`int`, *optional*, defaults to 14):
        The spatial patch size of the vision encoder.
    spatial_merge_size (`int`, *optional*, defaults to 2):
        The size of the spatial merge operation applied by the patch merger.
    temporal_patch_size (`int`, *optional*, defaults to 1):
        The temporal patch size of the vision encoder.
    out_hidden_size (`int`, *optional*, defaults to 2560):
        The output hidden size of the vision encoder, matching the text model hidden size.
    layer_norm_eps (`float`, *optional*, defaults to 1e-05):
        The epsilon used by the vision encoder layer normalization layers.
    """

    model_type = "llava_onevision1_5_vision"
    base_config_key = "vision_config"

    depth: int = 24
    hidden_size: int = 1024
    hidden_act: str = "gelu"
    intermediate_size: int = 4096
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 1
    out_hidden_size: int = 2560
    layer_norm_eps: float = 1e-05
    initializer_range: float = 0.02


@auto_docstring(checkpoint="lmms-lab/LLaVA-OneVision-1.5-4B-Instruct")
@strict
class LlavaOnevision1_5TextConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import LlavaOnevision1_5TextModel, LlavaOnevision1_5TextConfig

    >>> # Initializing a LLaVA-OneVision-1.5 text configuration
    >>> configuration = LlavaOnevision1_5TextConfig()

    >>> # Initializing a model from the configuration
    >>> model = LlavaOnevision1_5TextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llava_onevision1_5_text"
    base_config_key = "text_config"
    default_theta = 5000000.0

    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    pad_token_id: int | None = None
    use_sliding_window: bool = False
    sliding_window: int | None = 4096
    max_window_layers: int = 28
    layer_types: list[str] | None = None

    def __post_init__(self, **kwargs):
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="lmms-lab/LLaVA-OneVision-1.5-4B-Instruct")
@strict
class LlavaOnevision1_5Config(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import LlavaOnevision1_5ForConditionalGeneration, LlavaOnevision1_5Config

    >>> # Initializing a LLaVA-OneVision-1.5 style configuration
    >>> configuration = LlavaOnevision1_5Config()

    >>> # Initializing a model from the configuration
    >>> model = LlavaOnevision1_5ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llava_onevision1_5"
    sub_configs = {"vision_config": LlavaOnevision1_5VisionConfig, "text_config": LlavaOnevision1_5TextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151655
    video_token_id: int = 151656
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        super().__post_init__(**kwargs)


__all__ = ["LlavaOnevision1_5Config", "LlavaOnevision1_5TextConfig", "LlavaOnevision1_5VisionConfig"]
