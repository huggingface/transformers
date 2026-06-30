# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..qwen2.configuration_qwen2 import Qwen2Config


@auto_docstring(checkpoint="thisisiron/Ovis2-1B-hf")
@strict
class Ovis2VisionConfig(PreTrainedConfig):
    r"""
    hidden_stride (`int`, *optional*, defaults to 1):
        The stride of the hidden layer in the Vision Transformer.
    num_visual_indicator_tokens (`int`, *optional*, defaults to 5):
        Number of visual indicator tokens.
    tokenize_function (`str`, *optional*, defaults to `"softmax"`):
        The function used to tokenize the visual indicator tokens.
    """

    base_config_key = "vision_config"

    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 14
    rms_norm_eps: float = 1e-5
    attention_dropout: float | int = 0.0
    qkv_bias: bool = False
    mlp_bias: bool = False
    hidden_act: str = "silu"
    vocab_size: int = 16384
    hidden_stride: int = 1
    num_visual_indicator_tokens: int = 5
    initializer_range: float = 0.02
    tokenize_function: str = "softmax"


@auto_docstring(checkpoint="thisisiron/Ovis2-1B-hf")
@strict
class Ovis2Config(PreTrainedConfig):
    r"""
    visual_indicator_token_ids (`List[int]`, *optional*, defaults to `[151666, 151667, 151668, 151669, 151670]`):
        The visual indicator token ids to encode the image prompt.

    ```python
    >>> from transformers import Ovis2ForConditionalGeneration, Ovis2Config

    >>> # Initializing a Ovis2 style configuration
    >>> configuration = Ovis2Config()

    >>> # Initializing a model from the Ovis2-2B style configuration
    >>> model = Ovis2ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "ovis2"
    sub_configs = {"text_config": Qwen2Config, "vision_config": Ovis2VisionConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151665
    visual_indicator_token_ids: list[int] | tuple[int, ...] = (151666, 151667, 151668, 151669, 151670)
    vocab_size: int = 151643
    hidden_size: int = 1536
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = Ovis2VisionConfig(**self.vision_config)
        if self.vision_config is None:
            self.vision_config = Ovis2VisionConfig(num_visual_indicator_tokens=len(self.visual_indicator_token_ids))

        if isinstance(self.text_config, dict):
            self.text_config = Qwen2Config(**self.text_config)
        elif self.text_config is None:
            self.text_config = Qwen2Config()

        super().__post_init__(**kwargs)


__all__ = ["Ovis2VisionConfig", "Ovis2Config"]
