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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..qwen2.configuration_qwen2 import Qwen2Config


@auto_docstring(checkpoint="thisisiron/Ovis2-1B-hf")
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

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 2816,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 8,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 14,
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        qkv_bias: bool = False,
        mlp_bias: bool = False,
        hidden_act="silu",
        vocab_size=16384,
        hidden_stride=1,
        num_visual_indicator_tokens=5,
        initializer_range=0.02,
        tokenize_function="softmax",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size

        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.qkv_bias = qkv_bias
        self.mlp_bias = mlp_bias
        self.rms_norm_eps = rms_norm_eps
        self.vocab_size = vocab_size
        self.hidden_stride = hidden_stride
        self.num_visual_indicator_tokens = num_visual_indicator_tokens
        self.tokenize_function = tokenize_function
        self.initializer_range = initializer_range


@auto_docstring(checkpoint="thisisiron/Ovis2-1B-hf")
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

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=151665,
        visual_indicator_token_ids=[151666, 151667, 151668, 151669, 151670],
        vocab_size=151643,
        hidden_size=1536,
        tie_word_embeddings=True,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = Ovis2VisionConfig(**vision_config)
        elif isinstance(vision_config, Ovis2VisionConfig):
            self.vision_config = vision_config
        if vision_config is None:
            self.vision_config = Ovis2VisionConfig(num_visual_indicator_tokens=len(visual_indicator_token_ids))

        if isinstance(text_config, dict):
            self.text_config = Qwen2Config(**text_config)
        elif isinstance(text_config, Qwen2Config):
            self.text_config = text_config
        elif text_config is None:
            self.text_config = Qwen2Config()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.image_token_id = image_token_id
        self.visual_indicator_token_ids = visual_indicator_token_ids
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


__all__ = ["Ovis2VisionConfig", "Ovis2Config"]
