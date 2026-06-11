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
"""BLIP-2 model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="Salesforce/blip2-opt-2.7b")
@strict
class Blip2VisionConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import Blip2VisionConfig, Blip2VisionModel

    >>> # Initializing a Blip2VisionConfig with Salesforce/blip2-opt-2.7b style configuration
    >>> configuration = Blip2VisionConfig()

    >>> # Initializing a Blip2VisionModel (with random weights) from the Salesforce/blip2-opt-2.7b style configuration
    >>> model = Blip2VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "blip_2_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 1408
    intermediate_size: int = 6144
    num_hidden_layers: int = 39
    num_attention_heads: int = 16
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 14
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0
    initializer_range: float = 1e-10
    qkv_bias: bool = True


@auto_docstring(checkpoint="Salesforce/blip2-opt-2.7b")
@strict
class Blip2QFormerConfig(PreTrainedConfig):
    r"""
    cross_attention_frequency (`int`, *optional*, defaults to 2):
        The frequency of adding cross-attention to the Transformer layers.
    use_qformer_text_input (`bool`, *optional*, defaults to `False`):
        Whether to use BERT-style embeddings.

    Examples:

    ```python
    >>> from transformers import Blip2QFormerConfig, Blip2QFormerModel

    >>> # Initializing a BLIP-2 Salesforce/blip2-opt-2.7b style configuration
    >>> configuration = Blip2QFormerConfig()

    >>> # Initializing a model (with random weights) from the Salesforce/blip2-opt-2.7b style configuration
    >>> model = Blip2QFormerModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "blip_2_qformer"
    base_config_key = "qformer_config"

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    cross_attention_frequency: int = 2
    encoder_hidden_size: int = 1408
    use_qformer_text_input: bool = False


@auto_docstring(checkpoint="Salesforce/blip2-opt-2.7b")
@strict
class Blip2Config(PreTrainedConfig):
    r"""
    qformer_config (`dict`, *optional*):
        Dictionary of configuration options used to initialize [`Blip2QFormerConfig`].
    num_query_tokens (`int`, *optional*, defaults to 32):
        The number of query tokens passed through the Transformer.
    image_text_hidden_size (`int`, *optional*, defaults to 256):
        Dimensionality of the hidden state of the image-text fusion layer.

    Example:

    ```python
    >>> from transformers import (
    ...     Blip2VisionConfig,
    ...     Blip2QFormerConfig,
    ...     OPTConfig,
    ...     Blip2Config,
    ...     Blip2ForConditionalGeneration,
    ... )

    >>> # Initializing a Blip2Config with Salesforce/blip2-opt-2.7b style configuration
    >>> configuration = Blip2Config()

    >>> # Initializing a Blip2ForConditionalGeneration (with random weights) from the Salesforce/blip2-opt-2.7b style configuration
    >>> model = Blip2ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a Blip2Config from a Blip2VisionConfig, Blip2QFormerConfig and any PreTrainedConfig

    >>> # Initializing BLIP-2 vision, BLIP-2 Q-Former and language model configurations
    >>> vision_config = Blip2VisionConfig()
    >>> qformer_config = Blip2QFormerConfig()
    >>> text_config = OPTConfig()

    >>> config = Blip2Config(vision_config=vision_config, qformer_config=qformer_config, text_config=text_config)
    ```"""

    model_type = "blip-2"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "qformer_config": Blip2QFormerConfig, "vision_config": Blip2VisionConfig}

    vision_config: dict | PreTrainedConfig | None = None
    qformer_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    num_query_tokens: int = 32
    image_text_hidden_size: int = 256
    image_token_index: int | None = None
    initializer_factor: float = 1.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = CONFIG_MAPPING["opt"]()
            logger.info("text_config is None. Initializing the text config with default values (`OPTConfig`).")
        elif isinstance(self.text_config, dict):
            text_model_type = self.text_config.get("model_type", "opt")
            self.text_config = CONFIG_MAPPING[text_model_type](**self.text_config)

        if self.qformer_config is None:
            self.qformer_config = Blip2QFormerConfig()
            logger.info("qformer_config is None. Initializing the Blip2QFormerConfig with default values.")
        elif isinstance(self.qformer_config, dict):
            self.qformer_config = Blip2QFormerConfig(**self.qformer_config)

        if self.vision_config is None:
            self.vision_config = Blip2VisionConfig()
            logger.info("`vision_config` is `None`. initializing the `Blip2VisionConfig` with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = Blip2VisionConfig(**self.vision_config)

        self.qformer_config.encoder_hidden_size = self.vision_config.hidden_size
        self.use_decoder_only_language_model = self.text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        kwargs["is_encoder_decoder"] = self.text_config.is_encoder_decoder
        super().__post_init__(**kwargs)


__all__ = ["Blip2Config", "Blip2QFormerConfig", "Blip2VisionConfig"]
