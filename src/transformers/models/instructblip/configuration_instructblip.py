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
"""InstructBLIP model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="Salesforce/instructblip-flan-t5-xl")
class InstructBlipVisionConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import InstructBlipVisionConfig, InstructBlipVisionModel

    >>> # Initializing a InstructBlipVisionConfig with Salesforce/instructblip-flan-t5-xl style configuration
    >>> configuration = InstructBlipVisionConfig()

    >>> # Initializing a InstructBlipVisionModel (with random weights) from the Salesforce/instructblip-flan-t5-xl style configuration
    >>> model = InstructBlipVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "instructblip_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1408,
        intermediate_size=6144,
        num_hidden_layers=39,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        initializer_range=1e-10,
        qkv_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.qkv_bias = qkv_bias


@auto_docstring(checkpoint="Salesforce/instructblip-flan-t5-xl")
class InstructBlipQFormerConfig(PreTrainedConfig):
    r"""
    cross_attention_frequency (`int`, *optional*, defaults to 2):
        The frequency of adding cross-attention to the Transformer layers.
    encoder_hidden_size (`int`, *optional*, defaults to 1408):
        The hidden size of the hidden states for cross-attention.

    Examples:

    ```python
    >>> from transformers import InstructBlipQFormerConfig, InstructBlipQFormerModel

    >>> # Initializing a InstructBLIP Salesforce/instructblip-flan-t5-xl style configuration
    >>> configuration = InstructBlipQFormerConfig()

    >>> # Initializing a model (with random weights) from the Salesforce/instructblip-flan-t5-xl style configuration
    >>> model = InstructBlipQFormerModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "instructblip_qformer"
    base_config_key = "qformer_config"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        cross_attention_frequency=2,
        encoder_hidden_size=1408,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.cross_attention_frequency = cross_attention_frequency
        self.encoder_hidden_size = encoder_hidden_size


@auto_docstring(checkpoint="Salesforce/instructblip-flan-t5-xl")
class InstructBlipConfig(PreTrainedConfig):
    r"""
    qformer_config (`dict`, *optional*):
        Dictionary of configuration options used to initialize [`InstructBlipQFormerConfig`].
    num_query_tokens (`int`, *optional*, defaults to 32):
        The number of query tokens passed through the Transformer.

    Example:

    ```python
    >>> from transformers import (
    ...     InstructBlipVisionConfig,
    ...     InstructBlipQFormerConfig,
    ...     OPTConfig,
    ...     InstructBlipConfig,
    ...     InstructBlipForConditionalGeneration,
    ... )

    >>> # Initializing a InstructBlipConfig with Salesforce/instructblip-flan-t5-xl style configuration
    >>> configuration = InstructBlipConfig()

    >>> # Initializing a InstructBlipForConditionalGeneration (with random weights) from the Salesforce/instructblip-flan-t5-xl style configuration
    >>> model = InstructBlipForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a InstructBlipConfig from a InstructBlipVisionConfig, InstructBlipQFormerConfig and any PreTrainedConfig

    >>> # Initializing InstructBLIP vision, InstructBLIP Q-Former and language model configurations
    >>> vision_config = InstructBlipVisionConfig()
    >>> qformer_config = InstructBlipQFormerConfig()
    >>> text_config = OPTConfig()

    >>> config = InstructBlipConfig(vision_config=vision_config, qformer_config=qformer_config, text_config=text_config)
    ```"""

    model_type = "instructblip"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {
        "text_config": AutoConfig,
        "qformer_config": InstructBlipQFormerConfig,
        "vision_config": InstructBlipVisionConfig,
    }

    def __init__(
        self,
        vision_config=None,
        qformer_config=None,
        text_config=None,
        num_query_tokens=32,
        image_token_index=None,
        **kwargs,
    ):
        if text_config is None:
            text_config = CONFIG_MAPPING["opt"]()
            logger.info("text_config is None. Initializing the text config with default values (`OPTConfig`).")
        elif isinstance(text_config, dict):
            text_model_type = text_config.get("model_type", "opt")
            text_config = CONFIG_MAPPING[text_model_type](**text_config)

        if qformer_config is None:
            qformer_config = InstructBlipQFormerConfig()
            logger.info("qformer_config is None. Initializing the InstructBlipQFormerConfig with default values.")
        elif isinstance(qformer_config, dict):
            qformer_config = InstructBlipQFormerConfig(**qformer_config)

        if vision_config is None:
            vision_config = InstructBlipVisionConfig()
            logger.info("`vision_config` is `None`. initializing the `InstructBlipVisionConfig` with default values.")
        elif isinstance(vision_config, dict):
            vision_config = InstructBlipVisionConfig(**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config
        self.qformer_config = qformer_config

        self.num_query_tokens = num_query_tokens
        self.image_token_index = image_token_index
        self.qformer_config.encoder_hidden_size = self.vision_config.hidden_size
        self.use_decoder_only_language_model = self.text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        self.initializer_factor = 1.0
        self.initializer_range = 0.02
        super().__init__(**kwargs)


__all__ = ["InstructBlipConfig", "InstructBlipQFormerConfig", "InstructBlipVisionConfig"]
