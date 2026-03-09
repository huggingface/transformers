# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Blip model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="Salesforce/blip-vqa-base")
class BlipTextConfig(PreTrainedConfig):
    r"""
    label_smoothing (float, *optional*):
        A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets
        become a mixture of the original ground truth and a uniform distribution as described in
        `Rethinking the Inception Architecture for Computer Vision <https://huggingface.co/papers/1512.00567>`__. Default: :math:`0.0`.

    Example:

    ```python
    >>> from transformers import BlipTextConfig, BlipTextModel

    >>> # Initializing a BlipTextConfig with Salesforce/blip-vqa-base style configuration
    >>> configuration = BlipTextConfig()

    >>> # Initializing a BlipTextModel (with random weights) from the Salesforce/blip-vqa-base style configuration
    >>> model = BlipTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "blip_text_model"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=30524,
        hidden_size=768,
        encoder_hidden_size=768,
        intermediate_size=3072,
        projection_dim=768,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=512,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        bos_token_id=30522,
        eos_token_id=2,
        pad_token_id=0,
        sep_token_id=102,
        is_decoder=True,
        use_cache=True,
        label_smoothing=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.sep_token_id = sep_token_id
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.is_decoder = is_decoder
        self.use_cache = use_cache
        self.label_smoothing = label_smoothing


@auto_docstring(checkpoint="Salesforce/blip-vqa-base")
class BlipVisionConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import BlipVisionConfig, BlipVisionModel

    >>> # Initializing a BlipVisionConfig with Salesforce/blip-vqa-base style configuration
    >>> configuration = BlipVisionConfig()

    >>> # Initializing a BlipVisionModel (with random weights) from the Salesforce/blip-vqa-base style configuration
    >>> model = BlipVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "blip_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=12,
        image_size=384,
        patch_size=16,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=1e-10,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act


@auto_docstring(checkpoint="Salesforce/blip-vqa-base")
class BlipConfig(PreTrainedConfig):
    r"""
    label_smoothing (float, *optional*):
        A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets
        become a mixture of the original ground truth and a uniform distribution as described in
        `Rethinking the Inception Architecture for Computer Vision <https://huggingface.co/papers/1512.00567>`__. Default: :math:`0.0`.
    image_text_hidden_size (`int`, *optional*, defaults to 256):
        Dimensionality of the hidden state of the image-text fusion layer.

    Example:

    ```python
    >>> from transformers import BlipConfig, BlipModel

    >>> # Initializing a BlipConfig with Salesforce/blip-vqa-base style configuration
    >>> configuration = BlipConfig()

    >>> # Initializing a BlipPModel (with random weights) from the Salesforce/blip-vqa-base style configuration
    >>> model = BlipModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a BlipConfig from a BlipTextConfig and a BlipVisionConfig

    >>> # Initializing a BLIPText and BLIPVision configuration
    >>> config_text = BlipTextConfig()
    >>> config_vision = BlipVisionConfig()

    >>> config = BlipConfig(text_config=config_text, vision_config=config_vision)
    ```"""

    model_type = "blip"
    sub_configs = {"text_config": BlipTextConfig, "vision_config": BlipVisionConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        image_text_hidden_size=256,
        label_smoothing=0.0,
        tie_word_embeddings=True,
        **kwargs,
    ):
        if text_config is None:
            text_config = BlipTextConfig()
            logger.info("`text_config` is `None`. Initializing the `BlipTextConfig` with default values.")
        elif isinstance(text_config, dict):
            text_config = BlipTextConfig(**text_config)

        if vision_config is None:
            vision_config = BlipVisionConfig()
            logger.info("`vision_config` is `None`. initializing the `BlipVisionConfig` with default values.")
        elif isinstance(vision_config, dict):
            vision_config = BlipVisionConfig(**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config

        self.text_config.encoder_hidden_size = self.vision_config.hidden_size

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0
        self.initializer_range = 0.02
        self.image_text_hidden_size = image_text_hidden_size
        self.label_smoothing = label_smoothing
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


__all__ = ["BlipConfig", "BlipTextConfig", "BlipVisionConfig"]
