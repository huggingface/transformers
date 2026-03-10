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
"""OWLv2 model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/owlv2-base-patch16")
# Copied from transformers.models.owlvit.configuration_owlvit.OwlViTTextConfig with OwlViT->Owlv2, owlvit-base-patch32->owlv2-base-patch16, owlvit->owlv2, OWL-ViT->OWLv2
class Owlv2TextConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import Owlv2TextConfig, Owlv2TextModel

    >>> # Initializing a Owlv2TextModel with google/owlv2-base-patch16 style configuration
    >>> configuration = Owlv2TextConfig()

    >>> # Initializing a Owlv2TextConfig from the google/owlv2-base-patch16 style configuration
    >>> model = Owlv2TextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "owlv2_text_model"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=49408,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=16,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=0,
        bos_token_id=49406,
        eos_token_id=49407,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor


@auto_docstring(checkpoint="google/owlv2-base-patch16")
# Copied from transformers.models.owlvit.configuration_owlvit.OwlViTVisionConfig with OwlViT->Owlv2, owlvit-base-patch32->owlv2-base-patch16, owlvit->owlv2, OWL-ViT->OWLv2, 32->16
class Owlv2VisionConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import Owlv2VisionConfig, Owlv2VisionModel

    >>> # Initializing a Owlv2VisionModel with google/owlv2-base-patch16 style configuration
    >>> configuration = Owlv2VisionConfig()

    >>> # Initializing a Owlv2VisionModel model from the google/owlv2-base-patch16 style configuration
    >>> model = Owlv2VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "owlv2_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=768,
        patch_size=16,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor


@auto_docstring(checkpoint="google/owlv2-base-patch16")
# Copied from transformers.models.owlvit.configuration_owlvit.OwlViTConfig with OwlViT->Owlv2, owlvit-base-patch32->owlv2-base-patch16, owlvit->owlv2, OWL-ViT->OWLv2
class Owlv2Config(PreTrainedConfig):
    model_type = "owlv2"
    sub_configs = {"text_config": Owlv2TextConfig, "vision_config": Owlv2VisionConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        return_dict=True,
        **kwargs,
    ):
        if text_config is None:
            text_config = Owlv2TextConfig()
            logger.info("`text_config` is `None`. initializing the `Owlv2TextConfig` with default values.")
        elif isinstance(text_config, dict):
            text_config = Owlv2TextConfig(**text_config)

        if vision_config is None:
            vision_config = Owlv2VisionConfig()
            logger.info("`vision_config` is `None`. initializing the `Owlv2VisionConfig` with default values.")
        elif isinstance(vision_config, dict):
            vision_config = Owlv2VisionConfig(**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.return_dict = return_dict
        self.initializer_factor = 1.0
        super().__init__(**kwargs)


__all__ = ["Owlv2Config", "Owlv2TextConfig", "Owlv2VisionConfig"]
