# Copyright 2024 Microsoft Research & University of Wisconsin-Madison and the HuggingFace Inc. team. All rights reserved.
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
"""PaliGemmamodel configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/paligemma-3b-pt-224")
class PaliGemmaConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import PaliGemmaForConditionalGeneration, PaliGemmaConfig, SiglipVisionConfig, GemmaConfig

    >>> # Initializing a Siglip-like vision config
    >>> vision_config = SiglipVisionConfig()

    >>> # Initializing a PaliGemma config
    >>> text_config = GemmaConfig()

    >>> # Initializing a PaliGemma paligemma-3b-224 style configuration
    >>> configuration = PaliGemmaConfig(vision_config, text_config)

    >>> # Initializing a model from the paligemma-3b-224 style configuration
    >>> model = PaliGemmaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "paligemma"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        tie_word_embeddings: bool | None = True,
        **kwargs,
    ):
        self.image_token_index = image_token_index
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.tie_word_embeddings = tie_word_embeddings
        self.is_encoder_decoder = False

        if isinstance(self.vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "siglip_vision_model")
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"](
                intermediate_size=4096,
                hidden_size=1152,
                patch_size=14,
                image_size=224,
                num_hidden_layers=27,
                num_attention_heads=16,
                vocab_size=257152,
                vision_use_head=False,
            )

        self.text_config = text_config
        if isinstance(self.text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "gemma")
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["gemma"](
                hidden_size=2048,
                num_hidden_layers=18,
                intermediate_size=16384,
                num_attention_heads=8,
                num_key_value_heads=1,
                is_encoder_decoder=False,
                vocab_size=vocab_size,
            )

        # BC: `use_bidirectional_attention` was originally unset in PaliGemma1 (backbone = Gemma1) AND PaliGemma2
        # (backbone = Gemma2). Both PaliGemmas want to default to True.
        if self.text_config.use_bidirectional_attention is None:
            self.text_config.use_bidirectional_attention = True

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim
        super().__init__(**kwargs)


__all__ = ["PaliGemmaConfig"]
