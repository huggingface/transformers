# Copyright 2023 Microsoft Research & University of Wisconsin-Madison and the HuggingFace Inc. team. All rights reserved.
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
"""VipLlava model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="llava-hf/vip-llava-7b-hf")
class VipLlavaConfig(PreTrainedConfig):
    r"""
    projector_layernorm_eps (`float`, *optional*, defaults to 1e-05):
        The layer norm epsilon of the projector layernorm
    vision_feature_layers (`Union[int, list[int]]`, *optional*, defaults to `[-2, -5, -8, -11, 6]`):
        The vision feature layer, or list of layers to select the vision features from.

    Example:

    ```python
    >>> from transformers import VipLlavaForConditionalGeneration, VipLlavaConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a VipLlava vipllava-7b style configuration
    >>> configuration = VipLlavaConfig(vision_config, text_config)

    >>> # Initializing a model from the vipllava-7b style configuration
    >>> model = VipLlavaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vipllava"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=32000,
        projector_hidden_act="gelu",
        projector_layernorm_eps=1e-5,
        vision_feature_layers=[-2, -5, -8, -11, 6],
        image_seq_length=576,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.projector_layernorm_eps = projector_layernorm_eps
        self.vision_feature_layers = vision_feature_layers
        self.image_seq_length = image_seq_length
        self.vision_config = vision_config
        self.tie_word_embeddings = tie_word_embeddings

        if isinstance(self.vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "clip_vision_model")
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            self.vision_config = CONFIG_MAPPING["clip_vision_model"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=336,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
            )

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "llama")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config

        super().__init__(**kwargs)


__all__ = ["VipLlavaConfig"]
