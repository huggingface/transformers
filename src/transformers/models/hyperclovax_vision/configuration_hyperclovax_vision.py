# Copyright 2026 NAVER Corp. and the HuggingFace Inc. team. All rights reserved.
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
"""HyperClovaX model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...models.auto import CONFIG_MAPPING, AutoConfig
from ...utils import auto_docstring
from ..granite.configuration_granite import GraniteConfig


@auto_docstring(checkpoint="naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")
@strict(accept_kwargs=True)
class HyperClovaXConfig(GraniteConfig):
    r"""
    HyperClovaX language model configuration.  Extends [`GraniteConfig`] with
    MuP post-norm support and backward-compatible RoPE fields.

    ``rope_theta`` / ``rope_scaling`` are accepted for compatibility with
    existing hub checkpoints that store RoPE settings in the legacy Llama-style
    flat format.  They are converted to the ``rope_parameters`` dict expected
    by [`GraniteRotaryEmbedding`] during ``__post_init__``.

    Args:
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base period of the Rotary Position Embedding (RoPE).  Converted to
            ``rope_parameters["rope_theta"]`` when ``rope_parameters`` is not
            set explicitly.
        rope_scaling (`dict`, *optional*):
            Legacy RoPE scaling dict (Llama style).  Keys are merged into
            ``rope_parameters`` when ``rope_parameters`` is not set explicitly.
        head_dim (`int`, *optional*):
            Dimension of each attention head.  Defaults to
            ``hidden_size // num_attention_heads``.
    ```python
    >>> from transformers import HyperClovaXConfig, HyperClovaXModel

    >>> # Initializing a HyperClovaX configuration
    >>> configuration = HyperClovaXConfig()

    >>> # Initializing a model from the configuration
    >>> model = HyperClovaXModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "hyperclovax"
    use_post_norm: bool = False  # Peri-LN (post-norm)


@auto_docstring(checkpoint="naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")
@strict(accept_kwargs=True)
class HCXVisionConfig(PreTrainedConfig):
    r"""
    text_config (`dict` or [`HyperClovaXConfig`], *optional*):
        Configuration for the LLM backbone.  Defaults to [`HyperClovaXConfig`].
    vision_config (`dict` or config, *optional*):
        Configuration for the vision encoder.  Defaults to
        [`Qwen2_5_VLVisionConfig`].
    image_token_id (`int`, *optional*):
        Token ID used as a placeholder for image patches in the input
        sequence. Falls back to ``img_start_id`` for backward compatibility
        with older checkpoints.
    video_token_id (`int`, *optional*):
        Token ID used as a placeholder for video patches in the input
        sequence. Falls back to ``video_start_id`` for backward
        compatibility with older checkpoints.

    ```python
    >>> from transformers import HCXVisionConfig, HCXVisionForConditionalGeneration

    >>> # Initializing a HyperClovaX Vision configuration with defaults
    >>> configuration = HCXVisionConfig()

    >>> # Initializing a model from the configuration
    >>> model = HCXVisionForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "hyperclovax_vision"
    sub_configs = {"text_config": HyperClovaXConfig, "vision_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int | None = None
    video_token_id: int | None = None

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            model_type = self.vision_config.get("model_type", "qwen2_5_vl_vision")
            model_type = "qwen2_5_vl_vision" if model_type == "qwen2_5_vl" else model_type
            self.vision_config["model_type"] = model_type
            self.vision_config = CONFIG_MAPPING[model_type](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["qwen2_5_vl_vision"]()

        if isinstance(self.text_config, dict):
            model_type = self.text_config.get("model_type", "hyperclovax")
            self.text_config = CONFIG_MAPPING[model_type](**self.text_config)
        elif self.text_config is None:
            self.text_config = HyperClovaXConfig()

        if self.image_token_id is None:
            self.image_token_id = kwargs.pop("img_start_id", 128060)
        if self.video_token_id is None:
            self.video_token_id = kwargs.pop("video_start_id", 128061)
        super().__post_init__(**kwargs)


__all__ = ["HyperClovaXConfig", "HCXVisionConfig"]
