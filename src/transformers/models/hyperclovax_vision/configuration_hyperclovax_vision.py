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

    model_type = "hyperclovax_text"

    rope_theta: float | int = 10000.0
    rope_scaling: dict | None = None
    head_dim: int | None = None

    def __post_init__(self, **kwargs):
        self.rope_theta = float(self.rope_theta)
        # Convert legacy flat rope_theta / rope_scaling to rope_parameters dict
        # so that GraniteRotaryEmbedding can consume it.
        if self.rope_parameters is None:
            rope_params: dict = {"rope_type": "default", "rope_theta": self.rope_theta}
            if self.rope_scaling is not None:
                rope_params.update(self.rope_scaling)
            self.rope_parameters = rope_params

        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")
@strict(accept_kwargs=True)
class HCXVisionConfig(PreTrainedConfig):
    r"""
    Configuration for [`HCXVisionForConditionalGeneration`].  Combines a
    [`HyperClovaXConfig`] text backbone with a vision encoder config and a
    single-linear multimodal projector.

    Args:
        text_config (`dict` or [`HyperClovaXConfig`], *optional*):
            Configuration for the LLM backbone.  Defaults to
            [`HyperClovaXConfig`].
        vision_config (`dict` or config, *optional*):
            Configuration for the vision encoder.  Defaults to
            [`Qwen2_5_VLVisionConfig`].
        img_start_id (`int`, *optional*, defaults to 128060):
            Token ID used as a placeholder for image patches in the input
            sequence.
        video_start_id (`int`, *optional*, defaults to 128061):
            Token ID used as a placeholder for video patches in the input
            sequence.

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

    model_type = "hyperclovax"
    sub_configs = {"text_config": HyperClovaXConfig, "vision_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    img_start_id: int = 128060
    video_start_id: int = 128061

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            model_type = self.vision_config.get("model_type", "qwen2_5_vl_image")
            # "qwen2_5_vl" refers to the full VL model; we only need the vision encoder
            if model_type == "qwen2_5_vl":
                model_type = "qwen2_5_vl_image"
            vision_config = CONFIG_MAPPING[model_type](**self.vision_config)
        elif self.vision_config is None:
            vision_config = CONFIG_MAPPING["qwen2_5_vl_image"]()
        self.vision_config = vision_config

        if isinstance(self.text_config, dict):
            model_type = self.text_config.get("model_type", "hyperclovax_text")
            model_type = "hyperclovax_text" if model_type == "hyperclovax" else model_type
            text_config = CONFIG_MAPPING[model_type](**self.text_config)
        elif self.text_config is None:
            text_config = HyperClovaXConfig()
        self.text_config = text_config

        self.image_token_id = self.img_start_id
        self.video_token_id = self.video_start_id

        self.initializer_range = self.text_config.initializer_range

        # Accept old hub configs that used model_type="vlm"
        if kwargs.get("model_type") == "vlm":
            kwargs["model_type"] = "hyperclovax"

        super().__post_init__(**kwargs)


__all__ = ["HyperClovaXConfig", "HCXVisionConfig"]
