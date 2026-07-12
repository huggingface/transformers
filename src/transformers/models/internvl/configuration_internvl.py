# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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


import copy

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


# Language backbone architecture -> (native text model_type, InternVL image token id).
# The original `internvl_chat` checkpoints do not store these natively.
_INTERNVL_CHAT_LM_MAPPING = {
    "Qwen2ForCausalLM": ("qwen2", 151667),
    "InternLM2ForCausalLM": ("llama", 92546),
}


def _convert_internvl_chat_config_dict(config_dict: dict) -> dict:
    """Normalize an original ``internvl_chat`` config dict into the native layout.

    The ``OpenGVLab/InternVL2-*`` checkpoints ship a bespoke ``internvl_chat``
    config (``llm_config``/``vision_config`` with ``intern_vit_6b`` fields and a
    ``select_layer`` index). This maps those onto the fields expected by
    ``InternVLConfig`` so the checkpoints load with the native implementation
    instead of remote code. Mirrors the offline conversion in
    ``convert_internvl_weights_to_hf.py``.
    """
    config_dict = copy.deepcopy(config_dict)
    llm_config = config_dict.get("llm_config") or {}
    vision_config = config_dict.get("vision_config") or {}

    lm_arch = (llm_config.get("architectures") or ["Qwen2ForCausalLM"])[0]
    text_model_type, image_token_id = _INTERNVL_CHAT_LM_MAPPING.get(lm_arch, ("qwen2", 151667))
    llm_config["model_type"] = text_model_type
    llm_config.setdefault("use_cache", True)

    # InternViT -> InternVLVisionConfig field renames.
    if "attention_probs_dropout_prob" in vision_config:
        dropout = vision_config.pop("attention_probs_dropout_prob")
        vision_config["attention_dropout"] = dropout
        vision_config["projection_dropout"] = dropout
    if "qk_normalization" in vision_config:
        vision_config["use_qk_norm"] = vision_config.pop("qk_normalization")
    if "qkv_bias" in vision_config:
        vision_config["attention_bias"] = vision_config.pop("qkv_bias")
    vision_config["use_absolute_position_embeddings"] = True
    allowed = set(InternVLVisionConfig.__annotations__)
    vision_config = {k: v for k, v in vision_config.items() if k in allowed}

    return {
        "vision_config": vision_config,
        "text_config": llm_config,
        "image_token_id": image_token_id,
        "downsample_ratio": config_dict.get("downsample_ratio", 0.5),
        "vision_feature_layer": config_dict.get("select_layer", -1),
        "tie_word_embeddings": llm_config.get("tie_word_embeddings", False),
    }


@auto_docstring(checkpoint="OpenGVLab/InternVL3-1B-hf")
@strict
class InternVLVisionConfig(PreTrainedConfig):
    r"""
    projection_dropout (`float`, *optional*, defaults to 0.0):
        Dropout probability for the projection layer.
    norm_type (`str`, *optional*, defaults to `"layer_norm"`):
        The type of normalization to use in the encoder. Can be `"layer_norm"` or `"rms_norm"`.
    use_mask_token (`bool`, *optional*, defaults to `False`):
        Whether to use a mask token for masked image modeling
    use_mean_pooling (`bool`, *optional*, defaults to `True`):
        Whether to mean pool the final hidden states of the patches instead of using the final hidden state of the
        CLS token, before applying the classification head.

    Example:

    ```python
    >>> from transformers import InternVLVisionConfig, InternVLVisionModel

    >>> # Initializing a InternVLVisionModel OpenGVLab/InternVL3-1B-hf style configuration
    >>> configuration = InternVLVisionConfig()

    >>> # Initializing a model (with random weights) from the OpenGVLab/InternVL3-1B-hf configuration
    >>> model = InternVLVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "internvl_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    attention_bias: bool = False
    use_qk_norm: bool = False
    intermediate_size: int = 4096
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_dropout: float | int = 0.0
    projection_dropout: float | int = 0.0
    initializer_range: float = 0.02
    norm_type: str = "layer_norm"
    layer_norm_eps: float = 1e-06
    image_size: int | list[int] | tuple[int, ...] = (448, 448)
    patch_size: int | list[int] | tuple[int, ...] = (14, 14)
    num_channels: int = 3
    use_mask_token: bool = False
    use_absolute_position_embeddings: bool = True
    layer_scale_init_value: float = 0.1
    use_mean_pooling: bool = True

    def __post_init__(self, **kwargs):
        self.image_size = (
            self.image_size if isinstance(self.image_size, (list, tuple)) else (self.image_size, self.image_size)
        )
        self.patch_size = (
            self.patch_size if isinstance(self.patch_size, (list, tuple)) else (self.patch_size, self.patch_size)
        )
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="OpenGVLab/InternVL3-1B-hf")
@strict
class InternVLConfig(PreTrainedConfig):
    r"""
    downsample_ratio (`float`, *optional*, defaults to 0.5):
        Factor by which to downsample the image.

    Example:

    ```python
    >>> from transformers import InternVLForConditionalGeneration, InternVLConfig

    >>> # Initializing a InternVL style configuration
    >>> configuration = InternVLConfig()

    >>> # Initializing a model (with random weights) from the OpenGVLab/InternVL3-1B-hf configuration
    >>> model = InternVLForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "internvl"
    sub_configs = {"text_config": AutoConfig, "vision_config": InternVLVisionConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151667
    image_seq_length: int = 256
    downsample_ratio: float = 0.5
    projector_hidden_act: str = "gelu"
    vision_feature_layer: int | list[int] = -1
    vision_feature_select_strategy: str = "default"
    tie_word_embeddings: bool = True

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        # Original `internvl_chat` checkpoints are remapped onto the native layout
        # so they can be loaded without remote code.
        if config_dict.get("model_type") == "internvl_chat" or "llm_config" in config_dict:
            config_dict = _convert_internvl_chat_config_dict(config_dict)
        return super().from_dict(config_dict, **kwargs)

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = InternVLVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = InternVLVisionConfig()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen2")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen2"]()

        super().__post_init__(**kwargs)


__all__ = ["InternVLVisionConfig", "InternVLConfig"]
