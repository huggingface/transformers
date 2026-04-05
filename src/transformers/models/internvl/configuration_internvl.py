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


from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


_LEGACY_INTERNVL_TEXT_KEYS_TO_DROP = (
    "_name_or_path",
    "_attn_implementation_autoset",
    "auto_map",
    "use_bfloat16",
    "use_flash_attn",
    "laux_allreduce",
    "moe_coeff_ratio",
    "moe_intermediate_size",
    "moe_output_scale",
    "noisy_gate_policy",
    "shared_expert_intermediate_size",
    "use_residual",
    "use_moe",
    "use_rts",
    "use_weighted_residual",
    "moe_config",
    "num_experts",
    "num_routed_experts",
    "num_shared_experts",
    "capacity_factor",
    "eval_capacity_factor",
)

_LEGACY_INTERNVL_VISION_KEYS_TO_DROP = (
    "_name_or_path",
    "architectures",
    "drop_path_rate",
    "dropout",
    "initializer_factor",
    "model_type",
    "use_bfloat16",
    "use_flash_attn",
)


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
    attribute_map = {
        "qkv_bias": "attention_bias",
        "qk_normalization": "use_qk_norm",
        "initializer_factor": "layer_scale_init_value",
    }

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
        if isinstance(self.image_size, list):
            self.image_size = tuple(self.image_size)
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="OpenGVLab/InternVL3-1B-hf")
@strict
class InternVLConfig(PreTrainedConfig):
    r"""
    downsample_ratio (`float`, *optional*, defaults to 0.5):
        Factor by which to downsample the image.
    force_image_size (`int`, *optional*):
        Legacy InternVL image size override used by older `internvl_chat` checkpoints.
    min_dynamic_patch (`int`, *optional*, defaults to 1):
        Minimum number of image patches when using dynamic image tiling.
    max_dynamic_patch (`int`, *optional*, defaults to 12):
        Maximum number of image patches when using dynamic image tiling.
    dynamic_image_size (`bool`, *optional*, defaults to `False`):
        Whether dynamic image tiling is enabled.
    use_thumbnail (`bool`, *optional*, defaults to `False`):
        Whether to append a thumbnail image during dynamic tiling.
    ps_version (`str`, *optional*, defaults to `"v2"`):
        Legacy pixel shuffle mode kept for backward compatibility with `internvl_chat` checkpoints.

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
    attribute_map = {"select_layer": "vision_feature_layer"}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151667
    image_seq_length: int = 256
    downsample_ratio: float = 0.5
    projector_hidden_act: str = "gelu"
    vision_feature_layer: int | list[int] = -1
    vision_feature_select_strategy: str = "default"
    tie_word_embeddings: bool = True
    force_image_size: int | None = None
    min_dynamic_patch: int = 1
    max_dynamic_patch: int = 12
    dynamic_image_size: bool = False
    use_thumbnail: bool = False
    ps_version: str = "v2"

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

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        if config_dict.get("model_type") == "internvl_chat":
            config_dict = cls._remap_old_internvl_chat_config(config_dict)
        return super().from_dict(config_dict, **kwargs)

    @classmethod
    def _remap_old_internvl_chat_config(cls, config_dict):
        config_dict = dict(config_dict)
        config_dict["model_type"] = "internvl_chat"
        config_dict.pop("auto_map", None)

        if "llm_config" in config_dict and "text_config" not in config_dict:
            text_config = dict(config_dict.pop("llm_config"))
            config_dict["text_config"] = text_config
            if "tie_word_embeddings" not in config_dict:
                config_dict["tie_word_embeddings"] = text_config.get("tie_word_embeddings", False)
        elif "llm_config" in config_dict:
            config_dict.pop("llm_config")

        if "select_layer" in config_dict and "vision_feature_layer" not in config_dict:
            config_dict["vision_feature_layer"] = config_dict.pop("select_layer")
        elif "select_layer" in config_dict:
            config_dict.pop("select_layer")

        vision_config = config_dict.get("vision_config")
        if isinstance(vision_config, dict) and "force_image_size" in config_dict:
            vision_config.setdefault("image_size", config_dict["force_image_size"])

        if isinstance(config_dict.get("text_config"), dict):
            text_config, image_token_id = cls._remap_old_text_config(config_dict["text_config"])
            config_dict["text_config"] = text_config
            config_dict.setdefault("image_token_id", image_token_id)

        if isinstance(config_dict.get("vision_config"), dict):
            config_dict["vision_config"] = cls._remap_old_vision_config(config_dict["vision_config"])

        return config_dict

    @staticmethod
    def _remap_old_text_config(text_config):
        text_config = dict(text_config)
        for key in _LEGACY_INTERNVL_TEXT_KEYS_TO_DROP:
            text_config.pop(key, None)
        text_config["use_cache"] = text_config.get("use_cache", True)

        architectures = set(text_config.get("architectures", []))
        is_internlm2 = text_config.get("model_type") == "internlm2" or "InternLM2ForCausalLM" in architectures
        if is_internlm2:
            bias = text_config.pop("bias", None)
            text_config["model_type"] = "llama"
            rope_parameters = dict(
                text_config.pop("rope_parameters", None) or text_config.pop("rope_scaling", {}) or {}
            )
            rope_parameters.setdefault("rope_type", rope_parameters.pop("type", "default"))
            rope_parameters.setdefault(
                "rope_theta", text_config.get("rope_theta", rope_parameters.pop("base", 10000.0))
            )
            text_config["rope_parameters"] = rope_parameters
            if bias is not None:
                text_config.setdefault("attention_bias", bias)
                text_config.setdefault("mlp_bias", bias)
            image_token_id = 92546
        else:
            text_config["model_type"] = text_config.get("model_type", "qwen2")
            image_token_id = 151648

        return text_config, image_token_id

    @staticmethod
    def _remap_old_vision_config(vision_config):
        vision_config = dict(vision_config)
        if "attention_probs_dropout_prob" in vision_config:
            attention_dropout = vision_config.pop("attention_probs_dropout_prob")
            vision_config.setdefault("attention_dropout", attention_dropout)
            vision_config.setdefault("projection_dropout", attention_dropout)
        if "qk_normalization" in vision_config:
            vision_config["use_qk_norm"] = vision_config.pop("qk_normalization")
        if "qkv_bias" in vision_config:
            vision_config["attention_bias"] = vision_config.pop("qkv_bias")
        for key in _LEGACY_INTERNVL_VISION_KEYS_TO_DROP:
            vision_config.pop(key, None)
        return vision_config


__all__ = ["InternVLVisionConfig", "InternVLConfig"]
