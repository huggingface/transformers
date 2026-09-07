# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


# Language backbone architecture -> native text `model_type`. The original checkpoints ship the
# backbone under a bespoke `llm_config`, so the native type has to be derived from `architectures`.
# `InternLM2ForCausalLM` maps to `llama` for the same reason the offline converter does: the
# InternLM2 decoder is llama-shaped once the fused `wqkv` is split. Phi3 (InternVL2-4B) is
# deliberately absent: its `llm_config` stores `original_max_position_embeddings` beside
# `rope_scaling` rather than inside it, which current `Phi3Config` rejects. Listing it here would
# trade a clear "unsupported backbone" error for a confusing rope `KeyError`.
_BACKBONE_TO_TEXT_MODEL_TYPE = {
    "Qwen2ForCausalLM": "qwen2",
    "InternLM2ForCausalLM": "llama",
    "LlamaForCausalLM": "llama",
}

# INTERIM: `<IMG_CONTEXT>` ids for checkpoints whose `config.json` predates `image_token_id`.
# Each value was read from that repo's `added_tokens.json`. Once every `OpenGVLab/InternVL2-*`
# repo carries `image_token_id`, this table and the warning below can both be deleted.
_BACKBONE_TO_IMAGE_TOKEN_ID = {
    "Qwen2ForCausalLM": 151648,  # InternVL2-1B
    "InternLM2ForCausalLM": 92546,  # InternVL2-2B, -8B, -26B
    "LlamaForCausalLM": 64000,  # InternVL2-40B
}

# Renames handled by `InternVLVisionConfig.__post_init__`; kept so they survive the field filter.
_INTERN_VIT_ALIASES = {"qk_normalization", "qkv_bias"}


def _convert_internvl_chat_config_dict(config_dict: dict) -> dict:
    """Normalize an original ``internvl_chat`` config dict onto the native ``InternVL`` layout.

    The ``OpenGVLab/InternVL2-*`` checkpoints ship a bespoke config (``llm_config`` plus an
    ``intern_vit_6b`` ``vision_config`` and a ``select_layer`` index). Only the fields that
    actually differ from the native defaults are emitted, so ``downsample_ratio`` (0.5) and
    ``select_layer`` (-1) are dropped -- every published checkpoint matches the default.
    """
    config_dict = copy.deepcopy(config_dict)
    llm_config = config_dict.get("llm_config") or {}
    vision_config = config_dict.get("vision_config") or {}

    architectures = llm_config.get("architectures") or []
    lm_arch = architectures[0] if architectures else None
    if lm_arch not in _BACKBONE_TO_TEXT_MODEL_TYPE:
        raise ValueError(
            f"Unsupported InternVL2 language backbone {lm_arch!r}. Expected one of "
            f"{sorted(_BACKBONE_TO_TEXT_MODEL_TYPE)}. If this is a new backbone, it needs an entry "
            "in `_BACKBONE_TO_TEXT_MODEL_TYPE`."
        )

    llm_config["model_type"] = _BACKBONE_TO_TEXT_MODEL_TYPE[lm_arch]
    llm_config.setdefault("use_cache", True)

    # Drop the InternViT-only keys (`model_type`, `architectures`, flash-attn flags, ...) but keep
    # the aliases that `InternVLVisionConfig.__post_init__` renames onto native field names.
    allowed = set(CONFIG_MAPPING["internvl_vision"].__annotations__) | _INTERN_VIT_ALIASES
    vision_config = {k: v for k, v in vision_config.items() if k in allowed}
    vision_config["use_absolute_position_embeddings"] = True

    if "image_token_id" in config_dict:
        image_token_id = config_dict["image_token_id"]
    else:
        image_token_id = _BACKBONE_TO_IMAGE_TOKEN_ID[lm_arch]
        logger.warning_once(
            "This checkpoint's `config.json` has no `image_token_id`, so it was inferred from the "
            f"language backbone ({lm_arch} -> {image_token_id}). Newer checkpoints store the id "
            "directly; if image inputs are ignored, check that the id matches `<IMG_CONTEXT>` in "
            "the checkpoint's `added_tokens.json`."
        )

    converted = {
        "vision_config": vision_config,
        "text_config": llm_config,
        "image_token_id": image_token_id,
    }
    # Only carry it when the source sets it, so the class default applies otherwise.
    if "tie_word_embeddings" in llm_config:
        converted["tie_word_embeddings"] = llm_config["tie_word_embeddings"]
    return converted


@auto_docstring(checkpoint="OpenGVLab/InternVL2-1B")
@strict
class InternVL2Config(PreTrainedConfig):
    r"""
    downsample_ratio (`float`, *optional*, defaults to 0.5):
        Factor by which to downsample the image.

    Configuration for the original ``OpenGVLab/InternVL2-*`` checkpoints.

    These ship the bespoke ``internvl_chat`` layout, which this class normalizes onto the native
    [`InternVLConfig`] fields so the checkpoints load into [`InternVLForConditionalGeneration`]
    without `trust_remote_code`. The modeling and processor classes are auto-mapped to InternVL's.

    Example:

    ```python
    >>> from transformers import InternVL2Config, InternVLForConditionalGeneration

    >>> configuration = InternVL2Config()
    >>> model = InternVLForConditionalGeneration(configuration)  # underlying architecture is InternVL
    >>> configuration = model.config
    ```"""

    model_type = "internvl_chat"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

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
        # Only the original layout carries `llm_config`; an already-normalized dict passes through.
        if "llm_config" in config_dict:
            config_dict = _convert_internvl_chat_config_dict(config_dict)
        return super().from_dict(config_dict, **kwargs)

    def __post_init__(self, **kwargs):
        # Resolved through the auto mapping rather than imported: this folder holds only a config,
        # and `utils/check_modeling_structure.py` (TRF009) forbids importing another model's code.
        vision_config_class = CONFIG_MAPPING["internvl_vision"]
        if isinstance(self.vision_config, dict):
            self.vision_config = vision_config_class(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = vision_config_class()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen2")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen2"]()

        super().__post_init__(**kwargs)


__all__ = ["InternVL2Config"]
