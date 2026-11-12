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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...utils import auto_docstring
from ..auto import AutoConfig


# InternLM2 (2B/8B/26B) needs its fused `wqkv` split at load time, which is not supported yet.
_LLM_ARCHITECTURE_TO_MODEL_TYPE = {
    "Qwen2ForCausalLM": "qwen2",
    "LlamaForCausalLM": "llama",
}


@auto_docstring(checkpoint="OpenGVLab/InternVL2-1B")
@strict
class InternVL2Config(PreTrainedConfig):
    r"""
    downsample_ratio (`float`, *optional*, defaults to 0.5):
        Factor by which to downsample the image.

    Example:

    ```python
    >>> from transformers import InternVL2Config, InternVLForConditionalGeneration

    >>> configuration = InternVL2Config()
    >>> model = InternVLForConditionalGeneration(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "internvl_chat"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="qwen2"),
        "vision_config": SubConfigSpec(config_class=AutoConfig, model_type="internvl_vision"),
    }

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151667
    image_seq_length: int = 256
    downsample_ratio: float = 0.5
    projector_hidden_act: str = "gelu"
    vision_feature_layer: int | list[int] = -1
    vision_feature_select_strategy: str = "default"
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        # BC: the original checkpoints store the language model under `llm_config`
        llm_config = kwargs.pop("llm_config", None)
        if self.text_config is None and llm_config is not None:
            architecture = llm_config["architectures"][0]
            if architecture not in _LLM_ARCHITECTURE_TO_MODEL_TYPE:
                raise ValueError(
                    f"Unsupported InternVL2 language model {architecture}, expected one of "
                    f"{list(_LLM_ARCHITECTURE_TO_MODEL_TYPE)}."
                )
            self.text_config = {**llm_config, "model_type": _LLM_ARCHITECTURE_TO_MODEL_TYPE[architecture]}
            self.tie_word_embeddings = llm_config.get("tie_word_embeddings", False)

        # BC: and the vision model under InternViT's own model type and field names
        if isinstance(self.vision_config, dict) and self.vision_config.get("model_type") == "intern_vit_6b":
            self.vision_config = {**self.vision_config, "model_type": "internvl_vision"}
            if "qk_normalization" in self.vision_config:
                self.vision_config["use_qk_norm"] = self.vision_config.pop("qk_normalization")
            if "qkv_bias" in self.vision_config:
                self.vision_config["attention_bias"] = self.vision_config.pop("qkv_bias")

        super().__post_init__(**kwargs)


__all__ = ["InternVL2Config"]
