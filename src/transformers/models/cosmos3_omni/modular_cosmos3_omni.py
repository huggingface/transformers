# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig
from ..qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from ..qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLPreTrainedModel


# Model arch is identical to Qwen3-VL, we need the config to override its `model_type`
# and match officially released ckpt from the hub


@auto_docstring(checkpoint="nvidia/Cosmos3-Nano")
@strict
class Cosmos3OmniConfig(Qwen3VLConfig):
    r"""
    Example:

    ```python
    >>> from transformers import Cosmos3OmniForConditionalGeneration, Cosmos3OmniConfig

    >>> # Initializing a Cosmos3 style configuration
    >>> configuration = Cosmos3OmniConfig()

    >>> # Initializing a model from the Cosmos3 style configuration
    >>> model = Cosmos3OmniForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "cosmos3_omni"
    sub_configs = {"vision_config": AutoConfig, "text_config": AutoConfig}

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            model_type = self.vision_config.pop("model_type", "qwen3_vl_vision")
            if model_type == "qwen3_vl":
                model_type = "qwen3_vl_vision"
            self.vision_config = CONFIG_MAPPING[model_type](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["qwen3_vl_vision"]()

        if isinstance(self.text_config, dict):
            model_type = self.text_config.get("model_type", "qwen3_vl_text")
            self.text_config = CONFIG_MAPPING[model_type](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3_vl_text"]()

        PreTrainedConfig.__post_init__(**kwargs)


_COSMOS3_DROPPED_UNIFIED_CHECKPOINT_KEYS = [
    # Generator (image / video diffusion) MoT expert + cross-modal projections
    r"\.add_q_proj\.",
    r"\.add_k_proj\.",
    r"\.add_v_proj\.",
    r"\.to_add_out\.",
    r"\.norm_added_q\.",
    r"\.norm_added_k\.",
    r"moe_gen",
    r"^proj_out\.",
    r"^proj_in\.",
    r"^time_embedder\.",
    # Sound tower
    r"^audio_proj_out\.",
    r"^audio_proj_in\.",
    r"^audio_modality_embed$",
    # Action tower
    r"^action_proj_out\.",
    r"^action_proj_in\.",
    r"^action_modality_embed$",
]


class Cosmos3OmniPreTrainedModel(Qwen3VLPreTrainedModel):
    # Unified Cosmos3 checkpoint also carries the Generator tower, sound/action towers,
    # and cross-modal adapters; those parameters are dropped when loading the Reasoner.
    _keys_to_ignore_on_load_unexpected = _COSMOS3_DROPPED_UNIFIED_CHECKPOINT_KEYS

    # So that modular doesn't drag unnecessary deps
    _no_split_modules = None
    _can_record_outputs = None

    def _init_weights(self, module):
        PreTrainedModel._init_weights(module)


class Cosmos3OmniForConditionalGeneration(Qwen3VLForConditionalGeneration):
    pass


__all__ = [
    "Cosmos3OmniConfig",
    "Cosmos3OmniForConditionalGeneration",
    "Cosmos3OmniPreTrainedModel",
    "Cosmos3OmniModel",  # noqa: F822
]
