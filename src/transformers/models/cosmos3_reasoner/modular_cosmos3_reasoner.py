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

from ...utils import auto_docstring
from ..qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig
from ..qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
)
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor


class Cosmos3ReasonerVisionConfig(Qwen3VLVisionConfig):
    model_type = "cosmos3_omni_vision"


class Cosmos3ReasonerTextConfig(Qwen3VLTextConfig):
    model_type = "cosmos3_omni_text"


@auto_docstring(checkpoint="nvidia/Cosmos3-Nano")
@strict
class Cosmos3ReasonerConfig(Qwen3VLConfig):
    r"""
    Configuration for the [Cosmos3](https://huggingface.co/nvidia/Cosmos3-Nano) Reasoner tower.

    The Reasoner tower is architecturally identical to Qwen3-VL, so this config inherits all
    fields from [`Qwen3VLConfig`] and only changes `model_type` so that conversion mappings
    and key-renaming rules dispatch correctly when loading a unified Cosmos3 checkpoint.
    """

    model_type = "cosmos3_omni"
    sub_configs = {"vision_config": Cosmos3ReasonerVisionConfig, "text_config": Cosmos3ReasonerTextConfig}


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


class Cosmos3ReasonerPreTrainedModel(Qwen3VLPreTrainedModel):
    # Unified Cosmos3 checkpoint also carries the Generator tower, sound/action towers,
    # and cross-modal adapters; those parameters are dropped when loading the Reasoner.
    _keys_to_ignore_on_load_unexpected = _COSMOS3_DROPPED_UNIFIED_CHECKPOINT_KEYS


class Cosmos3ReasonerVisionModel(Qwen3VLVisionModel):
    pass


class Cosmos3ReasonerTextModel(Qwen3VLTextModel):
    pass


class Cosmos3ReasonerModel(Qwen3VLModel):
    pass


class Cosmos3ReasonerForConditionalGeneration(Qwen3VLForConditionalGeneration):
    pass


class Cosmos3ReasonerProcessor(Qwen3VLProcessor):
    pass


__all__ = [
    "Cosmos3ReasonerConfig",
    "Cosmos3ReasonerTextConfig",
    "Cosmos3ReasonerVisionConfig",
    "Cosmos3ReasonerVisionModel",
    "Cosmos3ReasonerForConditionalGeneration",
    "Cosmos3ReasonerModel",
    "Cosmos3ReasonerPreTrainedModel",
    "Cosmos3ReasonerProcessor",
    "Cosmos3ReasonerTextModel",
]
