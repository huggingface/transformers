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
"""Cosmos3 model — loads the Reasoner tower of a Cosmos3 MoT checkpoint into Qwen3-VL."""

from huggingface_hub.dataclasses import strict

from ...utils import auto_docstring
from ..qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from ..qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLModel


@auto_docstring(checkpoint="nvidia/Cosmos3-Nano")
@strict
class Cosmos3Config(Qwen3VLConfig):
    r"""
    Configuration for the [Cosmos3](https://huggingface.co/nvidia/Cosmos3-Nano) Reasoner tower.

    The Reasoner tower is architecturally identical to Qwen3-VL, so this config inherits all
    fields from [`Qwen3VLConfig`] and only changes `model_type` so that conversion mappings
    and key-renaming rules dispatch correctly when loading a unified Cosmos3 checkpoint.
    """

    model_type = "cosmos3_omni"


_COSMOS3_DROPPED_UNIFIED_CHECKPOINT_KEYS = [
    # Generator (image / video diffusion) MoT expert + cross-modal projections
    r"_moe_gen",
    r"^llm2vae\.",
    r"^vae2llm\.",
    r"^time_embedder\.",
    # Sound tower
    r"^llm2sound\.",
    r"^sound2llm\.",
    r"^sound_modality_embed$",
    # Action tower
    r"^llm2action\.",
    r"^action2llm\.",
    r"^action_modality_embed$",
]


class Cosmos3Model(Qwen3VLModel):
    config: Cosmos3Config

    # Base-model loading from a unified Cosmos3 checkpoint drops the Generator tower,
    # cross-modal adapters, and the causal-LM head.
    _keys_to_ignore_on_load_unexpected = _COSMOS3_DROPPED_UNIFIED_CHECKPOINT_KEYS + [
        r"^lm_head\.weight$"
    ]


class Cosmos3ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    config: Cosmos3Config

    # The unified Cosmos3 checkpoint stores both the Reasoner tower (loaded here) and the
    # Generator tower / cross-modal adapters (dropped). These patterns silence the
    # "unexpected keys" warning for parameters that belong to the dropped components.
    _keys_to_ignore_on_load_unexpected = _COSMOS3_DROPPED_UNIFIED_CHECKPOINT_KEYS


__all__ = [
    "Cosmos3Config",
    "Cosmos3ForConditionalGeneration",
    "Cosmos3Model",
]
