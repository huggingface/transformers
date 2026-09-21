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
"""What a component is, shared by the three stages that pass one along.

A decomposition produces components, an exporter turns each into a graph, and a runtime drives them. The
vocabulary lives here so all three agree by construction rather than by recognising each other's strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .metadata import ExportMetadata


class ComponentRole(str, Enum):
    """What a component *is* to a runtime, as opposed to what it is called.

    A decomposition names its components for the reader (`"image_encoder"`, `"text_decoder"`); the role is
    what the runtime and the loader dispatch on, so neither has to recognise names. `DECODE` decides the
    shape of the whole export: one that has a decode graph is driven through `generate`.
    """

    MODEL = "model"
    # The text stack, driven once per step. `prefill` and `decode` name the *call* a graph was traced at,
    # not the module it wraps — for a decoder-only model both wrap the whole model, and only the inputs
    # differ (a prompt on an empty cache, a continuation on a full one). A decode graph whose query axis
    # stayed symbolic serves the prompt as well, and then it is the only text graph an export ships.
    DECODE = "decode"
    # The prompt's own graph, shipped only where the decode graph cannot stand in for it.
    PREFILL = "prefill"
    # An encoder-decoder's encoder, which may also compute the decoder's cross-attention cache.
    ENCODER = "encoder"
    # `input_ids -> inputs_embeds`, so the runtime can scatter modality features into the embeddings.
    EMBED_TOKENS = "embed_tokens"
    # One modality's `get_<modality>_features` — its tower and projector (`image_encoder`, `audio_encoder`).
    MODALITY_ENCODER = "modality_encoder"
    # A modality embedded once ahead of the loop, each step reading its own window (voxtral_realtime).
    STREAMING_EMBEDDER = "streaming_embedder"

    @classmethod
    def of(cls, name: str) -> ComponentRole:
        """The role a component name carries.

        The one place a name is read as a role, and only where a role is missing: a decomposition states the
        role of what it produces, and this covers a manifest written before it did.
        """
        by_name = {
            "model": cls.MODEL,
            "decode": cls.DECODE,
            "text_decoder": cls.DECODE,
            "prefill": cls.PREFILL,
            "encoder": cls.ENCODER,
            "embed_tokens": cls.EMBED_TOKENS,
        }
        if name in by_name:
            return by_name[name]
        if name.endswith("_encoder"):
            return cls.MODALITY_ENCODER
        if name.endswith("_embedder"):
            return cls.STREAMING_EMBEDDER
        raise ValueError(f"No role for a component named {name!r}; name it for what it does, or pass a role.")


@dataclass
class Component:
    """One piece of a decomposed model, ready to export: what to trace, and what to trace it with."""

    name: str
    module: Any
    inputs: dict[str, Any]
    role: ComponentRole


@dataclass
class ExportedComponent:
    """One exported graph, with what the trace recorded about it and what it is for.

    The three travel together because they are one thing: a graph whose metadata went missing is a graph a
    runtime has to guess about (precision, cache kwarg, mask layout — each guess has been wrong at least
    once), and a graph whose role went missing is one the loader has to recognise by name.
    """

    name: str
    artifact: Any
    metadata: ExportMetadata
    role: ComponentRole
    files: dict[str, Any] = field(default_factory=dict)
