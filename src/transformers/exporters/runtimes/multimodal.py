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
"""Running a multimodal export: its components chained as the model's own forward chains them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...utils import is_torch_available
from ..decompose import (
    _MODALITY_SPECS,
    DEEPSTACK_INPUTS,
    ModalitySpec,
    dense_deepstack,
    flatten_anyres_patches,
    pack_anyres_features,
)
from ..precompute import _find_config_attr, get_rope_index_from_config, precompute_export_inputs
from ..utils import cast_leaf_tensors
from .base import ExportedModel, ModelRunner


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ...configuration_utils import PreTrainedConfig


MULTIMODAL_COMPONENTS = frozenset(
    {"language_model", "text_embeddings_model", *(spec.component for spec in _MODALITY_SPECS)}
)


class ExportedMultimodalModel(ExportedModel):
    """A multimodal export ([`~HfExporter.export_multimodal`]) called like the model it came from: the token ids are
    embedded, each modality's features are scattered into its placeholder rows, and `language_model` runs on the
    result.
    """

    def __init__(self, runners: dict[str, ModelRunner], config: PreTrainedConfig | None = None):
        super().__init__(runners["language_model"], config)
        self.text_runner = runners["text_embeddings_model"]
        self.modality_runners = [
            (spec, runners[spec.component]) for spec in _MODALITY_SPECS if spec.component in runners
        ]

    def __call__(self, **kwargs) -> Any:
        embedded, features = self.embed(**kwargs)
        inputs = dict(kwargs)
        # M-RoPE positions come from the token ids and grids, which the language model no longer takes.
        if (
            inputs.get("position_ids") is None
            and (rope_index := get_rope_index_from_config(self.config, kwargs)) is not None
        ):
            inputs["position_ids"] = rope_index[0]
        inputs.update(embedded)
        traced = self.runner.metadata.kwargs
        outputs = self.runner(**{name: value for name, value in inputs.items() if name in traced})
        # The model returns its image features too, which the language model never sees.
        if features.get("vision_embeddings_model") is not None and "image_hidden_states" in getattr(
            outputs, "__dataclass_fields__", {}
        ):
            outputs["image_hidden_states"] = features["vision_embeddings_model"]
        # And Granite4-Vision its per-layer ones.
        if "deepstack_features" in features and "deepstack_features" in getattr(outputs, "__dataclass_fields__", {}):
            outputs["deepstack_features"] = features["deepstack_features"]
        return outputs

    def embed(self, **kwargs) -> tuple[dict, dict]:
        """`input_ids` as `inputs_embeds` (plus `per_layer_inputs` when the embedder returns them), each present
        modality's features scattered into its placeholder rows, and what a deepstack decoder adds inside its layers;
        and the features, by component (a Granite4-Vision tower's per-image ones under `deepstack_features`)."""
        input_ids = kwargs["input_ids"]
        embedded = self.text_runner(input_ids=input_ids)
        embedded = dict(embedded) if isinstance(embedded, dict) else {"inputs_embeds": embedded}
        inputs_embeds = embedded["inputs_embeds"]
        features, deepstack = {}, []
        for spec, runner in self.modality_runners:
            mask = (input_ids == getattr(self.config, spec.token_field)).unsqueeze(-1).to(inputs_embeds.device)
            if all(kwargs.get(key) is None for key in spec.feature_keys) or not mask.any():
                continue
            features[spec.component], layers = self._features(spec, runner, kwargs)
            if features[spec.component] is None:
                # All of Granite4-Vision's features are added inside the decoder, over zeroed placeholder rows; the
                # model returns them too, one tensor per image.
                inputs_embeds = inputs_embeds.masked_fill(mask, 0.0)
                features["deepstack_features"] = list(layers.items())
                layers = {layer: torch.cat(images) for layer, images in layers.items()}
            else:
                inputs_embeds = inputs_embeds.masked_scatter(mask, features[spec.component].to(inputs_embeds))
            if layers is not None:
                deepstack.append((mask[..., 0], layers))
        embedded["inputs_embeds"] = inputs_embeds
        embedded.update(self._deepstack_inputs(deepstack, inputs_embeds))
        return embedded, features

    def _deepstack_inputs(self, deepstack: list, inputs_embeds: torch.Tensor) -> dict[str, Any]:
        """What a deepstack decoder adds inside its layers, from each modality's `(rows, features by layer)`, in the
        dense form its graph was traced with (`dense_deepstack`): zeros on a step without images."""
        traced = self.runner.metadata.kwargs
        for mask_name, spec in DEEPSTACK_INPUTS.items():
            if spec.features in traced:
                recorded = traced[spec.features]
                layers = (
                    [int(layer) for layer in recorded["leaves"]] if spec.by_layer else range(len(recorded["items"]))
                )
                return dense_deepstack(mask_name, inputs_embeds, layers, deepstack)
        return {}

    def _features(self, spec: ModalitySpec, runner: ModelRunner, kwargs: dict) -> tuple[torch.Tensor | None, Any]:
        """Run one modality's graph on its own inputs, as its getter was called: features for its placeholder rows,
        and a deepstack tower's per-layer features (`None` otherwise)."""
        traced = runner.metadata.kwargs
        feed = {
            ("grid_thw" if key == spec.grid_key else key): value
            for key, value in kwargs.items()
            if value is not None and (key in spec.input_keys or key == spec.grid_key or key in traced)
        }
        image_sizes = kwargs.get("image_sizes")
        if image_sizes is not None and "image_sizes" not in traced and "pixel_values" in feed:
            feed["pixel_values"] = flatten_anyres_patches(self.config, feed["pixel_values"], image_sizes)
        feed = precompute_export_inputs(self.config, feed)
        for name in traced.keys() - feed.keys():
            if (value := _find_config_attr(self.config, name)) is not None:
                feed[name] = value
        feed = cast_leaf_tensors(feed, dtype=runner.dtype, device=runner.device)
        outputs = runner(**{name: value for name, value in feed.items() if name in traced})
        if not isinstance(outputs, dict):
            return outputs, None
        if "deepstack_features" in outputs:
            return outputs["features"], list(outputs["deepstack_features"])
        newline = outputs.get("image_newline")
        if "image_features" in outputs:
            return pack_anyres_features(self.config, outputs["image_features"], image_sizes, newline), None
        # Granite4-Vision's anyres tower: `image_features.<decoder layer>`, all added inside the decoder, per image.
        layers = {
            int(name.rsplit(".", 1)[1]): pack_anyres_features(self.config, value, image_sizes, newline, per_image=True)
            for name, value in outputs.items()
            if name.startswith("image_features.")
        }
        return None, layers
