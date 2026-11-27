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

from typing import Any

from ..modeling_outputs import BaseModelOutputWithPooling
from ..utils import is_torch_available
from .base import ExportedModel
from .decompose import (
    _MODALITY_SPECS,
    DEEPSTACK_INPUTS,
    ModalitySpec,
    _find_config_attr,
    dense_deepstack,
    flatten_anyres_patches,
    pack_anyres_features,
)
from .runners.base import ModelRunner
from .utils import cast_leaf_tensors, with_data_dependent_inputs


if is_torch_available():
    import torch


MULTIMODAL_COMPONENTS = frozenset(
    {"language_model", "text_embeddings_model", *(spec.component for spec in _MODALITY_SPECS)}
)


class ExportedMultimodalModel(ExportedModel):
    """A multimodal export ([`~HfExporter.export_multimodal`]) called like the model it came from: the token ids are
    embedded, each modality's features are scattered into its placeholder rows, and `language_model` runs on the
    result.
    """

    def __call__(self, share_inputs: bool = True, share_outputs: bool = False, **kwargs) -> Any:
        """Run the components on a processor's whole output, returning what the source model returns.
        `share_inputs` and `share_outputs` go to the `language_model` runner, as for [`ExportedModel.__call__`]."""
        # What the model computes with data-dependent ops (M-RoPE positions, encoder inputs), from its code alone
        kwargs = with_data_dependent_inputs(self.weightless_model, kwargs)
        input_ids = kwargs["input_ids"]
        # The embedder's and modality graphs' outputs are read within this call (what the model returns is copied
        # out), so a backend may hand out its own buffers.
        embedded = self.runners["text_embeddings_model"](input_ids=input_ids, share_outputs=True)
        inputs_embeds = embedded["inputs_embeds"] if isinstance(embedded, dict) else embedded
        features, deepstack = {}, []
        for spec in _MODALITY_SPECS:
            given = [key for key in spec.feature_keys if kwargs.get(key) is not None]
            if not given:
                continue
            # GLM-Image, Chameleon, ...: their image features become token ids, scattered before the embedding
            if spec.modality == "image" and getattr(self.config, "vq_config", None) is not None:
                raise NotImplementedError(
                    f"{self.config.model_type} fills its image placeholders with VQ token ids, not features, which "
                    "this runtime doesn't assemble yet: run the export's components yourself."
                )
            runner = self.runners.get(spec.component)
            if runner is None:
                raise ValueError(
                    f"Got `{given[0]}`, but this export has no `{spec.component}`: export the model with inputs that "
                    "include this modality to run it."
                )
            features[spec.component], layers = self._features(spec, runner, kwargs)
            mask = (input_ids == getattr(self.config, spec.token_field)).unsqueeze(-1).to(inputs_embeds.device)
            if features[spec.component] is None:
                # All of Granite4-Vision's features are added inside the decoder, over zeroed placeholder rows; the
                # model returns them too, one tensor per decoder layer in its config's order.
                inputs_embeds = inputs_embeds.masked_fill(mask, 0.0)
                layers = {layer: torch.cat(images) for layer, images in layers.items()}
                features["deepstack_features"] = list(layers.values())
            else:
                inputs_embeds = inputs_embeds.masked_scatter(mask, features[spec.component].to(inputs_embeds))
            if layers is not None:
                deepstack.append((mask[..., 0], layers))
        inputs = dict(kwargs)
        inputs["inputs_embeds"] = inputs_embeds
        inputs.update(self._deepstack_inputs(deepstack, inputs_embeds))
        language_model = self.runners["language_model"]
        traced = language_model.metadata.kwargs
        outputs = language_model(
            share_inputs=share_inputs,
            share_outputs=share_outputs,
            **{name: value for name, value in inputs.items() if name in traced},
        )
        fields = getattr(outputs, "__dataclass_fields__", {})
        # The model returns each modality's features too, which the language model never sees: as its encoder's
        # `pooler_output` (`image_outputs`, ...), or as `image_hidden_states` (`audio_hidden_states`, ...) where a
        # model still returns those.
        for spec in _MODALITY_SPECS:
            if features.get(spec.component) is None:
                continue
            if f"{spec.modality}_outputs" in fields:
                pooled = BaseModelOutputWithPooling(pooler_output=features[spec.component].clone())
                outputs[f"{spec.modality}_outputs"] = pooled
            if f"{spec.modality}_hidden_states" in fields:
                outputs[f"{spec.modality}_hidden_states"] = features[spec.component].clone()
        # And Granite4-Vision its per-layer ones.
        if "deepstack_features" in features and "deepstack_features" in fields:
            outputs["deepstack_features"] = [
                layer_features.clone() for layer_features in features["deepstack_features"]
            ]
        return outputs

    def _deepstack_inputs(self, deepstack: list, inputs_embeds: torch.Tensor) -> dict[str, Any]:
        """What a deepstack decoder adds inside its layers, from each modality's `(rows, features by layer)`, in the
        dense form its graph was traced with (`dense_deepstack`): zeros on a step without images."""
        traced = self.runners["language_model"].metadata.kwargs
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
        # Its data-dependent inputs, under the names its graph was traced with (`image_cu_seqlens` -> `cu_seqlens`)
        for key, value in kwargs.items():
            name = key.removeprefix(f"{spec.modality}_")
            if name != key and key != spec.grid_key and value is not None and name in traced:
                feed[name] = value
        for name in traced.keys() - feed.keys():
            if (value := _find_config_attr(self.config, name)) is not None:
                feed[name] = value
        feed = cast_leaf_tensors(feed, dtype=runner.dtype, device=runner.device)
        outputs = runner(share_outputs=True, **{name: value for name, value in feed.items() if name in traced})
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
