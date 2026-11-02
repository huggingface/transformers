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
from ..decompose import _MODALITY_SPECS, ModalitySpec, flatten_anyres_patches, pack_anyres_features
from ..precompute import _find_config_attr, precompute_export_inputs
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
        input_ids = kwargs["input_ids"]
        embedded = self.text_runner(input_ids=input_ids)
        inputs_embeds = embedded["inputs_embeds"] if isinstance(embedded, dict) else embedded
        features = {}
        for spec, runner in self.modality_runners:
            if all(kwargs.get(key) is None for key in spec.feature_keys):
                continue
            features[spec.component] = self._features(spec, runner, kwargs)
            mask = (input_ids == getattr(self.config, spec.token_field)).unsqueeze(-1).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(mask, features[spec.component].to(inputs_embeds))
        # M-RoPE positions come from the token ids and grids, which the language model no longer takes.
        inputs = precompute_export_inputs(self.config, kwargs)
        inputs["inputs_embeds"] = inputs_embeds
        traced = self.runner.metadata.kwargs
        outputs = self.runner(**{name: value for name, value in inputs.items() if name in traced})
        # The model returns its image features too, which the language model never sees.
        if "vision_embeddings_model" in features and "image_hidden_states" in getattr(
            outputs, "__dataclass_fields__", {}
        ):
            outputs["image_hidden_states"] = features["vision_embeddings_model"]
        return outputs

    def _features(self, spec: ModalitySpec, runner: ModelRunner, kwargs: dict) -> torch.Tensor:
        """Run one modality's graph on its own inputs, as its getter was called: features for its placeholder rows."""
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
            return outputs
        if "image_features" not in outputs:
            raise NotImplementedError(
                f"{spec.component} returns per-layer (deepstack) features, which the runtime can't feed to the language model yet."
            )
        return pack_anyres_features(self.config, outputs["image_features"], image_sizes, outputs.get("image_newline"))
