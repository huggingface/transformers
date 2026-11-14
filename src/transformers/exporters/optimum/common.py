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
"""What the optimum libraries' loaders share: finding an export's files, and the generation mixin every optimum export
runs with, supplying what its decode graph takes that `generate` doesn't give it."""

from __future__ import annotations

from pathlib import Path
from typing import Unpack

from ...utils.hub import DownloadKwargs, cached_file, hf_api
from ...utils.import_utils import is_torch_available


if is_torch_available():
    import torch


def _find(pretrained_model_name_or_path, files, **kwargs: Unpack[DownloadKwargs]) -> str | None:
    """The first of `files` the export holds."""
    for file in files:
        if cached_file(pretrained_model_name_or_path, file, _raise_exceptions_for_missing_entries=False, **kwargs):
            return file
    return None


def _architecture(config) -> str | None:
    return (config.architectures or [None])[0]


def _repo_files(pretrained_model_name_or_path, **kwargs: Unpack[DownloadKwargs]) -> list[str]:
    """Every file the export holds, by its path in it."""
    if Path(pretrained_model_name_or_path).is_dir():
        root = Path(pretrained_model_name_or_path)
        return [str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()]
    return hf_api().list_repo_files(
        pretrained_model_name_or_path, revision=kwargs.get("revision"), token=kwargs.get("token")
    )


class _OptimumGenerationMixin:
    """What any optimum export's decode graph takes that `generate` doesn't give it: an all-ones padding mask, which
    `generate` drops (optimum-intel's and optimum-onnx's causal LMs take one), and `cache_position`, which it no longer
    passes (optimum-executorch's graphs take one). Each library's own mixin goes before it."""

    def _validate_model_kwargs(self, model_kwargs: dict):
        """Also accept a processor's token types (Gemma 3's), which a graph with its attention built in doesn't take."""
        if "token_type_ids" not in self._decode_metadata().kwargs:
            model_kwargs = {key: value for key, value in model_kwargs.items() if key != "token_type_ids"}
        super()._validate_model_kwargs(model_kwargs)

    def prepare_inputs_for_generation(self, *args, attention_mask=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, attention_mask=attention_mask, **kwargs)
        traced = self._decode_metadata().kwargs
        # The running sequence, of which the step is the tail
        sequence = args[0] if args else kwargs["input_ids"]
        step = model_inputs.get("decoder_input_ids" if self.config.is_encoder_decoder else "input_ids")
        mask = traced.get("attention_mask", {})
        if not self.config.is_encoder_decoder and mask.get("rank") == 2 and model_inputs.get("attention_mask") is None:
            model_inputs["attention_mask"] = torch.ones(
                sequence.shape, dtype=getattr(torch, mask["dtype"]), device=sequence.device
            )
        if "cache_position" in traced and step is not None:
            model_inputs["cache_position"] = torch.arange(
                sequence.shape[1] - step.shape[1], sequence.shape[1], device=step.device
            )
        return model_inputs
