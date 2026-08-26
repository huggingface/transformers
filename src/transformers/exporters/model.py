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
"""Running a single exported graph: `ExportedModel`."""

from __future__ import annotations

from pathlib import Path

import torch

from ..models.auto import AutoConfig
from ..utils import logging
from ..utils.generic import ModelOutput
from .base import ModelRunner, load_export_runners, split_download_kwargs


logger = logging.get_logger(__name__)


class ExportedModel:
    """Call one exported graph the way you would call the model it came from.

    Most exports are not generative: a sequence classifier, a token classifier, an encoder, a feature
    extractor — one graph, one forward. `ExportedGenerator` is for the decomposed, cache-driven case; this
    is for everything that is just a forward pass, and it is what [`~HfExporter.export`] produces.

    Example:
        exported = OnnxExporter().export(model, inputs, OnnxConfig(dynamic=True))
        exported.save_pretrained("out/")

        classifier = ExportedModel.from_pretrained("out/")
        logits = classifier(input_ids=ids, attention_mask=mask).logits
    """

    def __init__(self, runner: ModelRunner, config=None):
        self.runner = runner
        self.config = config

    def __repr__(self) -> str:
        return f"{type(self).__name__}(runner={type(self.runner).__name__})"

    @property
    def device(self) -> torch.device:
        return torch.device(self.runner.device)

    @property
    def dtype(self) -> torch.dtype:
        return self.runner.dtype

    @property
    def input_names(self) -> tuple[str, ...]:
        """What the graph takes — the kwargs this accepts, as traced."""
        return self.runner.input_names

    def __call__(self, **kwargs) -> ModelOutput:
        """Run the graph. Kwargs the trace never saw are dropped rather than refused, so a caller can pass
        a processor's whole output the way it would to the eager model."""
        declared = set(self.input_names)
        if declared:
            unused = [name for name in kwargs if name not in declared]
            if unused:
                logger.warning_once(
                    f"Ignoring {unused}, which this graph was not traced with (it takes {sorted(declared)})."
                )
            kwargs = {name: value for name, value in kwargs.items() if name in declared}
        return ModelOutput(**self.runner(**kwargs))

    @classmethod
    def from_pretrained(cls, save_directory: str | Path, **kwargs) -> ExportedModel:
        """Load a single-component export — a local directory or a Hub repo — written by
        [`~ExporterOutput.save_pretrained`]."""
        download_kwargs, _ = split_download_kwargs(dict(kwargs))
        runners, _ = load_export_runners(save_directory, **kwargs)
        if len(runners) != 1:
            raise ValueError(
                f"{save_directory} describes {len(runners)} components ({sorted(runners)}); use "
                "`ExportedGenerator.from_pretrained` for a decomposed export, or "
                "`AutoExportedModel.from_pretrained` to pick by what was saved."
            )
        runner = next(iter(runners.values()))
        return cls(runner, AutoConfig.from_pretrained(save_directory, **download_kwargs))
