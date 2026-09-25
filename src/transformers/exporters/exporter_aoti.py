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
"""AOTInductor export: the `torch.export` graph, compiled ahead of time."""

from __future__ import annotations

import io
import json
from collections.abc import MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils import logging
from ..utils.import_utils import is_torch_available
from .configs import AotiConfig, ExportFormat
from .exporter_dynamo import DynamoExporter
from .metadata import EXPORT_METADATA_KEY


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch


class AotiExporter(DynamoExporter):
    """Compile a model to a standalone AOTInductor package.

    The same `torch.export` trace the [`DynamoExporter`] produces, handed to Inductor, which generates and
    compiles kernels for it and packages them as a `.pt2`. What that buys over the traced program is the
    compilation: a `torch.export` program replays its ATen graph through the ordinary eager kernels, where
    this runs generated ones, with no warm-up to pay at load.

    What it costs is portability. The package holds machine code for the device it was compiled on, so —
    unlike every other backend here — it cannot be moved afterwards, and `ModelRunner.to` refuses.
    """

    export_format = ExportFormat.AOTI
    artifact_suffix = ".pt2"

    def export_artifact(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, Any],
        config: AotiConfig | dict[str, Any],
    ) -> tuple[bytes, dict]:
        if isinstance(config, dict):
            config = AotiConfig(**config)
        elif not isinstance(config, AotiConfig):
            raise TypeError(f"Expected config to be an AotiConfig or dict, got {type(config)}")

        program, metadata = super().export_artifact(model, sample_inputs, config)
        # Into a buffer, not a file: a package is a self-contained archive, so an export that has not been
        # saved yet is these bytes and nothing on disk to keep alive or clean up.
        package = io.BytesIO()
        torch._inductor.aoti_compile_and_package(
            program,
            package_path=package,
            # `aot_inductor.metadata` is a `{str: str}` map baked into the archive and read back by the
            # loaded model, so the trace's own account of itself travels *inside* the file the way ONNX's
            # `metadata_props` and ExecuTorch's constant method do — where `torch.export.save` drops it.
            inductor_configs={
                **(config.inductor_configs or {}),
                "aot_inductor.metadata": {EXPORT_METADATA_KEY: json.dumps(metadata)},
            },
        )
        return package.getvalue(), metadata

    @classmethod
    def save_artifact(cls, artifact: bytes, path: Path) -> None:
        """Write the package out. It carries its own metadata, so there is nothing to write beside it."""
        Path(path).write_bytes(artifact)
