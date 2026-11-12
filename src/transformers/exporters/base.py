# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
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
"""Abstract base class for all Transformers exporters."""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..generation import GenerationConfig
from ..models.auto.configuration_auto import AutoConfig
from ..utils import CONFIG_NAME, GENERATION_CONFIG_NAME, PushToHubMixin, logging
from ..utils.hub import cached_file, cached_files, hf_api
from ..utils.import_utils import is_torch_available
from .configs import ExportConfigMixin, ExportFormat
from .metadata import ExportMetadata, register_treespec_types
from .quantizers.utils import capture_calibration_inputs
from .utils import check_packages, decompose_for_generation


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ..cache_utils import Cache
    from ..configuration_utils import PreTrainedConfig
    from ..modeling_utils import PreTrainedModel
    from .runners.base import ModelRunner


logger = logging.get_logger(__name__)

# Backend, file and recorded metadata per component. Kept outside the programs because not every format
# can carry the metadata (`torch.export` cannot).
EXPORT_MANIFEST_FILE = "export.json"
EXPORT_SCHEMA_VERSION = 1


@dataclass
class ExportedComponent:
    """One exported graph: the backend's program, and what the exporter recorded about it."""

    program: Any
    metadata: ExportMetadata


class ExportedModel(PushToHubMixin):
    """An exported model: its graphs (`components`, `{name: ExportedComponent}`) and the configs they were traced
    against, called the way you'd call the model it came from.

    [`~HfExporter.export`] produces a single `"model"` component, whose program is `.program`;
    [`~HfExporter.export_for_generation`] produces one component per generation stage.

    ```python
    >>> exported = OnnxExporter().export(model, inputs, OnnxConfig(dynamic=True))
    >>> exported.save_pretrained("exported")
    >>> outputs = ExportedModel.from_pretrained("exported", device="cuda")(**inputs)
    >>> outputs.last_hidden_state
    ```
    """

    def __init__(
        self,
        components: dict[str, ExportedComponent],
        export_format: ExportFormat | str,
        config: PreTrainedConfig | None = None,
        generation_config: GenerationConfig | None = None,
    ):
        self.components = dict(components)
        self.export_format = export_format
        self.config = config
        self.generation_config = generation_config
        self._runners: dict[str, ModelRunner] | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(format={str(self.export_format)!r}, components={list(self.components)})"

    def _single_component(self) -> str:
        """The name of the export's one component: `.program` and calling it need a single graph."""
        if len(self.components) != 1:
            raise ValueError(
                f"This export has {len(self.components)} components ({list(self.components)}): `.program` and "
                "calling it need a single-graph export (`HfExporter.export`); take a component's program from "
                "`.components[name]`."
            )
        return next(iter(self.components))

    @property
    def program(self):
        """The backend's own program, for a single-graph export."""
        return self.components[self._single_component()].program

    @property
    def runners(self) -> dict[str, ModelRunner]:
        """The runner of each component, on the backend's default device unless [`~ExportedModel.to`] picked one."""
        if self._runners is None:
            self.to()
        return self._runners

    @property
    def device(self) -> torch.device:
        """Where the export runs."""
        return next(iter(self.runners.values())).device

    @property
    def dtype(self) -> torch.dtype:
        """The precision the export was traced at."""
        return getattr(torch, next(iter(self.components.values())).metadata.export_dtype)

    def to(self, device=None, **runner_kwargs) -> ExportedModel:
        """Load the programs on their backend's runtime, on `device` (the backend's default when `None`).
        `runner_kwargs` go to the backend's runner: `providers` and `session_options` for ONNX Runtime, `config`
        (compile properties) and `torch_threads` (split the CPU with torch work around the model) for OpenVINO."""
        from .auto import get_runner_class

        runner_class = get_runner_class(self.export_format)
        self._runners = {
            name: runner_class.from_program(component.program, component.metadata, device, **runner_kwargs)
            for name, component in self.components.items()
        }
        return self

    def __call__(self, share_inputs: bool = True, share_outputs: bool = False, **kwargs) -> Any:
        """Run a single-graph export, returning what the source model returns. Kwargs the export was not traced with
        are dropped, so a processor's whole output can be passed. `share_inputs` and `share_outputs` go to the runner
        (see [`ModelRunner.__call__`]): with `share_outputs`, the outputs are valid until the next call."""
        name = self._single_component()
        runner = self.runners[name]
        traced_kwargs = runner.metadata.kwargs
        unused = [key for key in kwargs if key not in traced_kwargs]
        if unused:
            logger.warning_once(f"Dropping inputs the export was not traced with: {unused}.")
        inputs = {key: value for key, value in kwargs.items() if key in traced_kwargs}
        return runner(share_inputs=share_inputs, share_outputs=share_outputs, **inputs)

    def save_pretrained(self, save_directory: str | Path, push_to_hub: bool = False, **kwargs) -> None:
        """Write the components (one file each), the configs, and a manifest naming each component's file and the
        metadata recorded when tracing it. [`~ExportedModel.from_pretrained`] loads them back.

        Args:
            save_directory (`str` or `Path`):
                The directory to write to, created if it does not exist.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether to upload the saved files to the Hub too, to `repo_id` (defaults to the name of
                `save_directory` in your namespace).
            kwargs (`dict[str, Any]`, *optional*):
                Passed to the Hub upload: `repo_id`, `commit_message`, `token`, `private`, ...
        """
        from .auto import get_exporter_class

        exporter = get_exporter_class(self.export_format)
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", directory.name)
            repo_id = hf_api().create_repo(repo_id, exist_ok=True, **kwargs).repo_id
            files_timestamps = self._get_files_timestamps(directory)

        entries = {}
        # Written aside, then moved over the old files: a program loaded from this directory may still read them
        # (OpenVINO maps its weights), and a moved file leaves the one it replaces readable.
        with tempfile.TemporaryDirectory(dir=directory) as staging:
            for name, component in self.components.items():
                filename = f"{exporter.program_prefix}{name}{exporter.program_suffix}"
                exporter.save_program(component.program, Path(staging) / filename)
                entries[name] = {"file": filename, "metadata": dataclasses.asdict(component.metadata)}
            for file in Path(staging).iterdir():
                os.replace(file, directory / file.name)

        manifest = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "export_format": str(self.export_format),
            "components": entries,
        }
        (directory / EXPORT_MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if self.config is not None:
            self.config.save_pretrained(directory)
        if self.generation_config is not None:
            self.generation_config.save_pretrained(directory)
        if push_to_hub:
            self._upload_modified_files(
                directory, repo_id, files_timestamps, commit_message=commit_message, token=kwargs.get("token")
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        device=None,
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        subfolder: str = "",
        trust_remote_code: bool = False,
        **runner_kwargs,
    ) -> ExportedModel:
        """Load what [`~ExportedModel.save_pretrained`] wrote: each component through its backend's
        [`~HfExporter.load_program`], with its metadata, and the configs saved next to them. The programs are loaded
        on their backend's runtime right away, ready to run.

        Args:
            pretrained_model_name_or_path (`str` or `Path`):
                A local directory, or a Hub repo id. From the Hub, only the files the manifest names are
                downloaded, so an export can sit next to the model's own weights.
            device (`str` or `torch.device`, *optional*):
                Where to run it (the backend's default when `None`).
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `hf auth login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

                </Tip>
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether to load the config of a model whose code lives in its repo, as for
                [`~AutoConfig.from_pretrained`].
            runner_kwargs (*optional*):
                The backend's runner options, as for [`~ExportedModel.to`]: `providers` and `session_options` for
                ONNX Runtime, `config` and `torch_threads` for OpenVINO.
        """
        from .auto import EXPORT_BACKENDS, get_exporter_class, get_runner_class

        download_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "local_files_only": local_files_only,
            "token": token,
            "revision": revision,
            "subfolder": subfolder,
        }
        manifest_file = cached_file(pretrained_model_name_or_path, EXPORT_MANIFEST_FILE, **download_kwargs)
        manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
        if manifest.get("schema_version", 1) > EXPORT_SCHEMA_VERSION:
            logger.warning(
                f"This export was saved in a newer format (schema {manifest['schema_version']}) than this version of "
                f"transformers reads ({EXPORT_SCHEMA_VERSION}): some fields might be ignored or misinterpreted."
            )
        exporter = get_exporter_class(manifest["export_format"])
        has_runner = EXPORT_BACKENDS[exporter.export_format].runner is not None
        runner_options = []
        if has_runner:
            parameters = inspect.signature(get_runner_class(exporter.export_format).from_program).parameters
            runner_options = [name for name in parameters if name not in ("program", "metadata", "device")]
        unknown = [name for name in runner_kwargs if name not in runner_options]
        if unknown:
            raise TypeError(f"Unknown kwargs {unknown}: the {exporter.export_format} runner takes {runner_options}.")
        program_files = [
            file for entry in manifest["components"].values() for file in exporter.program_files(entry["file"])
        ]
        cached_files(pretrained_model_name_or_path, program_files, **download_kwargs)
        cached_files(
            pretrained_model_name_or_path,
            [CONFIG_NAME, GENERATION_CONFIG_NAME],
            _raise_exceptions_for_missing_entries=False,
            **download_kwargs,
        )
        directory = Path(manifest_file).parent
        components = {}
        for name, entry in manifest["components"].items():
            metadata = ExportMetadata.from_dict(entry["metadata"])
            # A program whose inputs or outputs hold a cache can't be read back until the cache is a pytree node.
            for spec in (metadata.in_spec, metadata.out_spec):
                if spec is not None:
                    register_treespec_types(spec)
            components[name] = ExportedComponent(exporter.load_program(directory / entry["file"]), metadata)
        config = (
            AutoConfig.from_pretrained(directory, trust_remote_code=trust_remote_code)
            if (directory / CONFIG_NAME).exists()
            else None
        )
        generation_config = (
            GenerationConfig.from_pretrained(directory) if (directory / GENERATION_CONFIG_NAME).exists() else None
        )
        model = cls(components, exporter.export_format, config=config, generation_config=generation_config)
        if has_runner:
            model.to(device, **runner_kwargs)
        return model


class HfExporter(ABC):
    """
    Abstract base class for all Transformers exporters.

    To add a backend, subclass and implement [`~HfExporter.export_program`] to trace one graph, and
    [`~HfExporter.save_program`] / [`~HfExporter.load_program`] to write one out and read it back. The public
    [`~HfExporter.export`] and [`~HfExporter.export_for_generation`] build an [`ExportedModel`] on top of
    them.
    """

    export_format: ExportFormat | str
    config_class: type[ExportConfigMixin]
    # A saved component's file is `<program_prefix><name><program_suffix>` (`model.onnx`, `openvino_decode.xml`).
    program_suffix: str
    # Only for formats whose extension doesn't name them (OpenVINO's `.xml` / `.bin`).
    program_prefix: str = ""

    required_packages: list[str] = []
    # Hard minimum versions — the exporter raises below these (features it relies on are absent).
    min_versions: dict[str, str] = {}
    # Versions the exporter is validated against — a mismatch only warns.
    tested_versions: dict[str, str] = {}

    def __init__(self):
        self.validate_environment()

    def validate_environment(self):
        """Check `required_packages` are installed and warn on version drift from `tested_versions`."""
        check_packages(self.__class__.__name__, self.required_packages, self.min_versions, self.tested_versions)

    @abstractmethod
    def export_program(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, torch.Tensor | Cache],
        config: ExportConfigMixin,
    ) -> tuple[Any, ExportMetadata]:
        """
        Trace one graph. The backend extension point, paired with [`~HfExporter.save_program`].

        Args:
            model ([`PreTrainedModel`]):
                The model to export.
            sample_inputs (`dict[str, torch.Tensor | Cache]`):
                **Forward** kwargs — what you'd pass to `model(**sample_inputs)`. These are used
                directly as the example inputs during tracing. For an autoregressive decode-step
                export, this means you need to include `cache`, etc.
                If you only have generation-style inputs, use [`~HfExporter.export_for_generation`]
                instead — it runs `model.generate` for you and exports each stage.
            config ([`~transformers.exporters.configs.ExportConfigMixin`]):
                Backend-specific configuration.

        Returns:
            `tuple[Any, ExportMetadata]`: The backend's program object (`ExportedProgram`, `onnx_ir.Model`,
            `ExecutorchProgramManager`, `openvino.Model`) and the metadata recorded while tracing it
            ([`~exporters.metadata.build_export_metadata`]), returned separately since only some formats can
            carry it.
        """

    @classmethod
    @abstractmethod
    def save_program(cls, program: Any, path: str | Path) -> None:
        """Write one program returned by [`~HfExporter.export_program`] to `path`, in the backend's file format."""

    @classmethod
    @abstractmethod
    def load_program(cls, path: str | Path) -> Any:
        """Read back a program written by [`~HfExporter.save_program`]. It can differ from what
        [`~HfExporter.export_program`] returned: a runtime `Program` for ExecuTorch, whose `ExecutorchProgramManager`
        can't be rebuilt from the file."""

    @classmethod
    def program_files(cls, filename: str) -> list[str]:
        """The files [`~HfExporter.save_program`] writes for a program saved as `filename`: the file itself, plus
        any the format keeps beside it (weights, external data)."""
        return [filename]

    def _export_component(self, model, sample_inputs, config: ExportConfigMixin | dict[str, Any]) -> ExportedComponent:
        if isinstance(config, dict):
            config = self.config_class(**config)
        elif not isinstance(config, self.config_class):
            raise TypeError(f"Expected config to be a {self.config_class.__name__} or dict, got {type(config)}")
        program, metadata = self.export_program(model, sample_inputs, config=config)
        metadata.export_config = json.loads(config.to_json_string())
        return ExportedComponent(program, metadata)

    def export(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, torch.Tensor | Cache],
        config: ExportConfigMixin | dict[str, Any],
    ) -> ExportedModel:
        """
        Export the model as one graph.

        Takes the same **forward** kwargs as [`~HfExporter.export_program`].

        Returns:
            [`ExportedModel`]: A single `"model"` component; the backend's program is `.program`, and
            `save_pretrained` writes it with its metadata and the model config.
        """
        component = self._export_component(model, sample_inputs, config)
        # A decomposed component (e.g. `FSMTEncoder`) is a plain `nn.Module` with no config.
        return ExportedModel({"model": component}, self.export_format, config=getattr(model, "config", None))

    def export_for_generation(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, torch.Tensor | Cache],
        config: ExportConfigMixin | dict[str, ExportConfigMixin],
        *,
        generation_config: GenerationConfig | None = None,
        multi_token_decode: bool = False,
    ) -> ExportedModel:
        """
        Decompose a generative model and export each component independently.

        Thin wrapper around [`~exporters.utils.decompose_for_generation`] that calls
        [`~HfExporter.export_program`] on every returned `(submodel, forward_inputs)` pair. If you need
        the intermediate `(submodel, forward_inputs)` pairs (for verification, custom inputs,
        skipping a stage, …), call [`~exporters.utils.decompose_for_generation`] directly.

        Args:
            model ([`PreTrainedModel`]):
                The generative model to export. Must support `model.generate(**sample_inputs)`.
            sample_inputs (`dict[str, torch.Tensor | Cache]`):
                **Generate** kwargs — what you'd pass to `model.generate(**sample_inputs)`
                (typically `input_ids` + `attention_mask`, plus any modality inputs like
                `pixel_values` / `input_features` for multi-modal models). Per-stage forward
                kwargs are captured internally.
            config ([`~transformers.exporters.configs.ExportConfigMixin`] or `dict[str, ExportConfigMixin]`):
                Backend-specific configuration. Pass a single config to apply to every
                component, or a `dict` keyed by component name (e.g. `"image_encoder"`,
                `"language_model"`, `"lm_head"`, `"decode"`) to override per-component —
                all component names must be present in the dict.
            generation_config ([`~generation.GenerationConfig`], *optional*):
                Forwarded to the `generate()` capture (defaults to the model's own). Pass one with
                `cache_implementation="static"` to export against a fixed-size `StaticCache`.
            multi_token_decode (`bool`, *optional*, defaults to `False`):
                Whether the `decode` component processes multiple query tokens at once — a dynamic
                query axis (prefill on an empty cache, continuation-from-past otherwise) — vs the
                classic single-token step (see [`~exporters.utils.decompose_for_generation`]). Only
                stays dynamic under a dynamic-shape export (`config.dynamic=True`).

        Returns:
            [`ExportedModel`]: One [`ExportedComponent`] per stage, with the same keys as
            [`~exporters.utils.decompose_for_generation`].
        """
        components = decompose_for_generation(
            model,
            sample_inputs,
            generation_config=generation_config,
            multi_token_decode=multi_token_decode,
        )

        if isinstance(config, dict):
            missing = set(components) - set(config)
            if missing:
                raise ValueError(
                    f"Per-component `config` dict is missing entries for: {sorted(missing)}. "
                    f"Expected one entry per component: {sorted(components)}."
                )
            configs = config
        else:
            configs = dict.fromkeys(components, config)
            quantizer = config.quantizer
            if quantizer is not None and quantizer.calibration_dataset:
                # Generate-level samples: capture each component's own forward inputs from them.
                calibration = capture_calibration_inputs(
                    model,
                    quantizer.calibration_dataset,
                    generation_config=generation_config,
                    multi_token_decode=multi_token_decode,
                )
                configs = {
                    name: dataclasses.replace(config, quantizer=quantizer.with_calibration(calibration[name]))
                    for name in components
                }

        exported: dict[str, ExportedComponent] = {}
        for name, (submodel, subinputs) in components.items():
            try:
                exported[name] = self._export_component(submodel, subinputs, configs[name])
            except Exception as e:
                raise RuntimeError(
                    f"{self.__class__.__name__}.export_program failed on component '{name}' "
                    f"(submodel={type(submodel).__name__}, input keys={list(subinputs)})."
                ) from e

        return ExportedModel(
            exported, self.export_format, config=getattr(model, "config", None), generation_config=generation_config
        )
