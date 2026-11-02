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

import copy
import dataclasses
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Unpack

from ..generation import GenerationConfig
from ..models.auto.configuration_auto import AutoConfig
from ..utils import CONFIG_NAME, GENERATION_CONFIG_NAME, PushToHubMixin, logging
from ..utils.hub import DownloadKwargs, cached_file, cached_files, hf_api
from .components import Component
from .configs import ExportConfigMixin, ExportFormat
from .decompose import decode_graph_name, decompose_for_generation, decompose_multimodal
from .metadata import ExportMetadata, export_config_metadata, register_treespec_types
from .quantizers.utils import capture_calibration_inputs
from .runtimes.base import ExportedModel
from .utils import check_packages


if TYPE_CHECKING:
    import torch

    from ..cache_utils import Cache
    from ..configuration_utils import PreTrainedConfig
    from ..modeling_utils import PreTrainedModel


logger = logging.get_logger(__name__)

# Backend, file and recorded metadata per component. Kept outside the artifacts because not every format
# can carry the metadata (`torch.export` cannot).
EXPORT_MANIFEST_FILE = "export.json"


@dataclass
class ExportedComponent:
    """One exported graph: the backend's program, and what the exporter recorded about it."""

    program: Any
    metadata: ExportMetadata


class ExportArtifacts(Mapping, PushToHubMixin):
    """What an export produced: a `Mapping` over `{name: ExportedComponent}`, plus the configs they were traced
    against.

    [`~HfExporter.export`] produces a single `"model"` component, whose program is `.program`;
    [`~HfExporter.export_multimodal`] one component per modality, plus the text embeddings and the model on
    `inputs_embeds`; [`~HfExporter.export_for_generation`] one component per generation stage.
    """

    def __init__(
        self,
        components: dict[str, ExportedComponent],
        export_format: ExportFormat | str,
        config: PreTrainedConfig | None = None,
        generation_config: GenerationConfig | None = None,
        generation: bool = False,
    ):
        self.components = dict(components)
        self.export_format = export_format
        self.config = config
        self.generation_config = generation_config
        self.generation = generation

    def __getitem__(self, name: str) -> ExportedComponent:
        return self.components[name]

    def __iter__(self):
        return iter(self.components)

    def __len__(self) -> int:
        return len(self.components)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(format={str(self.export_format)!r}, components={list(self)})"

    @property
    def program(self):
        """The backend's own program, for a single-graph export."""
        if len(self.components) != 1:
            raise ValueError(
                f"This export has {len(self.components)} components ({list(self.components)}), so there is "
                "no single `.program`; index it by name and take that component's."
            )
        return next(iter(self.components.values())).program

    def runtime(self, device=None, **runner_kwargs) -> ExportedModel:
        """Run this export on `device` (the backend's default when `None`), as an [`ExportedModel`]: a single graph
        ([`~HfExporter.export`]), or a multimodal split ([`~HfExporter.export_multimodal`]) as an
        [`ExportedMultimodalModel`]. `runner_kwargs` go to the backend's runners: `providers` and `session_options`
        for ONNX Runtime, `config` (compile properties) for OpenVINO."""
        from .auto import get_runner_class
        from .runtimes.multimodal import MULTIMODAL_COMPONENTS, ExportedMultimodalModel

        multimodal = {"language_model", "text_embeddings_model"} <= self.components.keys() <= MULTIMODAL_COMPONENTS
        if self.generation:
            raise ValueError(
                f"This is a generation export ({list(self.components)}), run by a generation loop; that runtime isn't "
                "available yet. `runtime()` runs a single graph (`HfExporter.export`) or a multimodal split "
                "(`HfExporter.export_multimodal`)."
            )
        if len(self.components) != 1 and not multimodal:
            raise ValueError(
                f"This export has components {list(self.components)}, whose runtime isn't available yet. `runtime()` "
                "runs a single graph (`HfExporter.export`) or a multimodal split (`HfExporter.export_multimodal`)."
            )
        runner_class = get_runner_class(self.export_format)
        runners = {
            name: runner_class.from_program(component.program, component.metadata, device, **runner_kwargs)
            for name, component in self.components.items()
        }
        if multimodal:
            model = ExportedMultimodalModel(runners, self.config)
        else:
            model = ExportedModel(next(iter(runners.values())), self.config)
        model.artifacts = self
        return model

    def save_pretrained(self, save_directory: str | Path, push_to_hub: bool = False, **kwargs) -> None:
        """Write the components (one file each), the configs, and a manifest naming each component's file and the
        metadata recorded when tracing it. [`~ExportArtifacts.from_pretrained`] loads them back.

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

        components = {}
        for name, component in self.components.items():
            filename = f"{exporter.program_prefix}{name}{exporter.program_suffix}"
            exporter.save_program(component.program, directory / filename)
            components[name] = {"file": filename, "metadata": dataclasses.asdict(component.metadata)}

        manifest = {
            "schema_version": 1,
            "export_format": str(self.export_format),
            "generation": self.generation,
            "components": components,
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
        cls, pretrained_model_name_or_path: str | Path, **kwargs: Unpack[DownloadKwargs]
    ) -> ExportArtifacts:
        """Load what [`~ExportArtifacts.save_pretrained`] wrote: each component through its backend's
        [`~HfExporter.load_program`], with its metadata, and the configs saved next to them.

        Args:
            pretrained_model_name_or_path (`str` or `Path`):
                A local directory, or a Hub repo id. From the Hub, only the files the manifest names are
                downloaded, so an export can sit next to the model's own weights.
            kwargs ([`~utils.hub.DownloadKwargs`], *optional*):
                Download kwargs, as for [`~PreTrainedModel.from_pretrained`] (`subfolder`, `revision`, `token`, ...).
        """
        from .auto import get_exporter_class

        manifest_file = cached_file(pretrained_model_name_or_path, EXPORT_MANIFEST_FILE, **kwargs)
        manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
        exporter = get_exporter_class(manifest["export_format"])
        program_files = [
            file for entry in manifest["components"].values() for file in exporter.program_files(entry["file"])
        ]
        cached_files(pretrained_model_name_or_path, program_files, **kwargs)
        cached_files(
            pretrained_model_name_or_path,
            [CONFIG_NAME, GENERATION_CONFIG_NAME],
            _raise_exceptions_for_missing_entries=False,
            **kwargs,
        )
        directory = Path(manifest_file).parent
        components = {}
        for name, entry in manifest["components"].items():
            metadata = ExportMetadata(**entry["metadata"])
            # A program whose inputs or outputs hold a cache can't be read back until the cache is a pytree node.
            for spec in (metadata.in_spec, metadata.out_spec):
                if spec is not None:
                    register_treespec_types(spec)
            components[name] = ExportedComponent(exporter.load_program(directory / entry["file"]), metadata)
        config = AutoConfig.from_pretrained(directory) if (directory / CONFIG_NAME).exists() else None
        generation_config = (
            GenerationConfig.from_pretrained(directory) if (directory / GENERATION_CONFIG_NAME).exists() else None
        )
        return cls(
            components,
            exporter.export_format,
            config=config,
            generation_config=generation_config,
            generation=manifest.get("generation", False),
        )


class HfExporter(ABC):
    """
    Abstract base class for all Transformers exporters.

    To add a backend, subclass and implement [`~HfExporter.export_program`] to trace one graph, and
    [`~HfExporter.save_program`] / [`~HfExporter.load_program`] to write one out and read it back. The public
    [`~HfExporter.export`] and [`~HfExporter.export_for_generation`] build an [`ExportArtifacts`] on top of
    them.
    """

    export_format: ExportFormat | str
    config_class: type[ExportConfigMixin]
    # A saved component's file is `<program_prefix><name><program_suffix>` (`model.onnx`, `openvino_decode.xml`).
    program_suffix: str
    # Only for formats whose extension doesn't name them (OpenVINO's `.xml` / `.bin`).
    program_prefix: str = ""
    # `export_for_generation`'s default: whether an encoder-decoder's decode graph writes its own cross-attention cache.
    decoder_writes_cross_cache: bool = False

    required_packages: list[str] = []
    # Hard minimum versions — the exporter raises below these (features it relies on are absent).
    min_versions: dict[str, str] = {}
    # Versions the exporter is validated against — a mismatch only warns.
    tested_versions: dict[str, str] = {}

    def __init__(self):
        self.validate_environment()

    def validate_environment(self, *args, **kwargs):
        """Check `required_packages` are installed and warn on version drift from `tested_versions`."""
        check_packages(self.__class__.__name__, self.required_packages, self.min_versions, self.tested_versions)

    def _as_config(self, config: ExportConfigMixin | dict[str, Any]) -> ExportConfigMixin:
        """`config` as this exporter's `config_class`, built from a dict when handed one."""
        if isinstance(config, dict):
            return self.config_class(**config)
        if not isinstance(config, self.config_class):
            raise TypeError(f"Expected config to be a {self.config_class.__name__} or dict, got {type(config)}")
        return config

    @abstractmethod
    def export_program(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, torch.Tensor | Cache],
        config: ExportConfigMixin,
    ) -> tuple[Any, dict[str, Any]]:
        """
        Trace one graph. The backend extension point, paired with [`~HfExporter.save_program`].

        Args:
            model ([`PreTrainedModel`]):
                The model to export.
            sample_inputs (`dict[str, torch.Tensor | Cache]`):
                **Forward** kwargs — what you'd pass to `model(**sample_inputs)`. These are used
                directly as the example inputs during tracing. For an autoregressive decode-step
                export, this means you need to include `past_key_values`, `cache_position`, etc.
                If you only have generation-style inputs, use [`~HfExporter.export_for_generation`]
                instead — it runs `model.generate` for you and exports each stage.
            config ([`~transformers.exporters.configs.ExportConfigMixin`]):
                Backend-specific configuration.

        Returns:
            `tuple[Any, dict]`: The backend's program object (`ExportedProgram`, `ONNXProgram`,
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
        config = self._as_config(config)
        program, metadata = self.export_program(model, sample_inputs, config=config)
        return ExportedComponent(program, ExportMetadata(**metadata, export_config=export_config_metadata(config)))

    def _export_components(
        self, components: dict[str, Component], config: ExportConfigMixin | dict[str, ExportConfigMixin]
    ) -> dict[str, ExportedComponent]:
        """Export each component with `config`, or with its entry in a per-component `config` dict."""
        if isinstance(config, dict):
            missing = set(components) - set(config)
            if missing:
                raise ValueError(
                    f"Per-component `config` dict is missing entries for: {sorted(missing)}. "
                    f"Expected one entry per component: {sorted(components)}."
                )
        exported: dict[str, ExportedComponent] = {}
        for name, component in components.items():
            component_config = config[name] if isinstance(config, dict) else config
            try:
                exported[name] = self._export_component(component.module, component.inputs, component_config)
            except Exception as e:
                raise RuntimeError(
                    f"{self.__class__.__name__}.export_program failed on component '{name}' "
                    f"(submodel={type(component.module).__name__}, input keys={list(component.inputs)})."
                ) from e
        return exported

    def export(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, torch.Tensor | Cache],
        config: ExportConfigMixin | dict[str, Any],
    ) -> ExportArtifacts:
        """
        Export the model as one graph.

        Takes the same **forward** kwargs as [`~HfExporter.export_program`].

        Returns:
            [`ExportArtifacts`]: A single `"model"` component; the backend's program is `.program`, and
            `save_pretrained` writes it with its metadata and the model config (none for a plain `nn.Module`, such
            as a decomposed `FSMTEncoder`).
        """
        component = self._export_component(model, sample_inputs, config)
        return ExportArtifacts({"model": component}, self.export_format, config=getattr(model, "config", None))

    def export_multimodal(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, torch.Tensor | Cache],
        config: ExportConfigMixin | dict[str, ExportConfigMixin],
    ) -> ExportArtifacts:
        """
        Split a multi-modal model and export each component.

        Takes **forward** kwargs, like [`~HfExporter.export`]. The model is split by
        [`~exporters.decompose.decompose_multimodal`]: the token embeddings (`"text_embeddings_model"`), one graph per
        modality present (`"vision_embeddings_model"`, `"audio_embeddings_model"`, ...), and the model itself on
        `inputs_embeds` (`"language_model"`), its head included. Scattering the features into the embeddings stays
        outside the graphs.

        Args:
            model ([`PreTrainedModel`]):
                The multi-modal model to export.
            sample_inputs (`dict[str, torch.Tensor]`):
                **Forward** kwargs, typically a processor's output: `input_ids`, `attention_mask` and the modality
                inputs (`pixel_values`, `input_features`, ...).
            config ([`~transformers.exporters.configs.ExportConfigMixin`] or `dict[str, ExportConfigMixin]`):
                Backend settings: one for all components, or a dict naming every component. With one config whose
                quantizer has a `calibration_dataset` of forward kwargs, each component calibrates on its own inputs
                to those samples.

        Returns:
            [`ExportArtifacts`]: One [`ExportedComponent`] per component.
        """
        components = decompose_multimodal(model, sample_inputs)
        quantizer = getattr(config, "quantizer", None)
        if not isinstance(config, dict) and quantizer is not None and quantizer.calibration_dataset:
            calibration: dict[str, list] = {}
            for sample in quantizer.calibration_dataset:
                for name, component in decompose_multimodal(model, sample).items():
                    calibration.setdefault(name, []).append(component.inputs)
            config = {
                name: dataclasses.replace(config, quantizer=quantizer.with_calibration(calibration[name]))
                for name in components
            }
        return ExportArtifacts(
            self._export_components(components, config),
            self.export_format,
            config=getattr(model, "config", None),
        )

    def export_for_generation(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, torch.Tensor | Cache],
        config: ExportConfigMixin | dict[str, ExportConfigMixin],
        *,
        generation_config: GenerationConfig | None = None,
        multi_token_decode: bool | None = None,
        decoder_writes_cross_cache: bool | None = None,
        keep_all_logits: bool = False,
    ) -> ExportArtifacts:
        """
        Decompose a generative model into the components a generation loop drives, and export each.

        The model is run on **generate** kwargs to capture what each component is called with
        ([`~exporters.decompose.decompose_for_generation`]). The decode graph also serves the prompt, so no prompt
        graph duplicates the model's weights.

        Args:
            model ([`PreTrainedModel`]):
                The generative model to export. Must support `model.generate(**sample_inputs)`.
            sample_inputs (`dict[str, torch.Tensor | Cache]`):
                **Generate** kwargs: typically `input_ids` + `attention_mask`, plus any modality inputs.
            config ([`~transformers.exporters.configs.ExportConfigMixin`] or `dict[str, ExportConfigMixin]`):
                Backend settings: one for all components, or a dict naming every component.
            generation_config ([`~generation.GenerationConfig`], *optional*):
                The config the capture generates under, which fixes the cache the graphs are traced against (pass
                `cache_implementation="static"` for a fixed-size `StaticCache`). Saved with the export.
            multi_token_decode (`bool`, *optional*):
                Whether the decode graph takes several query tokens at once, so it takes the whole prompt in one call.
                Defaults to whether the decode config is dynamic; refused on a static one, whose decode graph takes
                the prompt a token at a time.
            decoder_writes_cross_cache (`bool`, *optional*):
                For an encoder-decoder, whether the decode graph computes its own cross-attention cache instead of
                the encoder graph. Defaults to the exporter's `decoder_writes_cross_cache`.
            keep_all_logits (`bool`, *optional*, defaults to `False`):
                Whether the graphs return logits for every position, which assisted (speculative) decoding needs,
                instead of the last position only, which greedy and sampled decoding read.

        Returns:
            [`ExportArtifacts`]: One [`ExportedComponent`] per component, named as optimum-intel does: a causal LM's
            `"model"`; an encoder-decoder's `"encoder_model"` and `"decoder_model"`; a multi-modal model's
            `"language_model"`, `"text_embeddings_model"` and one `"<modality>_embeddings_model"` per modality.
        """
        generation_config, multi_token_decode, decoder_writes_cross_cache = self._resolve_generation_options(
            model, config, generation_config, multi_token_decode, decoder_writes_cross_cache
        )
        components = decompose_for_generation(
            model,
            sample_inputs,
            generation_config=generation_config,
            multi_token_decode=multi_token_decode,
            decoder_writes_cross_cache=decoder_writes_cross_cache,
        )
        # The decode graph serves the prefill step: it isn't a graph of its own.
        components.pop("prefill")
        if keep_all_logits:
            # Without the captured `logits_to_keep` the forward defaults to 0, i.e. every position.
            for component in components.values():
                component.inputs.pop("logits_to_keep", None)

        if not isinstance(config, dict):
            quantizer = config.quantizer
            if quantizer is not None and quantizer.calibration_dataset:
                # Generate-level samples: capture each component's own forward inputs from them.
                calibration = capture_calibration_inputs(
                    model,
                    quantizer.calibration_dataset,
                    generation_config=generation_config,
                    multi_token_decode=multi_token_decode,
                    decoder_writes_cross_cache=decoder_writes_cross_cache,
                )
                config = {
                    name: dataclasses.replace(config, quantizer=quantizer.with_calibration(calibration[name]))
                    for name in components
                }

        return ExportArtifacts(
            self._export_components(components, config),
            self.export_format,
            config=getattr(model, "config", None),
            generation_config=generation_config,
            generation=True,
        )

    def _resolve_generation_options(
        self, model, config, generation_config, multi_token_decode, decoder_writes_cross_cache
    ) -> tuple[GenerationConfig | None, bool, bool]:
        """`export_for_generation`'s defaults, checked.

        `generation_config` is completed from the model's own (e.g. `forced_eos_token_id`), as `generate` does: the
        export saves it, and a runtime generating with it alone would otherwise diverge from eager.
        """
        if generation_config is not None and getattr(model, "generation_config", None) is not None:
            generation_config = copy.deepcopy(generation_config)
            generation_config.update(
                **model.generation_config.to_dict(), defaults_only=True, allow_custom_entries=True
            )
            # The model's sampling defaults (qwen3's `temperature`) don't apply to a greedy config, and would make it
            # unsaveable: `save_pretrained` validates strictly.
            if not generation_config.do_sample:
                defaults = GenerationConfig()
                for name in ("temperature", "top_p", "top_k", "min_p", "typical_p", "epsilon_cutoff", "eta_cutoff"):
                    setattr(generation_config, name, getattr(defaults, name))
        decode_config = config.get(decode_graph_name(config)) if isinstance(config, dict) else config
        dynamic = bool(getattr(decode_config, "dynamic", False) or getattr(decode_config, "dynamic_shapes", None))
        if multi_token_decode is None:
            multi_token_decode = dynamic
        elif multi_token_decode and not dynamic:
            raise ValueError(
                "`multi_token_decode=True` needs a dynamic export: a static one freezes the decode graph's query axis "
                "at the captured length."
            )
        if decoder_writes_cross_cache is None:
            decoder_writes_cross_cache = self.decoder_writes_cross_cache
        return generation_config, multi_token_decode, decoder_writes_cross_cache
