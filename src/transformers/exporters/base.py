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
import functools
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING

from packaging import version

from ..generation.configuration_utils import GenerationMode
from ..models.auto import AutoConfig
from ..utils import cached_file, logging
from ..utils.generic import ModelOutput
from ..utils.import_utils import _is_package_available, is_torch_available
from .components import ExportedComponent
from .configs import ExportConfigMixin, ExportFormat
from .decompose import decompose_for_generation
from .metadata import ExportMetadata
from .utils import runner_feed


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch

# Backend, file and recorded metadata per component. Kept outside the artifacts because not every format
# can carry the metadata (`torch.export` cannot).
EXPORT_MANIFEST_FILE = "export.json"


# Where an export's draft model for assisted generation is saved, as an export of its own.
ASSISTANT_SUBFOLDER = "assistant"


if TYPE_CHECKING:
    if is_torch_available():
        import torch

        from ..cache_utils import Cache
        from ..generation import GenerationConfig
        from ..modeling_utils import PreTrainedModel


HUB_DOWNLOAD_KWARGS = frozenset(
    {"cache_dir", "force_download", "local_files_only", "token", "revision", "subfolder", "proxies"}
)


def split_download_kwargs(kwargs: dict) -> tuple[dict, dict]:
    """Split `from_pretrained` kwargs into the ones that resolve files and the ones that build runners."""
    download = {name: kwargs.pop(name) for name in list(kwargs) if name in HUB_DOWNLOAD_KWARGS}
    return download, kwargs


def resolve_export_file(pretrained_model_name_or_path, filename: str, **download_kwargs) -> str | None:
    """Locate one of an export's files in a local directory or a Hub repo; `None` when it is absent."""
    return cached_file(
        pretrained_model_name_or_path,
        filename,
        _raise_exceptions_for_missing_entries=False,
        **download_kwargs,
    )


def read_export_manifest(pretrained_model_name_or_path, **download_kwargs) -> dict:
    """Read and check a saved export's manifest.

    An export without recorded metadata is refused: a runner would otherwise infer precision and cache
    layout from names and shapes, and silently get them wrong.
    """
    path = resolve_export_file(pretrained_model_name_or_path, EXPORT_MANIFEST_FILE, **download_kwargs)
    if path is None:
        raise OSError(
            f"No `{EXPORT_MANIFEST_FILE}` in {pretrained_model_name_or_path}. Exported models are loaded "
            "from what `ExportArtifacts.save_pretrained` wrote; to assemble runners yourself, build each one "
            "with `ModelRunner.from_pretrained` and pass them to `ExportedGenerator`."
        )
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    components = manifest.get("components") or {}
    if not components:
        raise OSError(f"{path} lists no components.")
    missing = [name for name, entry in components.items() if not entry.get("metadata")]
    if missing:
        raise OSError(
            f"{path} has no recorded export metadata for {missing}. Re-save the export with "
            "`ExportArtifacts.save_pretrained`, which writes it; loading without it would fall back to "
            "inferring precision and cache layout from the graph's names and shapes."
        )
    return manifest


def load_export_runners(pretrained_model_name_or_path, **kwargs) -> dict[str, ModelRunner]:
    """Resolve a saved export into `{component: runner}`, each handed its recorded metadata."""
    from .auto import export_backend

    download_kwargs, runner_kwargs = split_download_kwargs(kwargs)
    manifest = read_export_manifest(pretrained_model_name_or_path, **download_kwargs)
    runner_class = export_backend(manifest["export_format"], "runner")
    return {
        component: runner_class.from_pretrained(
            resolve_export_file(pretrained_model_name_or_path, entry["file"], **download_kwargs),
            export_metadata=entry["metadata"],
            **runner_kwargs,
        )
        for component, entry in manifest["components"].items()
    }


class ExportArtifacts(Mapping):
    """What an export produced: a `Mapping` over `{name: ExportedComponent}`, plus the configs they were traced
    against.

    `generation_config` declares the cache the graphs were traced against, so a load without it would build
    the wrong one.
    """

    def __init__(
        self,
        components: dict[str, ExportedComponent],
        export_format: ExportFormat,
        config: object | None = None,
        generation_config: GenerationConfig | None = None,
        assistant: ExportArtifacts | None = None,
    ):
        self.components = dict(components)
        self.export_format = export_format
        self.config = config
        self.generation_config = generation_config
        self.assistant = assistant

    def __getitem__(self, name: str) -> ExportedComponent:
        return self.components[name]

    def __iter__(self):
        return iter(self.components)

    def __len__(self) -> int:
        return len(self.components)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(format={self.export_format.value!r}, components={list(self.components)})"

    def can_generate(self) -> bool:
        """Whether these artifacts are driven through `generate`, i.e. whether there is a decode graph."""
        return "decode" in self.components

    @property
    def artifact(self):
        """The backend's own program, for a single-graph export."""
        if len(self.components) != 1:
            raise ValueError(
                f"This export has {len(self.components)} components ({list(self.components)}), so there is "
                "no single `.artifact`; index it by name and take that component's."
            )
        return next(iter(self.components.values())).artifact

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Write the components (one file each), the configs, and the manifest that makes the directory loadable.

        Reload with [`~exporters.ExportedGenerator.from_pretrained`].
        """
        from .auto import export_backend

        backend = export_backend(self.export_format, "exporter")
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)

        components = {}
        for name, component in self.components.items():
            filename = f"{name}{backend.artifact_suffix}"
            backend.save_artifact(component.artifact, directory / filename)
            components[name] = {"file": filename, "metadata": component.metadata.raw}

        (directory / EXPORT_MANIFEST_FILE).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "export_format": self.export_format.value,
                    "components": components,
                },
                indent=2,
            )
            + "\n"
        )
        if self.config is not None:
            self.config.save_pretrained(directory)
        if self.generation_config is not None:
            self.generation_config.save_pretrained(directory)
        if self.assistant is not None:
            self.assistant.save_pretrained(directory / ASSISTANT_SUBFOLDER)

    def runtime(self, **kwargs):
        """Run without going to disk: an [`ExportedGenerator`] for an export with a decode graph, else an
        [`ExportedModel`]. `kwargs` (e.g. `device=`) go to each runner, which gets the metadata a load passes."""
        from .auto import export_backend

        runner_class = export_backend(self.export_format, "runner")
        runners = {
            name: runner_class.from_artifact(component.artifact, export_metadata=component.metadata.raw, **kwargs)
            for name, component in self.components.items()
        }
        if self.can_generate():
            from .generator import ExportedGenerator

            return ExportedGenerator(runners, self.config, self.generation_config)

        return ExportedModel(next(iter(runners.values())), self.config)


class HfExporter(ABC):
    """
    Abstract base class for all Transformers exporters.

    To add a backend, subclass and implement its two halves: [`~HfExporter.export_artifact`] to trace one
    graph, and [`~HfExporter.save_artifact`] to write one out. The public [`~HfExporter.export`] and
    [`~HfExporter.export_for_generation`] build an [`ExportArtifacts`] on top of them.
    """

    export_format: ExportFormat
    artifact_suffix: str
    config_class: type

    required_packages: list[str] = []
    # Hard minimums (raise below these); `tested_versions` only warns on mismatch.
    min_versions: dict[str, str] = {}
    tested_versions: dict[str, str] = {}

    # `export_for_generation`'s default: whether a merged decode graph writes its own cross-attention cache.
    decoder_writes_cross_cache: bool = False

    def __init__(self):
        self.validate_environment()

    def _as_config(self, config):
        """`config` as this exporter's `config_class`, built from a dict when handed one."""
        if isinstance(config, dict):
            return self.config_class(**config)
        if not isinstance(config, self.config_class):
            raise TypeError(f"Expected config to be a {self.config_class.__name__} or dict, got {type(config)}")
        return config

    def validate_environment(self, *args, **kwargs):
        """Check `required_packages` are installed and warn on version drift from `tested_versions`."""
        # Local-version suffixes (`+cu126`) are stripped: patches target the public API, not the build.
        missing, drift = [], []
        for pkg in self.required_packages:
            exists, installed = _is_package_available(pkg, return_version=True)
            if not exists:
                missing.append(pkg)
                continue
            tested = self.tested_versions.get(pkg)
            if tested is not None and installed != "N/A":
                installed_base = installed.split("+", 1)[0]
                tested_base = tested.split("+", 1)[0]
                if installed_base != tested_base:
                    drift.append((pkg, installed_base, tested_base))

        if missing:
            specs = ", ".join(
                f"{pkg}=={self.tested_versions[pkg]}" if pkg in self.tested_versions else pkg for pkg in missing
            )
            raise ImportError(f"To use {type(self).__name__}, please install the following dependencies: {specs}")

        outdated = []
        for pkg, minimum in self.min_versions.items():
            _, installed = _is_package_available(pkg, return_version=True)
            if installed == "N/A" or version.parse(installed.split("+", 1)[0]) < version.parse(minimum):
                outdated.append(f"{pkg}>={minimum} (found {installed})")
        if outdated:
            raise ImportError(f"{type(self).__name__} requires newer versions of: {', '.join(outdated)}")

        if drift:
            details = ", ".join(f"{pkg}: installed {got}, tested {want}" for pkg, got, want in drift)
            logger.warning(
                f"{type(self).__name__} is experimental and patches many backend internals; "
                f"behaviour may differ from what was validated. Version drift detected — {details}. "
                f"If you hit issues, try the tested versions."
            )

    @abstractmethod
    def export_artifact(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, torch.Tensor | Cache],
        config: ExportConfigMixin,
    ) -> tuple[object, dict]:
        """Trace one graph. The backend extension point, paired with [`~HfExporter.save_artifact`].

        Returns `(artifact, metadata)`: the backend's program object and what the trace recorded about it
        (`build_export_metadata`), returned separately since only some formats can carry it.

        Args:
            model ([`PreTrainedModel`]):
                The model to export.
            sample_inputs (`dict[str, torch.Tensor | Cache]`):
                **Forward** kwargs, as passed to `model(**sample_inputs)`; a decode step includes
                `past_key_values`, `cache_position`, etc. For generate kwargs, use
                [`~HfExporter.export_for_generation`].
            config ([`~transformers.exporters.configs.ExportConfigMixin`]):
                Backend-specific configuration.
        """

    def export(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, torch.Tensor | Cache],
        config: ExportConfigMixin,
    ) -> ExportArtifacts:
        """Export the model as one graph, as an [`ExportArtifacts`] that can save and run itself.

        Takes the same **forward** kwargs as [`~HfExporter.export_artifact`]; the backend's program is
        `output.artifact`.
        """
        artifact, metadata = self.export_artifact(model, sample_inputs, config)
        component = ExportedComponent(artifact, ExportMetadata.from_dict(metadata))
        # A decomposed component (e.g. `FSMTEncoder`) is a plain `nn.Module` with no config.
        return ExportArtifacts({"model": component}, self.export_format, config=getattr(model, "config", None))

    def export_for_generation(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, torch.Tensor | Cache],
        config: ExportConfigMixin | dict[str, ExportConfigMixin],
        *,
        generation_config: GenerationConfig | None = None,
        assistant_model: PreTrainedModel | None = None,
        decoder_writes_cross_cache: bool | None = None,
        multi_token_decode: bool | None = None,
        keep_all_logits: bool | None = None,
    ) -> ExportArtifacts:
        """Decompose a generative model into the components a generation loop drives, and export each.

        The model is run on **generate** kwargs to capture what each component is called with.

        Args:
            model ([`PreTrainedModel`]):
                The generative model to export. Must support `model.generate(**sample_inputs)`.
            sample_inputs (`dict[str, torch.Tensor | Cache]`):
                **Generate** kwargs: typically `input_ids` + `attention_mask`, plus any modality inputs.
            config ([`~transformers.exporters.configs.ExportConfigMixin`] or `dict[str, ExportConfigMixin]`):
                Backend settings: one for all components, or a dict naming every component.
            generation_config ([`~generation.GenerationConfig`], *optional*):
                The config the capture generates under, which fixes the cache the graphs are traced against.
                Saved with the export.
            assistant_model ([`PreTrainedModel`], *optional*):
                A draft model for assisted generation, exported alongside on the same `sample_inputs`, returned as
                [`ExportArtifacts.assistant`] and saved in its `assistant/` subfolder. Pass its runtime to `generate`
                as `assistant_model`, as with any draft. With a per-component `config` dict, its config is the
                `"assistant"` entry.
            decoder_writes_cross_cache (`bool`, *optional*):
                For an encoder-decoder with a multi-token decode: whether the decode graph computes its own
                cross-attention cache instead of the encoder graph. Defaults to the exporter class attribute.
            multi_token_decode (`bool`, *optional*):
                Whether the decode graph takes several query tokens at once, so it also serves the prompt.
                Defaults to whether the decode config is dynamic; refused on a static one. Pass `False` to keep a
                single-token decode beside a prompt graph.
            keep_all_logits (`bool`, *optional*):
                Whether the graphs return logits for every position, which assisted (speculative) decoding needs.
                Defaults to whether an `assistant_model` is given or `generation_config` asks for assisted generation
                (prompt lookup, early exit, MTP); otherwise only the last position, which is all greedy and sampled
                decoding read.
        """
        generation_config, decoder_writes_cross_cache, multi_token_decode, keep_all_logits = (
            self._resolve_generation_options(
                model,
                config,
                generation_config=generation_config,
                assistant_model=assistant_model,
                decoder_writes_cross_cache=decoder_writes_cross_cache,
                multi_token_decode=multi_token_decode,
                keep_all_logits=keep_all_logits,
            )
        )

        parts = decompose_for_generation(
            model,
            sample_inputs,
            generation_config=generation_config,
            multi_token_decode=multi_token_decode,
            decoder_writes_cross_cache=decoder_writes_cross_cache,
        )
        if keep_all_logits:
            # Without the captured `logits_to_keep` the forward defaults to 0, i.e. every position
            for part in parts.values():
                part.inputs.pop("logits_to_keep", None)
        if isinstance(config, dict):
            missing = set(parts) - set(config)
            if missing:
                raise ValueError(
                    f"Per-component `config` dict is missing entries for: {sorted(missing)}. "
                    f"Expected one entry per component: {sorted(parts)}."
                )
            configs = config
        else:
            configs = dict.fromkeys(parts, config)

        components: dict[str, ExportedComponent] = {}
        for name, part in parts.items():
            try:
                artifact, metadata = self.export_artifact(part.module, part.inputs, config=configs[name])
            except Exception as e:
                raise RuntimeError(
                    f"{type(self).__name__}.export_artifact failed on component '{name}' "
                    f"(submodel={type(part.module).__name__}, input keys={list(part.inputs)})."
                ) from e
            components[name] = ExportedComponent(artifact, ExportMetadata.from_dict(metadata))

        assistant = None
        if assistant_model is not None:
            if isinstance(config, dict) and "assistant" not in config:
                raise ValueError('A per-component `config` dict needs an `"assistant"` entry for `assistant_model`.')
            assistant = self.export_for_generation(
                assistant_model,
                sample_inputs,
                config["assistant"] if isinstance(config, dict) else config,
                multi_token_decode=multi_token_decode,
            )
        return ExportArtifacts(
            components,
            self.export_format,
            config=getattr(model, "config", None),
            generation_config=generation_config,
            assistant=assistant,
        )

    def _resolve_generation_options(
        self,
        model,
        config,
        *,
        generation_config: GenerationConfig | None,
        assistant_model: PreTrainedModel | None,
        decoder_writes_cross_cache: bool | None,
        multi_token_decode: bool | None,
        keep_all_logits: bool | None,
    ) -> tuple[GenerationConfig | None, bool, bool, bool]:
        """Resolve and check `export_for_generation`'s `None` options.

        `generation_config` is filled from the model's own (e.g. `forced_eos_token_id`), as `generate` does;
        the runtime only gets this config, so without the merge beam search diverges from eager.
        """
        if generation_config is not None and getattr(model, "generation_config", None) is not None:
            generation_config = copy.deepcopy(generation_config)
            generation_config.update(
                **model.generation_config.to_dict(), defaults_only=True, allow_custom_entries=True
            )
        if decoder_writes_cross_cache is None:
            decoder_writes_cross_cache = self.decoder_writes_cross_cache
        decode_config = config.get("decode") if isinstance(config, dict) else config
        dynamic = bool(getattr(decode_config, "dynamic", False))
        if multi_token_decode is None:
            multi_token_decode = dynamic
        elif multi_token_decode and not dynamic:
            raise ValueError(
                "`multi_token_decode=True` needs a dynamic export: a static one freezes the decode graph's query axis "
                "at the captured length, which no decode step feeds."
            )
        if generation_config is not None and generation_config.use_mtp:
            raise ValueError(
                "`use_mtp` is not supported for exported models: the MTP drafter reads the main model's hidden "
                "states and loads its own weights from it, and an exported decode graph returns logits only."
            )
        if keep_all_logits is None:
            keep_all_logits = assistant_model is not None or (
                generation_config is not None
                and generation_config.get_generation_mode() == GenerationMode.ASSISTED_GENERATION
            )
        return generation_config, decoder_writes_cross_cache, multi_token_decode, keep_all_logits

    @classmethod
    @abstractmethod
    def save_artifact(cls, artifact, path: Path) -> None:
        """Write one exported component to `path` in the backend's file format."""


class ModelRunner(ABC):
    """Wraps one exported artifact so it forwards like its source module: `runner(**kwargs) -> {name: tensor}`.

    `kwargs` are the exported forward's kwargs, with `past_key_values` as a `Cache` for a decode graph; each
    backend adapts it to how its graph carries the cache. Outputs are keyed by leaf name (`logits`,
    `past_key_values.…`) on every backend.
    """

    # Answers what the handle cannot (precision, cache geometry); where both can, the handle wins.
    export_metadata: ExportMetadata = ExportMetadata()

    # Whether the graph keeps its cache as internal state (OpenVINO) instead of taking and returning it.
    owns_state: bool = False
    # Cache leaves kept as state rather than inputs.
    state_paths: frozenset[str] = frozenset()

    @functools.cached_property
    def input_names(self) -> tuple[str, ...]:
        """The graph's inputs in flat order, as recorded at export; runners whose handle names them assign it."""
        return self.export_metadata.input_names

    @functools.cached_property
    def device(self) -> torch.device:
        """Where this graph runs and its outputs land: recorded at export, else CPU; runners may assign it."""
        return self.export_metadata.device or torch.device("cpu")

    @functools.cached_property
    def dtype(self) -> torch.dtype:
        """Precision the graph computes at, as recorded at export; runners may assign it.

        The cache is sized from this, so it is refused rather than guessed: fp32 cache leaves fed to a
        half-precision export are rejected by the backend.
        """
        if dtype := self.export_metadata.dtype:
            return dtype
        raise ValueError(
            "This runner was built without its export metadata, so the precision it was exported at is "
            "unknown, and guessing it silently corrupts a half-precision export. Load it with "
            "`ExportedGenerator.from_pretrained` / `AutoExportedModel.from_pretrained`, or run it from the "
            "`ExportArtifacts` the export returned — both carry the metadata."
        )

    @functools.cached_property
    def cache_inputs(self) -> tuple[str, ...]:
        """Every kwarg this graph takes a cache under, in traced order.

        Usually one; voxtral_realtime's decode also takes `encoder_past_key_values`. Read from the recorded
        kwargs, else matched against each backend's input naming.
        """
        recorded = self.export_metadata.kwargs
        if recorded:
            containers = tuple(name for name, spec in recorded.items() if spec.get("container") == "cache")
            # A model-specific state class (xlstm) records its own class name, so empty falls through to the name scan.
            if containers:
                return containers
        declared = tuple(
            kwarg
            for kwarg in ("cache_params", "past_key_values")
            if any(name.removeprefix("input.").startswith(kwarg) for name in self.input_names)
        )
        # A stateful graph's cache is named only by its state paths; without them it would never be fed.
        # A graph can fold one cache and take another as input, hence the union.
        folded = tuple(path.split(".", 1)[0] for path in sorted(self.state_paths) if "." in path)
        return tuple(dict.fromkeys(declared + folded))

    def declares(self, name: str, value=None) -> bool:
        """Whether this graph takes the feed entry `name`, directly or as the pytree whose leaves it names.

        Backends name flattened leaves differently (`encoder_outputs.last_hidden_state` for ONNX,
        `encoder_outputs_last_hidden_state` for ExecuTorch). Only a non-tensor is flattened, so a tensor must
        be named outright. Every caller should use this rather than an exact name match.
        """
        if name in self.input_names or name in self.cache_inputs:
            return True
        return not isinstance(value, torch.Tensor) and any(
            declared.removeprefix("input.").startswith((f"{name}.", f"{name}_")) for declared in self.input_names
        )

    @functools.cached_property
    def cache_input(self) -> str | None:
        """The kwarg of the primary cache the generation loop feeds (`"cache_params"`, `"past_key_values"`), or
        `None`."""
        return self.cache_inputs[0] if self.cache_inputs else None

    def to(self, device) -> ModelRunner:
        """The runner to use for `device`; returned rather than moved in place since some backends must reopen
        (ORT) or cannot move at all (ExecuTorch)."""
        raise ValueError(
            f"{type(self).__name__} is bound to {self.device} by the runtime that loaded it. Load the "
            f"artifact again with `device={device!r}` to run it elsewhere."
        )

    @classmethod
    def from_artifact(cls, artifact, export_metadata=None, **kwargs) -> ModelRunner:
        """Build the runner from an in-memory artifact, as `ExportArtifacts.runtime()` does."""
        raise NotImplementedError(
            f"{cls.__name__} cannot be built from an in-memory artifact. Implement `from_artifact` to run "
            "an export without saving it first."
        )

    @classmethod
    def from_pretrained(cls, path: str | Path, **kwargs) -> ModelRunner:
        """Build the runner from one saved artifact, the inverse of [`~HfExporter.save_artifact`]."""
        raise NotImplementedError(
            f"{cls.__name__} cannot be loaded from disk. Implement `from_pretrained` to build it from a "
            "saved artifact."
        )

    @abstractmethod
    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        """Run the graph on `kwargs`; return its outputs as `{leaf_name: tensor}`."""


class ExportedModel:
    """Call one exported graph the way you would call the model it came from.

    For single-forward exports (classifiers, encoders) produced by [`~HfExporter.export`]; the decomposed,
    cache-driven case is `ExportedGenerator`.

    Example:
        exported_artifacts = OnnxExporter().export(model, inputs, OnnxConfig(dynamic=True))
        exported_artifacts.save_pretrained("out/")

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
        return self.runner.device

    @property
    def dtype(self) -> torch.dtype:
        return self.runner.dtype

    def can_generate(self) -> bool:
        """`False`: one graph is one forward."""
        return False

    @property
    def input_modalities(self) -> list[str] | str:
        """The source model's input modalities, read off the config."""
        return getattr(self.config, "input_modalities", "text")

    def to(self, device) -> ExportedModel:
        """Move to `device` if this graph's backend can."""
        if torch.device(device) != self.device:
            self.runner = self.runner.to(device)
        return self

    @property
    def input_names(self) -> tuple[str, ...]:
        """The kwargs the graph takes, as traced."""
        return self.runner.input_names

    def __call__(self, **kwargs) -> ModelOutput:
        """Run the graph. Kwargs the trace never saw are dropped, so a processor's whole output can be passed."""
        return ModelOutput(**self.runner(**runner_feed(self.runner, kwargs, warn_unused=True)))

    @classmethod
    def from_pretrained(cls, save_directory: str | Path, **kwargs) -> ExportedModel:
        """Load a single-component export (local directory or Hub repo) written by
        [`~ExportArtifacts.save_pretrained`]."""
        download_kwargs, _ = split_download_kwargs(dict(kwargs))
        runners = load_export_runners(save_directory, **kwargs)
        if len(runners) != 1:
            raise ValueError(
                f"{save_directory} describes {len(runners)} components ({sorted(runners)}); use "
                "`ExportedGenerator.from_pretrained` for a decomposed export, or "
                "`AutoExportedModel.from_pretrained` to pick by what was saved."
            )
        runner = next(iter(runners.values()))
        return cls(runner, AutoConfig.from_pretrained(save_directory, **download_kwargs))
