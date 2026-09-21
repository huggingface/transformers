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

import functools
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from packaging import version

from ..models.auto import AutoConfig
from ..utils import cached_file, logging
from ..utils.generic import ModelOutput
from ..utils.import_utils import _is_package_available, is_torch_available
from .configs import ExportConfigMixin, ExportFormat
from .decompose import decompose_for_generation
from .metadata import EXPORT_METADATA_KEY, ExportMetadata
from .utils import runner_feed


logger = logging.get_logger(__name__)

# Everything a saved export says about itself: which backend wrote it, which file each component is, and
# what the exporter recorded about each component's graph. One file, because those are one question — and
# because the alternative, leaning on each backend's file format to carry the metadata, silently leaves a
# runner inferring precision and cache layout when the format cannot (`torch.export` cannot).
EXPORT_MANIFEST_FILE = "export.json"


if TYPE_CHECKING:
    if is_torch_available():
        import torch

        from ..cache_utils import Cache
        from ..generation import GenerationConfig
        from ..modeling_utils import PreTrainedModel


# What `from_pretrained` takes for the download rather than for the runner it builds.
HUB_DOWNLOAD_KWARGS = frozenset(
    {"cache_dir", "force_download", "local_files_only", "token", "revision", "subfolder", "proxies"}
)


def split_download_kwargs(kwargs: dict) -> tuple[dict, dict]:
    """Split `from_pretrained` kwargs into the ones that resolve files and the ones that build runners."""
    download = {name: kwargs.pop(name) for name in list(kwargs) if name in HUB_DOWNLOAD_KWARGS}
    return download, kwargs


def resolve_export_file(pretrained_model_name_or_path, filename: str, **download_kwargs) -> str | None:
    """Locate one of an export's files, in a local directory or a Hub repo.

    `cached_file` resolves both, so a saved export is loadable from the Hub the way a checkpoint is; returns
    `None` when the file is simply not there, which is how an optional one (a `generation_config.json`) is
    checked for.
    """
    return cached_file(
        pretrained_model_name_or_path,
        filename,
        _raise_exceptions_for_missing_entries=False,
        **download_kwargs,
    )


def read_export_manifest(pretrained_model_name_or_path, **download_kwargs) -> dict:
    """Read and check a saved export's manifest, so both loaders fail the same way.

    Takes a local directory or a Hub repo id. An export that lost its recorded metadata is refused rather
    than loaded: a runner without it still runs, inferring the precision, the cache kwarg and the mask
    layout from names and shapes, and silently getting them wrong is the failure this payload exists to
    prevent.
    """
    path = resolve_export_file(pretrained_model_name_or_path, EXPORT_MANIFEST_FILE, **download_kwargs)
    if path is None:
        raise OSError(
            f"No `{EXPORT_MANIFEST_FILE}` in {pretrained_model_name_or_path}. Exported models are loaded "
            "from what `ExporterOutput.save_pretrained` wrote; to assemble runners yourself, build each one "
            "with `ModelRunner.from_pretrained` and pass them to `ExportedGenerator.from_runners`."
        )
    manifest = json.loads(Path(path).read_text())
    components = manifest.get("components") or {}
    if not components:
        raise OSError(f"{path} lists no components.")
    missing = [name for name, entry in components.items() if not entry.get("metadata")]
    if missing:
        raise OSError(
            f"{path} has no recorded export metadata for {missing}. Re-save the export with "
            "`ExporterOutput.save_pretrained`, which writes it; loading without it would fall back to "
            "inferring precision and cache layout from the graph's names and shapes."
        )
    return manifest


def load_export_runners(pretrained_model_name_or_path, **kwargs) -> tuple[dict[str, ModelRunner], dict]:
    """Resolve a saved export into `{component: runner}` plus its manifest.

    The part both loaders share: read the manifest, pick the runner for the format it names, and build one
    per component from the file it names — each handed the metadata the save recorded for it, so a reloaded
    runner knows what the one built straight from the export knows.
    """
    from .auto import export_backend

    download_kwargs, runner_kwargs = split_download_kwargs(kwargs)
    manifest = read_export_manifest(pretrained_model_name_or_path, **download_kwargs)
    runner_class = export_backend(manifest["export_format"], "runner")
    runners = {
        component: runner_class.from_pretrained(
            resolve_export_file(pretrained_model_name_or_path, entry["file"], **download_kwargs),
            export_metadata=entry["metadata"],
            **runner_kwargs,
        )
        for component, entry in manifest["components"].items()
    }
    return runners, manifest


class ExporterOutput(Mapping):
    """What an export produced: each component's artifact, what the trace recorded about it, and the configs.

    A `Mapping` over `{component: artifact}`, so indexing and iteration reach the thing you would hand a
    backend. `metadata` holds the matching `{component: dict}` — the trace's own account of each graph
    (`build_export_metadata`), which travels here rather than hidden inside the artifacts: the backends carry
    it differently in their files and `torch.export` cannot carry it at all, so a save that read it back out
    of them would depend on the format rather than on the export.

    `generation_config` is part of the product, not decoration: it declares the cache the graphs were traced
    against, so a load without it would build the wrong one.
    """

    def __init__(
        self,
        artifacts: dict[str, object],
        metadata: dict[str, dict],
        export_format: ExportFormat,
        config: object | None = None,
        generation_config: GenerationConfig | None = None,
        kind: str = "model",
    ):
        self.artifacts = dict(artifacts)
        self.metadata = dict(metadata)
        self.export_format = export_format
        self.config = config
        self.generation_config = generation_config
        self.kind = kind

    def __getitem__(self, component: str):
        return self.artifacts[component]

    def __iter__(self):
        return iter(self.artifacts)

    def __len__(self) -> int:
        return len(self.artifacts)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(format={self.export_format.value!r}, components={list(self.artifacts)})"

    @property
    def backend(self) -> type[HfExporter]:
        """The exporter class for this format — what knows the suffix and how to write the files. Resolved
        from the format the way a load resolves its runner, rather than held as a reference back to the
        instance that produced this."""
        from .auto import export_backend

        return export_backend(self.export_format, "exporter")

    @property
    def artifact(self):
        """The one artifact, for a single-component export ([`~HfExporter.export`]). Raises for a decomposed
        model, where there is no single graph to mean."""
        if len(self.artifacts) != 1:
            raise ValueError(
                f"This export has {len(self.artifacts)} components ({list(self.artifacts)}); index it by "
                "component name instead of asking for `.artifact`."
            )
        return next(iter(self.artifacts.values()))

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Write the components, the configs, and the manifest that makes the directory loadable.

        One file per component, named after it, so the sidecars a large ONNX graph spills stay distinct.
        Reload with [`~exporters.ExportedGenerator.from_pretrained`].
        """
        backend = self.backend
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)

        components = {}
        for name, artifact in self.artifacts.items():
            filename = f"{name}{backend.artifact_suffix}"
            backend.save_artifact(artifact, directory / filename)
            components[name] = {"file": filename, "metadata": self.metadata.get(name, {})}

        (directory / EXPORT_MANIFEST_FILE).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "export_format": self.export_format.value,
                    # What to load this back as: a decomposed, cache-driven export is driven through
                    # `generate`, a single graph is just called. Recorded rather than inferred from the
                    # component names, which are the exporter's business and not a contract.
                    "kind": self.kind,
                    "components": components,
                },
                indent=2,
            )
            + "\n"
        )
        # Written as the model's own files, so a load reads them the way it reads any checkpoint's.
        if self.config is not None:
            self.config.save_pretrained(directory)
        if self.generation_config is not None:
            self.generation_config.save_pretrained(directory)

    def runners(self, components: Iterable[str] | None = None, **kwargs) -> dict[str, ModelRunner]:
        """A runner per artifact, built in memory — the same runners a load builds, handed the same
        metadata, so an export can be checked in the process that produced it.

        `components` limits which ones are built (opening a session has a cost); `kwargs` go to each
        runner, `device=` among them.
        """
        from .auto import export_backend

        runner_class = export_backend(self.export_format, "runner")
        wanted = self.artifacts if components is None else {name: self.artifacts[name] for name in components}
        return {
            name: runner_class.from_artifact(artifact, export_metadata=self.metadata.get(name), **kwargs)
            for name, artifact in wanted.items()
        }

    def runner(self, **kwargs) -> ModelRunner:
        """The one runner, for a single-component export — `.runners()` is the decomposed form, as
        `.artifact` is to `[...]`."""
        if len(self.artifacts) != 1:
            raise ValueError(
                f"This export has {len(self.artifacts)} components ({list(self.artifacts)}); use "
                "`.runners()` and pick by component name."
            )
        return next(iter(self.runners(**kwargs).values()))

    def runtime(self, **kwargs):
        """Something runnable, without going to disk: an [`ExportedGenerator`] for a decomposed export, an
        [`ExportedModel`] for a single graph. Dispatches on `kind`, exactly as a load does."""
        runners = self.runners(**kwargs)
        if self.kind == "generation":
            from .generator import ExportedGenerator

            return ExportedGenerator.from_runners(runners, self.config, self.generation_config)

        return ExportedModel(next(iter(runners.values())), self.config)


class HfExporter(ABC):
    """
    Abstract base class for all Transformers exporters.

    To add a backend, subclass and implement its two halves: [`~HfExporter.export_artifact`] to trace one
    graph, and [`~HfExporter.save_artifact`] to write one out. The public [`~HfExporter.export`] and
    [`~HfExporter.export_for_generation`] build an [`ExporterOutput`] on top of them.
    """

    # What this backend is, and what its artifacts are called on disk. Both required of a concrete
    # exporter: one that can trace a graph can name the file it writes.
    export_format: ExportFormat
    artifact_suffix: str

    # What it needs installed to run.
    required_packages: list[str] = []
    # Hard minimum versions — the exporter raises below these (features it relies on are absent).
    min_versions: dict[str, str] = {}
    # Versions the exporter is validated against — a mismatch only warns.
    tested_versions: dict[str, str] = {}

    def __init__(self):
        self.validate_environment()

    def validate_environment(self, *args, **kwargs):
        """Check `required_packages` are installed and warn on version drift from `tested_versions`."""
        # Single pass: ``_is_package_available`` returns both existence and version, so we collect
        # missing packages and drift in one loop and report them all at the end (rather than failing
        # on the first miss). The local-version suffix (``+cu126``, ``+cpu``) is stripped — patches
        # target the public API, not the build.
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

        # Enforce hard minimums; collect all violations and report once, rather than failing on the first.
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

        Returns `(artifact, metadata)`: the backend's own program object, and what the trace recorded about
        it (`build_export_metadata`). The metadata is returned rather than left inside the artifact because
        only some formats can carry it — handing it back means every caller has it regardless.

        Args:
            model ([`PreTrainedModel`]):
                The model to export.
            sample_inputs (`dict[str, torch.Tensor | Cache]`):
                **Forward** kwargs — what you'd pass to `model(**sample_inputs)`, used directly as the
                example inputs during tracing. For an autoregressive decode step that means including
                `past_key_values`, `cache_position`, etc. If you only have generation-style inputs, use
                [`~HfExporter.export_for_generation`], which runs `model.generate` for you.
            config ([`~transformers.exporters.configs.ExportConfigMixin`]):
                Backend-specific configuration.
        """

    def export(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, torch.Tensor | Cache],
        config: ExportConfigMixin,
    ) -> ExporterOutput:
        """Export the model as one graph, as an [`ExporterOutput`] that can save and run itself.

        Takes the same **forward** kwargs as [`~HfExporter.export_artifact`]; reach the backend's own
        program object through `output.artifact`. No `generation_config`: one graph is called, not
        generated from, so there is no cache contract to record — that is
        [`~HfExporter.export_for_generation`]'s business.
        """
        artifact, metadata = self.export_artifact(model, sample_inputs, config)
        # `getattr`, because a single graph is often a decomposed component rather than a whole model, and
        # those are plain `nn.Module`s: an encoder-decoder's `FSMTEncoder`, an RNN-T's decoder. Such an
        # export simply saves no `config.json`.
        return ExporterOutput(
            {"model": artifact},
            {"model": metadata},
            self.export_format,
            kind="model",
            config=getattr(model, "config", None),
        )

    def export_for_generation(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, torch.Tensor | Cache],
        config: ExportConfigMixin | dict[str, ExportConfigMixin],
        generation_config: GenerationConfig | None = None,
        multi_token_decode: bool = False,
    ) -> dict[str, object]:
        """
        Decompose a generative model and export each component independently.

        Thin wrapper around [`~exporters.utils.decompose_for_generation`] that calls
        [`~HfExporter.export`] on every returned `(submodel, forward_inputs)` pair. If you need
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
                `"text_decoder"`, `"decode"`) to override per-component —
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
            `dict[str, Any]`: `{component_name: backend_specific_artifact}` — same keys as
            [`~exporters.utils.decompose_for_generation`]. Values are whatever
            [`~HfExporter.export`] returns for the concrete backend (`ExportedProgram`,
            `ONNXProgram`, `ExecutorchProgramManager`).
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

        artifacts: dict[str, object] = {}
        metadata: dict[str, dict] = {}
        for name, (submodel, subinputs) in components.items():
            try:
                artifacts[name], metadata[name] = self.export_artifact(submodel, subinputs, config=configs[name])
            except Exception as e:
                raise RuntimeError(
                    f"{type(self).__name__}.export_artifact failed on component '{name}' "
                    f"(submodel={type(submodel).__name__}, input keys={list(subinputs)})."
                ) from e

        # Carrying the configs the components were traced with is what lets the result save itself: the
        # `generation_config` in particular declares the cache the graphs were traced against, and a save
        # that lost it would leave a load guessing.
        return ExporterOutput(
            artifacts,
            metadata,
            self.export_format,
            kind="generation",
            config=getattr(model, "config", None),
            generation_config=generation_config
            if generation_config is not None
            else getattr(model, "generation_config", None),
        )

    @classmethod
    @abstractmethod
    def save_artifact(cls, artifact, path: Path) -> None:
        """Write one exported component to `path`. Implemented per backend, which owns its file format.

        A class method: what a format writes is a property of the format, not of a particular exporter
        instance, which is what lets an [`ExporterOutput`] save itself knowing only which backend made it.
        """


class ModelRunner(ABC):
    """Wraps one exported artifact so it forwards like the module it was exported from:
    `runner(**kwargs) -> {name: tensor}`, torch tensors in and out.

    `kwargs` are the exported forward's kwargs — including `past_key_values` as a `Cache` object for a
    decode graph; each backend adapts it to however its graph carries the cache (flattened `input.<name>`
    tensors, positional buffers, or the pytree itself). The returned dict maps output leaf names
    (`logits`, `past_key_values.…`) to tensors, so callers read outputs the same way for every backend.

    `input_names` lists the graph's declared inputs — a caller feeds only what the graph takes;
    `text_input` and `mask_inputs` are derived from it so a generation loop can key its feed. `device` /
    `dtype` are where the backend lands its output tensors — the generator reads them off the decode
    runner, so nobody has to pass them in.
    """

    # What the exporter recorded about this graph (`build_export_metadata`), parsed — the trace's own
    # account of itself, and what every accessor below reads. Empty for an artifact written without it.
    #
    # Who owns which fact, once, for all of them: the *handle* answers what it can observe about itself —
    # what it declares, in what order, at what shapes — and a runner states that by assigning the attribute
    # in `__init__`, which seeds the accessor below. The *metadata* answers what no handle can state: the
    # precision the graph computes in, the cache's per-layer geometry and sizes, the rank of a mask that
    # was traced away. Where both could answer, the handle wins, because it is the thing that will refuse
    # the call. An accessor is the fallback for an artifact whose runner said nothing.
    export_metadata: ExportMetadata = ExportMetadata()
    device: torch.device | str = "cpu"

    @functools.cached_property
    def input_names(self) -> tuple[str, ...]:
        """What this graph takes, in the flat order it takes them, as recorded at export.

        Handles that name their own inputs assign this instead (see the ownership rule above): ONNX exposes
        a mutated input under the name `generate` uses rather than the `input.`-prefixed one its session
        declares, and a dynamo module *is* the program, so its input spec is the record."""
        return self.export_metadata.input_names

    @functools.cached_property
    def dtype(self) -> torch.dtype:
        """Precision the graph computes at, as recorded at export.

        The generation layer sizes the cache it feeds from this, and a half-precision export (a grouped-mm
        MoE, a varlen-attention VLM) takes half-precision cache leaves — hand it the fp32 default and the
        feed is refused when the artifact binds its inputs. An artifact carrying no metadata cannot say, so
        it says so rather than sniffing whichever of its tensors happens to be floating point. A runner whose
        handle knows better assigns `self.dtype` instead, which seeds this."""
        if dtype := self.export_metadata.dtype:
            return dtype
        logger.warning_once(
            f"This artifact carries no `{EXPORT_METADATA_KEY}`, so the precision it was exported at is "
            f"unknown; assuming {torch.float32}. Re-export it to record the precision."
        )
        return torch.float32

    @functools.cached_property
    def cache_inputs(self) -> tuple[str, ...]:
        """Every kwarg this graph takes a cache under, in the order the trace recorded them.

        Usually one, but a model that caches its *encoder* separately declares two — voxtral_realtime's
        decode takes `past_key_values` and `encoder_past_key_values` — and a backend that has to expand
        each cache's leaves by name needs all of them, not just the first (leaving the second unfilled is
        a `KeyError` on an input the method declares).

        Read off the recorded kwargs when the artifact carries metadata — the ones traced as a cache
        container, whatever they are called. Otherwise matched across each backend's naming of the same
        thing: dynamo takes the whole pytree under the bare name, ONNX flattens it to
        `input.<kwarg>.<path>` inputs, ExecuTorch to `<kwarg>_<leaf index>`."""
        recorded = self.export_metadata.kwargs
        if recorded:
            containers = tuple(name for name, spec in recorded.items() if spec.get("container") == "cache")
            # A model-specific state class (xlstm's `cache_params`) records its own class name rather than
            # "cache", so an empty answer here means "not recorded as a cache", not "no cache" — fall through
            # to the name scan the way an artifact without metadata does.
            if containers:
                return containers
        return tuple(
            kwarg
            for kwarg in ("cache_params", "past_key_values")
            if any(name.removeprefix("input.").startswith(kwarg) for name in self.input_names)
        )

    def declares(self, name: str, value=None) -> bool:
        """Whether this graph takes the feed entry `name` — directly, or as the pytree whose leaves it names.

        A pytree kwarg (`encoder_outputs`, a mask dict, the cache) goes in under its *kwarg* name while each
        backend names the leaves its own way (`encoder_outputs.last_hidden_state` for ONNX,
        `encoder_outputs_last_hidden_state` for ExecuTorch, the kwarg itself for dynamo), so the kwarg's own
        name is not always among `input_names`. Only a non-tensor is flattened, so a plain tensor must be
        named outright — `input_features` is not declared by a graph that only takes `input_features_mask`.

        The one rule for "does this graph take this", asked here by every caller: a feed built against a
        weaker rule (an exact name match) silently loses whatever a flattening backend renamed.
        """
        if name in self.input_names or name == self.cache_input:
            return True
        return not isinstance(value, torch.Tensor) and any(
            declared.removeprefix("input.").startswith((f"{name}.", f"{name}_")) for declared in self.input_names
        )

    @functools.cached_property
    def cache_input(self) -> str | None:
        """The kwarg this graph takes its *primary* cache under — `"cache_params"` for a recurrent model
        (fixed-size conv / recurrent state), `"past_key_values"` otherwise, `None` for a graph with no
        cache. This is the one the generation loop grows and feeds each step; see `cache_inputs` for the
        rest.

        Cached: the metadata and the declared names are both fixed once the runner is built, and the
        generation loop asks for this on every step."""
        return self.cache_inputs[0] if self.cache_inputs else None

    @property
    def kv_geometry(self) -> dict[int, tuple[int, int, int]]:
        """`{layer index: (num_kv_heads, key_head_dim, value_head_dim)}` the graph's cache was traced with —
        empty when it takes none, and missing a layer whose state is not keys and values (a recurrent
        layer's conv / SSM buffers). What the runtime sizes a cache to, since a config cannot always say."""
        return self.export_metadata.kv_geometry

    def to(self, device) -> ModelRunner:
        """The runner to use for `device`, or a refusal saying why this backend has none.

        Returned rather than moved in place because not every backend can move: a `torch.export` program is
        a module and moves, an ORT session is fixed to the execution provider it was created with and is
        reopened instead, and a `.pte` is bound to the backend it was lowered for and cannot be either.
        """
        raise ValueError(
            f"{type(self).__name__} is bound to {self.device} by the runtime that loaded it. Load the "
            f"artifact again with `device={device!r}` to run it elsewhere."
        )

    @staticmethod
    def resolve_metadata(injected, read_baked) -> ExportMetadata:
        """The metadata a load passed in, else what the artifact itself carries.

        `read_baked` is called only when needed: reading it back out of an artifact costs something on
        some backends (ExecuTorch executes a baked constant method to get at it).
        """
        return ExportMetadata.from_dict(injected) if injected is not None else read_baked()

    @classmethod
    def from_artifact(cls, artifact, export_metadata=None, **kwargs) -> ModelRunner:
        """Build the runner from an artifact still in memory — what an export hands back.

        The in-memory counterpart of `from_pretrained`: `ExporterOutput.runtime()` calls this so a model
        can be driven straight after exporting it, without a save and a reload in between.
        """
        raise NotImplementedError(
            f"{cls.__name__} cannot be built from an in-memory artifact. Implement `from_artifact` to run "
            "an export without saving it first."
        )

    @classmethod
    def from_pretrained(cls, path: str | Path, **kwargs) -> ModelRunner:
        """Build the runner from one saved artifact — the inverse of [`~HfExporter.save_artifact`].

        Each backend loads its own file into whatever it runs (an ORT session, an unlifted module, a
        loaded `.pte`) and hands it to `__init__`, so a reloaded runner is the one the exporter produced.
        """
        raise NotImplementedError(
            f"{cls.__name__} cannot be loaded from disk. Implement `from_pretrained` to build it from a "
            "saved artifact."
        )

    @abstractmethod
    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        """Run the graph on `kwargs`; return its outputs as `{leaf_name: tensor}`."""


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

    def can_generate(self) -> bool:
        """`False`: one graph is one forward. The generative case is `ExportedGenerator`, which drives the
        component graphs through `generate`."""
        return False

    @property
    def input_modalities(self) -> list[str] | str:
        """What the model this came from takes, for anything that reports on a loaded model (pipelines do).
        Read off the config, since the graph knows only tensor names."""
        return getattr(self.config, "input_modalities", "text")

    def to(self, device) -> ExportedModel:
        """Move to `device` if this graph's backend can; whether it can is the runner's to answer."""
        if torch.device(device) != self.device:
            self.runner = self.runner.to(device)
        return self

    @property
    def input_names(self) -> tuple[str, ...]:
        """What the graph takes — the kwargs this accepts, as traced."""
        return self.runner.input_names

    def __call__(self, **kwargs) -> ModelOutput:
        """Run the graph. Kwargs the trace never saw are dropped rather than refused, so a caller can pass
        a processor's whole output the way it would to the eager model."""
        return ModelOutput(**self.runner(**runner_feed(self.runner, kwargs, warn_unused=True)))

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
