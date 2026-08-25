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
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import TYPE_CHECKING

import torch
from packaging import version

from ..utils import logging
from ..utils.import_utils import _is_package_available, is_torch_available
from .configs import ExportConfigMixin
from .utils import EXPORT_METADATA_KEY, ExportMetadata, decompose_for_generation


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    if is_torch_available():
        import torch

        from ..cache_utils import Cache
        from ..generation import GenerationConfig
        from ..modeling_utils import PreTrainedModel


class HfExporter(ABC):
    """
    Abstract base class for all Transformers exporters.

    Subclass and implement [`~HfExporter.export`] to add a new export backend.
    """

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
    def export(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, torch.Tensor | Cache],
        config: ExportConfigMixin,
    ):
        """
        Export the model and return the backend-specific program object.

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
            Backend-specific export artifact.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement `export`. Pick a concrete exporter "
            "(`DynamoExporter`, `OnnxExporter`, `ExecutorchExporter`), or override `export` "
            "in your subclass with a backend-specific tracing pipeline that consumes `config` "
            "and returns the runtime artifact."
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

        exported: dict[str, object] = {}
        for name, (submodel, subinputs) in components.items():
            try:
                exported[name] = self.export(submodel, subinputs, config=configs[name])
            except Exception as e:
                raise RuntimeError(
                    f"{type(self).__name__}.export failed on component '{name}' "
                    f"(submodel={type(submodel).__name__}, input keys={list(subinputs)})."
                ) from e

        return exported


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
    # account of itself, and what every accessor below reads. Empty for an artifact written without it;
    # those runners assign the fields they can answer themselves.
    export_metadata: ExportMetadata = ExportMetadata()
    device: torch.device | str = "cpu"

    @functools.cached_property
    def mask_dict_ranks(self) -> dict[str, int] | None:
        """`{attention type: rank}` when the graph took a *dict* of masks instead of one (mixed full/sliding
        attention), which a model builds inside its forward — so the runtime has to hand one in.

        Recorded at export. A backend whose handle can still recover it for an artifact written before that
        overrides this — and they do not all read the same place: dynamo takes the dict as one kwarg and keeps
        the per-type keys in its pytree child spec, while ONNX and ExecuTorch flatten it to one input per
        type."""
        return self.export_metadata.mask_ranks

    @functools.cached_property
    def input_names(self) -> tuple[str, ...]:
        """What this graph takes, in the flat order it takes them, as recorded at export.

        A runner whose handle names them itself assigns `self.input_names` instead, which seeds this: ONNX
        exposes a mutated input under the name `generate` uses rather than the `input.`-prefixed one its
        session declares, and a dynamo module is the program, so its own input spec is the record."""
        return self.export_metadata.input_names

    @functools.cached_property
    def input_shapes(self) -> dict[str, tuple[int | None, ...]]:
        """Shape per input, `None` per symbolic axis — the shapes the *trace* saw, as recorded.

        A runner assigns `self.input_shapes` instead when what its handle *declares* is the load-bearing
        fact: ONNX sizes a not-yet-created cache entry from the declared shape, and only the session says
        which axes it left symbolic."""
        return self.export_metadata.shapes

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

    @functools.cached_property
    def cache_input(self) -> str | None:
        """The kwarg this graph takes its *primary* cache under — `"cache_params"` for a recurrent model
        (fixed-size conv / recurrent state), `"past_key_values"` otherwise, `None` for a graph with no
        cache. This is the one the generation loop grows and feeds each step; see `cache_inputs` for the
        rest.

        Cached: the metadata and the declared names are both fixed once the runner is built, and the
        generation loop asks for this on every step."""
        return self.cache_inputs[0] if self.cache_inputs else None

    @functools.cached_property
    def text_input(self) -> str:
        """The graph's text input: `"decoder_input_ids"` (encoder-decoder decode), `"inputs_embeds"`
        (multi-modal decode) or `"input_ids"`."""
        return next((n for n in ("decoder_input_ids", "inputs_embeds") if n in self.input_names), "input_ids")

    @functools.cached_property
    def mask_inputs(self) -> tuple[str, ...]:
        """The graph's attention-mask input name(s) — several for mixed full/sliding attention."""
        return tuple(
            n
            for n in self.input_names
            if n == "attention_mask" or n.startswith(("attention_mask.", "attention_mask_"))
        )

    @functools.cached_property
    def decoder_mask_input(self) -> str | None:
        """`"decoder_attention_mask"` when the graph declares one. An encoder-decoder splits the two masks:
        `attention_mask` covers the *encoder's* sequence (what cross-attention reads) while this one covers
        the decoder's own — so the causal mask belongs here, and `generate` does not hand it over (the eager
        model builds it inside the forward the graph starts after)."""
        return "decoder_attention_mask" if "decoder_attention_mask" in self.input_names else None

    @functools.cached_property
    def mask_rank(self) -> int | None:
        """The rank the graph's `attention_mask` was traced with, `None` when it takes none.

        `generate` upgrades a 2D padding mask to the 4D causal mask for any compileable cache, assuming the
        model's forward wants one — but an exported graph starts *after* whatever mask building its model
        does, so only the trace can say which it took. An alibi model (bloom) reads the 2D padding mask
        directly and compares its width to the cache length, so a 4D mask fails a guard rather than
        mismatching a shape."""
        shape = self.input_shapes.get("attention_mask")
        return len(shape) if shape is not None else None

    @abstractmethod
    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        """Run the graph on `kwargs`; return its outputs as `{leaf_name: tensor}`."""
