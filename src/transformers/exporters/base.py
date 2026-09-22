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

from abc import ABC, abstractmethod
from collections.abc import Callable, MutableMapping
from typing import TYPE_CHECKING, Any

from packaging import version

from ..utils import logging
from ..utils.import_utils import _is_package_available, is_torch_available, requires
from .configs import ExportConfigMixin
from .utils import decompose_for_generation


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    from ..image_processing_backends import TorchvisionBackend

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

    @requires(backends=("torch",))
    def export_end_to_end(
        self,
        model: PreTrainedModel,
        preprocess: TorchvisionBackend | Callable | None,
        post_process: Callable | None,
        sample_inputs: MutableMapping[str, torch.Tensor],
        config: ExportConfigMixin | dict[str, Any],
    ):
        """
        Export preprocessing, the model forward, and post-processing as a **single** graph.

        Chains the three stages into one [`~exporters.utils.EndToEndModel`] and calls
        [`~HfExporter.export`] on it, so the artifact takes raw inputs (uint8 images, an audio
        waveform, …) and returns final outputs (top-k labels, rescaled boxes, …). The runtime then
        needs no Python-side processor. If you need the intermediate module (to run it eagerly for
        verification, or to reuse it with a different config), build `EndToEndModel` directly.

        Both stages are traced, so anything either of them computes in Python runs once at trace
        time and is frozen into the graph. Tokenizers are out entirely. The Transformers image and
        video processors are torchvision-backed and do trace, but one that derives its output size
        in Python from the input size (an aspect-ratio-preserving resize) freezes that arithmetic
        into a shape guard, pinning the graph to the traced resolution; a processor with a fixed
        target size stays resolution-agnostic.

        Data-dependent output *shapes* (`scores[scores > threshold]`) are a softer constraint:
        `torch.export` resolves the surviving count to an unbacked symbol, so the stock detection
        post-processors trace as-is, but a backend that requires static shapes will reject the
        result — prefer a fixed-size top-k there and let the caller threshold.

        A Transformers post-processor can be passed as-is: the arguments that decide between its
        traceable and its eager branch are filled in by
        [`~exporters.utils.bind_static_post_process_kwargs`], so `return_traceable_outputs=True` and a
        constant-folded `target_sizes` need no `functools.partial` from the caller. Reshape the
        tensor tuple the artifact returns with the processor's matching `build_*_outputs` method.

        Args:
            model ([`PreTrainedModel`]):
                The model to export.
            preprocess (`TorchvisionBackend` or `Callable`, *optional*):
                A torchvision-backed image or video processor, called with `return_tensors="pt"`,
                or any callable mapping the raw inputs it claims to a mapping of `model.forward`
                kwargs. `None` passes them straight to the forward. Pass an `nn.Module` when the
                preprocessing owns tensors so they're lifted into the graph as buffers.
            post_process (`Callable`, *optional*):
                Called as `post_process(model_outputs, **claimed_inputs)`; its return value becomes
                the graph output. `None` returns the model outputs unchanged.
            sample_inputs (`dict[str, torch.Tensor]`):
                **Raw** inputs — what you'd pass to the pipeline, not to `model.forward`. Each key
                is routed to the stage that declares it as a parameter; a stage declaring `**kwargs`
                takes everything the other stage doesn't declare explicitly (see
                [`~exporters.utils.route_end_to_end_inputs`]). A key claimed by neither stage raises.
                A `target_sizes` is traced as a constant rather than routed (see
                [`~exporters.utils.bind_static_post_process_kwargs`]).
            config ([`~transformers.exporters.configs.ExportConfigMixin`]):
                Backend-specific configuration, forwarded to [`~HfExporter.export`].

        Returns:
            Backend-specific export artifact, as returned by [`~HfExporter.export`].
        """
        from .utils import EndToEndModel, bind_static_post_process_kwargs

        post_process, sample_inputs = bind_static_post_process_kwargs(post_process, sample_inputs)
        end_to_end_model = EndToEndModel(
            model, input_names=list(sample_inputs), preprocess=preprocess, post_process=post_process
        )
        try:
            return self.export(end_to_end_model, sample_inputs, config=config)
        except Exception as e:
            raise RuntimeError(
                f"{type(self).__name__}.export failed on the end-to-end graph for {type(model).__name__} "
                f"(preprocess inputs={end_to_end_model.preprocess_names}, post-process inputs="
                f"{end_to_end_model.post_process_names}). Export the model alone with `export` to tell a "
                f"model-side failure from a pre/post-processing one — the stages are traced too, so a "
                f"data-dependent op in either of them fails here."
            ) from e
