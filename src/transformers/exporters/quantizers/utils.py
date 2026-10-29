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
"""Calibration helpers of the quantizers: per-component calibration inputs for `export_for_generation`, and the feeds
that map a forward-kwargs sample to the inputs of the model each quantizer calibrates."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from ..utils import decompose_for_generation, get_leaf_tensors


if TYPE_CHECKING:
    from ...modeling_utils import PreTrainedModel


def capture_calibration_inputs(
    model: PreTrainedModel,
    calibration_dataset: Iterable[dict[str, Any]],
    generation_config: Any = None,
    multi_token_decode: bool = False,
) -> dict[str, list[dict]]:
    """Run each generate-style sample through `decompose_for_generation` and collect every component's forward
    kwargs: `{component_name: [forward_inputs, ...]}`, one calibration set per component."""
    calibration: dict[str, list[dict]] = {}
    for sample in calibration_dataset:
        components = decompose_for_generation(
            model, sample, generation_config=generation_config, multi_token_decode=multi_token_decode
        )
        for name, (_submodel, forward_inputs) in components.items():
            calibration.setdefault(name, []).append(forward_inputs)
        # A multi-token decode graph also serves the prefill step, so it calibrates on the prefill inputs too.
        if multi_token_decode and "prefill" in components and "decode" in components:
            calibration["decode"].append(components["prefill"][1])
    return calibration


def traced_inputs(sample: dict[str, Any], traced_names: list[str]) -> dict[str, Any]:
    """A calibration sample's traced inputs, deep-copied (a calibration forward writes the cache in place). Any other
    key, such as an output flag `prepare_for_export` popped from the sample inputs, is dropped."""
    missing = [name for name in traced_names if name not in sample]
    if missing:
        raise ValueError(
            f"A calibration sample lacks the traced inputs {missing} (it has {sorted(sample)}). Calibration samples "
            "are the exported forward's kwargs (`export_for_generation` with a single config is the exception: it "
            "takes generate kwargs)."
        )
    return copy.deepcopy({name: sample[name] for name in traced_names})


def onnx_feed(input_names: list[str], sample) -> dict[str, np.ndarray]:
    """`sample`'s tensor leaves as the ONNX model's inputs, by input name (mutated inputs carry an `input.` prefix)."""
    leaves = {path: tensor.cpu().numpy() for path, tensor in get_leaf_tensors(sample).items()}
    paths = {name: name.removeprefix("input.") for name in input_names}
    return {name: leaves[path] for name, path in paths.items() if path in leaves}


def openvino_feed(ov_model: Any, sample) -> dict[str, Any]:
    """`sample`'s forward kwargs as the converted model's inputs: tensor leaves by port name (folded state takes none),
    the identity `beam_idx`, and scalars as they are."""
    leaves = {path: tensor.cpu().numpy() for path, tensor in get_leaf_tensors(sample).items()}
    batch = next(iter(leaves.values())).shape[0] if leaves else 1

    def port_value(name: str) -> Any:
        path = name.removeprefix("input.")
        if path in leaves:
            return leaves[path]
        if name == "beam_idx":
            return np.arange(batch, dtype=np.int32)
        if name in sample:
            return np.array(sample[name])
        return None

    feed = {}
    for port in ov_model.inputs:
        # A port can carry several names: feed it under the first one that has a value
        for name in port.get_names():
            if (value := port_value(name)) is not None:
                feed[name] = value
                break
    return feed
