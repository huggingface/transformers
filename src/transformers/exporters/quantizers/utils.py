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

from ..decompose import decode_graph_name, decompose_for_generation
from ..utils import BATCH_INPUTS, get_leaf_tensors


if TYPE_CHECKING:
    from ...modeling_utils import PreTrainedModel


def capture_calibration_inputs(
    model: PreTrainedModel,
    calibration_dataset: Iterable[dict[str, Any]],
    *,
    multi_token_decode: bool,
    generation_config: Any = None,
    decoder_writes_cross_cache: bool = False,
) -> dict[str, list[dict]]:
    """Run each generate-style sample through `decompose_for_generation` and collect every exported component's
    forward kwargs: `{component_name: [forward_inputs, ...]}`, one calibration set per component."""
    calibration: dict[str, list[dict]] = {}
    for sample in calibration_dataset:
        components = decompose_for_generation(
            model,
            sample,
            generation_config=generation_config,
            multi_token_decode=multi_token_decode,
            decoder_writes_cross_cache=decoder_writes_cross_cache,
        )
        prefill = components.pop("prefill")
        for name, component in components.items():
            calibration.setdefault(name, []).append(component.inputs)
        # A multi-token decode graph also runs the prefill step, so it calibrates on that call too; a single-token
        # one only ever takes one token.
        if multi_token_decode:
            calibration[decode_graph_name(components)].append(prefill.inputs)
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
    # Off a text input: the first leaf may be an m-rope `position_ids`, `[sections, batch, positions]`.
    batch_input = next((leaves[name] for name in BATCH_INPUTS if name in leaves), next(iter(leaves.values()), None))
    batch = batch_input.shape[0] if batch_input is not None else 1

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
