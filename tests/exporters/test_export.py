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

import copy
import functools
import inspect
import os
import re

import pytest
import torch
from parameterized import parameterized

from transformers import set_seed
from transformers.exporters.exporter_dynamo import DynamoConfig, DynamoExporter
from transformers.exporters.exporter_executorch import ExecutorchConfig, ExecutorchExporter
from transformers.exporters.exporter_onnx import OnnxConfig, OnnxExporter
from transformers.exporters.exporter_openvino import OpenVINOConfig, OpenVINOExporter
from transformers.exporters.utils import (
    decompose_for_generation,
    decompose_multimodal,
    get_leaf_tensors,
    is_multimodal,
)
from transformers.testing_utils import (
    require_executorch,
    require_onnxruntime,
    require_onnxscript,
    require_openvino,
    require_torch_greater_or_equal,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    slow,
    torch_device,
)


# ──────────────────────────── skip lists ────────────────────────────
#
# A single mapping ``EXPORT_SKIPS[scope][model_class_name] = reason`` drives every skip.
# A ``scope`` key is a **dotted set of tags** drawn from the active context — the backend
# (``"onnx"`` / ``"openvino"`` / ``"executorch"``), ``"generate"``, and ``"dynamic"``. It applies
# when *all* its tags are active, so tag order doesn't matter and any combination composes:
# ``"all"`` (no tags) always applies, ``"dynamic"`` matches any dynamic export, ``"onnx.dynamic"``
# matches dynamic ONNX, ``"openvino.generate.dynamic"`` matches only OpenVINO generate-under-dynamic.
# ``_scope_applies`` does the subset check; ``_should_skip`` returns ``True`` as soon as the model
# is found under any applicable scope. Reasons live next to the model name so the "why" travels
# with the entry.
#
# Adding a new skip: pick the tightest tag-set that covers the failure and add a ``"Name": "reason"``
# entry under that dotted key (creating the key if needed — no matcher change required).


EXPORT_SKIPS: dict[str, dict[str, str]] = {
    # Every backend, every variant.
    "all": {},
    # Any backend (incl. plain `torch.export`), dynamic-shape variant only.
    "dynamic": {
        "Sam2Model": (
            "`torch.export` of the Hiera vision backbone under dynamic shapes exceeds the 1000s "
            "test timeout on every backend (Dynamo/ONNX/OpenVINO/ExecuTorch) — 12 attention blocks "
            "× 3 Q-pool stage transitions on symbolic H/W. Static exports fine."
        ),
        "Sam2VisionModel": "Same Hiera-backbone dynamic-shape `timeout` as `Sam2Model`.",
        "HieraModel": (
            "Hiera mask-unit window `reroll` produces nested symbolic floordivs that `torch.export` "
            "can't guard under dynamic shapes (same backbone family as `Sam2Model`). Static exports fine."
        ),
        "HieraBackbone": "Same Hiera `reroll` dynamic-shape failure as `HieraModel`.",
        "HieraForImageClassification": "Same Hiera `reroll` dynamic-shape failure as `HieraModel`.",
        "HieraForPreTraining": "Same Hiera `reroll` dynamic-shape failure as `HieraModel`.",
    },
    # Every backend, generate path only.
    "generate": {},
    # ONNX, every variant.
    "onnx": {
        "HunYuanVLModel": (
            "ONNX export trips an int32-overflow `GuardOnDataDependentSymNode` (`64*h*w`) in the vision "
            "patch-merger conv on symbolic spatial dims. Plain `torch.export` (dynamo) exports fine."
        ),
        "HunYuanVLForConditionalGeneration": "Same ONNX vision-conv int32 guard as `HunYuanVLModel`.",
    },
    # ONNX, generate path only.
    "onnx.generate": {},
    # ONNX, dynamic-shape only.
    "onnx.dynamic": {
        "SwinModel": (
            "Shifted-window attention on symbolic H/W: `torch.export` + onnxscript exceed the "
            "1000s test timeout under dynamic shapes (static exports fine)."
        ),
        "SwinBackbone": "Same shifted-window `timeout` as `SwinModel`.",
        "SwinForImageClassification": "Same shifted-window `timeout` as `SwinModel`.",
        "SwinForMaskedImageModeling": "Same shifted-window `timeout` as `SwinModel`.",
        "Swinv2Model": "Same shifted-window `timeout` as `SwinModel`.",
        "Swinv2ForImageClassification": "Same shifted-window `timeout` as `SwinModel`.",
        "Swinv2ForMaskedImageModeling": "Same shifted-window `timeout` as `SwinModel`.",
        "Swinv2Backbone": "Same shifted-window `timeout` as `SwinModel`.",
        "DonutSwinModel": "Same shifted-window `timeout` as `SwinModel`.",
        "DonutSwinForImageClassification": "Same shifted-window `timeout` as `SwinModel`.",
        "MaskFormerModel": "Same shifted-window (Swin backbone) `timeout` as `SwinModel`.",
        "MaskFormerForInstanceSegmentation": "Same `timeout` as `MaskFormerModel`.",
        "Mask2FormerModel": (
            "Deformable-attention pixel decoder exceeds the 1000s test timeout under dynamic shapes."
        ),
        "Mask2FormerForUniversalSegmentation": "Same `timeout` as `Mask2FormerModel`.",
        "FunnelModel": (
            "onnxscript's constant-folding optimizer raises `Bitwidth not available for ONNX data type: "
            "STRING` on funnel's dynamic-shape graph. `torch.export`/OpenVINO and static ONNX all export fine."
        ),
        "FunnelForMaskedLM": "Same onnxscript optimizer `STRING` failure as `FunnelModel`.",
        "FunnelForPreTraining": "Same onnxscript optimizer `STRING` failure as `FunnelModel`.",
        "FunnelForQuestionAnswering": "Same onnxscript optimizer `STRING` failure as `FunnelModel`.",
        "FunnelForTokenClassification": "Same onnxscript optimizer `STRING` failure as `FunnelModel`.",
        "FunnelBaseModel": "Same onnxscript optimizer `STRING` failure as `FunnelModel`.",
        "FunnelForMultipleChoice": "Same onnxscript optimizer `STRING` failure as `FunnelModel`.",
        "FunnelForSequenceClassification": "Same onnxscript optimizer `STRING` failure as `FunnelModel`.",
    },
    # ExecuTorch, every variant.
    "executorch": {
        "BarkFineModel": (
            "ExecuTorch memory planning miscomputes the tensor spec (`buffer of size N, expected nbytes of M`) — a dtype-size mismatch in the lowered program."
        ),
        "ClvpModelForConditionalGeneration": (
            "A pass-through output aliases an input (`Output node is already in the inputs`)."
        ),
        "ColQwen2ForRetrieval": (
            "ExecuTorch dim-order lowering requires a copying view (`Cannot view a tensor ... with shape/strides`)."
        ),
        "DabDetrModel": ("XNNPACK partitioner: `Attempting to convert non-NHWC compatible node to NHWC`."),
        "DabDetrForObjectDetection": "Same `nhwc` failure as `DabDetrModel`.",
        "Ernie4_5_VLMoeModel": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "Ernie4_5_VLMoeForConditionalGeneration": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "FlavaForPreTraining": ("XNNPACK partitioner: `Invalid partition, found dependency cycles`."),
        "GPT2Model": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "GPT2LMHeadModel": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "GPT2DoubleHeadsModel": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "GPT2ForQuestionAnswering": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "GPT2ForSequenceClassification": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "GPT2ForTokenClassification": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "Gemma3nModel": "Same `spec` failure as `BarkFineModel`.",
        "Gemma3nForConditionalGeneration": "Same `spec` failure as `BarkFineModel`.",
        "Glm46VModel": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "Glm46VForConditionalGeneration": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "Glm4vModel": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "Glm4vForConditionalGeneration": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "Glm4vMoeModel": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "Glm4vMoeForConditionalGeneration": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "GlmImageModel": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "GlmImageForConditionalGeneration": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "GlmOcrModel": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "GlmOcrForConditionalGeneration": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "GroundingDinoModel": ("Lowering exceeds the test timeout under dynamic shapes."),
        "GroundingDinoForObjectDetection": "Same `timeout` failure as `GroundingDinoModel`.",
        "InstructBlipModel": "Same `spec` failure as `BarkFineModel`.",
        "InstructBlipForConditionalGeneration": "Same `spec` failure as `BarkFineModel`.",
        "InstructBlipVideoForConditionalGeneration": "Same `spec` failure as `BarkFineModel`.",
        "InstructBlipVideoModel": "Same `spec` failure as `BarkFineModel`.",
        "MMGroundingDinoModel": "Same `timeout` failure as `GroundingDinoModel`.",
        "MMGroundingDinoForObjectDetection": "Same `timeout` failure as `GroundingDinoModel`.",
        "MiniMaxM3VLModel": ("Serialization rejects an i64 constant (`bad number for type int32`)."),
        "MiniMaxM3SparseForConditionalGeneration": "Same `int32` failure as `MiniMaxM3VLModel`.",
        "PPDocLayoutV3ForObjectDetection": ("Delegation drops a referenced weight (`KeyError` on a state-dict key)."),
        "PaddleOCRVLForConditionalGeneration": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "PerceptionLMModel": "Same `passthrough` failure as `ClvpModelForConditionalGeneration`.",
        "PerceptionLMForConditionalGeneration": "Same `passthrough` failure as `ClvpModelForConditionalGeneration`.",
        "Qwen2VLModel": "Same `spec` failure as `BarkFineModel`.",
        "Qwen2VLForConditionalGeneration": "Same `spec` failure as `BarkFineModel`.",
        "Qwen2_5OmniThinkerForConditionalGeneration": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "Qwen2_5_VLModel": "Same `spec` failure as `BarkFineModel`.",
        "Qwen2_5_VLForConditionalGeneration": "Same `spec` failure as `BarkFineModel`.",
        "Qwen3OmniMoeThinkerForConditionalGeneration": "Same `view` failure as `ColQwen2ForRetrieval`.",
        "Qwen3_5Model": "Same `spec` failure as `BarkFineModel`.",
        "Qwen3_5ForConditionalGeneration": "Same `spec` failure as `BarkFineModel`.",
        "Qwen3_5ForSequenceClassification": "Same `spec` failure as `BarkFineModel`.",
        "Qwen3_5ForTokenClassification": "Same `spec` failure as `BarkFineModel`.",
        "Qwen3_5MoeModel": "Same `spec` failure as `BarkFineModel`.",
        "Qwen3_5MoeForConditionalGeneration": "Same `spec` failure as `BarkFineModel`.",
    },
    "executorch.generate": {
        "PPFormulaNetForConditionalGeneration": (
            "ExecuTorch memory planning miscomputes the tensor spec (`buffer of size N, expected nbytes of M`) — a dtype-size mismatch in the lowered program."
        ),
    },
    "executorch.dynamic": {
        "BigBirdModel": ("Lowering exceeds the test timeout under dynamic shapes."),
        "BigBirdForPreTraining": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForMaskedLM": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForCausalLM": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForMultipleChoice": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForQuestionAnswering": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForSequenceClassification": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForTokenClassification": "Same `timeout` failure as `BigBirdModel`.",
        "DepthProModel": (
            "`_ViewSpec is incompatible with its base` — mixed shape dynamism between a view and its base."
        ),
        "DepthProForDepthEstimation": "Same `viewspec` failure as `DepthProModel`.",
        "DonutSwinModel": (
            "ExecuTorch memory planning overflows under unbounded dynamic shapes (`mem_offset does not fit in 64 bits`)."
        ),
        "DonutSwinForImageClassification": "Same `overflow` failure as `DonutSwinModel`.",
        "Mask2FormerModel": "Same `timeout` failure as `BigBirdModel`.",
        "Mask2FormerForUniversalSegmentation": "Same `timeout` failure as `BigBirdModel`.",
        "MaskFormerModel": "Same `timeout` failure as `BigBirdModel`.",
        "MaskFormerForInstanceSegmentation": "Same `timeout` failure as `BigBirdModel`.",
        "MaskFormerSwinModel": "Same `overflow` failure as `DonutSwinModel`.",
        "MaskFormerSwinBackbone": "Same `overflow` failure as `DonutSwinModel`.",
        "MllamaModel": "Same `overflow` failure as `DonutSwinModel`.",
        "MllamaForConditionalGeneration": "Same `overflow` failure as `DonutSwinModel`.",
        "PvtModel": "Same `viewspec` failure as `DepthProModel`.",
        "PvtForImageClassification": "Same `viewspec` failure as `DepthProModel`.",
        "Sam2Model": ("Delegation drops a referenced weight (`KeyError` on a state-dict key)."),
        "Sam2VisionModel": "Same `timeout` failure as `BigBirdModel`.",
        "Swin2SRModel": "Same `overflow` failure as `DonutSwinModel`.",
        "Swin2SRForImageSuperResolution": "Same `overflow` failure as `DonutSwinModel`.",
        "SwinModel": "Same `overflow` failure as `DonutSwinModel`.",
        "SwinBackbone": "Same `overflow` failure as `DonutSwinModel`.",
        "SwinForImageClassification": "Same `overflow` failure as `DonutSwinModel`.",
        "SwinForMaskedImageModeling": "Same `overflow` failure as `DonutSwinModel`.",
        "Swinv2Model": "Same `overflow` failure as `DonutSwinModel`.",
        "Swinv2ForImageClassification": "Same `overflow` failure as `DonutSwinModel`.",
        "Swinv2ForMaskedImageModeling": "Same `overflow` failure as `DonutSwinModel`.",
        "Swinv2Backbone": "Same `overflow` failure as `DonutSwinModel`.",
        "VitDetModel": "Same `viewspec` failure as `DepthProModel`.",
        "VitDetBackbone": "Same `viewspec` failure as `DepthProModel`.",
        "Wav2Vec2BertForCTC": ("`flatc` schema compilation fails when serializing the program."),
        "Wav2Vec2BertModel": "Same `flatc` failure as `Wav2Vec2BertForCTC`.",
        "Wav2Vec2BertForSequenceClassification": "Same `flatc` failure as `Wav2Vec2BertForCTC`.",
        "Wav2Vec2BertForAudioFrameClassification": "Same `flatc` failure as `Wav2Vec2BertForCTC`.",
        "Wav2Vec2BertForXVector": "Same `flatc` failure as `Wav2Vec2BertForCTC`.",
    },
    # OpenVINO, every variant.
    "openvino": {
        "TapasModel": "OpenVINO has no conversion rule for `aten.scatter_reduce.two` (tapas segment reduction).",
        "TapasForMaskedLM": "Same OpenVINO `scatter_reduce` gap as `TapasModel`.",
        "TapasForQuestionAnswering": "Same OpenVINO `scatter_reduce` gap as `TapasModel`.",
        "TapasForSequenceClassification": "Same OpenVINO `scatter_reduce` gap as `TapasModel`.",
        "HunYuanVLModel": "OpenVINO conversion of the vision stack fails (same family as the ONNX/ExecuTorch gaps).",
        "HunYuanVLForConditionalGeneration": "Same OpenVINO gap as `HunYuanVLModel`.",
        "Kimi_K25Model": (
            "OpenVINO CPU plugin fails to compile (`to_shape was called on a dynamic shape`) — a node in "
            "the vision/MLA stack keeps a fully dynamic shape even under static export (data-dependent "
            "vision token count from `image_grid_thw`)."
        ),
        "Kimi_K25ForConditionalGeneration": "Same OpenVINO `to_shape`/dynamic-shape compile failure as `Kimi_K25Model`.",
    },
    # OpenVINO, generate path only.
    "openvino.generate": {},
    # OpenVINO, dynamic-shape only.
    "openvino.dynamic": {
        "BigBirdModel": ("OpenVINO conversion exceeds the 1000s test timeout under dynamic shapes."),
        "BigBirdForPreTraining": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForMaskedLM": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForCausalLM": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForMultipleChoice": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForQuestionAnswering": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForSequenceClassification": "Same `timeout` failure as `BigBirdModel`.",
        "BigBirdForTokenClassification": "Same `timeout` failure as `BigBirdModel`.",
        "MaskFormerModel": "Shifted-window (Swin) backbone exceeds the 1000s test timeout under dynamic shapes.",
        "MaskFormerForInstanceSegmentation": "Same `timeout` as `MaskFormerModel`.",
        "Mask2FormerModel": "Deformable-attention pixel decoder exceeds the 1000s test timeout under dynamic shapes.",
        "Mask2FormerForUniversalSegmentation": "Same `timeout` as `Mask2FormerModel`.",
        "GroundingDinoModel": ("Deformable-attention encoder exceeds the 1000s test timeout under dynamic shapes."),
        "GroundingDinoForObjectDetection": "Same `timeout` as `GroundingDinoModel`.",
        "MMGroundingDinoModel": "Same `timeout` as `GroundingDinoModel`.",
        "MMGroundingDinoForObjectDetection": "Same `timeout` as `GroundingDinoModel`.",
    },
}


# Parameterization for export tests: runs once with dynamic=True and once with dynamic=False.
DYNAMIC_EXPORT_PARAMS = parameterized.expand(
    [(False,), (True,)],
    name_func=lambda f, _, p: f"{f.__name__}_{'dynamic' if p.args[0] else 'static'}",
)

# Maximum time (in seconds) for a single export test before it is killed.
EXPORT_TEST_TIMEOUT = 1000

# Minimum torch version the exporters target — older releases lack `torch.export` features the
# exporters rely on, so the export sweep is skipped (not failed) below this. Sourced from the
# exporter itself so the test and the runtime check can't drift apart.
MIN_EXPORT_TORCH_VERSION = DynamoExporter.min_versions["torch"]


# ──────────────────────────── helpers ────────────────────────────


def _cast_inputs(obj, device, dtype):
    """Recursively move tensors to `device`, casting floating-point tensors to `dtype`."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, dtype=dtype) if obj.is_floating_point() else obj.to(device=device)
    if isinstance(obj, dict):
        return {k: _cast_inputs(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cast_inputs(v, device, dtype) for v in obj)
    return obj


def disable_hub_kernels(test_fn):
    """Force `is_kernels_available()` to `False` for the duration of an export test.

    Export must trace the pure-PyTorch path, never a Hub kernel (`mamba-ssm`, `causal-conv1d`, …): those
    need optional deps (`einops`, triton, …) and aren't exportable anyway. Kernels load lazily on the first
    (eager) forward — outside the exporter's own trace-time patch — so the whole test is wrapped. With
    `is_kernels_available()` False, `lazy_load_kernel` short-circuits to `None` and the fallback runs.
    (Also keeps a `pytest -n` sweep from hammering the `kernels-community` Hub org into a 429.)
    """

    @functools.wraps(test_fn)
    def wrapper(*args, **kwargs):
        from transformers.integrations import hub_kernels
        from transformers.utils import import_utils

        # `lazy_load_kernel` gates on `hub_kernels`'s own binding; patch the canonical def too.
        targets = [(hub_kernels, "is_kernels_available"), (import_utils, "is_kernels_available")]
        saved = [(obj, name, getattr(obj, name)) for obj, name in targets]
        for obj, name in targets:
            setattr(obj, name, lambda *args, **kwargs: False)
        try:
            return test_fn(*args, **kwargs)
        finally:
            for obj, name, original in saved:
                setattr(obj, name, original)

    return wrapper


def _clean_inputs_for_export(inputs_dict, config):
    """Strip None values and export-incompatible keys from an inputs dict. Mutates config in-place."""
    inputs_dict = {k: v for k, v in inputs_dict.items() if v is not None}
    for key in ("labels", "future_values", "return_loss"):
        inputs_dict.pop(key, None)
    config.return_loss = False
    return inputs_dict


def _run_onnx_program(onnx_program, inputs) -> dict:
    """Run an ONNX program and return outputs as a `{name: tensor}` dict."""
    set_seed(1234)
    onnx_inputs = get_leaf_tensors(inputs)
    onnx_outputs = onnx_program(**onnx_inputs)
    onnx_names = (re.sub(r"^output\.", "", node.name) for node in onnx_program.model_proto.graph.output)
    return dict(zip(onnx_names, onnx_outputs))


def _stage_openvino_artifact(ov_model, model_dir: str, component: str, config=None) -> None:
    """Stage the OpenVINO IR (+ ``config.json`` once per model) into the shared artifact tree as
    ``<model_dir>/<component>.{xml,bin}`` + ``<model_dir>/config.json``.

    Only the dynamic export is staged (the caller passes ``model_dir=None`` for static runs) — its
    symbolic shapes are the more useful reference and it avoids the two modes overwriting each other.

    The tree lives at ``OPENVINO_ARTIFACTS_DIR`` — created and exported by conftest's
    ``pytest_configure`` when ``PUSH_OPENVINO_ARTIFACTS`` is set. Under ``pytest -n`` every worker
    inherits that path and stages into the same tree; the controller uploads it once in one commit
    at session finish (see conftest). This avoids the per-process / per-worker upload races that an
    in-process ``atexit`` upload would cause.
    """
    import openvino

    root = os.environ.get("OPENVINO_ARTIFACTS_DIR")
    if root is None:  # conftest didn't set up staging (flag off) — nothing to do
        return
    dest = os.path.join(root, model_dir)
    os.makedirs(dest, exist_ok=True)
    openvino.save_model(ov_model, os.path.join(dest, f"{component}.xml"))
    config_path = os.path.join(dest, "config.json")
    if config is not None and not os.path.exists(config_path):
        config.to_json_file(config_path)


def _run_openvino_model(ov_model, inputs, model_dir=None, component=None, config=None) -> dict:
    """Compile an OpenVINO model and run it, returning outputs as a `{name: array}` dict.

    Feeds the tensor leaves that survived as input ports (stateful folding removes cache
    inputs), seeds folded state variables from the sample cache leaves so outputs correspond
    to the same inputs eager saw, supplies the identity `beam_idx`, and passes scalar kwargs
    through under their FX placeholder names.

    When ``PUSH_OPENVINO_ARTIFACTS`` is set, stages the IR under ``<model_dir>/<component>`` for a
    single Hub upload at process exit.
    """
    import numpy as np
    import openvino

    if model_dir is not None and os.environ.get("PUSH_OPENVINO_ARTIFACTS"):
        _stage_openvino_artifact(ov_model, model_dir, component, config)

    set_seed(1234)
    compiled = openvino.compile_model(ov_model, "AUTO")
    request = compiled.create_infer_request()
    leaves = {path: tensor.cpu() for path, tensor in get_leaf_tensors(inputs).items()}
    batch = next(iter(leaves.values())).shape[0] if leaves else 1

    feed = {}
    for port in compiled.inputs:
        # Passthrough tensors carry both an input and an output name — check every alias.
        for name in port.get_names():
            path = re.sub(r"^input\.", "", name)
            if path in leaves:
                feed[name] = leaves[path]
            elif name == "beam_idx":
                feed[name] = np.arange(batch, dtype=np.int32)
            elif name in inputs:
                feed[name] = np.array(inputs[name])
            else:
                continue
            break

    # Folded state variables read zeros on the first infer — seed them from the sample leaves
    # (cast to the variable's dtype: the exporter may retype state, e.g. i64 lengths to i32).
    # The variable id is ``input.<path>output.<path>``.
    def _state_path(state):
        return state.name[len("input.") : (len(state.name) - len("input.output.")) // 2 + len("input.")]

    for state in request.query_state():
        path = _state_path(state)
        if path in leaves:
            state.state = openvino.Tensor(leaves[path].numpy().astype(state.state.data.dtype, copy=False))

    results = request.infer(feed)
    outputs = {}
    for port in compiled.outputs:
        # Compilation may merge a named output tensor with an intermediate that kept its
        # numeric id — prefer the human-readable alias over ``get_any_name``'s sorted-first.
        names = sorted(port.get_names())
        name = next((n for n in names if not n.isdigit()), names[0])
        outputs[re.sub(r"^output\.", "", name)] = results[port]

    # Folded state tensors are outputs too — read them back so the returned dict covers the
    # same leaves eager returns.
    for state in request.query_state():
        outputs[_state_path(state)] = state.state.data.copy()

    return outputs


def _scope_applies(mapping: dict[str, dict[str, str]], active: set[str], name: str) -> bool:
    """Whether ``name`` is listed under any scope in ``mapping`` applicable to ``active``.

    A dotted scope key is a set of required tags; it applies when every tag is in ``active``
    (``"all"`` = no tags = always). Tag order is irrelevant and any combination composes.
    """
    return any(
        (set() if scope == "all" else set(scope.split("."))) <= active and name in entries
        for scope, entries in mapping.items()
    )


# ──────────────────────────── mixins ────────────────────────────


class ExportTesterMixin:
    """Mixin providing non-generative export tests for Dynamo, ONNX, and ExecuTorch backends.

    Mixed into [`ModelTesterMixin`] so every model test class that inherits from it
    automatically runs these export tests against all entries in `all_model_classes`.

    Expected attributes provided by [`ModelTesterMixin`]:
    - `all_model_classes` — iterable of model class objects to test.
    - `model_tester` — object with `prepare_config_and_inputs_for_common()` (and optionally
      `prepare_config_and_inputs_for_model_class()`).
    - `test_torch_exportable` — bool; set to `False` to skip all export tests for the model.
    - `_prepare_for_class(inputs_dict, model_class)` — adjusts inputs per model class.

    Tests are parameterised over `dynamic=True` / `dynamic=False` via `DYNAMIC_EXPORT_PARAMS`.
    Multi-modal models (detected by `is_multimodal`) are automatically decomposed and each
    submodule is tested independently.
    """

    def _skip_if_not_exportable(self):
        """Skip the test if the model architecture is not exportable."""
        if not self.test_torch_exportable:
            self.skipTest(reason="Model architecture is not Dynamo exportable/traceable")

        with open(inspect.getfile(self.all_model_classes[0]), "r") as f:
            source_code = f.read()
            # TODO: add use_experts_implementation support to remaining MoE models
            if "for expert" in source_code and "use_experts_implementation" not in source_code:
                self.skipTest(reason="Model architecture uses eager MoE implementation which is not torch exportable")

    def _should_skip(self, model_class, generate=False, dynamic=False, backend=None):
        """Return True if this model class should be skipped for export tests.

        Builds the active tag-set from ``(backend, generate, dynamic)`` and returns True if
        ``EXPORT_SKIPS`` lists the model under any applicable scope (see ``_scope_applies``).
        """
        active = set()
        if backend:
            active.add(backend)
        if generate:
            active.add("generate")
        if dynamic:
            active.add("dynamic")
        return _scope_applies(EXPORT_SKIPS, active, model_class.__name__)

    def _prepare_export_model_and_inputs(self, model_class):
        """Create model and forward inputs ready for export.

        Returns:
            `(config, components)`: the full model's init config, and a `{name: (model, inputs)}`
            mapping (one entry per component). The config is returned separately because components
            can be decomposed submodels carrying their own sub-configs, not the top-level one.
        """
        if hasattr(self.model_tester, "prepare_config_and_inputs_for_model_class"):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
        else:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict = self._prepare_for_class(inputs_dict, model_class)
        inputs_dict = _clean_inputs_for_export(inputs_dict, config)

        set_config_for_less_flaky_test(config)
        model = model_class(config).eval().to(torch_device)
        set_model_for_less_flaky_test(model)

        # Cast inputs to model device/dtype
        try:
            model_param = next(iter(model.parameters()))
            inputs_dict = _cast_inputs(inputs_dict, model_param.device, model_param.dtype)
        except StopIteration:
            pass

        if is_multimodal(model):
            return model.config, decompose_multimodal(model, inputs_dict)
        return model.config, {"model": (model, inputs_dict)}

    def _collect_eager_outputs(self, components):
        """Run eager forward for each component and return a ``{name: leaf_tensors}`` dict."""
        eager_outputs = {}
        for name, (model, inputs) in components.items():
            with torch.no_grad():
                set_seed(1234)
                eager_outputs[name] = get_leaf_tensors(model(**copy.deepcopy(inputs)))
                assert eager_outputs[name], f"Eager outputs are empty for {name}."
        return eager_outputs

    def _check_outputs_close(self, actual, expected, atol, rtol, check_device=True):
        """Assert outputs are close, allowing up to 5% element-level mismatch."""
        try:
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, check_device=check_device)
        except AssertionError as e:
            mismatched_percentage = re.findall(r"Mismatched elements: (\d+) / (\d+)", str(e))
            if mismatched_percentage:
                mismatched, total = map(int, mismatched_percentage[0])
                if mismatched / total < 0.05:
                    return  # allow up to 5%
            raise e

    # ──────────────────── torch.export tests ─────────────────────

    @DYNAMIC_EXPORT_PARAMS
    @slow
    @pytest.mark.torch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_torch_export(self, dynamic, atol=1e-4, rtol=1e-4):
        """Export each model class with ``torch.export`` and verify outputs match eager within tolerance."""
        self._skip_if_not_exportable()

        exporter = DynamoExporter()
        config = DynamoConfig(dynamic=dynamic)

        for model_class in self.all_model_classes:
            if self._should_skip(model_class, dynamic=dynamic):
                continue

            model_config, components = self._prepare_export_model_and_inputs(model_class)
            eager_outputs = self._collect_eager_outputs(components)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    exported_program = exporter.export(model, inputs, config=config)

                    with torch.no_grad():
                        set_seed(1234)
                        exported_outputs = get_leaf_tensors(exported_program.module()(**copy.deepcopy(inputs)))
                        self.assertTrue(exported_outputs, f"Exported outputs are empty for {name}.")

                    self._check_outputs_close(exported_outputs, eager_outputs[name], atol=atol, rtol=rtol)

    # ──────────────────────── ONNX tests ─────────────────────────

    @DYNAMIC_EXPORT_PARAMS
    @slow
    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_onnx_export(self, dynamic):
        """Export each model class to ONNX and verify output names match eager."""
        self._skip_if_not_exportable()

        for model_class in self.all_model_classes:
            if self._should_skip(model_class, dynamic=dynamic, backend="onnx"):
                continue

            exporter = OnnxExporter()
            config = OnnxConfig(dynamic=dynamic)

            model_config, components = self._prepare_export_model_and_inputs(model_class)
            eager_outputs = self._collect_eager_outputs(components)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    onnx_program = exporter.export(model, inputs, config=config)
                    onnx_outputs = _run_onnx_program(onnx_program, inputs)
                    self.assertTrue(onnx_outputs, f"ONNX outputs are empty for {name}.")
                    self.assertEqual(set(onnx_outputs.keys()), set(eager_outputs[name].keys()))

    # ──────────────────── OpenVINO tests ─────────────────────────

    @slow
    @DYNAMIC_EXPORT_PARAMS
    @require_openvino
    @pytest.mark.openvino_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @disable_hub_kernels
    def test_openvino_export(self, dynamic):
        """Export each model class to OpenVINO IR and verify output names match eager."""
        self._skip_if_not_exportable()
        exporter = OpenVINOExporter()
        config = OpenVINOConfig(dynamic=dynamic)

        for model_class in self.all_model_classes:
            if self._should_skip(model_class, dynamic=dynamic, backend="openvino"):
                continue

            model_config, components = self._prepare_export_model_and_inputs(model_class)
            eager_outputs = self._collect_eager_outputs(components)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    ov_model = exporter.export(model, inputs, config=config)
                    ov_outputs = _run_openvino_model(
                        ov_model, inputs, model_class.__name__ if dynamic else None, name, model_config
                    )
                    self.assertTrue(ov_outputs, f"OpenVINO outputs are empty for {name}.")
                    self.assertEqual(set(ov_outputs.keys()), set(eager_outputs[name].keys()))

    # ──────────────────── ExecuTorch tests ───────────────────────

    @DYNAMIC_EXPORT_PARAMS
    @slow
    @require_executorch
    @pytest.mark.executorch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_executorch_export(self, dynamic):
        """Export each model class to ExecuTorch (xnnpack on CPU, cuda on GPU) and verify no errors."""

        self._skip_if_not_exportable()
        exporter = ExecutorchExporter()
        config = ExecutorchConfig(dynamic=dynamic)

        for model_class in self.all_model_classes:
            if self._should_skip(model_class, dynamic=dynamic, backend="executorch"):
                continue

            model_config, components = self._prepare_export_model_and_inputs(model_class)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    exporter.export(model, inputs, config=config)


class ExportGenerateTesterMixin(ExportTesterMixin):
    """Mixin providing generation-aware export tests for torch.export, ONNX, and ExecuTorch backends.

    Inherits ``ExportTesterMixin`` for the shared exportability gate / skip logic / input prep, and
    is mixed into a model test class alongside ``GenerationTesterMixin``.

    Required attributes on the host class (in addition to those from ``ExportTesterMixin``):
    - ``all_generative_model_classes`` — iterable of generative model class objects to test.
    - ``prepare_config_and_inputs_for_generate()`` — returns ``(config, inputs_dict)`` suitable
      for ``model.generate()``.

    Each generative model is decomposed into prefill and decode components via
    :func:`decompose_prefill_decode`.  Multi-modal models additionally decompose the prefill
    stage into individual submodules via :func:`decompose_multimodal`.
    """

    def _prepare_export_generate_model_and_inputs(self, model_class):
        """Decompose a generative model into exportable components.

        For multi-modal models: decomposes the prefill stage into individual submodules plus the decode stage.
        For decoder-only models: returns prefill and decode components.

        Returns:
            `(config, components)`: the full model's init config and the `{name: (model, inputs)}`
            mapping (see `_prepare_export_model_and_inputs`).
        """
        config, inputs_dict = self.prepare_config_and_inputs_for_generate()
        inputs_dict = _clean_inputs_for_export(inputs_dict, config)

        set_config_for_less_flaky_test(config)
        model = model_class(config).eval().to(torch_device)
        set_model_for_less_flaky_test(model)

        return model.config, decompose_for_generation(model, inputs_dict)

    # ──────────────────── torch.export tests ─────────────────────

    @DYNAMIC_EXPORT_PARAMS
    @slow
    @pytest.mark.torch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_torch_export_generate(self, dynamic, atol=1e-4, rtol=1e-4):
        """Export prefill and decode stages with ``torch.export`` and verify outputs match eager."""
        self._skip_if_not_exportable()

        exporter = DynamoExporter()
        config = DynamoConfig(dynamic=dynamic)

        for model_class in self.all_generative_model_classes:
            if self._should_skip(model_class, generate=True):
                continue

            model_config, components = self._prepare_export_generate_model_and_inputs(model_class)
            eager_outputs = self._collect_eager_outputs(components)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    exported_program = exporter.export(model, inputs, config=config)

                    with torch.no_grad():
                        set_seed(1234)
                        exported_outputs = get_leaf_tensors(exported_program.module()(**copy.deepcopy(inputs)))
                        self.assertTrue(exported_outputs, "Exported outputs are empty.")

                    self._check_outputs_close(exported_outputs, eager_outputs[name], atol=atol, rtol=rtol)

    # ──────────────────────── ONNX tests ─────────────────────────

    @DYNAMIC_EXPORT_PARAMS
    @slow
    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_onnx_export_generate(self, dynamic):
        """Export prefill and decode stages to ONNX and verify output names match eager."""
        self._skip_if_not_exportable()

        for model_class in self.all_generative_model_classes:
            if self._should_skip(model_class, generate=True, dynamic=dynamic, backend="onnx"):
                continue

            exporter = OnnxExporter()
            config = OnnxConfig(dynamic=dynamic)

            model_config, components = self._prepare_export_generate_model_and_inputs(model_class)
            eager_outputs = self._collect_eager_outputs(components)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    onnx_program = exporter.export(model, inputs, config=config)
                    set_seed(1234)
                    onnx_outputs = _run_onnx_program(onnx_program, inputs)
                    self.assertTrue(onnx_outputs, "ONNX outputs are empty.")
                    self.assertEqual(set(onnx_outputs.keys()), set(eager_outputs[name].keys()))

    # ──────────────────── OpenVINO tests ─────────────────────────

    @slow
    @DYNAMIC_EXPORT_PARAMS
    @require_openvino
    @pytest.mark.openvino_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @disable_hub_kernels
    def test_openvino_export_generate(self, dynamic):
        """Export prefill and decode stages to OpenVINO IR and verify output names match eager."""
        self._skip_if_not_exportable()
        exporter = OpenVINOExporter()
        config = OpenVINOConfig(dynamic=dynamic)

        for model_class in self.all_generative_model_classes:
            if self._should_skip(model_class, generate=True, dynamic=dynamic, backend="openvino"):
                continue

            model_config, components = self._prepare_export_generate_model_and_inputs(model_class)
            eager_outputs = self._collect_eager_outputs(components)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    ov_model = exporter.export(model, inputs, config=config)
                    ov_outputs = _run_openvino_model(
                        ov_model, inputs, model_class.__name__ if dynamic else None, name, model_config
                    )
                    self.assertTrue(ov_outputs, "OpenVINO outputs are empty.")
                    self.assertEqual(set(ov_outputs.keys()), set(eager_outputs[name].keys()))

    # ──────────────────── ExecuTorch tests ───────────────────────

    @DYNAMIC_EXPORT_PARAMS
    @slow
    @require_executorch
    @pytest.mark.executorch_export_test
    @pytest.mark.timeout(EXPORT_TEST_TIMEOUT)
    @require_torch_greater_or_equal(MIN_EXPORT_TORCH_VERSION)
    @disable_hub_kernels
    def test_executorch_export_generate(self, dynamic):
        """Export prefill and decode stages to ExecuTorch and verify no errors."""

        self._skip_if_not_exportable()
        exporter = ExecutorchExporter()
        config = ExecutorchConfig(dynamic=dynamic)

        for model_class in self.all_generative_model_classes:
            if self._should_skip(model_class, generate=True, dynamic=dynamic, backend="executorch"):
                continue

            model_config, components = self._prepare_export_generate_model_and_inputs(model_class)

            for name, (model, inputs) in components.items():
                with self.subTest(f"{model_class.__name__}/{name}"):
                    exporter.export(model, inputs, config=config)
