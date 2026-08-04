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
"""Shared helpers to *run* exported programs (ONNX in ORT, ExecuTorch in its runtime).

Used by both the per-model export tests (`test_export.py`, which check outputs against eager) and the
quantization tests (`test_quantization.py`, which only check the quantized graph executes). Kept out of
either test module so neither imports the other; the filename doesn't match pytest's `test_*` pattern,
so it isn't collected.
"""

import re

from transformers import set_seed
from transformers.exporters.utils import get_leaf_tensors
from transformers.utils.import_utils import is_torch_available


if is_torch_available():
    import torch


def run_onnx_program(onnx_program, inputs) -> dict:
    """Run an ONNX program and return outputs as a `{name: tensor}` dict."""
    set_seed(1234)
    onnx_inputs = get_leaf_tensors(inputs)
    onnx_outputs = onnx_program(**onnx_inputs)
    onnx_names = (re.sub(r"^output\.", "", node.name) for node in onnx_program.model_proto.graph.output)
    return dict(zip(onnx_names, onnx_outputs))


def run_executorch_program(program_manager, inputs):
    """Load and run an ExecuTorch program, returning its outputs — or ``None`` to skip this component.

    ``None`` means "move on to the next component" and is returned when either:
    - the export is valid but ExecuTorch's own runtime can't service it — a missing portable kernel
      (``0x14``), an oversized arena (``0x21`` / ``bad_alloc``), or a portable-kernel / XNNPACK-delegate
      failure at execute (``0x12`` / ``0x1``): a runtime limitation, not a transformers export defect; or
    - the inputs couldn't be reconstructed for this program (a derived symint slot with no eager leaf).

    Otherwise the model's declared outputs are returned for the caller to check against eager.
    ``torch.export`` also appends mutated inputs (in-place-modified ``pixel_values``, recurrent state,
    …) to the program outputs; those are dropped here — keeping only ``USER_OUTPUT`` slots — so the
    result matches eager's returned leaves.

    Inputs are bound *positionally* against the program's declared slots (``num_inputs`` /
    ``input_tensor_meta``), filled in order from the eager pytree leaves — tensor leaves for tensor
    slots, scalars for the rest.
    """
    from executorch.runtime import Runtime, Verification

    set_seed(1234)
    leaves = torch.utils._pytree.tree_leaves(inputs)
    # The runtime rejects non-contiguous inputs, so materialise tensor leaves. `int` covers `bool`.
    tensors = [t.contiguous() for t in leaves if isinstance(t, torch.Tensor)]
    scalars = (t for t in leaves if isinstance(t, (int, float)))

    # Load — surfaces ExecuTorch resource limits (missing portable kernel / oversized arena).
    try:
        program = Runtime.get().load_program(program_manager.buffer, verification=Verification.Minimal)
        method = program.load_method("forward")
    except (RuntimeError, MemoryError) as e:
        if is_executorch_runtime_limit(e):
            return None
        raise

    # Each slot declares its shape; match it to an eager tensor leaf of that shape so the right tensor
    # lands in the right slot (count alone isn't enough — a wrong-shape tensor crashes conv/copy
    # kernels at execute). Under dynamic shapes the declared shape is an upper bound and won't match a
    # leaf, so fall back to the next unused leaf (leaf order tracks the program's input order). If a
    # slot can't be filled — a derived symint, or no leaf of the right shape — reconstruction isn't
    # possible; return None and rely on the load check rather than run with bogus inputs.
    args = []
    for i in range(method.metadata.num_inputs()):
        try:
            shape = tuple(method.metadata.input_tensor_meta(i).sizes())
        except Exception:  # non-tensor slot
            args.append(next(scalars, None))
        else:
            match = next((t for t in tensors if tuple(t.shape) == shape), tensors[0] if tensors else None)
            if match is not None:
                tensors.remove(match)
            args.append(match)
        if args[-1] is None:
            return None

    try:
        outputs = method.execute(args)
    except (RuntimeError, MemoryError) as e:
        if is_executorch_runtime_limit(e):
            return None
        raise

    # Drop `torch.export`'s appended mutated-input outputs, keeping only the model's `USER_OUTPUT`s
    # (in program-output order). Then keep tensors only, mirroring eager's `get_leaf_tensors`, so the
    # returned outputs line up with eager's returned leaves for the caller's count check.
    exported_program = program_manager.exported_program
    exported_program = exported_program() if callable(exported_program) else exported_program
    output_kinds = [spec.kind.name for spec in exported_program.graph_signature.output_specs]
    if len(output_kinds) == len(outputs):
        outputs = [out for out, kind in zip(outputs, output_kinds) if kind == "USER_OUTPUT"]
    return [out for out in outputs if isinstance(out, torch.Tensor)]


# ExecuTorch runtime error codes that mean "the export is valid (it produced a loadable program) but
# ExecuTorch's own portable runtime / XNNPACK backend can't service it" — a runtime limitation, not a
# transformers export defect (which surfaces earlier as a `torch.export` error or later as an output
# mismatch). Load: 0x14 missing portable kernel, 0x21 arena can't be allocated, 0x1 XNNPACK partition
# won't compile (`xnn_status_unsupported_parameter`). Execute: 0x12 portable-kernel InvalidArgument
# (constant_pad_nd/convolution/upsample_aa out-tensor sizing), 0x1 XNNPACK delegate failure, 0x10
# XNNPACK delegate can't resize a static tensor to the runtime shape. The execute-phase codes surface
# from either `execute()` or `set_inputs()` (binding the runtime inputs is part of `Method.execute`).
_ET_LOAD_LIMIT_CODES = {"0x1", "0x14", "0x21"}
_ET_EXECUTE_LIMIT_CODES = {"0x1", "0x10", "0x12"}


def is_executorch_runtime_limit(exc):
    """True if ``exc`` is a known ExecuTorch runtime limitation (missing kernel / arena / kernel bug)."""
    msg = str(exc)
    if isinstance(exc, MemoryError) or "bad_alloc" in msg:
        return True
    load = re.search(r"Failed to load method forward, error: 0x:?([0-9a-fA-F]+)", msg)
    if load and f"0x{load.group(1)}" in _ET_LOAD_LIMIT_CODES:
        return True
    execute = re.search(r"(?:execute\(\)|set_inputs\(\) for method '\w+') failed with error 0x([0-9a-fA-F]+)", msg)
    return bool(execute and f"0x{execute.group(1)}" in _ET_EXECUTE_LIMIT_CODES)
