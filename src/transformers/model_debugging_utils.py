# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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

import functools
import json
import os
import re
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from typing import Optional

from transformers.utils.import_utils import requires

from .utils import is_torch_available


if is_torch_available():
    import torch
    import torch.distributed.tensor


from .utils import logging


logger = logging.get_logger(__name__)

# Note to code inspectors: this toolbox is intended for people who add models to `transformers`.
_torch_distributed_available = torch.distributed.is_available()


def _is_rank_zero():
    """Return True if rank=0 or we aren't running distributed."""
    if not (_torch_distributed_available and torch.distributed.is_initialized()):
        return True
    return torch.distributed.get_rank() == 0


MEMORY_ADDRESS_REGEX = re.compile(r"object at 0x[0-9A-Fa-f]+")


def _sanitize_repr_for_diff(x_str: str) -> str:
    """
    Replace memory addresses in an object's repr with a stable placeholder
    so that beautiful JSON diffs won't be ruined by ephemeral addresses.
    """
    return MEMORY_ADDRESS_REGEX.sub("object at 0xXXXXXXXX", x_str)


def _dtensor_repr(x):
    """Return a stable string representation for a DTensor-like object."""
    if _is_rank_zero():
        return f"DTensor (rank0) -> {repr(x._local_tensor)}"
    return "DTensor(non-rank0)"


def _serialize_io(value):
    """
    Recursively build a JSON-serializable Python structure from `value`.
    Tensors and DTensors become sanitized repr strings.
    Lists/tuples/dicts are recursed into.
    All memory addresses are replaced with a stable placeholder.

    Args:
        value: Any Python object, often including torch Tensors, lists, dicts, etc.

    Returns:
        A nested Python structure (list, dict, or sanitized string) that is safe to json.dump.
    """
    if isinstance(value, (list, tuple)):
        return [_serialize_io(v) for v in value]

    if isinstance(value, dict):
        return {k: _serialize_io(v) for k, v in value.items()}

    if hasattr(value, "_local_tensor"):
        # DTensor-like handling, just use local tensor attribute
        torch.set_printoptions(sci_mode=True)
        val_repr = _repr_to_list(value)
        out = {
            "shape": repr(value._local_tensor.shape),
            "dtype": repr(value._local_tensor.dtype),
            "value": val_repr,
        }
        if value._local_tensor.dtype in {torch.float16, torch.float32, torch.bfloat16}:
            value = value._local_tensor.clone()
            out.update(
                {
                    "mean": _sanitize_repr_for_diff(repr(value.mean())),
                    "std": _sanitize_repr_for_diff(repr(value.std())),
                    "min": _sanitize_repr_for_diff(repr(value.min())),
                    "max": _sanitize_repr_for_diff(repr(value.max())),
                }
            )
        return out

    if isinstance(value, torch.Tensor):
        torch.set_printoptions(sci_mode=True)
        val_repr = _repr_to_list(value)
        out = {
            "shape": repr(value.shape),
            "dtype": repr(value.dtype),
            "value": val_repr,
        }
        if value.dtype in {torch.float16, torch.float32, torch.bfloat16}:
            out.update(
                {
                    "mean": _sanitize_repr_for_diff(repr(value.mean())),
                    "std": _sanitize_repr_for_diff(repr(value.std())),
                    "min": _sanitize_repr_for_diff(repr(value.min())),
                    "max": _sanitize_repr_for_diff(repr(value.max())),
                }
            )
        return out

    return _sanitize_repr_for_diff(repr(value))


def _repr_to_list(val):
    return _sanitize_repr_for_diff(repr(val)).splitlines()


def _repr_to_list(value: torch.Tensor):
    torch.set_printoptions(sci_mode=True, linewidth=120)
    with StringIO() as buf, redirect_stdout(buf):
        print(value)  # to redirected stdout to avoid line splits
        raw = buf.getvalue()
    return _sanitize_repr_for_diff(raw).splitlines()


def prune_outputs_if_children(node):
    # if there are children, remove this node's "outputs"
    # so we only see outputs at the leaf level
    if node.get("children"):
        node.pop("outputs", None)
        for child in node["children"]:
            prune_outputs_if_children(child)


LAYER_SUFFIX_RE = re.compile(r"(.*)\.(\d+)$")  # should be generic enough, ends with a number


def is_layer_block(node):
    match = LAYER_SUFFIX_RE.match(node.get("module_path", ""))
    if not match or not node.get("children"):
        return False
    number = match.group(2)
    return any(f".{number}." in child.get("module_path", "") for child in node["children"])


def prune_intermediate_layers(node):
    if not node.get("children"):
        return

    layer_blocks = [(i, child) for i, child in enumerate(node["children"]) if is_layer_block(child)]

    if len(layer_blocks) > 2:
        to_remove = [i for i, _ in layer_blocks[1:-1]]
        node["children"] = [child for i, child in enumerate(node["children"]) if i not in to_remove]

    for child in node["children"]:
        prune_intermediate_layers(child)


def log_model_debug_trace(debug_path, model):
    if debug_path:
        try:
            os.makedirs(debug_path, exist_ok=True)
            base = os.path.join(debug_path, model._debugger_module_dump_name + "_debug_tree")
        except Exception as e:
            raise ValueError(f"Unexpected or existing debug_path={debug_path}. {e}")
    else:
        base = model._debugger_module_dump_name + "_debug_tree"

    logger.info(f"Writing model trace at {base}.json")
    full_path = base + "_FULL_TENSORS.json"
    summary_path = base + "_SUMMARY.json"

    prune_outputs_if_children(model._call_tree)

    with open(full_path, "w") as f:
        json.dump(model._call_tree, f, indent=2)

    # summary-only version for readability - traversing the tree again #TODO optimize?
    def strip_values(node):
        def clean(val):
            if isinstance(val, dict):
                val.pop("value", None)
                for v in val.values():
                    clean(v)
            elif isinstance(val, list):
                for item in val:
                    clean(item)

        clean(node.get("inputs", {}))
        clean(node.get("outputs", {}))

        for child in node.get("children", []):
            strip_values(child)

    tree_copy = json.loads(json.dumps(model._call_tree))  # deep copy
    strip_values(tree_copy)

    with open(summary_path, "w") as f:
        json.dump(tree_copy, f, indent=2)


def _attach_debugger_logic(model, class_name, debug_path: str):
    # Prepare data structures on the model object
    model._call_tree = {"module_path": class_name, "inputs": None, "outputs": None, "children": []}
    model._debugger_model_call_stack = []
    model._debugger_module_dump_name = class_name  # used for final JSON filename

    def wrap_forward(module, full_path):
        orig_forward = module.forward

        @functools.wraps(orig_forward)
        def wrapped_forward(*inps, **kws):
            if _is_rank_zero():
                dict_inputs = {"args": inps, "kwargs": kws}
                dict_inputs = {k: dict_inputs[k] for k in dict_inputs if len(dict_inputs[k]) > 0}
                node = {
                    "module_path": full_path,
                    "inputs": _serialize_io(dict_inputs),
                    "outputs": None,
                    "children": [],
                }
                model._debugger_model_call_stack.append(node)
            with torch.no_grad():
                out = orig_forward(*inps, **kws)

            if _is_rank_zero():
                if sum(1 for _ in module.named_children()) > 0:
                    node["outputs"] = None
                else:
                    node["outputs"] = _serialize_io(out)

                finished = model._debugger_model_call_stack.pop()
                # prune empty vertices here as well (mostly empty children nodes)
                if not finished["children"]:
                    finished.pop("children")

                if model._debugger_model_call_stack:
                    model._debugger_model_call_stack[-1]["children"].append(finished)
            return out

        module.forward = wrapped_forward

    # wrap all submodules
    for name, submodule in model.named_modules():
        if name == "":
            continue
        wrap_forward(submodule, f"{class_name}.{name}")

    # wrap top-level forward
    real_top_forward = model.forward

    @functools.wraps(real_top_forward)
    def top_wrapped_forward(*inps, **kws):
        if _is_rank_zero():
            top_node = {
                "module_path": f"{class_name} (top-level)",
                "inputs": _serialize_io({"args": inps, "kwargs": kws}),
                "outputs": None,
                "children": [],
            }
            model._debugger_model_call_stack.append(top_node)

        out = real_top_forward(*inps, **kws)
        if _is_rank_zero() and model._debugger_model_call_stack:
            top_node["outputs"] = _serialize_io(out)
            finished = model._debugger_model_call_stack.pop()
            model._call_tree["inputs"] = finished["inputs"]
            model._call_tree["outputs"] = finished["outputs"]
            model._call_tree["children"] = finished["children"]
            # prune empty stuff for visibility
            [model._call_tree.pop(k, None) for k in list(model._call_tree.keys()) if not model._call_tree[k]]

            # prune layers that are not 0 or last
            prune_intermediate_layers(model._call_tree)
            # Write final JSON trace here
            log_model_debug_trace(debug_path=debug_path, model=model)
        return out

    model.forward = top_wrapped_forward


@requires(backends=("torch",))
@contextmanager
def model_addition_debugger_context(model, debug_path: Optional[str] = None):
    """
    # Model addition debugger - context manager for model adders
    This context manager is a power user tool intended for model adders.
    It tracks all forward calls within a model forward and logs a slice of each input and output on a nested Json.
    To note, this context manager enforces `torch.no_grad()`.

    ## Usage

    add the context manager to a model to debug

    ```python
    import torch
    from PIL import Image
    import requests
    from transformers import LlavaProcessor, LlavaForConditionalGeneration
    from transformers.model_debugging_utils import model_addition_debugger_context
    torch.random.manual_seed(673)

    # load pretrained model and processor
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = LlavaProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, low_cpu_mem_usage=True)

    # create random image input
    random_image = Image.fromarray(torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8).numpy())

    # prompt
    prompt = "<image>Describe this image."

    # process inputs
    inputs = processor(text=prompt, images=random_image, return_tensors="pt")

    # call forward method (not .generate!)
    with model_addition_debugger_context(model):
        output = model.forward(**inputs)
    ```

    """
    orig_forwards = {m: m.forward for _, m in model.named_modules()}
    orig_forwards[model] = model.forward
    _attach_debugger_logic(model, model.__class__.__name__, debug_path)
    try:
        yield model
    finally:
        for module_instance, forward_method in orig_forwards.items():
            module_instance.forward = forward_method
