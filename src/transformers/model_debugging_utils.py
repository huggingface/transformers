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
from contextlib import contextmanager
from typing import Optional

from transformers.utils.import_utils import export

from .utils import is_torch_available


if is_torch_available():
    import torch
    import torch.distributed.tensor
    from torch import nn

    from .modeling_utils import PreTrainedModel

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
        return {
            "shape": repr(value._local_tensor.shape),
            "dtype": repr(value._local_tensor.dtype),
            "value": _sanitize_repr_for_diff(repr(value)),
        }

    if isinstance(value, torch.Tensor):
        # standard PyTorch Tensor
        # return also the shape of such
        return {"shape": repr(value.shape), "dtype": repr(value.dtype), "value": _sanitize_repr_for_diff(repr(value))}

    # fallback for everything else (bool, int, float, None, or custom class)
    return _sanitize_repr_for_diff(repr(value))


def prune_outputs_if_children(node):
    # if there are children, remove this node's "outputs"
    # so we only see outputs at the leaf level
    if node.get("children"):
        node.pop("outputs", None)
        for child in node["children"]:
            prune_outputs_if_children(child)


def log_model_debug_trace(debug_path, model):
    if debug_path:
        try:
            os.makedirs(debug_path, exist_ok=False)
            output_path = os.path.join(debug_path, model._debugger_module_dump_name + "_debug_tree.json")
        except Exception as e:
            raise ValueError(f"Unexpected or existing debug_path={debug_path}. {e}")
    else:
        output_path = model._debugger_module_dump_name + "_debug_tree.json"
    logger.info(f"Writing model trace at {output_path}")
    with open(output_path, "w") as outfile:
        prune_outputs_if_children(model._call_tree)
        json.dump(model._call_tree, outfile, indent=2)


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
            with torch.inference_mode():
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

        return out

    model.forward = top_wrapped_forward

    # Final hook for writing JSON on forward-end
    def final_hook(_, inputs, outputs):
        if _is_rank_zero() and model._debugger_model_call_stack:
            finished = model._debugger_model_call_stack.pop()
            model._call_tree["inputs"] = finished["inputs"]
            model._call_tree["outputs"] = finished["outputs"]
            model._call_tree["children"] = finished["children"]

        if _is_rank_zero():
            log_model_debug_trace(debug_path=debug_path, model=model)

    model.register_forward_hook(final_hook)
    # Optionally also for a couple possible hooks that have specific names. It should be just one.
    # This means modules that are not typically called "forward" within the model. But we should not need to recurse
    # through them.
    possible_model_calls = ["language_model", "model"]
    for model_call in possible_model_calls:
        this_model_call = getattr(model, model_call, None)
        if this_model_call and isinstance(this_model_call, (nn.Module, PreTrainedModel)):
            this_model_call.register_forward_hook(final_hook)
            break  # exit the loop after finding one (unsure, but should be just one call.)


@export(backends=("torch",))
def model_addition_debugger(cls):
    """
    # Model addition debugger - a model adder tracer
    This decorator is a power user tool intended for model adders.
    It tracks all forward calls within a model forward and logs a slice of each input and output on a nested Json.
    To note, this decorator enforces `torch.inference_mode()`.
    ## Usage

    add decorator to your model class
    ```python
    from ...modeling_utils import model_addition_debugger

    @model_addition_debugger
    class MyModel(nn.Module) # Can inherit from PreTrainedModel too
        # ... nothing else changes
    ```
    Then, in a separate script (example is for Llava)

    ```python
    import torch
    from PIL import Image
    import requests
    from transformers import LlavaProcessor, LlavaForConditionalGeneration
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
    with torch.no_grad():
        output = model.forward(**inputs)
    ```

    """
    orig_init = cls.__init__

    @functools.wraps(cls.__init__)
    def wrapped_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        _attach_debugger_logic(self, cls.__name__)

    cls.__init__ = wrapped_init
    return cls


@export(backends=("torch",))
@contextmanager
def model_addition_debugger_context(model, debug_path: Optional[str] = None):
    """
    # Model addition debugger - context manager for model adders
    This context manager is a power user tool intended for model adders.
    It tracks all forward calls within a model forward and logs a slice of each input and output on a nested Json.
    To note, this context manager enforces `torch.inference_mode()`.

    ## Usage

    add the context manager to a model to debug

    ```python
    import torch
    from PIL import Image
    import requests
    from transformers import LlavaProcessor, LlavaForConditionalGeneration
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
    _attach_debugger_logic(model, model.__class__.__name__, debug_path)
    try:
        yield model
    finally:
        pass
