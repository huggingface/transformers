# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""
Contains the logic for automatic additional output capture with our forward decorators.
This mostly describe the hooks used and the logic to make capture thread/context safe.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING

from .import_utils import is_torchdynamo_compiling, requires


if TYPE_CHECKING:
    from torch import nn

    from ..modeling_utils import PreTrainedModel


_CAN_RECORD_REGISTRY = {}


@dataclass
@requires(backends=("torch",))
class OutputRecorder:
    """
    Configuration for recording outputs from a model via hooks.

    Attributes:
        target_class (Type): The class (e.g., nn.Module) to which the hook will be attached.
        index (Optional[int]): If the output is a tuple/list, optionally record only at a specific index.
        layer_name (Optional[str]): Name of the submodule to target (if needed), e.g., "transformer.layer.3.attn".
        class_name (Optional[str]): Name of the class to which the hook will be attached. Could be the suffix of class name in some cases.
        capture_initial_hidden_state  (bool): Whether to prepend the first module's input as the initial hidden state.
    """

    target_class: type[nn.Module]
    index: int = 0
    layer_name: str | None = None
    class_name: str | None = None
    capture_initial_hidden_state: bool = True


class CompileableContextVar:
    """
    Compatibility wrapper around a ContextVar for use with `torch.compile`.

    Uses context-local state outside tracing and a shared value when set during tracing. The shared fallback avoids
    ContextVar mutation during tracing, but does not provide thread/context isolation.
    """

    def __init__(self, name):
        self.context_var = ContextVar(name, default=None)
        self.global_var = None
        self.compiling = False

    def get(self):
        # Set was called before and compilation was already detected
        if self.compiling:
            return self.global_var
        else:
            return self.context_var.get()

    def set(self, value):
        if is_torchdynamo_compiling():
            self.global_var = value
            self.compiling = True
            return None
        else:
            return self.context_var.set(value)

    def reset(self, token):
        if self.compiling or token is None:
            self.global_var = None
            self.compiling = False
        else:
            self.context_var.reset(token)

    @contextmanager
    def temporary_value(self, value):
        """Temporarily use `value` in this Python context, restoring the previous binding on exit.

        Enter outside tracing, while `get()` reads `context_var`.
        This leaves `global_var` and `compiling` unchanged.
        """
        token = self.context_var.set(value)
        try:
            yield
        finally:
            self.context_var.reset(token)


# Context-local outside tracing; the compilation fallback uses shared state.
_active_collector = CompileableContextVar("output_collector")


def _append_captured_output(collector, key, initial_args, output, index):
    """Append one output, preserving initial and selected hidden-state positions."""
    values = collector[key]
    selected = collector.get("_hidden_states_layers") if key == "hidden_states" else None
    if initial_args is not None and not values and selected is None:
        values.append(initial_args[0])
    if selected is not None and len(values) not in selected:
        values.append(None)
        return
    if not isinstance(output, tuple):
        values.append(output)
    elif output[index] is not None:
        values.append(output[index])


@contextmanager
def _fresh_output_collection():
    """
    Collect requested router, attention and hidden-state outputs in fresh lists for each region.

    Enter from eager code around the call to a compiled region. Fresh lists prevent Dynamo from specializing on
    the number of outputs collected by earlier regions. Merge them into the enclosing collector on exit, including
    when the region raises. Resolve hidden-state layer positions on exit, outside the compiled region.
    Native ContextVar support (#46370) does not remove compilation guards rightfully triggering on a growing list.
    Whole-model tracing captures the enclosing loop and list growth together, so leave its collector unchanged.
    """
    if is_torchdynamo_compiling() or _active_collector.compiling:
        yield
        return

    collector = _active_collector.get()
    if collector is None:
        yield
        return

    keys = tuple(key for key in ("router_logits", "attentions", "hidden_states") if key in collector)
    if not keys:
        yield
        return

    local_collector = dict(collector)
    for key in keys:
        local_collector[key] = []
    if "hidden_states" in keys:
        local_collector["_pending_hidden_states"] = []
    try:
        with _active_collector.temporary_value(local_collector):
            yield
    finally:
        for key in keys:
            if key == "hidden_states":
                pending_hidden_states = local_collector["_pending_hidden_states"]
                parent_pending = collector.get("_pending_hidden_states")
                if parent_pending is not None:
                    parent_pending.extend(pending_hidden_states)
                else:
                    for initial_args, output, index in pending_hidden_states:
                        _append_captured_output(collector, key, initial_args, output, index)
            else:
                collector[key].extend(local_collector[key])


def install_output_capuring_hook(
    module: nn.Module, key: str, index: int, capture_initial_hidden_state: bool = True
) -> None:
    """Install the forward hook needed to capture the output described by `key` and `index` in `module`."""

    def output_capturing_hook(module, args, output):
        # Get the current thread-local collector
        collected_outputs = _active_collector.get()
        # If it's None or not a key we want to capture, simply return, the hook is inactive
        if collected_outputs is None or key not in collected_outputs.keys():
            return

        initial_args = None
        if (
            key == "hidden_states"
            and capture_initial_hidden_state
            and collected_outputs.get("_hidden_states_layers") is None
        ):
            initial_args = args[:1]
        pending_hidden_states = collected_outputs.get("_pending_hidden_states")
        if key == "hidden_states" and pending_hidden_states is not None:
            pending_hidden_states.append((initial_args, output, index))
        else:
            _append_captured_output(collected_outputs, key, initial_args, output, index)

    module.register_forward_hook(output_capturing_hook)


def recursively_install_hooks(
    parent_module: nn.Module, module_name: str, capture_tasks: list[tuple[str, OutputRecorder]]
) -> None:
    """
    Recursively install all output capturing hooks on all submodules of `parent_module`.
    Note that we need to use this recursive approach instead of simply iterating over all modules, because we want
    to respect the `capture_tasks` of all individual submodels (`PreTrainedModel` instances) in the graph. That is, once
    we reach a submodel in the graph, its children should use this submodel's `capture_tasks`, but other parts of the graph
    should not.
    """
    from ..modeling_utils import PreTrainedModel

    # First dispatch to children if needed
    for name, module in parent_module.named_children():
        # Keep dispatching the same `capture_tasks`
        if not isinstance(module, PreTrainedModel):
            recursively_install_hooks(module, f"{module_name}.{name}", capture_tasks)
        # New Submodel: we need to dispatch its own `capture_tasks`
        else:
            install_all_output_capturing_hooks(module, prefix=f"{module_name}.{name}")

    # Potentially install the hook on current `parent_module`
    for key, specs in capture_tasks:
        # Check if the spec matches the target class
        match_target_class = specs.target_class is not None and isinstance(parent_module, specs.target_class)
        # This check is for multimodals where only backbone layer suffix is available
        match_class_name = specs.class_name is not None and module_name.endswith(specs.class_name)

        if match_target_class or match_class_name:
            # If the spec has a specified layer name, check it
            if specs.layer_name is not None:
                # Format the target layer name to have one dot on both sides
                target_layer_name = specs.layer_name.strip(".")
                target_layer_name = "." + target_layer_name + "."
                # Match it against the module name (with a trailing dot to match in case the target is trailing)
                matches = target_layer_name in module_name + "."
                if not matches:
                    continue

            install_output_capuring_hook(parent_module, key, specs.index, specs.capture_initial_hidden_state)


def install_all_output_capturing_hooks(model: PreTrainedModel, prefix: str | None = None) -> None:
    """
    Install the output recording hooks on all the modules in `model`. This will take care of correctly dispatching
    the `_can_record_outputs` property of each individual submodels in case of composite models.
    """
    # _can_record_outputs is None by default
    capture_flags = _CAN_RECORD_REGISTRY.get(str(model.__class__)) or {}  # there is a weak ref for executorch

    capture_tasks = []
    for key, layer_specs in capture_flags.items():
        if not isinstance(layer_specs, list):
            layer_specs = [layer_specs]
        for specs in layer_specs:
            if not isinstance(specs, OutputRecorder):
                index = 0 if "hidden_states" in key else 1
                class_name = None if not isinstance(specs, str) else specs
                target_class = specs if not isinstance(specs, str) else None
                specs = OutputRecorder(target_class=target_class, index=index, class_name=class_name)
            capture_tasks.append((key, specs))

    # Install the hooks
    prefix = prefix if prefix is not None else ""
    recursively_install_hooks(model, prefix, capture_tasks)
    # Mark the model as already hooked
    setattr(model, "_output_capturing_hooks_installed", True)


# We need this to make sure we don't have race conditions when installing hooks, resulting in them being installed
# several times
_hook_installation_lock = threading.Lock()


def maybe_install_capturing_hooks(model: PreTrainedModel) -> None:
    """
    Check if the model already has output capturing hooks installed, and install them if it is not already the
    case.
    Note that this is thread-safe, in case 2 (or more) threads want to install them concurrently.
    """
    # First check
    if getattr(model, "_output_capturing_hooks_installed", False):
        return

    with _hook_installation_lock:
        # Second check, in case several threads entered this function concurrently and did not return on the
        # previous check
        if getattr(model, "_output_capturing_hooks_installed", False):
            return
        # This will install the hooks and mark the model as hooked
        install_all_output_capturing_hooks(model)


def capture_outputs(func=None, *, tie_last_hidden_states=True):
    """
    Decorator to intercept specific layer outputs through hooks. The hooks are installed only once and lazily,
    the first time output capture is requested with the `output_xxx` kwargs/config.
    The implementation is fully context/thread safe, except when using `torch.compile`, as dynamo is unable to trace
    through `ContextVar` methods.

    Args:
        tie_last_hidden_states (`bool`, *optional*, defaults to `True`):
            Whether to overwrite `out.hidden_states[-1]` with the `out.last_hidden_state`. This is true for all language models
            and should be toggled off only if `out.hidden_states[-1]` has to be the hidden state before last layer norm, which
            is needed for some vision models (e.g. CLIP, SigLIP). A model config can override this default per-model by setting
            `config.tie_last_hidden_states`.
    """

    def wrapped_fn(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Pop it so that internal modules always return a dict even if False is requested
            return_dict = kwargs.pop("return_dict", getattr(self.config, "return_dict", True))

            # _can_record_outputs is None by default
            capturable_flags = _CAN_RECORD_REGISTRY.get(str(self.__class__)) or {}
            recordable_keys = {
                f"output_{k}": kwargs.get(f"output_{k}", getattr(self.config, f"output_{k}", False))
                for k in capturable_flags
            }
            # For BC as cross-attentions used to be captured with `output_attentions`
            if "cross_attentions" in capturable_flags:
                recordable_keys["output_cross_attentions"] = kwargs.get(
                    "output_attentions", getattr(self.config, "output_attentions", False)
                )
            # The sam model variants need this annoying exception as well...
            if "mask_decoder_attentions" in capturable_flags:
                recordable_keys["output_mask_decoder_attentions"] = kwargs.get(
                    "output_attentions", getattr(self.config, "output_attentions", False)
                )

            collected_outputs = {k.replace("output_", ""): [] for k, v in recordable_keys.items() if v}
            # We accept a list of layer indices as `output_hidden_states`, to capture only specific layer outputs - in this case
            # we need to add the layers to the `collected_outputs`'s dict to tell the hook which ones we need
            if "output_hidden_states" in recordable_keys and isinstance(
                recordable_keys["output_hidden_states"], (list, tuple, set)
            ):
                collected_outputs["_hidden_states_layers"] = set(recordable_keys["output_hidden_states"])
            # Make sure hooks are installed if we need to collect outputs
            if len(collected_outputs) > 0:
                maybe_install_capturing_hooks(self)
            # Let's activate the output collector hooks if needed!
            output_token = _active_collector.set(collected_outputs)

            # Run the forward
            try:
                outputs = func(self, *args, **kwargs)
            # Reset the states
            finally:
                _active_collector.reset(output_token)

            hidden_states_layers = collected_outputs.pop("_hidden_states_layers", None)
            # Inject collected outputs into model output (return everything as tuples for BC)
            for key in collected_outputs:
                if key == "hidden_states":
                    tie_last = getattr(self.config, "tie_last_hidden_states", None)
                    tie_last = tie_last_hidden_states if tie_last is None else tie_last
                    if not tie_last or (hidden_states_layers is not None and collected_outputs[key][-1] is None):
                        pass
                    elif hasattr(outputs, "vision_hidden_states"):
                        collected_outputs[key] = collected_outputs[key][:-1]
                        collected_outputs[key].append(outputs.vision_hidden_states)
                    elif hasattr(outputs, "last_hidden_state"):
                        collected_outputs[key] = collected_outputs[key][:-1]
                        collected_outputs[key].append(outputs.last_hidden_state)

                outputs[key] = tuple(collected_outputs[key])

            if return_dict is False:
                outputs = outputs.to_tuple()

            return outputs

        return wrapper

    if func is not None:
        return wrapped_fn(func)
    return wrapped_fn
