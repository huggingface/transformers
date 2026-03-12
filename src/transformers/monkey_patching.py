# Copyright 2026 The HuggingFace Inc. team.
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

import re
import sys
import threading
from contextlib import contextmanager

from .utils import is_torch_available, logging
from .utils.output_capturing import OutputRecorder


if is_torch_available():
    import torch.nn as nn

logger = logging.get_logger(__name__)

_monkey_patch_mapping_cache: dict[str, type[nn.Module]] = {}
_compiled_patterns_cache: dict[str, re.Pattern] = {}
_monkey_patch_lock = threading.Lock()


def _compile_pattern(pattern: str) -> re.Pattern | None:
    """
    Compile a regex pattern and cache it. Returns None if pattern is invalid.

    Args:
        pattern: The regex pattern string to compile

    Returns:
        Compiled regex pattern or None if invalid
    """
    if pattern in _compiled_patterns_cache:
        return _compiled_patterns_cache[pattern]

    try:
        compiled = re.compile(pattern)
        _compiled_patterns_cache[pattern] = compiled
        return compiled
    except re.error as e:
        logger.warning(f"Invalid regex pattern '{pattern}': {e}. Treating as non-pattern.")
        return None


def _find_replacement_class(class_name: str, mapping: dict[str, type[nn.Module]]) -> type[nn.Module] | None:
    """
    Find replacement class for a given class name, checking exact matches first, then regex patterns.

    Args:
        class_name: The class name to find a replacement for
        mapping: Dictionary of patterns/names to replacement classes

    Returns:
        The replacement class if found, None otherwise
    """
    # First check for exact match (highest priority)
    if class_name in mapping:
        return mapping[class_name]

    # Then check regex patterns
    for pattern, replacement_class in mapping.items():
        # Skip if already matched as exact
        if pattern == class_name:
            continue

        # Try to compile and match as regex
        compiled_pattern = _compile_pattern(pattern)
        if compiled_pattern is not None and compiled_pattern.search(class_name):
            return replacement_class

    return None


def register_patch_mapping(mapping: dict[str, type[nn.Module]], overwrite: bool = False) -> None:
    """
    Register patch mappings to enable automatic patching during model creation using `from_pretrained`,
    `from_config` or within the `apply_patches` context manager.

    Use this to register class replacements that will be automatically applied when loading any model.
    This is useful for quantization library compatibility, structural optimizations, and architectural
    experimentation. The mapping is global, can grow with multiple calls, and can be cleared entirely.

    Args:
        mapping (`Dict[str, type[nn.Module]]`):
            Mapping from original class names (or regex patterns) to replacement classes. Supports:
            - Exact class names: `"Qwen2MoeExperts"` → `CustomExperts`
            - Regex patterns: `".*Attention"` matches `LlamaAttention`, `MistralAttention`, etc.,
            or `"^Llama\\d+Attention$"` matches `Llama2Attention`, `Llama3Attention`, etc.

            Exact matches take precedence over patterns. Patterns are matched using `re.search()`,
            so they can match anywhere in the class name unless you use anchors (`^` for start, `$` for end).
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether to overwrite existing mappings for class names that are already registered.

    Example:
        ```python
        from transformers import AutoModelForCausalLM
        from transformers.monkey_patching import register_patch_mapping

        # Define custom expert implementation
        class SequentialExperts(nn.Module):
            ...

        # Register exact class name
        register_patch_mapping(
            mapping={"Qwen2MoeExperts": SequentialExperts}
        )

        # Register with regex pattern to match multiple classes
        register_patch_mapping(
            mapping={".*Attention": CustomAttention}  # Matches LlamaAttention, MistralAttention, etc.
        )

        # Match specific model versions
        register_patch_mapping(
            mapping={"^Llama\\d+Attention$": CustomLlamaAttention}  # Matches Llama2Attention, Llama3Attention
        )

        # The patch will be automatically applied during loading
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        ```

    Note:
        For weight conversions, use [`~transformers.register_checkpoint_conversion_mapping`] instead.
    """
    global _monkey_patch_mapping_cache
    with _monkey_patch_lock:
        for class_name, replacement_class in mapping.items():
            # Validate that replacement_class is actually a class and is a subclass of nn.Module
            if not isinstance(replacement_class, type):
                raise TypeError(
                    f"Replacement for '{class_name}' must be a class, got {type(replacement_class).__name__}"
                )
            if not issubclass(replacement_class, nn.Module):
                raise TypeError(
                    f"Replacement class for '{class_name}' must be a subclass of nn.Module, "
                    f"got {replacement_class.__name__} which inherits from {[c.__name__ for c in replacement_class.__mro__[1:]]}"
                )

            if class_name in _monkey_patch_mapping_cache and not overwrite:
                raise ValueError(
                    f"Class '{class_name}' already has a patch mapping registered. Use overwrite=True to replace it."
                )
            _monkey_patch_mapping_cache[class_name] = replacement_class


def unregister_patch_mapping(keys: list[str]) -> None:
    """
    Unregister patch mappings to disable automatic patching.

    This removes specified mappings from the global registry, preventing them from being applied
    during model loading. You must provide the exact same name or pattern that was used during registration.

    Args:
        keys (`List[str]`):
            List of mapping keys (class names or regex patterns) to remove from the patch mapping
            (e.g., `["Qwen2MoeExperts"]` or `[".*Attention"]`).

    Example:
        ```python
        from transformers import AutoModelForCausalLM
        from transformers.monkey_patching import register_patch_mapping, unregister_patch_mapping

        # Register a patch
        register_patch_mapping(
            mapping={"Qwen2MoeExperts": CustomExperts}
        )

        # Unregister the patch
        unregister_patch_mapping(["Qwen2MoeExperts"])

        # The patch will no longer be applied during loading
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")
        ```
    """
    global _monkey_patch_mapping_cache
    with _monkey_patch_lock:
        for key in keys:
            if key not in _monkey_patch_mapping_cache:
                raise ValueError(
                    f"Class or pattern '{key}' not found in monkey patch mapping cache. "
                    f"Cannot unregister a class that is not registered."
                )
            del _monkey_patch_mapping_cache[key]


def get_patch_mapping() -> dict[str, type[nn.Module]]:
    """
    Get all registered patch mappings.

    Returns:
        `Dict[str, type[nn.Module]]`: Dictionary mapping class names or patterns to replacement classes.
    """
    with _monkey_patch_lock:
        return _monkey_patch_mapping_cache.copy()


def clear_patch_mapping() -> None:
    """
    Clear all registered patch mappings.

    This removes all registered mappings from the global registry.

    Example:
        ```python
        from transformers.monkey_patching import register_patch_mapping, clear_patch_mapping

        # Register some patches
        register_patch_mapping(
            mapping={"Qwen2MoeExperts": CustomExperts}
        )

        # Clear all patches
        clear_patch_mapping()
        ```
    """
    global _monkey_patch_mapping_cache
    with _monkey_patch_lock:
        _monkey_patch_mapping_cache.clear()


@contextmanager
def apply_patches():
    """
    Context manager to apply registered monkey patches within a block of code.

    This temporarily replaces original classes with their registered replacements during the execution of the block, and restores the original classes afterward.

    Example:
        ```python
        from transformers import Qwen2MoeModel, Qwen2MoeConfig
        from transformers.monkey_patching import register_patch_mapping, apply_patches

        # Register a patch
        register_patch_mapping(
            mapping={"Qwen2MoeExperts": CustomExperts}
        )

        # Apply patches within the context
        with apply_patches():
            # The model will use CustomExperts instead of Qwen2MoeExperts
            model = Qwen2MoeModel(Qwen2MoeConfig())

        # Outside the context, original classes are restored
        # The model will use Qwen2MoeExperts again
        model = Qwen2MoeModel(Qwen2MoeConfig())
        ```
    """
    mapping = get_patch_mapping()
    if not mapping:
        yield
        return

    original_classes = {}

    # Create list to avoid dict changed during iteration
    for module in list(sys.modules.values()):
        if module is None or not hasattr(module, "__name__"):
            continue
        if not module.__name__.startswith("transformers"):
            continue

        # Iterate through all attributes in transformers modules
        for attr_name in dir(module):
            # Check if this attribute name matches any pattern before accessing it
            replacement_class = _find_replacement_class(attr_name, mapping)
            if replacement_class is None:
                continue

            try:
                attr = getattr(module, attr_name)
                # Check if it's a class
                if not isinstance(attr, type):
                    continue

                original_classes[(module.__name__, attr_name)] = attr
                setattr(module, attr_name, replacement_class)
            except (AttributeError, TypeError, ImportError):
                # Skip attributes that can't be accessed or modules that can't be imported
                continue

    yield

    for (module_name, class_name), original_class in original_classes.items():
        module = sys.modules[module_name]
        setattr(module, class_name, original_class)


# _can_record_outputs is a class attribute so patching and unpatching it in the class won't work
# since the model instance will still reference the original class's _can_record_outputs.
def patch_output_recorders(model: nn.Module) -> None:
    """
    Patch the model instance's output recorders to use the registered replacement classes.

    This function updates output recorders in a model's submodules to use monkey-patched replacement
    classes. Output recorders are used by the transformers library to track intermediate outputs during
    forward passes (via the `_can_record_outputs` attribute). When classes are monkey-patched, these
    recorders need to be updated to reference the new classes.

    This is automatically called during model initialization when loading with `from_pretrained` or
    `from_config`. You typically don't need to call this manually unless you're constructing models
    in custom ways.

    Note:
        The `_can_record_outputs` attribute is a class-level attribute that maps output names to either:
        - `OutputRecorder` instances that have a `target_class` attribute
        - Class types directly

        This function patches both cases to use the replacement classes from the monkey patch registry.

    Args:
        model (`nn.Module`):
            The model instance whose output recorders should be patched. All submodules will be
            traversed to find and patch their `_can_record_outputs` attributes.

    Example:
        ```python
        from transformers import AutoModelForCausalLM
        from transformers.monkey_patching import register_patch_mapping, patch_output_recorders

        # Register a patch
        register_patch_mapping(mapping={"Qwen2MoeExperts": CustomExperts})

        # If you construct a model manually (without from_pretrained), patch recorders
        model = Qwen2MoeModel(config)
        patch_output_recorders(model)  # Updates output recorders to use CustomExperts
        ```
    """

    mapping = get_patch_mapping()
    if not mapping:
        return

    for submodule in model.modules():
        if hasattr(submodule, "_can_record_outputs") and submodule._can_record_outputs is not None:
            for output, recorder in submodule._can_record_outputs.items():
                if isinstance(recorder, OutputRecorder):
                    # Check if target class matches any registered pattern or exact name
                    replacement_class = _find_replacement_class(recorder.target_class.__name__, mapping)
                    if replacement_class is not None:
                        recorder.target_class = replacement_class
                elif isinstance(recorder, type):
                    # Check if class type matches any registered pattern or exact name
                    replacement_class = _find_replacement_class(recorder.__name__, mapping)
                    if replacement_class is not None:
                        submodule._can_record_outputs[output] = replacement_class
