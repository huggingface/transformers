"""Observe coarse dataflow by running one forward pass on the meta-device model.

Our structural edges are inferred from the module tree; this module *grounds* the top-level
flow in what actually executes. Meta tensors carry shapes but allocate ~no memory, so a
forward is cheap and safe to run for every model. We register forward hooks on the top-level
stages (a ``ModuleList`` stack is represented by its first element) and record their in/out
shapes in call order.

To stay config-parametric (this feeds an ``ArchitectureTemplate``, not a checkpoint-specific
graph), observed integer dims are **symbolized** back to config expressions where they match a
salient config value — e.g. ``[1, 8, 4096]`` becomes ``["B", "S", "config.hidden_size"]``.

Anything that can't run on meta (data-dependent control flow, missing inputs) simply yields no
dataflow block; the structural edges remain. No weights are ever loaded.
"""

from __future__ import annotations

from typing import Any


# Config fields whose (unique-ish) integer values we try to recognize inside observed shapes,
# in priority order so the most shape-like field wins on a value collision.
_SHAPE_CONFIG_FIELDS = (
    "hidden_size",
    "vocab_size",
    "intermediate_size",
    "head_dim",
    "max_position_embeddings",
    "num_attention_heads",
    "num_key_value_heads",
)

_DUMMY_SEQ_LEN = 8


class _Deadline:
    """Best-effort wall-clock timeout via SIGALRM so a looping forward cannot hang generation.

    No-op where SIGALRM is unavailable (non-main-thread or non-Unix)."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self._sig = None

    def __enter__(self):
        import signal

        self._sig = getattr(signal, "SIGALRM", None)
        if self._sig is not None:
            try:

                def _raise(*_):
                    raise TimeoutError("forward exceeded time budget")

                self._old = signal.signal(self._sig, _raise)
                signal.setitimer(signal.ITIMER_REAL, self.seconds)
            except (ValueError, OSError):
                self._sig = None
        return self

    def __exit__(self, *exc):
        if self._sig is not None:
            import signal

            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(self._sig, self._old)
        return False


def _shape_of(x: Any) -> list[int] | None:
    """First tensor shape found in ``x`` (tensor / tuple / list / ModelOutput-like)."""
    if hasattr(x, "shape"):
        try:
            return list(x.shape)
        except (TypeError, ValueError):
            return None
    if isinstance(x, (tuple, list)):
        for item in x:
            shape = _shape_of(item)
            if shape is not None:
                return shape
    if hasattr(x, "to_tuple"):
        try:
            return _shape_of(x.to_tuple())
        except Exception:
            return None
    return None


def _dummy_inputs(model: Any, config: Any, device: str) -> dict:
    """Minimal forward inputs matching the model's modality (no tokenizer/processor)."""
    import torch

    main = getattr(model, "main_input_name", None) or "input_ids"
    inputs: dict[str, Any] = {}
    if main in ("input_values", "input_features"):
        inputs[main] = torch.zeros(1, 1, 8000, device=device)
    elif main == "pixel_values":
        size = getattr(config, "image_size", None) or 64
        size = size if isinstance(size, int) else (size[0] if isinstance(size, (list, tuple)) else 64)
        channels = getattr(config, "num_channels", None) or 3
        inputs[main] = torch.zeros(1, channels, size, size, device=device)
    else:
        inputs["input_ids"] = torch.ones(1, _DUMMY_SEQ_LEN, dtype=torch.long, device=device)

    # Encoder-decoder models need decoder inputs to run a full forward.
    if getattr(config, "is_encoder_decoder", False):
        inputs["decoder_input_ids"] = torch.ones(1, _DUMMY_SEQ_LEN, dtype=torch.long, device=device)
    return inputs


def capture_dataflow(model: Any, config: Any) -> dict[str, Any] | None:
    """Run one meta forward, returning ``{input, output, stages}`` in call order, or None."""
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None

    model.eval()
    try:
        inputs = _dummy_inputs(model, config, "meta")
    except Exception:
        return None
    in_name = next(iter(inputs))
    in_shape = list(next(iter(inputs.values())).shape)

    order = {id(m): i for i, (_, m) in enumerate(model.named_children())}
    flow: list[dict] = []
    hooks = []

    def make_hook(stage_name: str, is_stack: bool):
        def hook(module, args, output):
            flow.append(
                {
                    "name": stage_name,
                    "class_name": type(module).__name__,
                    "is_stack": is_stack,
                    "in_shape": _shape_of(args),
                    "out_shape": _shape_of(output),
                    "_ord": order.get(id(module if not is_stack else module), 999),
                }
            )

        return hook

    for name, child in model.named_children():
        target, is_stack = child, False
        if isinstance(child, (nn.ModuleList, nn.Sequential)) and len(child) > 0:
            target, is_stack = child[0], True  # a stack is represented by its first element
        try:
            hooks.append(target.register_forward_hook(make_hook(name, is_stack)))
        except Exception:
            continue

    output = None
    try:
        with torch.no_grad(), _Deadline(8):
            output = model(**inputs)
    except BaseException:
        return None
    finally:
        for handle in hooks:
            handle.remove()

    if not flow:
        return None

    # De-duplicate: a stage hit more than once keeps its first call; preserve call order.
    seen, stages = set(), []
    for entry in flow:
        if entry["name"] in seen:
            continue
        seen.add(entry["name"])
        stages.append(entry)

    return {
        "input": {"name": in_name, "shape": in_shape},
        "output": {"shape": _shape_of(output)},
        "stages": stages,
    }


def _reverse_shape_map(config: Any) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for field in reversed(_SHAPE_CONFIG_FIELDS):  # reversed so earlier (higher-priority) wins
        value = getattr(config, field, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 1:
            mapping[value] = field
    return mapping


def symbolize_shape(shape: list[int] | None, config: Any) -> list | None:
    """Rewrite an observed integer shape into config-parametric tokens.

    ``[1, 8, 4096]`` -> ``["B", "S", "config.hidden_size"]``. Batch (leading 1) -> ``B``,
    the dummy sequence length -> ``S``, dims matching a salient config value -> ``config.<field>``;
    anything else is left as an integer.
    """
    if shape is None:
        return None
    reverse = _reverse_shape_map(config)
    out: list = []
    for i, dim in enumerate(shape):
        if i == 0 and dim == 1:
            out.append("B")
        elif dim == _DUMMY_SEQ_LEN:
            out.append("S")
        elif dim in reverse:
            out.append(f"config.{reverse[dim]}")
        else:
            out.append(dim)
    return out
