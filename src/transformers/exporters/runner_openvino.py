"""Driving an OpenVINO model: `OpenVINOModelRunner`."""

from __future__ import annotations

import re

from ..utils.import_utils import is_openvino_available, is_torch_available
from .base import ModelRunner
from .utils import get_leaf_tensors


if is_torch_available():
    import torch

if is_openvino_available():
    import numpy as np
    import openvino


# `AUTO` lets OpenVINO pick among the plugins it finds; a caller who wants one names it.
_DEFAULT_OV_DEVICE = "AUTO"


class OpenVINOModelRunner(ModelRunner):
    """`ModelRunner` backed by a compiled `openvino.Model` and one infer request.

    Two things set it apart from the other runners. The cache is not an input: the stateful transformation
    folds each round-tripped pair into an internal `ReadValue`/`Assign` variable, so a `past_key_values`
    kwarg is seeded into those variables instead of fed, and read back out of them afterwards. And the
    graph's ports carry the `input.` / `output.` prefixes the exporter used to disambiguate a name that is
    both — stripping them is what recovers the leaf names the caller uses.
    """

    def __init__(self, ov_model, export_metadata=None, device=None):
        self._model = ov_model
        self._device_name = _DEFAULT_OV_DEVICE if device is None else _ov_device_name(device)
        self._compiled = openvino.compile_model(ov_model, self._device_name)
        self._request = self._compiled.create_infer_request()
        # Nothing rides inside the IR: `openvino.save_model` writes the graph and the weights, and the
        # metadata travels with the artifacts instead (`ExportArtifacts`, the saved `export_metadata.json`).
        self.export_metadata = self.resolve_metadata(export_metadata, lambda: None)
        self.input_names = tuple(
            _leaf_name(name) for port in self._compiled.inputs for name in [_port_name(port)] if name != "beam_idx"
        )
        # OpenVINO hands back host tensors whichever plugin ran the graph.
        self.device = torch.device("cpu")

    @classmethod
    def from_artifact(cls, artifact, export_metadata=None, device=None, **kwargs) -> OpenVINOModelRunner:
        """Compile the `openvino.Model` an export handed back."""
        return cls(artifact, export_metadata=export_metadata, device=device, **kwargs)

    @classmethod
    def from_pretrained(cls, path, export_metadata=None, device=None, **kwargs) -> OpenVINOModelRunner:
        """Read a saved `.xml` (with its `.bin` beside it) and compile it."""
        return cls(openvino.Core().read_model(str(path)), export_metadata=export_metadata, device=device, **kwargs)

    def to(self, device) -> OpenVINOModelRunner:
        """Recompile for another plugin — a compiled model is bound to the one it was compiled for."""
        return type(self)(self._model, export_metadata=self.export_metadata, device=device)

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        leaves = {path: tensor.cpu() for path, tensor in get_leaf_tensors(kwargs).items()}
        batch = next(iter(leaves.values())).shape[0] if leaves else 1

        feed = {}
        for port in self._compiled.inputs:
            # A passthrough tensor carries both an input and an output name, so every alias is tried.
            for name in port.get_names():
                leaf = _leaf_name(name)
                if leaf in leaves:
                    feed[name] = leaves[leaf]
                elif name == "beam_idx":
                    # Greedy decoding reorders nothing, and the fused `Gather` reads this every step.
                    feed[name] = np.arange(batch, dtype=np.int32)
                elif name in kwargs:
                    # A scalar the trace baked as a port of its own (`max_seqlen`, a feature layer index).
                    feed[name] = np.array(kwargs[name])
                else:
                    continue
                break

        self._prime_state(leaves)
        results = self._request.infer(feed)

        outputs = {_leaf_name(_port_name(port)): torch.as_tensor(results[port]) for port in self._compiled.outputs}
        # The state is an output too: the loop reads its shape to know how far the cache has advanced, and
        # the next graph in the chain is seeded from it (the decode runner takes over what prefill left).
        # Copied, not aliased — the buffer behind it is the variable's own, and the next `infer` writes
        # through it.
        for state in self._request.query_state():
            outputs[_state_path(state)] = torch.from_numpy(state.state.data.copy())
        return outputs

    def _prime_state(self, leaves: dict) -> None:
        """Make the request's state match the sequence this call belongs to.

        The cache the caller holds is the authority for now, so it is written into the variables on every
        call and read back out after — which keeps `ExportedGenerator`'s loop identical to every other
        backend's, at the cost of copying the cache in and out per step.

        That cost is the whole point of a stateful export, so it should go: optimum-intel drives these
        models by never passing a cache at all, calling `reset_state()` on the first step and tracking the
        length itself. Doing the same here means teaching the generation loop that this backend owns its
        state, which is a change to `ExportedGenerator` rather than to this runner.
        """
        states = self._request.query_state()
        handed = {path: leaves[path] for state in states if (path := _state_path(state)) in leaves}
        if not handed:
            return
        for state in states:
            tensor = handed.get(_state_path(state))
            # An empty cache is what the variable already holds: a prefill writes the state rather than
            # reading it, and assigning a zero-length tensor over the shape the graph declares is refused.
            if tensor is not None and tensor.shape[-2]:
                # The exporter retypes state the CPU plugin refuses to hold (i64 lengths become i32).
                state.state = openvino.Tensor(tensor.numpy().astype(state.state.data.dtype, copy=False))


def _ov_device_name(device) -> str:
    """The OpenVINO plugin for a torch-style device — `cuda` has none, so it is refused rather than
    silently run on the CPU."""
    device = torch.device(device)
    if device.type == "cpu":
        return "CPU"
    if device.type in ("xpu", "gpu"):
        return "GPU" if device.index is None else f"GPU.{device.index}"
    raise ValueError(f"OpenVINO has no plugin for `{device.type}`; it runs on CPU, GPU (Intel) or NPU.")


def _port_name(port) -> str:
    """The readable name of a port. Compilation can merge a named tensor with an intermediate that kept its
    numeric id, and `get_any_name` returns the sorted-first of the two."""
    names = sorted(port.get_names())
    return next((name for name in names if not name.isdigit()), names[0])


def _leaf_name(name: str) -> str:
    """The kwarg-space name behind a port name — `disambiguate_io_names` only ever prefixes."""
    return re.sub(r"^(input|output)\.", "", name)


def _state_path(state) -> str:
    """The leaf a state variable stands for. `apply_make_stateful_transformation` names each variable by
    concatenating its input and output names, so the variable for `x` is `input.xoutput.x`."""
    name = state.name
    return name[len("input.") : (len(name) - len("input.output.")) // 2 + len("input.")]
