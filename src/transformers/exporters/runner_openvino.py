"""Driving an OpenVINO model: `OpenVINOModelRunner`."""

from __future__ import annotations

from ..utils.import_utils import is_openvino_available, is_torch_available
from .base import ModelRunner
from .metadata import ExportMetadata
from .utils import BATCH_INPUTS, get_leaf_tensors, leaf_name


if is_torch_available():
    import torch

if is_openvino_available():
    import numpy as np
    import openvino


_DEFAULT_OV_DEVICE = "AUTO"


class OpenVINOModelRunner(ModelRunner):
    """`ModelRunner` backed by a compiled `openvino.Model` and one infer request.

    In a stateful export the cache lives in `ReadValue`/`Assign` variables, so a `past_key_values` kwarg is
    seeded into them instead of fed. Port names carry the exporter's `input.` / `output.` prefixes.
    """

    def __init__(self, ov_model, export_metadata=None, device=None, ov_config=None):
        self._model = ov_model
        self._device_name = _DEFAULT_OV_DEVICE if device is None else _ov_device_name(device)
        # The CPU plugin quantizes a state-held cache to `u8` by default; pin it to the graph's precision.
        self._ov_config = {"KV_CACHE_PRECISION": _graph_precision(ov_model), **(ov_config or {})}
        self._compiled = openvino.compile_model(ov_model, self._device_name, self._ov_config)
        self._request = self._compiled.create_infer_request()
        # Encoders, `stateful=False` exports and prefill graphs have no variables and are fed like any backend.
        states = self._request.query_state()
        self.owns_state = bool(states)
        # Cached: querying the state every decode step costs a round trip per layer per token.
        self.state_paths = frozenset(_state_path(state) for state in states)
        infos = [variable.get_info() for variable in ov_model.get_variables()]
        # The CPU plugin hands a rank-0 variable (a static layer's `cumulative_length`) back as `[1]`.
        self._scalar_states = frozenset(info.variable_id for info in infos if info.data_shape.rank.get_length() == 0)
        # What each variable holds, read here rather than off `state.state`, which copies the variable out.
        self._state_types = {info.variable_id: info.data_type for info in infos}
        self._state_length = 0
        self.export_metadata = ExportMetadata.from_dict(export_metadata)
        self.input_names = tuple(
            leaf_name(name) for port in self._compiled.inputs for name in [_port_name(port)] if name != "beam_idx"
        )
        self._input_ports = [
            (port, [(name, leaf_name(name)) for name in port.get_names()], port.get_element_type())
            for port in self._compiled.inputs
        ]
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
        return type(self)(self._model, export_metadata=self.export_metadata, device=device, ov_config=self._ov_config)

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        leaves = {path: tensor.cpu() for path, tensor in get_leaf_tensors(kwargs).items()}
        batch = _feed_batch(leaves)

        feed = []
        for port, names, element_type in self._input_ports:
            # A passthrough tensor carries both an input and an output name, so every alias is tried.
            for name, leaf in names:
                if leaf in leaves:
                    value = _as_ov(_as_element_type(leaves[leaf], element_type))
                elif name == "beam_idx":
                    # Greedy decoding reorders nothing, and the fused `Gather` reads this every step.
                    value = np.arange(batch, dtype=np.int32)
                elif name in kwargs:
                    # A scalar the trace baked as a port of its own (`max_seqlen`, a feature layer index).
                    value = np.array(kwargs[name])
                else:
                    continue
                feed.append((port, _as_port_tensor(value, element_type)))
                break

        self._prime_state(leaves)
        # The state grows by the (decoder's) text width.
        text = next(
            (
                leaves[name]
                for name in ("decoder_input_ids", "decoder_inputs_embeds", "input_ids", "inputs_embeds")
                if name in leaves
            ),
            None,
        )
        query_length = text.shape[1] if text is not None and text.dim() > 1 else 0
        # Per port rather than a name-keyed dict, which the request resolves on every call.
        for port, tensor in feed:
            self._request.set_tensor(port, tensor)
        self._request.infer()

        # Recorded names first, positionally: a port the rename missed keeps an internal name (`embedding_0:0`).
        recorded = self.export_metadata.output_names
        names = recorded if len(recorded) == len(self._compiled.outputs) else None
        outputs = {
            # Copied: the request writes its next outputs into the same buffers.
            (names[index] if names else leaf_name(_port_name(port))): _as_torch(
                self._request.get_tensor(port).data.copy()
            )
            for index, port in enumerate(self._compiled.outputs)
        }
        if query_length:
            self._state_length += query_length
        return outputs

    def state_tensors(self, paths=None) -> dict[str, torch.Tensor]:
        """What the folded cache holds: every variable, or only those standing for `paths`. For inspection only."""
        tensors = {}
        for state in self._request.query_state():
            if paths is not None and _state_path(state) not in paths:
                continue
            tensor = _as_torch(state.state.data.copy())
            tensors[_state_path(state)] = tensor.reshape(()) if state.name in self._scalar_states else tensor
        return tensors

    def adopt_state(self, leaves: dict, length: int) -> None:
        """Seed the folded state from the cache another graph (the prompt graph) wrote, and set its length.

        Only the leaves this graph keeps as variables are taken.
        """
        shared = {path: tensor.cpu() for path, tensor in leaves.items() if path in self.state_paths}
        if shared:
            self._prime_state(shared)
            self._state_length = length

    @property
    def state_length(self) -> int:
        """How much of the sequence the plugin's variables hold."""
        return self._state_length

    def reset_state(self) -> None:
        """Start a new sequence: zero the variables and the length that tracks them."""
        self._request.reset_state()
        self._state_length = 0

    def _prime_state(self, leaves: dict) -> None:
        """Write cache leaves fed directly (a component call, `adopt_state`) into their variables."""
        # Assigning a zero-length tensor is refused, and an empty cache is what the variable already holds.
        handed = {
            path: tensor
            for path, tensor in leaves.items()
            if path in self.state_paths and (tensor.dim() < 2 or tensor.shape[-2])
        }
        if not handed:
            return
        for state in self._request.query_state():
            tensor = handed.get(_state_path(state))
            if tensor is not None:
                element_type = self._state_types[state.name]
                tensor = _as_element_type(tensor, element_type)
                state.state = (
                    _as_ov(tensor)
                    if tensor.dtype == torch.bfloat16
                    # The exporter retypes state the CPU plugin refuses to hold (i64 lengths become i32).
                    else openvino.Tensor(tensor.numpy().astype(element_type.to_dtype(), copy=False))
                )


def _graph_precision(ov_model) -> str:
    """The floating-point type this graph computes in, as a plugin property takes it (read off its outputs)."""
    for port in ov_model.outputs:
        element_type = port.get_element_type()
        if element_type.is_real():
            return element_type.get_type_name()
    return "f32"


def _feed_batch(leaves: dict) -> int:
    """How many sequences this call carries, for `beam_idx`.

    Read off a batch input, not the first leaf: an m-rope `position_ids` is `[sections, batch, positions]`.
    """
    for name in BATCH_INPUTS:
        tensor = leaves.get(name)
        if tensor is not None and tensor.dim():
            return tensor.shape[0]
    return next(iter(leaves.values())).shape[0] if leaves else 1


def _as_ov(tensor):
    """A torch tensor as something OpenVINO takes, contiguous (the feed is shared); `bfloat16` via a `uint16` view."""
    tensor = tensor.contiguous()
    if tensor.dtype == torch.bfloat16:
        return openvino.Tensor(tensor.view(torch.uint16).numpy(), list(tensor.shape), openvino.Type.bf16)
    return tensor.numpy()


def _as_port_tensor(value, element_type):
    """`value` as an `openvino.Tensor` of the port's type (cast, since `set_tensor` does not convert)."""
    if isinstance(value, openvino.Tensor):
        return value
    wanted = element_type.to_dtype() if element_type.is_static() else value.dtype
    # `asarray` rather than `ascontiguousarray`, which turns a rank-0 scalar (`logits_to_keep`) into `[1]`.
    array = np.asarray(value, dtype=wanted)
    return openvino.Tensor(array if array.flags.c_contiguous else array.copy(), shared_memory=True)


def _as_element_type(tensor, element_type):
    """A floating tensor in the type an OpenVINO port or variable declares, where the two differ.

    OpenVINO views a half port's feed bytes rather than converting, so a float32 tensor would be misread.
    """
    floats = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16, "f64": torch.float64}
    dtype = floats.get(element_type.get_type_name())
    if dtype is None or not tensor.is_floating_point() or tensor.dtype == dtype:
        return tensor
    return tensor.to(dtype)


def _as_torch(array):
    """The inverse: OpenVINO hands back `bfloat16` as raw `uint16`, which torch can view as what it is."""
    if array.dtype == np.uint16:
        return torch.from_numpy(array).view(torch.bfloat16)
    return torch.as_tensor(array)


def _ov_device_name(device) -> str:
    """The OpenVINO plugin for a torch-style device; `cuda` has none and is refused."""
    device = torch.device(device)
    if device.type == "cpu":
        return "CPU"
    if device.type in ("xpu", "gpu"):
        return "GPU" if device.index is None else f"GPU.{device.index}"
    raise ValueError(f"OpenVINO has no plugin for `{device.type}`; it runs on CPU, GPU (Intel) or NPU.")


def _port_name(port) -> str:
    """The readable name of a port: not OV's own `<node>:<port>` or bare-id names (`linear_14:0` vs `logits`)."""
    names = sorted(port.get_names())
    given = [name for name in names if ":" not in name and not name.isdigit()]
    return given[0] if given else names[0]


def _state_path(state) -> str:
    """The leaf a state variable stands for: the variable for `x` is named `input.xoutput.x`."""
    name = state.name
    return name[len("input.") : (len(name) - len("input.output.")) // 2 + len("input.")]
