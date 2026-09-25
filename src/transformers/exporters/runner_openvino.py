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

    def __init__(self, ov_model, export_metadata=None, device=None, ov_config=None):
        self._model = ov_model
        self._device_name = _DEFAULT_OV_DEVICE if device is None else _ov_device_name(device)
        # A folded cache is the plugin's to store, and the CPU one quantizes it to `u8` by default. That
        # never showed while the cache was fed in and out as real tensors — a plugin only quantizes what it
        # owns — so a stateful export would silently answer at a precision the model never asked for. Pin it
        # to what the graph computes in; a caller who wants the plugin's default passes `ov_config` to say so.
        self._ov_config = {"KV_CACHE_PRECISION": _graph_precision(ov_model), **(ov_config or {})}
        self._compiled = openvino.compile_model(ov_model, self._device_name, self._ov_config)
        self._request = self._compiled.create_infer_request()
        # Only a graph whose cache the stateful transformation actually folded owns its state. An encoder
        # has none, a `stateful=False` export keeps the cache as ports, and a prefill graph writes its cache
        # rather than carrying it — all of those are fed and read like any other backend's.
        states = self._request.query_state()
        self.owns_state = bool(states)
        # The leaves those variables stand for, named once. A hand-off is the only reason to touch the
        # state, and asking the request for its variables on every decode step to find there is nothing to
        # hand over costs one round trip per layer per token.
        self._state_paths = frozenset(_state_path(state) for state in states)
        # The CPU plugin hands a rank-0 variable (a static layer's `cumulative_length`) back as `[1]`.
        self._scalar_states = frozenset(
            variable.get_info().variable_id
            for variable in ov_model.get_variables()
            if variable.get_info().data_shape.rank.get_length() == 0
        )
        # How much of the sequence the variables hold. The loop asks for it rather than measuring a cache,
        # because there is no cache to measure — see `state_length`.
        self._state_length = 0
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
        return type(self)(self._model, export_metadata=self.export_metadata, device=device, ov_config=self._ov_config)

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        leaves = {path: tensor.cpu() for path, tensor in get_leaf_tensors(kwargs).items()}
        batch = _feed_batch(leaves)

        feed = {}
        for port in self._compiled.inputs:
            # A passthrough tensor carries both an input and an output name, so every alias is tried.
            for name in port.get_names():
                leaf = _leaf_name(name)
                if leaf in leaves:
                    feed[name] = _as_ov(_as_element_type(leaves[leaf], port.get_element_type()))
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
        # What this call adds to the sequence — the text width, which is what the variables grow by. The
        # decoder's text where there is one: an encoder-decoder's cache is its decoder's, fed as
        # `decoder_input_ids` beside an `input_ids` that only the encoder reads.
        text = next(
            (
                leaves[name]
                for name in ("decoder_input_ids", "decoder_inputs_embeds", "input_ids", "inputs_embeds")
                if name in leaves
            ),
            None,
        )
        query_length = text.shape[1] if text is not None and text.dim() > 1 else 0
        # `share_inputs`: the feed tensors outlive the call here, so OpenVINO can read them in place
        # instead of copying each one into its own buffer.
        results = self._request.infer(feed, share_inputs=True)

        # The trace's own names first, positionally: a port the rename could not reach keeps an internal
        # name (`embedding_0:0` for a component whose output is a bare tensor), and the graph's output order
        # is what the metadata recorded.
        recorded = self.export_metadata.output_names
        names = recorded if len(recorded) == len(self._compiled.outputs) else None
        outputs = {
            (names[index] if names else _leaf_name(_port_name(port))): _as_torch(results[port])
            for index, port in enumerate(self._compiled.outputs)
        }
        # The state is deliberately not among them. It stays in the plugin, which is what makes a stateful
        # export worth exporting: the loop asks `state_length` how far the sequence has got instead of
        # carrying the whole cache back and forth a step at a time.
        if query_length:
            self._state_length += query_length
        return outputs

    def state_tensors(self) -> dict[str, torch.Tensor]:
        """What the folded cache holds, read off the `Assign` sinks the transformation left.

        The call path does not do this — keeping the cache in the plugin is the point — but it is how a
        caller inspects the state, and how a test checks that the graph writes the leaves eager returns.
        """
        tensors = {}
        for state in self._request.query_state():
            tensor = _as_torch(state.state.data.copy())
            tensors[_state_path(state)] = tensor.reshape(()) if state.name in self._scalar_states else tensor
        return tensors

    @property
    def state_paths(self) -> frozenset[str]:
        """The cache leaves this graph keeps in variables rather than taking as inputs."""
        return self._state_paths

    def adopt_state(self, leaves: dict, length: int) -> None:
        """Seed the folded state from the cache another graph wrote, and say how far along it is.

        A prompt graph hands its keys and values back as outputs; a decode graph keeps them in variables.
        Nothing carries them across on its own, so the loop moves them once — without it the decode graph
        attends to a sequence that starts at its own first token, which reads as a plausible continuation
        of nothing.
        """
        # What the two graphs have in common, which need not be everything: a decode graph can hold state
        # the prompt never wrote (an encoder-decoder's cross-attention keys, written once by the encoder).
        # Nothing in common at all is the failure worth naming — the hand-off would silently seed nothing.
        shared = self._state_paths & leaves.keys()
        if not shared:
            raise RuntimeError(
                f"{type(self).__name__} was handed state naming none of its variables: "
                f"{sorted(self._state_paths)[:3]} not among {sorted(leaves)[:3]}. Seeding nothing would "
                "leave the graph attending to a sequence that starts at this step."
            )
        self._prime_state({path: leaves[path].cpu() for path in shared})
        self._state_length = length

    @property
    def state_length(self) -> int:
        """How much of the sequence the plugin's variables hold — what a cache's length would say, for a
        graph that has none."""
        return self._state_length

    def reset_state(self) -> None:
        """Start a new sequence: zero the variables and the length that tracks them."""
        self._request.reset_state()
        self._state_length = 0

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
        if not self._state_paths & leaves.keys():
            return
        states = self._request.query_state()
        handed = {path: leaves[path] for state in states if (path := _state_path(state)) in leaves}
        for state in states:
            tensor = handed.get(_state_path(state))
            # An empty cache is what the variable already holds: a prefill writes the state rather than
            # reading it, and assigning a zero-length tensor over the shape the graph declares is refused.
            # Only the keys and values can be empty in that sense — a counter rides in the cache beside them
            # with no sequence axis of its own, and is handed over whatever it holds.
            if tensor is not None and (tensor.dim() < 2 or tensor.shape[-2]):
                tensor = _as_element_type(tensor, state.state.get_element_type())
                state.state = (
                    _as_ov(tensor)
                    if tensor.dtype == torch.bfloat16
                    # The exporter retypes state the CPU plugin refuses to hold (i64 lengths become i32).
                    else openvino.Tensor(tensor.numpy().astype(state.state.data.dtype, copy=False))
                )


def _graph_precision(ov_model) -> str:
    """The floating-point type this graph computes in, as a plugin property takes it.

    Read off the graph rather than the config: an export saved with `compress_to_fp16` holds `f16` weights,
    and what the cache has to match is the type the attention actually runs at.
    """
    for port in ov_model.outputs:
        element_type = port.get_element_type()
        if element_type.is_real():
            return element_type.get_type_name()
    return "f32"


def _feed_batch(leaves: dict) -> int:
    """How many sequences this call carries, for the `beam_idx` the fused state gather reads.

    The text input says it. Whichever leaf comes first does not: an m-rope `position_ids` is
    `[sections, batch, positions]`, so a glm4v step would answer 4 for a batch of 2 and the plugin then
    refuses the gather against its state ("beam idx batch: 4 is not equal to batch of state: 2").
    """
    for name in ("input_ids", "inputs_embeds", "decoder_input_ids", "decoder_inputs_embeds", "attention_mask"):
        tensor = leaves.get(name)
        if tensor is not None and tensor.dim():
            return tensor.shape[0]
    return next(iter(leaves.values())).shape[0] if leaves else 1


def _as_ov(tensor):
    """A torch tensor as something OpenVINO takes.

    Laid out contiguously first: the call shares its feed rather than copying it, and a tensor that reached
    here as a view (a transpose, a slice down a stride) has no buffer to share.

    `bfloat16` has no numpy counterpart, so it goes through a `uint16` view with the element type named
    outright — the bits are the same, only numpy has no word for them.
    """
    tensor = tensor.contiguous()
    if tensor.dtype == torch.bfloat16:
        return openvino.Tensor(tensor.view(torch.uint16).numpy(), list(tensor.shape), openvino.Type.bf16)
    return tensor.numpy()


def _as_element_type(tensor, element_type):
    """A floating tensor in the type an OpenVINO port or variable declares, where the two differ.

    A graph that computes in float32 hands float32 on, and the next graph's port may still declare the half
    type the model was built in — a split vision encoder feeding its decoder, or a cache the generation loop
    kept in the model's dtype. OpenVINO reads a half port's feed by viewing its bytes rather than converting
    them, so a float32 tensor arriving there is read as twice as many elements; converting first is what
    eager would have done anyway, since its tensors never left the model's dtype.
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
    """The OpenVINO plugin for a torch-style device — `cuda` has none, so it is refused rather than
    silently run on the CPU."""
    device = torch.device(device)
    if device.type == "cpu":
        return "CPU"
    if device.type in ("xpu", "gpu"):
        return "GPU" if device.index is None else f"GPU.{device.index}"
    raise ValueError(f"OpenVINO has no plugin for `{device.type}`; it runs on CPU, GPU (Intel) or NPU.")


def _port_name(port) -> str:
    """The readable name of a port.

    Compilation can merge a named tensor with an intermediate that kept the name OV gave it, and the port
    then carries both. OV writes its own as `<node>:<port>` or a bare id, so the name the export chose is
    the one that looks like neither — picking by sort order would hand back `linear_14:0` over `logits`.
    """
    names = sorted(port.get_names())
    given = [name for name in names if ":" not in name and not name.isdigit()]
    return given[0] if given else names[0]


def _leaf_name(name: str) -> str:
    """The kwarg-space name behind a port name — `disambiguate_io_names` only ever prefixes."""
    return re.sub(r"^(input|output)\.", "", name)


def _state_path(state) -> str:
    """The leaf a state variable stands for. `apply_make_stateful_transformation` names each variable by
    concatenating its input and output names, so the variable for `x` is `input.xoutput.x`."""
    name = state.name
    return name[len("input.") : (len(name) - len("input.output.")) // 2 + len("input.")]
