"""Driving an ONNX Runtime session: `OnnxModelRunner`."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..utils.import_utils import is_torch_available
from .base import ModelRunner
from .cache import _read_cache_entry
from .metadata import (
    EXPORT_METADATA_KEY,
    ExportMetadata,
)
from .utils import (
    get_leaf_tensors,
)


if is_torch_available():
    import torch


def _session_device(device=None) -> torch.device:
    """Which device a session opened for `device` runs on, with the GPU index filled in.

    `cuda` names whichever GPU is current, and that is the one a caller's tensors are allocated on, so it is
    the one the session has to be opened for. Without a device to go by, CUDA is used when one is visible.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _ort_to_torch_dtype(ort_type: str | None) -> torch.dtype | None:
    """torch dtype for an ORT input/output type string (`"tensor(float)"`), or `None` if it names no
    tensor — the caller then keeps the tensor's own dtype.

    Resolved by *name* against `torch` rather than through `JitScalarType`: that helper aliases ONNX's
    plain integer types onto torch's quantized ones (`int32` -> `qint32`, `int8`/`uint8` -> `qint8`/
    `quint8`), and casting a real tensor to a quantized dtype raises `empty_strided not supported on
    quantized tensors` — which is what a grid VLM's int32 `cu_seqlens` hit. ONNX spells every type the
    way torch does apart from the two float aliases and the fp8 family's underscore.
    """
    match = re.fullmatch(r"tensor\((\w+)\)", ort_type or "")
    if match is None:
        return None
    name = {"float": "float32", "double": "float64"}.get(match.group(1), match.group(1))
    name = re.sub(r"^float8(e\d+m\d+.*)", r"float8_\1", name)
    dtype = getattr(torch, name, None)
    return dtype if isinstance(dtype, torch.dtype) else None


def _resolved_axis(dim: int | str, known: dict[str, int]) -> int | None:
    """One declared axis as a concrete size, or `None` when it cannot be worked out.

    ORT states an axis either as an int or as the *name* the export gave it, and a name can be arithmetic:
    a growing cache's output is `s36 + 1` where its input is `s36`. Folding those against the sizes read off
    the inputs is what lets such a graph be bound at all — the alternative is allocating nothing and feeding
    through the host. Sums and differences of known names and integers cover what the exporter emits; a form
    outside that (a product of two symbols, a floor-div) stays unresolved rather than being guessed at.
    """
    if isinstance(dim, int):
        return dim
    if dim in known:
        return known[dim]
    total, sign, tokens = 0, 1, dim.replace("+", " + ").replace("-", " - ").split()
    for token in tokens:
        if token in "+-":
            sign = 1 if token == "+" else -1
        elif token in known:
            total, sign = total + sign * known[token], 1
        elif token.isdigit():
            total, sign = total + sign * int(token), 1
        else:
            return None
    return total


def _ort_element_type(ort_type: str | None) -> int | None:
    """ONNX element type for an ORT input/output type string (`"tensor(float)"`), or `None` if it names no
    tensor.

    This is what io-binding wants for `element_type`. Its other accepted form is a numpy *type* — but numpy
    cannot spell `bfloat16` or the fp8 family (ORT's own `TypeHelper.ort_type_to_numpy_type` raises on them),
    while the ONNX enum covers every type ORT can carry. ONNX spells them the way the ORT string does, in
    upper case.
    """
    match = re.fullmatch(r"tensor\((\w+)\)", ort_type or "")
    if match is None:
        return None
    from onnx import TensorProto

    element_type = getattr(TensorProto, match.group(1).upper(), None)
    return element_type if isinstance(element_type, int) else None


class OnnxModelRunner(ModelRunner):
    """`ModelRunner` backed by an `onnxruntime.InferenceSession`. Torch tensors in, torch tensors out —
    on CUDA the session is fed by pointer (io-binding) and on CPU through a zero-copy numpy view, so no
    tensor is copied across the host boundary either way. The KV-cache rides as matched `input.<name>` /
    `output.<name>` graph inputs/outputs, so a `past_key_values` kwarg is flattened into the feed and the
    `output.` prefix stripped from the results (back to plain leaf names)."""

    def __init__(self, session, export_metadata=None, source=None):
        self._session = session
        # What `to()` reopens: the `.onnx` path or the `ONNXProgram` this session was made from. ORT fixes a
        # session's execution provider when the session is created, so moving device means creating another
        # one over the same model -- and a session does not hand its own model back. A program kept here
        # stays in memory alongside ORT's own copy of it; a session built elsewhere has no source and cannot
        # move.
        self._source = source
        self._output_names = [o.name for o in session.get_outputs()]
        self.export_metadata = self.resolve_metadata(
            export_metadata,
            lambda: ExportMetadata.from_json(session.get_modelmeta().custom_metadata_map.get(EXPORT_METADATA_KEY)),
        )
        # Where the session runs, and so where `__call__` lands its outputs. A CUDA session is pinned to one
        # GPU, and which one is read back off the provider rather than assumed: io-binding hands ORT raw
        # addresses, and an address on a GPU the session was not opened for is an illegal access, not a copy.
        options = session.get_provider_options().get("CUDAExecutionProvider")
        self.device = torch.device("cpu" if options is None else f"cuda:{options.get('device_id', 0)}")

        # The graph names its *mutated* inputs with an `input.` prefix — the cache leaves always, and any
        # plain kwarg the graph writes to (a merged multi-token decode mutates its `attention_mask`).
        # Stripping it back off is what recovers the kwarg-space name (`disambiguate_io_names` only ever
        # prefixes, so the transformation inverts), and walking the session's own inputs means one a graph
        # pass dropped as unused is simply absent rather than something to look up.
        self._session_names, self._input_dtypes, self.input_shapes, self._cache_paths = {}, {}, {}, {}
        for spec in session.get_inputs():
            # The whole dotted name, not just its first segment: a graph handed a *dict* of masks declares
            # one input per entry (`attention_mask.full_attention`, `attention_mask.linear_attention`), and
            # both would otherwise collapse onto `attention_mask` and lose the session name behind it.
            kwarg_name = spec.name.removeprefix("input.")
            kwarg, _, path = kwarg_name.partition(".")
            in_cache = kwarg in self.cache_inputs
            exposed = spec.name if in_cache else kwarg_name
            if in_cache:
                # Keyed by path, not position: a recurrent layer's states are `None` until a step produces
                # them, so the cache's tensor leaves are a *subset* of what the graph declares.
                self._cache_paths.setdefault(kwarg, {})[exposed] = path.split(".") if path else []
            self._session_names[exposed] = spec.name
            # ORT rejects a feed whose dtype differs from the declared one, and the masks the runtime builds
            # are not always the type the graph was traced with (a bool padding mask vs a float causal one).
            self._input_dtypes[exposed] = _ort_to_torch_dtype(spec.type)
            # The *declared* shape, which is not the traced one: it sizes a cache entry the model has not
            # created yet, and only this says which axes ORT left symbolic.
            self.input_shapes[exposed] = tuple(spec.shape)
        self.input_names = tuple(self._session_names)
        # The way back, for a feed keyed the way the session names things: `input_shapes` is keyed by the
        # name the *caller* uses, which for a cache leaf is the graph's own and for anything else is not.
        self._exposed_names = {session: exposed for exposed, session in self._session_names.items()}

        # Feed the session by *pointer* (io-binding) rather than by value. On CUDA that removes a device copy
        # of every input and every output — per step, the whole KV cache twice over. On CPU there was no copy
        # to remove (a `.numpy()` view shares the buffer), but binding still pays: the cache pair below can
        # share one buffer, so ORT stops allocating a full-size cache output on every step. Driven directly
        # rather than through `onnxruntime.transformers`' `CudaSession`, which is CUDA-only and builds its
        # numpy type map eagerly, raising on `bfloat16`.
        self._output_shapes = {spec.name: tuple(spec.shape) for spec in session.get_outputs()}
        specs = (*session.get_inputs(), *session.get_outputs())
        self._element_types = {spec.name: _ort_element_type(spec.type) for spec in specs}
        self._io_dtypes = {spec.name: _ort_to_torch_dtype(spec.type) for spec in specs}
        self._io_binding, self._shared_outputs, self._cuda_graph, self._binds = None, {}, False, False
        self._buffers: dict[str, torch.Tensor] = {}
        # A dtype numpy cannot spell cannot be bound at all (there is no public way to hand ORT an existing
        # pointer without an element type), so such a graph keeps the plain `run` path.
        if all(kind is not None for kind in self._element_types.values()):
            self._io_binding = session.io_binding()
            # A cache the graph mutates in place is declared as a matched `input.<name>`/`output.<name>` pair.
            # Where the two shapes are *identical* the write lands in the buffer it read from, so bind both to
            # one tensor and the update costs nothing (the documented static-cache trick). A growing cache
            # declares a longer output than its input, so the shapes differ and it is left alone — sharing
            # there would bind an output to a buffer too small to hold it.
            for spec in session.get_inputs():
                paired = "output." + spec.name.removeprefix("input.")
                if spec.name.startswith("input.") and self._output_shapes.get(paired) == tuple(spec.shape):
                    self._shared_outputs[spec.name] = paired
            # A decode step of one token over a static cache repeats one shape, which ORT can capture as a
            # CUDA graph and replay, dropping the per-step launch overhead. Capture happens on the first run
            # and replay needs the bound pointers to hold still, so those inputs get buffers this runner keeps
            # (below) instead of wherever the caller allocated them.
            #
            # Gated on the session's own `enable_cuda_graph` and nothing else. It is a provider option fixed
            # at creation, so it is the caller's decision; and once they have made it ORT captures whatever
            # the first run binds, whether we cooperate or not — binding fresh pointers each call would break
            # the replay. Whether an axis was *declared* static is beside the point: a batch dim left
            # symbolic by a dynamic export is still one constant value for a generation loop. What ORT
            # requires is that the shapes do not change between calls, and it says so itself if they do.
            # The same goes for the cache length: a static cache declares that axis symbolically (it is
            # `max_cache_len`, resolved per session) and then holds it constant for every step, which is
            # exactly the shape worth capturing.
            options = session.get_provider_options().get("CUDAExecutionProvider", {})
            self._cuda_graph = options.get("enable_cuda_graph") in ("1", 1, True)
            # Binding is on for any graph whose types it can express. An output whose *size* it cannot work
            # out no longer costs the graph its binding: that one output is left to ORT to allocate (see
            # `_bound_run`), which is strictly better than feeding the whole thing through the host to avoid
            # a single unsizeable tensor.
            self._binds = True

    @staticmethod
    def _providers_for(device=None) -> list[str | tuple[str, dict]]:
        """The providers to open a session with. `device` pins it; without one, CUDA is used only when a
        device is actually visible — an ORT build carries its CUDA provider whether or not the machine has a
        GPU, and asking for it without one prints a provider failure before falling back on its own.

        The GPU is named outright, where ORT would otherwise default to device 0 whatever the caller asked
        for. A session that runs on a different GPU than its inputs live on does not copy them across: it
        reads the address it was given on the GPU it was opened for, which faults.
        """
        import onnxruntime

        device = _session_device(device)
        if device.type != "cuda":
            return ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
            raise ValueError("This onnxruntime build has no CUDA provider, so the session cannot run on CUDA.")
        return [("CUDAExecutionProvider", {"device_id": device.index})]

    @classmethod
    def from_artifact(cls, artifact, export_metadata=None, device=None, providers=None, **kwargs) -> OnnxModelRunner:
        """Open an in-memory `ONNXProgram` as a session, without writing the proto out first."""
        import onnxruntime

        providers = providers or cls._providers_for(device)
        session = onnxruntime.InferenceSession(artifact.model_proto.SerializeToString(), providers=providers)
        return cls(session, export_metadata=export_metadata, source=artifact, **kwargs)

    @classmethod
    def from_pretrained(cls, path, export_metadata=None, device=None, providers=None, **kwargs) -> OnnxModelRunner:
        """Open a saved `.onnx` as an ORT session. `device` picks the providers (and so where the runner's
        outputs land); pass `providers` to choose them outright."""
        import onnxruntime

        providers = providers or cls._providers_for(device)
        return cls(
            onnxruntime.InferenceSession(str(path), providers=providers),
            export_metadata=export_metadata,
            source=path,
            **kwargs,
        )

    def to(self, device) -> OnnxModelRunner:
        """Another runner over the same model, opened on `device`'s providers.

        Not a move: a session's execution provider is fixed when it is created, so this opens a second one.
        That costs an ORT session load -- small next to the export, not free -- and it is why the runner is
        returned rather than mutated."""
        # `cuda` and `cuda:0` name the same session when the current GPU is 0, and reopening for the
        # difference in spelling would only pay the load again.
        if _session_device(device) == self.device:
            return self
        if self._source is None:
            return super().to(device)
        loader = self.from_pretrained if isinstance(self._source, (str, Path)) else self.from_artifact
        # The recorded payload, not the parsed object: what a load injects is the raw mapping.
        return loader(self._source, export_metadata=self.export_metadata.raw, device=device)

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        feed = self._flattened(kwargs)
        return self._bound_run(feed) if self._binds else self._host_run(feed)

    def _flattened(self, kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """The graph's inputs as flat tensors under the names it declares them by.

        Two things are not tensors on the way in: the cache, whose leaves the graph takes one by one, and any
        other pytree kwarg (`encoder_outputs`, a mask dict), whose leaves it names by dotted path.
        """
        # Every cache the graph declares, not just the one the generation loop grows: a model that caches its
        # encoder separately (voxtral_realtime) takes two, and the leaves of the second are `input.`-prefixed
        # exactly like the first — left to the generic pytree branch below they would be fed under an
        # unprefixed name the session never declared.
        batch = next((t.shape[0] for t in kwargs.values() if isinstance(t, torch.Tensor) and t.dim() > 0), 1)
        for cache_input in self.cache_inputs:
            cache = kwargs.pop(cache_input, None)
            if cache is None:
                continue
            for name, path in self._cache_paths.get(cache_input, {}).items():
                entry = _read_cache_entry(cache, path)
                # A slot the cache still holds as `None` is a state the model has not created yet (a
                # recurrent layer's conv / SSM state before its first step). The graph takes it as an
                # input all the same, so hand it the zeros its own lazy initialization would have made,
                # sized from the shape the graph declares — only the batch axis is symbolic there.
                if entry is None:
                    shape = [
                        batch if axis == 0 or not isinstance(dim, int) else dim
                        for axis, dim in enumerate(self.input_shapes.get(name, ()))
                    ]
                    entry = torch.zeros(*shape, dtype=self._input_dtypes.get(name) or self.dtype)
                kwargs[name] = entry.detach()
        for name in [n for n, v in kwargs.items() if not isinstance(v, torch.Tensor)]:
            kwargs.update({f"{name}.{leaf}": t for leaf, t in get_leaf_tensors(kwargs.pop(name)).items()})
        return {
            self._session_names.get(name, name): tensor.detach().to(self._input_dtypes.get(name) or tensor.dtype)
            for name, tensor in kwargs.items()
        }

    def _host_run(self, feed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the session by value, through host arrays. For a graph whose outputs cannot be sized in
        advance — there is nothing to bind them to, and a numpy view of a cpu tensor shares its buffer."""
        outputs = self._session.run(None, {name: tensor.cpu().numpy() for name, tensor in feed.items()})
        return {
            name.removeprefix("output."): torch.from_numpy(value).to(self.device)
            for name, value in zip(self._output_names, outputs)
        }

    def _scratch(self, tensor: torch.Tensor, name: str) -> int:
        """The address of a one-element buffer standing in for an empty `tensor`, kept for this runner's
        lifetime so the pointer outlives the run."""
        key = f"{name}\0scratch"
        scratch = self._buffers.get(key)
        if scratch is None or scratch.dtype != tensor.dtype or scratch.device != tensor.device:
            scratch = self._buffers[key] = torch.empty(1, dtype=tensor.dtype, device=tensor.device)
        return scratch.data_ptr()

    def _bound_run(self, feed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the session over io-binding: every tensor bound where it already lives, by pointer.

        The outputs need buffers up front, sized by resolving their symbolic axes against the inputs that
        declare the same axis name — which is what ties `logits`' sequence axis to `input_ids`'. Whether that
        is possible at all was settled in `__init__` (`_binds`).
        """
        device, known_axes = self.device, {}
        device_type, device_index = device.type, device.index or 0
        for name, tensor in feed.items():
            for axis, dim in enumerate(self.input_shapes.get(self._exposed_names.get(name, name), ())):
                if isinstance(dim, str) and axis < tensor.dim():
                    known_axes[dim] = tensor.shape[axis]

        # Every output needs a size before anything is bound.
        resolved = {
            name: tuple(_resolved_axis(dim, known_axes) for dim in declared)
            for name, declared in self._output_shapes.items()
        }
        self._io_binding.clear_binding_inputs()
        self._io_binding.clear_binding_outputs()
        outputs = {}
        for name in list(feed):
            # Assigned back, not just rebound: binding hands ORT a *pointer*, so a converted tensor that only
            # existed as a loop local would be freed before the run and leave the graph reading whatever took
            # its place (an embedding `Gather` on freed memory reports indices out of bounds).
            tensor = feed[name] = feed[name].to(device).contiguous()
            # Under a CUDA graph the captured pointers have to stay put, so the tensor is copied into a
            # buffer this runner keeps rather than bound where the caller happened to allocate it.
            if self._cuda_graph:
                buffer = self._buffers.get(name)
                if buffer is None or buffer.shape != tensor.shape or buffer.dtype != tensor.dtype:
                    buffer = self._buffers[name] = torch.empty_like(tensor)
                buffer.copy_(tensor)
                tensor = buffer
            # A rank-0 entry (a cache's `cumulative_length` counter) binds as `[1]`, not `[]`: ORT sizes the
            # binding from the shape it is given, and an empty one leaves it writing into nothing.
            bound_shape = list(tensor.shape) or [1]
            # ORT refuses a null pointer, which is what an empty tensor has — and empty is ordinary here: a
            # cache entry at prefill holds zero positions. Such a tensor lends a one-element scratch buffer's
            # address instead; the bound shape still says zero, so nothing is read or written through it.
            pointer = tensor.data_ptr() or self._scratch(tensor, name)
            self._io_binding.bind_input(
                name, device_type, device_index, self._element_types[name], bound_shape, pointer
            )
            if (paired := self._shared_outputs.get(name)) is not None:
                # One buffer for the pair: the graph's write lands in the tensor it just read, under the same
                # shape the input went in with — which is what the graph writes back, so a rank-0 counter's
                # output is `[1]` here too.
                self._io_binding.bind_output(
                    paired, device_type, device_index, self._element_types[paired], bound_shape, pointer
                )
                outputs[paired] = tensor
        # Bind order is what `get_outputs()` comes back in, so the ORT-allocated ones are read back by index.
        bound_order = list(outputs)
        ort_allocated = []
        for name in self._output_shapes:
            if name in self._shared_outputs.values():  # bound to its input's buffer above, so already sized
                continue
            bound_order.append(name)
            shape = resolved[name]
            # An axis `_resolved_axis` will not guess at (a floor-div — a streamed modality's window folds
            # its stride away like that) leaves no size to allocate against, so that one output is ORT's to
            # size and costs a copy. Everything else binds, the rest of the graph included.
            if None in shape:
                self._io_binding.bind_output(name, device_type=device_type, device_id=device_index)
                ort_allocated.append(name)
                continue
            buffer = self._buffers.get(name)
            if buffer is None or tuple(buffer.shape) != shape:
                buffer = self._buffers[name] = torch.empty(shape, dtype=self._io_dtypes[name], device=device)
            # Exactly as declared — an output's shape is *verified* against what the node produces, where an
            # input's is merely sized from what it is given, so padding a rank-0 output to `[1]` is refused.
            self._io_binding.bind_output(
                name,
                device_type,
                device_index,
                self._element_types[name],
                list(shape),
                buffer.data_ptr() or self._scratch(buffer, name),
            )
            outputs[name] = buffer
        self._session.run_with_iobinding(self._io_binding)
        if ort_allocated:
            values = self._io_binding.get_outputs()
            for name in ort_allocated:
                value = values[bound_order.index(name)]
                outputs[name] = torch.from_numpy(value.numpy()).to(device)
        return {name.removeprefix("output."): tensor for name, tensor in outputs.items()}
