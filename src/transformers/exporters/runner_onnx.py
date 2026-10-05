"""Driving an ONNX Runtime session: `OnnxModelRunner`."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..utils.import_utils import is_torch_available
from .base import ModelRunner
from .cache import _read_cache_entry
from .metadata import ExportMetadata
from .utils import get_leaf_tensors


if is_torch_available():
    import torch


def _session_device(device=None) -> torch.device:
    """The device a session opened for `device` runs on, with the GPU index filled in (`cuda` -> current GPU)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _ort_to_torch_dtype(ort_type: str | None) -> torch.dtype | None:
    """torch dtype for an ORT type string (`"tensor(float)"`), or `None` if it names no tensor.

    Resolved by name rather than through `JitScalarType`, which maps `int32`/`int8` onto quantized dtypes.
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

    A symbolic axis can be arithmetic (a growing cache's output is `s36 + 1`); only sums and differences of
    known names and integers are resolved, anything else (products, floor-div) stays `None`.
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
    """ONNX element type for an ORT type string, for io-binding's `element_type`, or `None` if it names no tensor.

    The ONNX enum rather than a numpy type, which cannot spell `bfloat16` or fp8.
    """
    match = re.fullmatch(r"tensor\((\w+)\)", ort_type or "")
    if match is None:
        return None
    from onnx import TensorProto

    element_type = getattr(TensorProto, match.group(1).upper(), None)
    return element_type if isinstance(element_type, int) else None


class OnnxModelRunner(ModelRunner):
    """`ModelRunner` backed by an `onnxruntime.InferenceSession`, fed by io-binding.

    The cache rides as matched `input.<name>` / `output.<name>` graph inputs/outputs: a `past_key_values` kwarg
    is flattened into the feed and the `output.` prefix stripped from the results."""

    def __init__(self, session, export_metadata=None, source=None):
        self._session = session
        # What `to()` reopens: a session's provider is fixed at creation and it does not hand its model back.
        self._source = source
        self._output_names = [o.name for o in session.get_outputs()]
        self.export_metadata = ExportMetadata.from_dict(export_metadata)
        # Read off the provider, not assumed: a pointer bound on the wrong GPU is an illegal access.
        options = session.get_provider_options().get("CUDAExecutionProvider")
        self.device = torch.device("cpu" if options is None else f"cuda:{options.get('device_id', 0)}")

        # Mutated inputs (cache leaves, a merged decode's `attention_mask`) carry an `input.` prefix; stripping
        # it recovers the kwarg name.
        self._session_names, self._cache_paths = {}, {}
        for spec in session.get_inputs():
            # The whole dotted name: a dict of masks declares one input per entry (`attention_mask.full_attention`).
            kwarg_name = spec.name.removeprefix("input.")
            kwarg, _, path = kwarg_name.partition(".")
            in_cache = kwarg in self.cache_inputs
            exposed = spec.name if in_cache else kwarg_name
            if in_cache:
                # Keyed by path: `None` recurrent states make the cache's leaves a subset of the graph's inputs.
                self._cache_paths.setdefault(kwarg, {})[exposed] = path.split(".") if path else []
            self._session_names[exposed] = spec.name
        self.input_names = tuple(self._session_names)

        # io-binding avoids a device copy per input/output on CUDA, and on CPU lets a cache pair share one
        # buffer. Not `onnxruntime.transformers.CudaSession`: CUDA-only and it raises on `bfloat16`.
        specs = (*session.get_inputs(), *session.get_outputs())
        self._element_types = {spec.name: _ort_element_type(spec.type) for spec in specs}
        # By session name. ORT rejects a feed whose dtype differs from the declared one (bool vs float masks); the
        # declared shapes size a cache entry the model has not created yet, and the output buffers.
        self._io_dtypes = {spec.name: _ort_to_torch_dtype(spec.type) for spec in specs}
        self._shapes = {spec.name: tuple(spec.shape) for spec in specs}
        self._io_binding, self._shared_outputs = None, {}
        # An input/output without an element type cannot be bound; such a graph keeps the plain `run` path.
        if all(kind is not None for kind in self._element_types.values()):
            self._io_binding = session.io_binding()
            # An in-place cache pair with identical shapes binds to one buffer; a growing cache's output is
            # longer than its input, so it is left unshared.
            for spec in session.get_inputs():
                paired = "output." + spec.name.removeprefix("input.")
                if spec.name.startswith("input.") and self._shapes.get(paired) == tuple(spec.shape):
                    self._shared_outputs[spec.name] = paired

    @staticmethod
    def _providers_for(device=None) -> list[str | tuple[str, dict]]:
        """The providers to open a session with; CUDA only when `device` asks for it or a GPU is visible.

        The GPU index is named outright (ORT defaults to device 0), and TF32 follows torch's matmul setting:
        ORT enables it by default, which drifts fp32 graphs from eager by ~1e-2.
        """
        import onnxruntime

        device = _session_device(device)
        if device.type != "cuda":
            return ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
            raise ValueError("This onnxruntime build has no CUDA provider, so the session cannot run on CUDA.")
        use_tf32 = int(torch.backends.cuda.matmul.allow_tf32)
        return [("CUDAExecutionProvider", {"device_id": device.index, "use_tf32": use_tf32})]

    # `Pad_Fusion` folds a zero `Pad` into a following `MaxPool`, which pads with -inf: wrong on all-negative
    # border windows (BiT's stem).
    _DISABLED_OPTIMIZERS = ["Pad_Fusion"]

    @classmethod
    def from_artifact(cls, artifact, export_metadata=None, device=None, **kwargs) -> OnnxModelRunner:
        """Open an in-memory `ONNXProgram` as a session, without writing the proto out first."""
        import onnxruntime

        session = onnxruntime.InferenceSession(
            artifact.model_proto.SerializeToString(),
            providers=cls._providers_for(device),
            disabled_optimizers=cls._DISABLED_OPTIMIZERS,
        )
        return cls(session, export_metadata=export_metadata, source=artifact, **kwargs)

    @classmethod
    def from_pretrained(cls, path, export_metadata=None, device=None, **kwargs) -> OnnxModelRunner:
        """Open a saved `.onnx` as an ORT session on `device`'s providers."""
        import onnxruntime

        return cls(
            onnxruntime.InferenceSession(
                str(path), providers=cls._providers_for(device), disabled_optimizers=cls._DISABLED_OPTIMIZERS
            ),
            export_metadata=export_metadata,
            source=path,
            **kwargs,
        )

    def to(self, device) -> OnnxModelRunner:
        """Another runner over the same model, opened on `device`'s providers (a new session, not a move)."""
        if _session_device(device) == self.device:
            return self
        if self._source is None:
            return super().to(device)
        loader = self.from_pretrained if isinstance(self._source, (str, Path)) else self.from_artifact
        return loader(self._source, export_metadata=self.export_metadata.raw, device=device)

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        feed = self._flattened(kwargs)
        return self._bound_run(feed) if self._io_binding is not None else self._host_run(feed)

    def _flattened(self, kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """The graph's inputs as flat tensors under their declared names; pytree kwargs flatten by dotted path."""
        # Every cache the graph declares: voxtral_realtime also caches its encoder, `input.`-prefixed too.
        batch = next((t.shape[0] for t in kwargs.values() if isinstance(t, torch.Tensor) and t.dim() > 0), 1)
        for cache_input in self.cache_inputs:
            cache = kwargs.pop(cache_input, None)
            if cache is None:
                continue
            for name, path in self._cache_paths.get(cache_input, {}).items():
                entry = _read_cache_entry(cache, path)
                # A `None` slot (recurrent state before its first step) is fed the zeros lazy init would make.
                if entry is None:
                    shape = [
                        batch if axis == 0 or not isinstance(dim, int) else dim
                        for axis, dim in enumerate(self._shapes.get(name, ()))
                    ]
                    entry = torch.zeros(*shape, dtype=self._io_dtypes.get(name) or self.dtype)
                kwargs[name] = entry.detach()
        for name in [n for n, v in kwargs.items() if not isinstance(v, torch.Tensor)]:
            kwargs.update({f"{name}.{leaf}": t for leaf, t in get_leaf_tensors(kwargs.pop(name)).items()})
        feed = {}
        for name, tensor in kwargs.items():
            session_name = self._session_names.get(name, name)
            feed[session_name] = tensor.detach().to(self._io_dtypes.get(session_name) or tensor.dtype)
        return feed

    def _host_run(self, feed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the session by value, through host arrays."""
        outputs = self._session.run(None, {name: tensor.cpu().numpy() for name, tensor in feed.items()})
        return {
            name.removeprefix("output."): torch.from_numpy(value).to(self.device)
            for name, value in zip(self._output_names, outputs)
        }

    def _bound_run(self, feed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the session over io-binding; output buffers are sized from the inputs' symbolic axes."""
        device, known_axes = self.device, {}
        device_type, device_index = device.type, device.index or 0
        for name, tensor in feed.items():
            for axis, dim in enumerate(self._shapes.get(name, ())):
                if isinstance(dim, str) and axis < tensor.dim():
                    known_axes[dim] = tensor.shape[axis]

        resolved = {
            name: tuple(_resolved_axis(dim, known_axes) for dim in self._shapes[name]) for name in self._output_names
        }
        self._io_binding.clear_binding_inputs()
        self._io_binding.clear_binding_outputs()
        outputs = {}
        # ORT refuses the null pointer of an empty tensor; a one-element stand-in, kept alive through the run.
        scratch = []

        def address(tensor: torch.Tensor) -> int:
            if tensor.data_ptr():
                return tensor.data_ptr()
            scratch.append(torch.empty(1, dtype=tensor.dtype, device=tensor.device))
            return scratch[-1].data_ptr()

        for name in list(feed):
            # Assigned back into `feed` so a converted tensor outlives the run (ORT holds only its pointer).
            on_host = device_type == "cuda" and not feed[name].is_floating_point() and name not in self._shared_outputs
            # Integers bind on host: ORT reads them on its CPU, and a CUDA pointer there gives `CUDA failure 700`.
            tensor = feed[name] = (feed[name].cpu() if on_host else feed[name].to(device)).contiguous()
            # A rank-0 input binds as `[1]`: ORT would otherwise write into nothing.
            bound_shape = list(tensor.shape) or [1]
            pointer = address(tensor)
            self._io_binding.bind_input(
                name,
                "cpu" if on_host else device_type,
                0 if on_host else device_index,
                self._element_types[name],
                bound_shape,
                pointer,
            )
            if (paired := self._shared_outputs.get(name)) is not None:
                self._io_binding.bind_output(
                    paired, device_type, device_index, self._element_types[paired], bound_shape, pointer
                )
                outputs[paired] = tensor
        # `get_outputs()` follows bind order.
        bound_order = list(outputs)
        ort_allocated = []
        for name in self._output_names:
            if name in self._shared_outputs.values():
                continue
            bound_order.append(name)
            shape = resolved[name]
            # An unresolvable axis leaves that one output for ORT to allocate.
            if None in shape:
                self._io_binding.bind_output(name, device_type=device_type, device_id=device_index)
                ort_allocated.append(name)
                continue
            # Fresh each call: a reused buffer would overwrite the outputs an earlier call returned.
            buffer = torch.empty(shape, dtype=self._io_dtypes[name], device=device)
            # Exactly as declared: ORT verifies output shapes, so a rank-0 output cannot be padded to `[1]`.
            self._io_binding.bind_output(
                name,
                device_type,
                device_index,
                self._element_types[name],
                list(shape),
                address(buffer),
            )
            outputs[name] = buffer
        # ORT runs on its own stream: sync both ways or half-written tensors are read (illegal access on busy GPUs).
        if device_type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        self._session.run_with_iobinding(self._io_binding)
        if device_type == "cuda":
            self._io_binding.synchronize_outputs()
        if ort_allocated:
            values = self._io_binding.get_outputs()
            for name in ort_allocated:
                value = values[bound_order.index(name)]
                outputs[name] = torch.from_numpy(value.numpy()).to(device)
        return {name.removeprefix("output."): tensor for name, tensor in outputs.items()}
