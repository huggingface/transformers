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

import math

import numpy as np

from .utils import is_torch_available, requires_backends


if is_torch_available():
    import torch


class AudioOps:
    """The contract an audio backend implements: array operations, nothing else.

    Every numerical stage of the pipeline lives in `BaseAudioProcessor`, written against these
    ops. A backend is a subclass implementing each *core* op below (every method of this class);
    it may also implement the names in `OPTIONAL_OPS`, which individual models require. It holds no
    model state. `NumpyAudioBackend` is the reference implementation: when this docstring and that code
    disagree, the code wins.

    Conventions:
    - `like` is an existing backend array whose *device* the new array should share (`randn` also
      takes its dtype from it). It may be `None`.
    - `dtype` is a numpy-style name: ``"float32"``, ``"float64"``, ``"int32"``, ``"int64"``,
      ``"bool"``, ``"complex64"``, ``"complex128"``. A backend resolves ``"float64"`` to its widest
      available float for the array's device (for example float32 on an Apple GPU) and
      ``"complex128"`` likewise (some backends have no complex128 at all). `dtype_name` reports the name an
      array actually has.
    - Axes follow numpy; ``axis=-1`` is the sample / frame axis. Reductions take `axis` as an int,
      a tuple of ints or `None` (all axes) and honour `keepdims`.
    - Ops return backend arrays, except `to_numpy` and `dtype_name`.
    - Binary ops (`maximum`, `matmul`, arithmetic) require operands of the *same* dtype; the base
      casts with `astype(x, dtype_name(y))` first. `masked_fill`'s mask broadcasts *to* `x`, never
      beyond it. `randn`'s `shape` is a tuple.
    - `log`, `log10`, `sqrt`, `pow` and `mean` take float input. `clamp_min` takes any real dtype
      (Fun-ASR calls it on int64); its `value` is a Python scalar of the array's kind or a 0-d array.

    Array protocol — what the base may do with a backend array besides calling ops:
    - arithmetic between two arrays of the same dtype, or between an array and a Python scalar
      *of the array's kind* (an int for int arrays, a float for float arrays, either for complex)
      on either side: ``+ - * /`` (no ``/`` on int arrays), and unary ``-``;
    - a Python float ``**`` a float array (scalar base);
    - comparisons ``< <= > >= == !=`` giving bool arrays, and ``~ & |`` on bool arrays;
    - indexing with ints, slices with a positive step, `None` and `...` (views or copies; never
      rely on aliasing — torch rejects negative steps, so reversal is not part of the protocol;
      indexing away every axis may yield a scalar on numpy: keep one axis or use `keepdims=True`);
    - the attributes ``.shape`` and ``.ndim``.
    Nothing else: no item assignment (build with `masked_fill`, `concat`, `arange`), no `.dtype`
    (use `dtype_name`), no method-style reductions (use the ops).
    """

    name: str = ""

    # ── plumbing ───────────────────────────────────────────────────────────────────────────────

    def asarray(self, x, *, like=None):
        """Backend array from a numpy array, a sequence, or an existing backend array, keeping its dtype (subject to
        the widest-float rule in the class docstring). May copy; never rely on aliasing."""
        raise NotImplementedError

    def to_numpy(self, x):
        """Host numpy array with the same values and dtype."""
        raise NotImplementedError

    def to_device(self, x, device):
        """`x` on `device` (a backend-specific designator). Backends without devices accept `None` and `"cpu"`
        (returning `x`) and raise on anything else."""
        raise NotImplementedError

    def to_device_many(self, seq, device):
        """``[to_device(x, device) for x in seq]`` as a list; an empty `seq` gives ``[]``. A backend may batch
        the transfer (torch stages CPU→CUDA copies through pinned chunks); the values are those of per-item
        `to_device`, and the items may share storage (torch returns views of one device buffer)."""
        raise NotImplementedError

    def astype(self, x, dtype):
        """`x` cast to `dtype` (see the class docstring for the widest-float rule)."""
        raise NotImplementedError

    def dtype_name(self, x) -> str:
        """The numpy-style name of `x`'s dtype (``"float32"``, ``"complex64"``, ``"bool"``, ...)."""
        raise NotImplementedError

    def zeros(self, shape, *, dtype="float32", like=None):
        """Zeros of `shape` and `dtype`, on `like`'s device."""
        raise NotImplementedError

    def arange(self, n, *, like=None):
        """Integers ``0 .. n-1`` (int64), on `like`'s device."""
        raise NotImplementedError

    def linspace(self, start, stop, num, *, dtype="float32", like=None):
        """`num` evenly spaced values from `start` to `stop` inclusive, computed in `dtype` by the backend's own
        kernel (the bits differ between libraries in `float32`), on `like`'s device."""
        raise NotImplementedError

    # ── assembly ───────────────────────────────────────────────────────────────────────────────

    def stack(self, seq):
        """Stack equally shaped arrays along a new leading axis."""
        raise NotImplementedError

    def concat(self, seq, axis=-1):
        """Concatenate along an existing axis."""
        raise NotImplementedError

    def pad(self, x, left, right, *, axis=-1, mode="constant", value=0.0):
        """Pad `axis` by `left` / `right` samples. `mode`: ``"constant"`` (with `value`),
        ``"reflect"`` (mirror without repeating the edge), ``"replicate"`` (repeat the edge).
        `reflect` requires `left` and `right` each at most `len - 1`."""
        raise NotImplementedError

    def pad_many(self, seq, length, *, side="right", value=0.0):
        """Each array of `seq` constant-padded on its last axis to `length` samples with `value`, as a list in
        input order. `side` is where the padding goes: ``"right"`` appends the missing ``length - len`` samples
        after the last one, ``"left"`` prepends them before the first; any other value raises `ValueError`,
        even when no array needs padding; an empty `seq` gives ``[]``. An array already
        `length` or longer is returned with its values unchanged (not necessarily the same object). A backend
        may assemble the batch in one allocation (torch on CUDA); the values are those of per-item `pad`."""
        raise NotImplementedError

    def reshape(self, x, shape):
        """`x` with the tuple `shape` (at most one ``-1``), elements in row-major order. May copy."""
        raise NotImplementedError

    def take_along_axis(self, x, indices, axis):
        """numpy's `take_along_axis`: ``out[..., i, ...] = x[..., indices[..., i, ...], ...]`` along the int
        `axis`. `indices` is an int64 array with `x`'s ndim; its axes other than `axis` are 1 or equal to `x`'s
        (`indices` broadcasts up to `x`, never `x` up to `indices`). A copy of the selected values, exact."""
        raise NotImplementedError

    def masked_fill(self, x, mask, value):
        """`x` with `value` written where the boolean `mask` is true; `mask` must broadcast *to* `x`'s shape (the
        result has `x`'s shape and dtype). Keeps `x`'s axis order, so a reduction over the result accumulates in
        that order (MLX: best effort)."""
        raise NotImplementedError

    def swapaxes(self, x, axis1, axis2):
        """`x` with two axes exchanged (may be a view)."""
        raise NotImplementedError

    def contiguous(self, x):
        """Return `x` in row-major contiguous memory; a copy only if needed. Layout is observable: a
        GEMM over a strided operand accumulates in a different order, which is why a stage may
        require it."""
        raise NotImplementedError

    # ── elementwise ────────────────────────────────────────────────────────────────────────────

    def log(self, x):
        """Elementwise natural logarithm. Float input."""
        raise NotImplementedError

    def log10(self, x):
        """Elementwise base-10 logarithm. Float input."""
        raise NotImplementedError

    def sqrt(self, x):
        """Elementwise square root. Float input."""
        raise NotImplementedError

    def abs(self, x):
        """Magnitude; for complex input the modulus, in the matching real dtype."""
        raise NotImplementedError

    def real(self, x):
        """Real part of a complex array, in the matching real dtype."""
        raise NotImplementedError

    def imag(self, x):
        """Imaginary part of a complex array, in the matching real dtype."""
        raise NotImplementedError

    def maximum(self, x, y):
        """Elementwise maximum of two broadcastable arrays."""
        raise NotImplementedError

    def minimum(self, x, y):
        """Elementwise minimum of two broadcastable arrays."""
        raise NotImplementedError

    def clamp_min(self, x, value):
        """Elementwise ``max(x, value)``. `value` is a Python scalar of the array's kind or a 0-d array;
        any real dtype."""
        raise NotImplementedError

    def exp(self, x):
        """Elementwise ``e**x``. Float input."""
        raise NotImplementedError

    def where(self, cond, x, y):
        """`x` where the boolean `cond` holds, else `y` (broadcast)."""
        raise NotImplementedError

    def pow(self, x, exponent):
        """Elementwise ``x ** exponent`` for a Python scalar `exponent`. Float input."""
        raise NotImplementedError

    # ── reductions ─────────────────────────────────────────────────────────────────────────────

    def sum(self, x, axis, *, keepdims=False):
        """Sum over `axis` (int, tuple of ints, or `None` for all axes). `axis=()` is not allowed (likewise for `mean` and `amax`)."""
        raise NotImplementedError

    def mean(self, x, axis, *, keepdims=False):
        """Mean over `axis` (int, tuple of ints, or `None`; `axis=()` is not allowed). Float input only."""
        raise NotImplementedError

    def amax(self, x, axis, *, keepdims=False):
        """Maximum over `axis` (int, tuple of ints, or `None`; `axis=()` is not allowed)."""
        raise NotImplementedError

    def amin(self, x, axis, *, keepdims=False):
        """Minimum over `axis` (int, tuple of ints, or `None`; `axis=()` is not allowed)."""
        raise NotImplementedError

    def argmax(self, x, axis):
        """int64 index of the maximum along the int `axis`; the first occurrence on ties."""
        raise NotImplementedError

    def argmin(self, x, axis):
        """int64 index of the minimum along the int `axis`; the first occurrence on ties."""
        raise NotImplementedError

    # ── linear algebra ─────────────────────────────────────────────────────────────────────────

    def matmul(self, a, b):
        """Batched matrix product with numpy broadcasting over leading axes. Two 1-d operands may
        give a numpy scalar rather than an array; the base only uses 2-d (or higher) operands."""
        raise NotImplementedError

    # ── signal ─────────────────────────────────────────────────────────────────────────────────

    def window(self, name, length, *, periodic, dtype=None, like=None, **params):
        """Analysis window of `length` samples. `name` is one of ``"hann"``, ``"hamming"``,
        ``"boxcar"``, ``"povey"`` (Hann to the 0.85) or ``"blackman"``; `periodic` builds
        `length + 1` points and drops the last. `params` carries `blackman_coeff`, required for
        ``"blackman"``; other keys are ignored. `dtype` is the dtype name the window is *returned*
        in (only torch also evaluates in it). `dtype=None` is the one place a backend default
        differs: it means the backend's native precision for a window built for `like` -- `like`'s
        dtype on torch (as `torch.hann_window(..., dtype=audio.dtype)`), float64 on numpy;
        outside backends use their native float (MLX: float32). `like` is a device hint, as for
        `zeros`: torch builds the window on `like`'s device; numpy and MLX ignore it. Each backend
        builds its own way, so float32 windows differ by ulps between backends; float64 ones agree
        to ~1e-16."""
        raise NotImplementedError

    def frame(self, x, frame_length, hop):
        """Overlapping frames of the last axis: ``(..., n_frames, frame_length)`` with
        ``n_frames = 1 + (len - frame_length) // hop``. May return a view.
        Read-only: callers multiply by a window into a new array and must not write into the frames.
        `x` must have at least `frame_length` samples."""
        raise NotImplementedError

    def rfft(self, x, n):
        """Real FFT of the last axis, `n` points, one-sided, unnormalised ("backward"). `n` shorter
        than the axis truncates, longer zero-pads. The output precision matches the input's
        (float32 → complex64, float64 → complex128 where the backend has it); callers wanting a
        wider FFT cast first."""
        raise NotImplementedError

    def resample(self, x, orig_sr, target_sr):
        """Resample the last axis from `orig_sr` to `target_sr` Hz with the backend's resampler,
        keeping the dtype. Sample rates are ints. Output length is ``len * target_sr // orig_sr``: resamplers disagree on
        the last sample at non-integer ratios, so every backend trims to that rule. Each backend resamples
        with its own library's filter (numpy: soxr HQ; torch: torchaudio's windowed sinc); they differ by
        up to ~1e-3 relative in-band and more near Nyquist; see the step-3 record."""
        raise NotImplementedError

    # ── randomness ─────────────────────────────────────────────────────────────────────────────

    def randn(self, shape, *, seed=None, like=None):
        """Standard normal samples of the tuple `shape`, in `like`'s dtype (float32 when `like` is
        `None`; `like` must be a float array) and on its device. An integer `seed` gives a private stream seeded with it; `None`
        draws from the library's global stream (so `torch.manual_seed` / `np.random.seed` apply).
        Streams differ between backends and between devices; only moments match."""
        raise NotImplementedError


CORE_OPS = tuple(name for name, value in vars(AudioOps).items() if callable(value) and not name.startswith("_"))

# Ops a backend may implement and a model may require (`required_ops`). No stub on `AudioOps`:
# their presence on a backend class means exactly "this backend implements it". The numpy
# implementation's docstring is the contract.
OPTIONAL_OPS = ("bilinear_resize",)


_AUDIO_BACKENDS: dict[str, type[AudioOps]] = {}


def missing_ops(backend_cls, names) -> list[str]:
    """Names in `names` that `backend_cls` (a backend class or instance) does not implement; a core
    op left as the contract's stub counts as missing."""
    if not isinstance(backend_cls, type):
        backend_cls = type(backend_cls)
    return [
        name
        for name in names
        if not callable(getattr(backend_cls, name, None))
        or getattr(backend_cls, name) is getattr(AudioOps, name, None)
    ]


def register_audio_backend(name: str, backend_cls: type[AudioOps]) -> None:
    """Make `backend_cls` loadable as ``backend=name``. Refused unless every core op is implemented."""
    if name in _AUDIO_BACKENDS:
        raise ValueError(f"Audio backend {name!r} is already registered.")
    missing = missing_ops(backend_cls, CORE_OPS)
    if missing:
        raise TypeError(
            f"Cannot register audio backend {name!r}: {backend_cls.__name__} is missing core ops {missing}."
        )
    own_name = vars(backend_cls).get("name")
    if own_name and own_name != name:
        raise ValueError(f"{backend_cls.__name__} is named {own_name!r}, cannot register it as {name!r}.")
    backend_cls.name = name
    _AUDIO_BACKENDS[name] = backend_cls


def get_audio_backend(name: str) -> type[AudioOps]:
    try:
        return _AUDIO_BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"Unknown audio backend {name!r}. Registered backends: {sorted(_AUDIO_BACKENDS)}. An outside "
            "backend registers itself when its module is imported (e.g. `import mlx_audio.transformers_backend`)."
        ) from None


def register_audio_op(backend: str, name: str):
    """Attach `fn` as op `name` on the registered backend `backend`, for code outside the library
    (remote-code processors, outside backends adding an optional op). In-library ops are plain
    methods in this module. Raises if `name` is a built-in op or attribute of the backend, or an op
    already registered with a different function. Registering a function with the same
    `__qualname__` again replaces it silently: remote code is re-imported as a new module on a new
    revision or `force_download`, and two repos may ship the same file under two module names.

    The function receives no `self`: it is attached as a staticmethod, so it cannot call the
    backend's other ops. An op attached to a shipped class is inherited by every subclass of it."""

    def decorator(fn):
        backend_cls = get_audio_backend(backend)
        if name.startswith("_"):
            raise ValueError(f"{name!r} is not a valid op name.")
        existing = getattr(backend_cls, name, None)
        if hasattr(backend_cls, name) and not (
            getattr(existing, "_registered_audio_op", False) and existing.__qualname__ == fn.__qualname__
        ):
            raise ValueError(f"Op {name!r} already exists on audio backend {backend!r}.")
        op = staticmethod(fn)
        # Marks the function, which a class attribute lookup returns, so a later registration can
        # tell an op registered here from a built-in one.
        op.__func__._registered_audio_op = True
        setattr(backend_cls, name, op)
        return fn

    return decorator


_NUMPY_PAD_MODES = {"constant": "constant", "reflect": "reflect", "replicate": "edge"}


def _check_axis(axis):
    """Shared by the reductions of every backend (`sum`, `mean`, `amax`, `amin`): `axis=()` reduces
    nothing in numpy but everything in torch; neither is what a caller means, so every backend rejects it."""
    if axis == ():
        raise ValueError("axis=() is not allowed; pass an int, a tuple of ints or None.")


def _side_widths(side, missing):
    """`(left, right)` pad widths that bring a waveform `missing` samples short to length on `side`."""
    if side == "right":
        return 0, missing
    if side == "left":
        return missing, 0
    raise ValueError(f"Invalid padding side: {side!r}; expected 'left' or 'right'.")


class NumpyAudioBackend(AudioOps):
    """The reference backend: every op, in numpy. No devices; `like` only matters for dtype."""

    name = "numpy"

    # ── plumbing ───────────────────────────────────────────────────────────────────────────────

    def asarray(self, x, *, like=None):
        return x if isinstance(x, np.ndarray) else np.asarray(x)

    def to_numpy(self, x):
        return np.asarray(x)

    def _check_host(self, device):
        # Every host spelling is accepted: "cpu", "cpu:0", torch.device("cpu").
        if device is not None and str(device).split(":")[0] != "cpu":
            raise ValueError(
                f"The {self.name!r} audio backend has no devices; got device={device!r}. Use backend='torch' to place "
                "the arrays on an accelerator."
            )

    def to_device(self, x, device):
        self._check_host(device)
        return x

    def to_device_many(self, seq, device):
        self._check_host(device)  # before the loop: an empty batch still refuses a device
        return list(seq)

    def astype(self, x, dtype):
        return x.astype(np.dtype(dtype), copy=False)

    def dtype_name(self, x):
        return x.dtype.name

    def zeros(self, shape, *, dtype="float32", like=None):
        return np.zeros(shape, dtype=np.dtype(dtype))

    def arange(self, n, *, like=None):
        return np.arange(n, dtype=np.int64)

    def linspace(self, start, stop, num, *, dtype="float32", like=None):
        return np.linspace(start, stop, num, dtype=np.dtype(dtype))

    # ── assembly ───────────────────────────────────────────────────────────────────────────────

    def stack(self, seq):
        return np.stack(list(seq))

    def concat(self, seq, axis=-1):
        return np.concatenate(list(seq), axis=axis)

    def pad(self, x, left, right, *, axis=-1, mode="constant", value=0.0):
        width = [(0, 0)] * x.ndim
        width[axis] = (left, right)
        if mode == "constant":
            return np.pad(x, width, mode="constant", constant_values=value)
        if mode not in _NUMPY_PAD_MODES:
            raise ValueError(f"Unknown pad mode {mode!r}.")
        return np.pad(x, width, mode=_NUMPY_PAD_MODES[mode])

    def pad_many(self, seq, length, *, side="right", value=0.0):
        _side_widths(side, 0)  # an unknown side raises even when nothing needs padding
        return [
            x if x.shape[-1] >= length else self.pad(x, *_side_widths(side, length - x.shape[-1]), value=value)
            for x in seq
        ]

    def reshape(self, x, shape):
        return np.reshape(x, shape)

    def take_along_axis(self, x, indices, axis):
        return np.take_along_axis(x, indices, axis)

    def masked_fill(self, x, mask, value):
        return np.where(mask, np.asarray(value, dtype=x.dtype), x)

    def swapaxes(self, x, axis1, axis2):
        return np.swapaxes(x, axis1, axis2)

    def contiguous(self, x):
        return np.ascontiguousarray(x)

    # ── elementwise ────────────────────────────────────────────────────────────────────────────
    #
    # Every result goes through `np.asarray`: a ufunc on 0-d input returns a numpy *scalar*
    # (`np.float32`), without the `.shape` / `.ndim` semantics the protocol promises; the wrap
    # guarantees an ndarray. It does not fix promotion: on numpy 1.x a 0-d array still promotes
    # against a Python float (float32 * 2.0 → float64), which is why the base keeps a reduced axis
    # (`keepdims=True`) wherever a reduction feeds arithmetic.

    def log(self, x):
        return np.asarray(np.log(x))

    def log10(self, x):
        return np.asarray(np.log10(x))

    def sqrt(self, x):
        return np.asarray(np.sqrt(x))

    def abs(self, x):
        return np.asarray(np.abs(x))

    def real(self, x):
        return np.asarray(np.real(x))

    def imag(self, x):
        return np.asarray(np.imag(x))

    def maximum(self, x, y):
        return np.asarray(np.maximum(x, y))

    def minimum(self, x, y):
        return np.asarray(np.minimum(x, y))

    def clamp_min(self, x, value):
        return np.asarray(np.maximum(x, np.asarray(value, dtype=x.dtype)))

    def exp(self, x):
        return np.asarray(np.exp(x))

    def where(self, cond, x, y):
        return np.asarray(np.where(cond, x, y))

    def pow(self, x, exponent):
        # the operator, not `np.power`: numpy fast-paths a scalar 2 to `square` and 0.5 to `sqrt`, as
        # the legacy extractors' `** power` did, and the ufunc does not
        return np.asarray(x**exponent)

    # ── reductions ─────────────────────────────────────────────────────────────────────────────

    def sum(self, x, axis, *, keepdims=False):
        _check_axis(axis)
        return np.asarray(np.sum(x, axis=axis, keepdims=keepdims))

    def mean(self, x, axis, *, keepdims=False):
        _check_axis(axis)
        return np.asarray(np.mean(x, axis=axis, keepdims=keepdims))

    def amax(self, x, axis, *, keepdims=False):
        _check_axis(axis)
        return np.asarray(np.amax(x, axis=axis, keepdims=keepdims))

    def amin(self, x, axis, *, keepdims=False):
        _check_axis(axis)
        return np.asarray(np.amin(x, axis=axis, keepdims=keepdims))

    def argmax(self, x, axis):
        return np.asarray(np.argmax(x, axis=axis)).astype(np.int64, copy=False)

    def argmin(self, x, axis):
        return np.asarray(np.argmin(x, axis=axis)).astype(np.int64, copy=False)

    # ── linear algebra ─────────────────────────────────────────────────────────────────────────

    def matmul(self, a, b):
        return np.matmul(a, b)

    # ── signal ─────────────────────────────────────────────────────────────────────────────────

    def window(self, name, length, *, periodic, dtype=None, like=None, **params):
        count = length + 1 if periodic else length
        fac = np.linspace(-np.pi, np.pi, count)
        if name == "hann":
            window = 0.5 + 0.5 * np.cos(fac)
        elif name == "hamming":
            window = 0.54 + 0.46 * np.cos(fac)
        elif name == "boxcar":
            window = np.ones(count)
        elif name == "povey":
            window = (0.5 + 0.5 * np.cos(fac)) ** 0.85
        elif name == "blackman":
            coeff = params["blackman_coeff"]
            indices = np.arange(count)
            angle = 2 * np.pi / (count - 1)
            window = coeff - 0.5 * np.cos(angle * indices) + (0.5 - coeff) * np.cos(2 * angle * indices)
        else:
            raise ValueError(f"Unknown window function {name!r}.")
        window = window[:length] if periodic else window
        return window.astype(np.dtype(dtype), copy=False) if dtype else window

    def frame(self, x, frame_length, hop):
        return np.lib.stride_tricks.sliding_window_view(x, frame_length, axis=-1)[..., ::hop, :]

    def rfft(self, x, n):
        # The contract keeps the input's precision; numpy 1.x computes a float32 FFT in float64 and
        # returns complex128, numpy 2 returns complex64 directly. The reference pins the *dtype*, and
        # values within the conformance tolerance.
        out = np.fft.rfft(x, n=n)
        return out.astype(np.complex64, copy=False) if x.dtype == np.float32 else out

    def resample(self, x, orig_sr, target_sr):
        requires_backends(self.resample, ["soxr"])
        import soxr

        if orig_sr == target_sr:
            return x
        # soxr wants (samples, channels); flatten leading axes to channels and restore them.
        leading = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1]).T
        out = soxr.resample(np.ascontiguousarray(flat), orig_sr, target_sr, quality="HQ").T
        out = out.reshape(*leading, out.shape[-1])
        return out[..., : x.shape[-1] * target_sr // orig_sr]  # the contract's length rule

    # ── randomness ─────────────────────────────────────────────────────────────────────────────

    def randn(self, shape, *, seed=None, like=None):
        dtype = like.dtype if like is not None else np.float32
        stream = np.random if seed is None else np.random.RandomState(seed)
        return stream.standard_normal(shape).astype(dtype, copy=False)

    # ── optional ops ───────────────────────────────────────────────────────────────────────────

    def bilinear_resize(self, x, size):
        """Bilinear resize of a 2-D ``(rows, cols)`` array to `size`, ``align_corners=False``
        (PyTorch / OpenCV convention): output pixel centres map to ``(i + 0.5) * in/out - 0.5``,
        clamped at 0, interpolating between the two nearest source rows and columns. Coordinates
        are computed in float32, as `torch.nn.functional.interpolate` does. Required by CLAP's
        fusion views (the shrink of a whole-clip mel to the encoder's frame count)."""
        out_rows, out_cols = size
        in_rows, in_cols = x.shape

        def coords(n_in, n_out):
            scale = np.float32(n_in) / np.float32(n_out)
            src = np.maximum((np.arange(n_out, dtype=np.float32) + np.float32(0.5)) * scale - np.float32(0.5), 0)
            i0 = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
            i1 = np.minimum(i0 + 1, n_in - 1)
            return i0, i1, (src - i0).astype(x.dtype)

        r0, r1, wr = coords(in_rows, out_rows)
        c0, c1, wc = coords(in_cols, out_cols)
        top = x[r0][:, c0] * (1 - wc) + x[r0][:, c1] * wc
        bottom = x[r1][:, c0] * (1 - wc) + x[r1][:, c1] * wc
        return top * (1 - wr)[:, None] + bottom * wr[:, None]


register_audio_backend("numpy", NumpyAudioBackend)


if is_torch_available():

    class TorchAudioBackend(AudioOps):
        """Every op in torch. Tensors live wherever `like` / the input lives; `"float64"` resolves
        to float32 on MPS, the library's convention (`torch.float64 if device.type != "mps" else
        torch.float32`, as in ViT-MAE, RT-DETR and Whisper generation), applied once, here."""

        name = "torch"

        _DTYPES = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool,
            "complex64": torch.complex64,
            "complex128": torch.complex128,
        }
        _WIDEST_ON_MPS = {"float64": "float32", "complex128": "complex64"}

        @staticmethod
        def _resolve_dtype(dtype: str, device_type: str) -> str:
            if device_type == "mps":
                return TorchAudioBackend._WIDEST_ON_MPS.get(dtype, dtype)
            return dtype

        @classmethod
        def _torch_dtype(cls, dtype: str, device_type: str):
            name = cls._resolve_dtype(dtype, device_type)
            if name not in cls._DTYPES:
                raise ValueError(f"Unknown dtype {dtype!r}; expected one of {sorted(cls._DTYPES)}.")
            return cls._DTYPES[name]

        @staticmethod
        def _device_of(like):
            return like.device if isinstance(like, torch.Tensor) else torch.device("cpu")

        # ── plumbing ───────────────────────────────────────────────────────────────────────────

        def asarray(self, x, *, like=None):
            if isinstance(x, torch.Tensor):
                result = x
            elif isinstance(x, np.ndarray):
                # `from_numpy` cannot wrap read-only or negative-stride arrays: this copies read-only
                # or negative-stride input (no warning); everything else is shared without a copy.
                if x.flags.writeable and all(stride >= 0 for stride in x.strides):
                    result = torch.from_numpy(x)
                else:
                    result = torch.from_numpy(np.array(x))
            else:
                # Sequences (`list[list[float]]`) are a documented input type; go through numpy so
                # the torch backend accepts everything the numpy one does.
                result = torch.from_numpy(np.asarray(x))
            return self.to_device(result, like.device) if isinstance(like, torch.Tensor) else result

        def to_numpy(self, x):
            return x.detach().cpu().numpy()

        def to_device(self, x, device):
            device = torch.device(device)
            # The widest-float rule lives in `_resolve_dtype`; this is the transfer side of it.
            if device.type == "mps" and x.dtype in (torch.float64, torch.complex128):
                x = x.to(torch.float32 if x.dtype == torch.float64 else torch.complex64)
            if x.device == device:
                return x
            if device.type == "cuda" and x.device.type == "cpu":
                # Pinned staging makes the copy asynchronous, so preprocessing overlaps a model
                # forward still running on the stream.
                return x.pin_memory().to(device, non_blocking=True)
            return x.to(device)

        _TRANSFER_CHUNK_BYTES = 8 * 2**20

        def to_device_many(self, seq, device):
            seq, device = list(seq), torch.device(device)
            if all(x.device == device for x in seq):
                return seq
            if (
                device.type != "cuda"
                or len(seq) == 1
                or any(x.device.type != "cpu" or x.ndim != 1 for x in seq)
                or len({x.dtype for x in seq}) > 1
            ):
                return [self.to_device(x, device) for x in seq]
            # Stage through pinned memory in ~8 MB chunks, so packing chunk k+1 on the host overlaps the DMA
            # of chunk k and only the last chunk's copy is exposed. Per-waveform pageable copies each blocked
            # the host, which made the transfer ~4x slower than the bytes it moves. The waveforms come back
            # as views of one device buffer.
            lengths = [x.shape[-1] for x in seq]
            flat = torch.empty(sum(lengths), dtype=seq[0].dtype, device=device)
            chunk_samples = self._TRANSFER_CHUNK_BYTES // flat.element_size()
            offset = start = pending = 0
            for end, length in enumerate(lengths, start=1):
                pending += length
                if pending < chunk_samples and end < len(seq):
                    continue
                host = torch.cat(seq[start:end], out=torch.empty(pending, dtype=flat.dtype, pin_memory=True))
                flat[offset : offset + pending].copy_(host, non_blocking=True)
                offset, start, pending = offset + pending, end, 0
            return list(flat.split(lengths))

        def astype(self, x, dtype):
            return x.to(self._torch_dtype(dtype, x.device.type))

        def dtype_name(self, x):
            return str(x.dtype).removeprefix("torch.")

        def zeros(self, shape, *, dtype="float32", like=None):
            device = self._device_of(like)
            return torch.zeros(shape, dtype=self._torch_dtype(dtype, device.type), device=device)

        def arange(self, n, *, like=None):
            return torch.arange(n, dtype=torch.int64, device=self._device_of(like))

        def linspace(self, start, stop, num, *, dtype="float32", like=None):
            device = self._device_of(like)
            return torch.linspace(start, stop, num, dtype=self._torch_dtype(dtype, device.type), device=device)

        # ── assembly ───────────────────────────────────────────────────────────────────────────

        def stack(self, seq):
            return torch.stack(list(seq))

        def concat(self, seq, axis=-1):
            return torch.cat(list(seq), dim=axis)

        def pad(self, x, left, right, *, axis=-1, mode="constant", value=0.0):
            axis = axis % x.ndim
            if mode == "constant":
                spec = [0, 0] * (x.ndim - 1 - axis) + [left, right]
                return torch.nn.functional.pad(x, spec, "constant", value)
            if mode not in ("reflect", "replicate"):
                raise ValueError(f"Unknown pad mode {mode!r}.")
            # `reflect` / `replicate` pad only the last axis of a (batch, channel, length) tensor.
            # For a non-last axis this copies once in and once out and returns a non-contiguous tensor.
            moved = x.movedim(axis, -1)
            flat = moved.reshape(1, -1, moved.shape[-1])
            padded = torch.nn.functional.pad(flat, (left, right), mode=mode)
            return padded.reshape(*moved.shape[:-1], padded.shape[-1]).movedim(-1, axis)

        def pad_many(self, seq, length, *, side="right", value=0.0):
            _side_widths(side, 0)  # an unknown side raises even when nothing needs padding
            seq = list(seq)
            # Batched assembly only pays off on an accelerator: on CPU the per-item pads are cheaper than a
            # scatter, and a single waveform has nothing to amortize.
            if (
                len(seq) <= 1
                or seq[0].device.type == "cpu"
                or any(x.ndim != 1 or x.device != seq[0].device for x in seq)
                or len({x.dtype for x in seq}) > 1
                or any(x.shape[-1] > length for x in seq)
            ):
                return [
                    x if x.shape[-1] >= length else self.pad(x, *_side_widths(side, length - x.shape[-1]), value=value)
                    for x in seq
                ]
            # One allocation and one scatter for the whole batch: per-waveform `pad` paid an allocation, a fill
            # and a copy per item, and the Python loop over rows kept the host busy while the device idled.
            # `masked_scatter_` fills the mask in row-major order, which is exactly the concatenation order for
            # either side, so the result is a pure copy -- bit-identical to per-item padding.
            device = seq[0].device
            lengths = self.asarray(np.asarray([x.shape[-1] for x in seq], dtype=np.int64), like=seq[0])
            positions = torch.arange(length, device=device)
            mask = positions >= length - lengths[:, None] if side == "left" else positions < lengths[:, None]
            batch = torch.full((len(seq), length), value, dtype=seq[0].dtype, device=device)
            return list(batch.masked_scatter_(mask, torch.cat(seq)))

        def reshape(self, x, shape):
            return x.reshape(shape)

        def take_along_axis(self, x, indices, axis):
            return torch.take_along_dim(x, indices, dim=axis)  # broadcasts like numpy

        def masked_fill(self, x, mask, value):
            # keep `x`'s axis order (dense strides), as `np.where` does when the mask broadcasts to `x`: out-of-place
            # `masked_fill` returns a row-major copy, and a reduction over a transposed batch then accumulates in
            # another order (Parakeet's per-feature normalization moved by an ulp against legacy)
            return x.clone(memory_format=torch.preserve_format).masked_fill_(mask, value)

        def swapaxes(self, x, axis1, axis2):
            return x.transpose(axis1, axis2)

        def contiguous(self, x):
            return x.contiguous()

        # ── elementwise ────────────────────────────────────────────────────────────────────────

        def log(self, x):
            return torch.log(x)

        def log10(self, x):
            return torch.log10(x)

        def sqrt(self, x):
            return torch.sqrt(x)

        def abs(self, x):
            return torch.abs(x)

        def real(self, x):
            return torch.real(x)

        def imag(self, x):
            return torch.imag(x)

        def maximum(self, x, y):
            return torch.maximum(x, y)

        def minimum(self, x, y):
            return torch.minimum(x, y)

        def clamp_min(self, x, value):
            return torch.clamp(x, min=value)

        def exp(self, x):
            return torch.exp(x)

        def where(self, cond, x, y):
            return torch.where(cond, x, y)

        def pow(self, x, exponent):
            return torch.pow(x, exponent)

        # ── reductions ─────────────────────────────────────────────────────────────────────────

        @staticmethod
        def _dims(x, axis):
            return tuple(range(x.ndim)) if axis is None else axis

        def sum(self, x, axis, *, keepdims=False):
            _check_axis(axis)
            return torch.sum(x, dim=self._dims(x, axis), keepdim=keepdims)  # bool sums to int64, as numpy

        def mean(self, x, axis, *, keepdims=False):
            _check_axis(axis)
            return torch.mean(x, dim=self._dims(x, axis), keepdim=keepdims)

        def amax(self, x, axis, *, keepdims=False):
            _check_axis(axis)
            return torch.amax(x, dim=self._dims(x, axis), keepdim=keepdims)

        def amin(self, x, axis, *, keepdims=False):
            _check_axis(axis)
            return torch.amin(x, dim=self._dims(x, axis), keepdim=keepdims)

        def argmax(self, x, axis):
            return torch.argmax(x, dim=axis)  # int64; the first occurrence on ties

        def argmin(self, x, axis):
            return torch.argmin(x, dim=axis)

        # ── linear algebra ─────────────────────────────────────────────────────────────────────

        def matmul(self, a, b):
            return torch.matmul(a, b)

        # ── signal ─────────────────────────────────────────────────────────────────────────────

        def window(self, name, length, *, periodic, dtype=None, like=None, **params):
            device = self._device_of(like)
            if dtype is not None:
                torch_dtype = self._torch_dtype(dtype, device.type)
            else:
                torch_dtype = like.dtype if isinstance(like, torch.Tensor) else torch.float32
            kwargs = {"dtype": torch_dtype, "device": device}
            if name == "hann":
                return torch.hann_window(length, periodic=periodic, **kwargs)
            if name == "hamming":
                return torch.hamming_window(length, periodic=periodic, **kwargs)
            if name == "boxcar":
                return torch.ones(length, **kwargs)
            if name == "povey":
                return torch.hann_window(length, periodic=periodic, **kwargs).pow(0.85)
            if name == "blackman":
                coeff = params["blackman_coeff"]
                angle = 2 * math.pi / (length if periodic else length - 1)
                indices = torch.arange(length, **kwargs)
                return coeff - 0.5 * torch.cos(angle * indices) + (0.5 - coeff) * torch.cos(2 * angle * indices)
            raise ValueError(f"Unknown window function {name!r}.")

        def frame(self, x, frame_length, hop):
            # A writable overlapping view; by contract callers never write into frames (numpy's view is read-only).
            return x.unfold(-1, frame_length, hop)

        def rfft(self, x, n):
            return torch.fft.rfft(x, n=n)

        def resample(self, x, orig_sr, target_sr):
            requires_backends(self.resample, ["torchaudio"])
            import torchaudio

            out = torchaudio.functional.resample(x, orig_freq=orig_sr, new_freq=target_sr)
            return out[..., : x.shape[-1] * target_sr // orig_sr]  # the contract's length rule

        # ── randomness ─────────────────────────────────────────────────────────────────────────

        def randn(self, shape, *, seed=None, like=None):
            # Streams differ by device as well as by backend: the same seed gives different numbers on CPU and CUDA.
            device = self._device_of(like)
            dtype = like.dtype if isinstance(like, torch.Tensor) else torch.float32
            if seed is None:
                return torch.randn(shape, dtype=dtype, device=device)  # the global stream (`torch.manual_seed`)
            generator = torch.Generator(device=device).manual_seed(seed)
            return torch.randn(shape, dtype=dtype, device=device, generator=generator)

        # ── optional ops ───────────────────────────────────────────────────────────────────────

        def bilinear_resize(self, x, size):
            resized = torch.nn.functional.interpolate(
                x[None, None], size=list(size), mode="bilinear", align_corners=False
            )
            return resized[0, 0]

    register_audio_backend("torch", TorchAudioBackend)
