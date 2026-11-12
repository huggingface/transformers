# [Temporary] CI-only debug plugin for the torch 2.15 CUDA graph capture failures (branch `torch_2_15_nvidia_ci`).
#
# Target failures (CI-only, never reproduced on SSH runners of the same pool):
#   - DeepseekV32ModelTest::test_static_cache_no_recompile_with_smaller_length
#       "Triton Error [CUDA]: operation failed due to a previous error during capture" (Triton launch inside capture)
#   - WhisperModelTest::test_onnx_export_generate_dynamic_static_cache
#       "info.status != cudaStreamCaptureStatusInvalidated INTERNAL ASSERT FAILED" (CUDACachingAllocator, at an `mm`)
# Both happen while inductor's cudagraph trees record (`CUDAGraphNode._record`), right after a mem-efficient SDPA.
#
# What this plugin does (all output via `logging.getLogger("ci_capture_debug").warning`, shown live through
# `log_cli`, and mirrored to `$_PATCHED_TESTING_METHODS_OUTPUT_DIR/capture_debug_<pid>.log`, uploaded with the reports):
#   - every test: start/end line with CUDA memory, live thread names
#   - every CUDA graph capture (all tests): capture_begin/capture_end with the raw CUDA capture status
#     (`cudaStreamIsCapturing` through ctypes: NONE / ACTIVE / INVALIDATED), thread names, memory, stack
#   - cudagraph trees: warmup (`CUDAWarmupNode.run`) and record (`CUDAGraphNode._record`) start/end/failure
#   - GC runs and ORT `InferenceSession` finalizations that happen DURING a capture (with stack)
#   - target tests only, "full" runs: a TorchDispatchMode that logs every aten op inside warmup and capture with
#     the capture status after it, plus every Triton kernel launch during capture; the first op/launch after which
#     the status becomes INVALIDATED is reported as the culprit. SDPA ops log shapes/strides/dtypes/alignment.
#   - target tests only: generate()/decompose_prefill_decode inputs/outputs summaries
#   - on failure of a target: full traceback, the last trace events, all thread stacks, CUDA memory summary,
#     nvidia-smi
#   - target tests are repeated `CAPDBG_REPEAT` times (default 5) right after the original, with
#     `torch._dynamo.reset()` before each repeat so each one compiles and records fresh. The original run is
#     "light" (no per-op tracing, as close to the plain CI run as possible); odd repeats are "full", even repeats light.
#
# Env knobs: CAPDBG_DISABLE=1, CAPDBG_TARGETS (comma-separated nodeid suffixes), CAPDBG_REPEAT, CAPDBG_RESET=0.

import collections
import contextlib
import copy
import ctypes
import faulthandler
import gc
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
import weakref

import pytest


logger = logging.getLogger("ci_capture_debug")

DEFAULT_TARGETS = (
    "DeepseekV32ModelTest::test_static_cache_no_recompile_with_smaller_length",
    "WhisperModelTest::test_onnx_export_generate_dynamic_static_cache",
)
TARGETS = tuple(t for t in os.environ.get("CAPDBG_TARGETS", ",".join(DEFAULT_TARGETS)).split(",") if t)
REPEAT = int(os.environ.get("CAPDBG_REPEAT", "5"))
RESET_BETWEEN_REPEATS = os.environ.get("CAPDBG_RESET", "1") != "0"
# Tensor value stats (`.item()`) and `cudaMemGetInfo` force device syncs. With them on, whisper passed 10/10 in CI (bug
# hidden?); without the plugin it failed again. Off by default so the plugin never syncs the device by itself.
STATS = os.environ.get("CAPDBG_STATS", "0") == "1"
# "patched": MINIMAL pure-Python patches + FULL tracing in odd repeats. "repeat_only": no patch at all (bisect: even
# the MINIMAL patches seemed to hide the whisper failure, 4/4 plugin-on passes vs 6/8 plugin-off failures).
PLUGIN_MODE = os.environ.get("CAPDBG_MODE", "repeat_only")

_CAPTURE_STATUS = {0: "NONE", 1: "ACTIVE", 2: "INVALIDATED"}


class _State:
    def __init__(self):
        self.t0 = time.time()
        self.fh = None
        self.test = None  # nodeid of the running test
        self.is_target = False
        self.full = False  # per-op tracing enabled for the running test
        self.repeat_index = 0
        self.capture_depth = 0
        self.capture_count = 0
        self.capture_events = []  # events of the current capture (full mode)
        self.last_status = None
        self.culprit_reported = False
        self.trace = collections.deque(maxlen=400)  # ring buffer of recent events (all modes)
        self.ort_counter = 0
        self.lock = threading.RLock()


S = _State()


# ───────────────────────────── logging helpers ─────────────────────────────


def _write_file(line):
    with S.lock:
        if S.fh is not None:
            try:
                S.fh.write(line + "\n")
                S.fh.flush()
            except Exception:  # noqa: S110  (debug logging must never break a test)
                pass


def log(msg, stack=False, live=True):
    """Log one event: ring buffer + file always, live logger (WARNING) unless `live=False`."""
    line = (
        f"[CAPDBG +{time.time() - S.t0:9.3f}s] [{threading.current_thread().name}] "
        f"[depth={S.capture_depth} status={_status_str()}] {msg}"
    )
    if stack:
        line += "\n" + "".join(traceback.format_stack(limit=30)[:-1])
    S.trace.append(line)
    _write_file(line)
    if live:
        logger.warning(line)


def _thread_names():
    return [f"{t.name}{'(d)' if t.daemon else ''}" for t in threading.enumerate()]


def dump_threads(title):
    frames = sys._current_frames()
    parts = [f"===== {title}: {threading.active_count()} threads ====="]
    for t in threading.enumerate():
        parts.append(f"--- thread {t.name} ident={t.ident} daemon={t.daemon}")
        frame = frames.get(t.ident)
        if frame is not None:
            parts.append("".join(traceback.format_stack(frame, limit=25)))
    log("\n".join(parts))


def dump_trace(title, n=120):
    events = list(S.trace)[-n:]
    log(f"===== {title}: last {len(events)} events =====\n" + "\n".join(events))


# ───────────────────────────── CUDA runtime via ctypes ─────────────────────────────


class _Cudart:
    """Raw CUDA runtime queries that are legal during stream capture (no sync, no allocation)."""

    def __init__(self):
        self.lib = None
        self.path = None

    def load(self):
        if self.lib is not None:
            return self.lib
        try:
            with open("/proc/self/maps", encoding="utf-8") as f:
                paths = sorted({line.split()[-1] for line in f if "libcudart.so" in line})
            for path in paths:
                lib = ctypes.CDLL(path)
                lib.cudaStreamIsCapturing.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
                lib.cudaStreamIsCapturing.restype = ctypes.c_int
                lib.cudaPeekAtLastError.restype = ctypes.c_int
                lib.cudaGetErrorName.restype = ctypes.c_char_p
                lib.cudaGetErrorName.argtypes = [ctypes.c_int]
                lib.cudaThreadExchangeStreamCaptureMode.argtypes = [ctypes.POINTER(ctypes.c_int)]
                lib.cudaThreadExchangeStreamCaptureMode.restype = ctypes.c_int
                self.lib, self.path = lib, path
                break
        except Exception as e:  # never break a test run
            self.path = f"load failed: {e!r}"
        return self.lib

    def stream_status(self, stream_ptr=None):
        lib = self.load()
        if lib is None:
            return None
        try:
            import torch

            if stream_ptr is None:
                if not torch.cuda.is_initialized():
                    return None
                stream_ptr = torch.cuda.current_stream().cuda_stream
            status = ctypes.c_int(-1)
            err = lib.cudaStreamIsCapturing(ctypes.c_void_p(stream_ptr), ctypes.byref(status))
            if err != 0:
                return f"query_err={self.err_name(err)}"
            return status.value
        except Exception as e:
            return f"query_exc={e!r}"

    def peek_error(self):
        lib = self.load()
        if lib is None:
            return None
        err = lib.cudaPeekAtLastError()
        return "cudaSuccess" if err == 0 else self.err_name(err)

    def thread_capture_mode(self):
        """Read this thread's capture mode (exchange twice so it is left unchanged)."""
        lib = self.load()
        if lib is None:
            return None
        mode = ctypes.c_int(0)  # cudaStreamCaptureModeGlobal
        lib.cudaThreadExchangeStreamCaptureMode(ctypes.byref(mode))
        previous = mode.value
        lib.cudaThreadExchangeStreamCaptureMode(ctypes.byref(mode))
        return {0: "global", 1: "thread_local", 2: "relaxed"}.get(previous, previous)

    def err_name(self, err):
        try:
            return f"{self.lib.cudaGetErrorName(err).decode()}({err})"
        except Exception:
            return str(err)


CUDART = _Cudart()


def _status_str(stream_ptr=None):
    # MINIMAL mode (everything except FULL repeats): no CUDA API call at all from the plugin.
    if not S.full or (S.capture_depth == 0 and stream_ptr is None):
        return "-"
    status = CUDART.stream_status(stream_ptr)
    return _CAPTURE_STATUS.get(status, status)


def _check_transition(where):
    """Call after each traced step during capture: report the first step after which the capture is INVALIDATED."""
    status = CUDART.stream_status()
    if status != S.last_status:
        log(
            f"!!! capture status changed {_CAPTURE_STATUS.get(S.last_status, S.last_status)} -> "
            f"{_CAPTURE_STATUS.get(status, status)} right after: {where}",
            stack=status == 2,
        )
        if status == 2 and not S.culprit_reported:
            S.culprit_reported = True
            log(
                f"!!! CULPRIT CANDIDATE (first step after which capture is INVALIDATED): {where} | "
                f"peek_error={CUDART.peek_error()} thread_mode={CUDART.thread_capture_mode()}"
            )
            dump_threads("threads when capture became INVALIDATED")
        S.last_status = status
    return status


# ───────────────────────────── tensor descriptions ─────────────────────────────


def _describe_tensor(t, stats):
    import torch

    s = f"Tensor{tuple(t.shape)} {t.dtype} {t.device} stride={t.stride()}"
    try:
        if t.device.type == "cuda" and t.numel():
            s += f" ptr%16={t.data_ptr() % 16} ptr%128={t.data_ptr() % 128} contig={t.is_contiguous()}"
    except Exception:  # noqa: S110
        pass
    if stats and STATS and t.numel() and S.capture_depth == 0:
        try:
            d = t.detach()
            if d.is_floating_point():
                f = d.float()
                s += (
                    f" min={f.min().item():.4g} max={f.max().item():.4g} mean={f.mean().item():.4g}"
                    f" nan={bool(torch.isnan(f).any())} inf={bool(torch.isinf(f).any())}"
                )
            elif d.dtype != torch.bool:
                s += f" min={d.min().item()} max={d.max().item()}"
                if d.numel() <= 16:
                    s += f" values={d.flatten().tolist()}"
        except Exception as e:
            s += f" (stats failed: {e!r})"
    return s


def describe(obj, prefix="", depth=0, stats=True, out=None, limit=60):
    import torch

    out = [] if out is None else out
    if depth > 5 or len(out) >= limit:
        return out
    if isinstance(obj, torch.Tensor):
        out.append(f"{prefix}: {_describe_tensor(obj, stats)}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            describe(v, f"{prefix}.{k}", depth + 1, stats, out, limit)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            describe(v, f"{prefix}[{i}]", depth + 1, stats, out, limit)
    elif hasattr(obj, "self_attention_cache"):  # EncoderDecoderCache
        out.append(f"{prefix}: {type(obj).__name__}")
        describe(obj.self_attention_cache, f"{prefix}.self", depth + 1, stats, out, limit)
        describe(obj.cross_attention_cache, f"{prefix}.cross", depth + 1, stats, out, limit)
    elif isinstance(getattr(obj, "layers", None), list):  # Cache
        out.append(f"{prefix}: {type(obj).__name__} n_layers={len(obj.layers)}")
        for i, layer in enumerate(obj.layers[:2]):
            out.append(f"{prefix}.layers[{i}]: {type(layer).__name__}")
            describe(getattr(layer, "keys", None), f"{prefix}.layers[{i}].keys", depth + 1, stats, out, limit)
    elif hasattr(obj, "to_tuple"):  # ModelOutput
        describe(dict(obj), f"{prefix}<{type(obj).__name__}>", depth + 1, stats, out, limit)
    elif obj is not None:
        out.append(f"{prefix}: {type(obj).__name__} {obj!r:.80}")
    return out


def log_obj(title, obj, stats=True):
    log(f"{title}:\n    " + "\n    ".join(describe(obj, title, stats=stats)))


def cuda_mem():
    try:
        import torch

        if not torch.cuda.is_initialized():
            return "cuda=uninitialized"
        free, total = torch.cuda.mem_get_info() if STATS and S.capture_depth == 0 else (-1, -1)
        return (
            f"alloc={torch.cuda.memory_allocated() / 2**20:.1f}MiB "
            f"reserved={torch.cuda.memory_reserved() / 2**20:.1f}MiB free={free / 2**20:.0f}/{total / 2**20:.0f}MiB"
        )
    except Exception as e:
        return f"mem=({e!r})"


# ───────────────────────────── per-op tracing (full mode) ─────────────────────────────


def _make_trace_mode():
    from torch.utils._python_dispatch import TorchDispatchMode

    class CaptureTraceMode(TorchDispatchMode):
        """Log every aten op with the capture status after it. Never touches device data."""

        def __init__(self, phase):
            super().__init__()
            self.phase = phase
            self.n = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            self.n += 1
            name = str(func)
            detail = ""
            if "scaled_dot_product" in name or "attention" in name:
                detail = " | " + " ; ".join(describe(list(args), "args", stats=False, limit=12))
            try:
                out = func(*args, **kwargs)
            except BaseException as e:
                log(f"[{self.phase}] op#{self.n} {name} RAISED {e!r}{detail}", stack=True)
                raise
            if S.capture_depth > 0:
                status = _check_transition(f"aten op #{self.n} {name}")
                log(
                    f"[{self.phase}] op#{self.n} {name} -> status={_CAPTURE_STATUS.get(status, status)}{detail}",
                    live=bool(detail) or status == 2,
                )
            else:
                log(f"[{self.phase}] op#{self.n} {name}{detail}", live=bool(detail))
            return out

    return CaptureTraceMode


_trace_mode_cls = None
CUDA_LOG_PATH = None


def _dump_cuda_driver_log(title, max_chars=20000):
    """Log the content of `$CUDA_LOG_FILE` (CUDA driver error log), if any."""
    path = CUDA_LOG_PATH
    if not path or not os.path.exists(path):
        log(f"===== {title}: no CUDA driver log at {path!r} =====")
        return
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    # Drop the noise (other threads calling cuCtxGetDevice without a context), keep the real errors.
    lines = [
        line
        for line in content.splitlines()
        if "No CUDA context is current" not in line and "CUDA_ERROR_INVALID_CONTEXT) from cuCtxGetDevice" not in line
    ]
    filtered = "\n".join(lines)
    log(
        f"===== {title} ({path}, {len(content)} chars; {len(lines)} lines after dropping INVALID_CONTEXT noise, "
        f"last {max_chars} chars shown) =====\n{filtered[-max_chars:]}"
    )


# ───────────────────────────── CUDA driver log callback: Python stack at the offending call ─────────────────────────────
#
# Run 43288869427 (CUDA_LOG_FILE) showed that, during the capture and on the capturing thread, something calls
# cuMemFree / cuMemFreeHost / cuCtxSynchronize ("API call conflicts with a stream capture sequence initiated from the
# calling thread"), i.e. CUDA resources (very likely ONNX Runtime's, from earlier ONNX tests) are destroyed mid-capture.
# This callback is invoked by the driver on the thread making the bad call: it dumps that thread's Python stack.
# It does nothing at all (no Python call) unless the driver logs something.

STACKS_PATH = None
_stacks_file = None
_cb_state = {"dumped": 0}
_cu_logs_callback = None  # keep the ctypes callback alive


def _install_cuda_logs_callback(out_dir):
    global STACKS_PATH, _stacks_file, _cu_logs_callback
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        log(f"cuLogsRegisterCallback: cannot load libcuda: {e!r}")
        return
    if not hasattr(libcuda, "cuLogsRegisterCallback"):
        log("cuLogsRegisterCallback: not available in this driver")
        return
    STACKS_PATH = os.path.join(out_dir, f"cuda_capture_conflict_stacks_{os.getpid()}.log")
    _stacks_file = open(STACKS_PATH, "w", encoding="utf-8")  # noqa: SIM115  (kept open for faulthandler)

    callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t)

    def _callback(data, level, message, length):
        # Only act on capture conflicts / invalidation; at most 12 dumps (the burst is ~25 calls in ~2 ms).
        if _cb_state["dumped"] >= 12 or not message:
            return
        if b"stream capture" not in message and b"nvalidated" not in message:
            return
        _cb_state["dumped"] += 1
        try:
            _stacks_file.write(
                f"\n===== [{time.time() - S.t0:9.3f}s] [{threading.current_thread().name} "
                f"ident={threading.get_ident()}] test={S.test} CUDA: {message.decode(errors='replace')}\n"
            )
            _stacks_file.write(f"gc enabled={gc.isenabled()} counts={gc.get_count()}\n")
            _stacks_file.flush()
            faulthandler.dump_traceback(file=_stacks_file, all_threads=False)
            _stacks_file.flush()
        except Exception:  # noqa: S110  (never raise from a driver callback)
            pass

    _cu_logs_callback = callback_type(_callback)
    handle = ctypes.c_void_p()
    libcuda.cuLogsRegisterCallback.argtypes = [callback_type, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    libcuda.cuLogsRegisterCallback.restype = ctypes.c_int
    result = libcuda.cuLogsRegisterCallback(_cu_logs_callback, None, ctypes.byref(handle))
    log(f"cuLogsRegisterCallback -> {result} (0 = CUDA_SUCCESS); Python stacks of capture conflicts -> {STACKS_PATH}")


def _dump_capture_conflict_stacks(title):
    if not STACKS_PATH or not os.path.exists(STACKS_PATH):
        log(f"===== {title}: no stacks file =====")
        return
    with open(STACKS_PATH, encoding="utf-8", errors="replace") as f:
        content = f.read()
    log(f"===== {title} ({STACKS_PATH}, {len(content)} chars) =====\n{content[-40000:]}")


_capture_trace_modes = []


# ───────────────────────────── patches ─────────────────────────────


def _patch_cuda_graph():
    import torch

    G = torch.cuda.CUDAGraph
    orig_begin, orig_end = G.capture_begin, G.capture_end

    def capture_begin(self, *args, **kwargs):
        S.capture_count += 1
        S.culprit_reported = False
        if S.full:
            log(
                f"CUDAGraph.capture_begin #{S.capture_count} test={S.test} args={args} kwargs={kwargs} "
                f"stream={torch.cuda.current_stream()} thread_mode_before={CUDART.thread_capture_mode()} "
                f"peek_error={CUDART.peek_error()} {cuda_mem()} threads={_thread_names()}",
                stack=True,
            )
        else:  # MINIMAL: pure Python only
            log(
                f"CUDAGraph.capture_begin #{S.capture_count} test={S.test} kwargs={kwargs} threads={_thread_names()}",
                live=S.is_target,
            )
        try:
            out = orig_begin(self, *args, **kwargs)
        except BaseException as e:
            log(f"capture_begin RAISED {e!r}", stack=True)
            raise
        S.capture_depth += 1
        if S.full:
            S.last_status = CUDART.stream_status()
            log(
                f"capture started: status={_CAPTURE_STATUS.get(S.last_status, S.last_status)} "
                f"thread_mode={CUDART.thread_capture_mode()}"
            )
        if S.full and _trace_mode_cls is not None:
            mode = _trace_mode_cls(f"capture#{S.capture_count}")
            mode.__enter__()
            _capture_trace_modes.append(mode)
        return out

    def capture_end(self, *args, **kwargs):
        if _capture_trace_modes:
            mode = _capture_trace_modes.pop()
            with contextlib.suppress(Exception):
                mode.__exit__(None, None, None)
            log(f"capture trace: {mode.n} aten ops traced")
        if S.full:
            status = CUDART.stream_status()
            log(
                f"CUDAGraph.capture_end #{S.capture_count} (before) status={_CAPTURE_STATUS.get(status, status)} "
                f"peek_error={CUDART.peek_error()}"
            )
            if status == 2:
                dump_trace(f"capture #{S.capture_count} is INVALIDATED at capture_end")
                dump_threads("threads at capture_end with INVALIDATED capture")
        try:
            return orig_end(self, *args, **kwargs)
        except BaseException as e:
            # Capture failed: CUDA queries are now harmless, so collect everything.
            log(
                f"capture_end #{S.capture_count} RAISED {e!r} | status={CUDART.stream_status()} "
                f"peek_error={CUDART.peek_error()} threads={_thread_names()}",
                stack=True,
            )
            dump_trace(f"trace at capture_end #{S.capture_count} failure")
            dump_threads(f"threads at capture_end #{S.capture_count} failure")
            raise
        finally:
            S.capture_depth = max(0, S.capture_depth - 1)
            log(
                f"CUDAGraph.capture_end #{S.capture_count} (after)" + (f" {cuda_mem()}" if S.full else ""),
                live=S.is_target,
            )

    G.capture_begin, G.capture_end = capture_begin, capture_end


def _patch_cudagraph_trees():
    try:
        from torch._inductor import cudagraph_trees as ct
    except Exception as e:
        log(f"cannot import cudagraph_trees: {e!r}")
        return

    orig_record = ct.CUDAGraphNode._record

    def _record(self, model, inputs):
        log(
            f"cudagraph_trees._record START graph_id={getattr(self, 'id', '?')} fn={getattr(model, '__name__', model)} "
            f"n_inputs={len(inputs)} kernel_free={getattr(self.wrapped_function, 'kernel_free_cudagraph', '?')}"
            + (f" {cuda_mem()}" if S.full else ""),
            live=S.is_target,
        )
        try:
            out = orig_record(self, model, inputs)
        except BaseException as e:
            log(
                f"cudagraph_trees._record FAILED graph_id={getattr(self, 'id', '?')}: {e!r}\n"
                + "".join(traceback.format_exception(e))
            )
            dump_trace("trace at _record failure")
            dump_threads("threads at _record failure")
            raise
        log(f"cudagraph_trees._record END graph_id={getattr(self, 'id', '?')}")
        return out

    ct.CUDAGraphNode._record = _record

    orig_warmup_run = ct.CUDAWarmupNode.run

    def warmup_run(self, new_inputs):
        log(
            f"cudagraph_trees warmup START fn={getattr(self.wrapped_function.model, '__name__', '?')} "
            f"n_inputs={len(new_inputs)}" + (f" {cuda_mem()}" if S.full else ""),
            live=S.is_target,
        )
        mode = None
        if S.full and _trace_mode_cls is not None:
            mode = _trace_mode_cls("warmup")
            mode.__enter__()
        try:
            out = orig_warmup_run(self, new_inputs)
        except BaseException as e:
            log(f"cudagraph_trees warmup FAILED: {e!r}", stack=True)
            raise
        finally:
            if mode is not None:
                mode.__exit__(None, None, None)
        log("cudagraph_trees warmup END" + (f" ({mode.n} aten ops)" if mode is not None else ""))
        return out

    ct.CUDAWarmupNode.run = warmup_run

    for name in (
        "record_function",
        "run_eager",
        "execute_node",
        "try_end_curr_recording",
        "try_end_curr_warmup",
        "try_end_curr_execution",
        "apply_checkpoint_execution_state_in_allocator",
    ):
        if hasattr(ct.CUDAGraphTreeManager, name):
            orig = getattr(ct.CUDAGraphTreeManager, name)

            def wrapper(self, *a, _orig=orig, _name=name, **k):
                log(
                    f"CUDAGraphTreeManager.{_name} path_state={getattr(self, 'path_state', '?')} "
                    f"current_node={type(getattr(self, 'current_node', None)).__name__}",
                    live=S.is_target,
                )
                return _orig(self, *a, **k)

            setattr(ct.CUDAGraphTreeManager, name, wrapper)

    orig_reset = ct.reset_cudagraph_trees

    def reset_cudagraph_trees():
        log("cudagraph_trees.reset_cudagraph_trees()")
        return orig_reset()

    ct.reset_cudagraph_trees = reset_cudagraph_trees


def _patch_triton_launch():
    try:
        from torch._inductor.runtime import triton_heuristics as th
    except Exception as e:
        log(f"cannot import triton_heuristics: {e!r}")
        return

    orig_run = th.CachingAutotuner.run

    def run(self, *args, **kwargs):
        if S.capture_depth == 0 or not S.full:
            return orig_run(self, *args, **kwargs)
        kname = getattr(getattr(self, "fn", None), "__name__", "?")
        before = CUDART.stream_status()
        try:
            out = orig_run(self, *args, **kwargs)
        except BaseException as e:
            log(
                f"triton launch {kname} RAISED {e!r} (status before launch={_CAPTURE_STATUS.get(before, before)})",
                stack=True,
            )
            raise
        status = _check_transition(f"triton kernel {kname}")
        log(f"triton launch {kname} -> status={_CAPTURE_STATUS.get(status, status)}", live=status == 2)
        return out

    th.CachingAutotuner.run = run


def _gc_callback(phase, info):
    if S.capture_depth == 0:
        return
    if phase == "start":
        log(f"!!! GC start DURING CAPTURE gen={info.get('generation')}", stack=True)
    elif S.full:
        log(f"GC stop during capture collected={info.get('collected')} -> status now={_status_str()}")
        _check_transition(f"GC (gen={info.get('generation')}, collected={info.get('collected')})")


def _patch_ort():
    try:
        from onnxruntime.capi import onnxruntime_inference_collection as oic
    except Exception as e:
        log(f"onnxruntime not available: {e!r}", live=False)
        return
    orig_init = oic.InferenceSession.__init__

    def __init__(self, *args, **kwargs):
        S.ort_counter += 1
        sid, created_in = S.ort_counter, S.test
        orig_init(self, *args, **kwargs)
        log(f"ORT InferenceSession #{sid} created providers={self.get_providers()}")

        def _finalized():
            during = S.capture_depth > 0
            log(
                f"ORT InferenceSession #{sid} (created in {created_in}) finalized"
                + (" !!! DURING CAPTURE !!!" if during else ""),
                stack=during,
                live=during or S.is_target,
            )
            if during and S.full:
                _check_transition(f"ORT session #{sid} finalizer")

        weakref.finalize(self, _finalized)

    oic.InferenceSession.__init__ = __init__


def _patch_transformers():
    try:
        from transformers.generation import utils as gu
    except Exception as e:
        log(f"cannot import transformers.generation.utils: {e!r}")
        return

    orig_generate = gu.GenerationMixin.generate

    def generate(self, *args, **kwargs):
        if not S.full:
            return orig_generate(self, *args, **kwargs)
        log(f"generate START {type(self).__name__} kwargs={sorted(kwargs)} {cuda_mem()}")
        log_obj("generate.inputs", {k: v for k, v in kwargs.items() if k != "generation_config"})
        try:
            out = orig_generate(self, *args, **kwargs)
        except BaseException as e:
            log(f"generate RAISED {e!r}")
            raise
        log_obj("generate.output", out)
        return out

    gu.GenerationMixin.generate = generate

    orig_crit = gu.GenerationMixin._valid_auto_compile_criteria

    def _valid_auto_compile_criteria(self, model_kwargs, generation_config):
        result = orig_crit(self, model_kwargs, generation_config)
        if S.full:
            log(f"_valid_auto_compile_criteria -> {result} compile_config={generation_config.compile_config}")
        return result

    gu.GenerationMixin._valid_auto_compile_criteria = _valid_auto_compile_criteria

    try:
        from transformers.exporters import utils as eu
    except Exception:
        return
    orig_decompose = eu.decompose_prefill_decode

    def decompose_prefill_decode(model, inputs, generation_config=None, multi_token_decode=False):
        log(
            f"decompose_prefill_decode model={type(model).__name__} multi_token_decode={multi_token_decode} "
            f"cache_implementation={getattr(generation_config, 'cache_implementation', None)}"
        )
        out = orig_decompose(model, inputs, generation_config=generation_config, multi_token_decode=multi_token_decode)
        for name, (_, comp_inputs) in out.items():
            log_obj(f"decompose.{name}", comp_inputs, stats=False)
        return out

    eu.decompose_prefill_decode = decompose_prefill_decode
    for mod in list(sys.modules.values()):
        if mod is not None and getattr(mod, "decompose_prefill_decode", None) is orig_decompose:
            mod.decompose_prefill_decode = decompose_prefill_decode


def _log_environment():
    try:
        import torch

        info = [
            f"torch={torch.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()}",
            f"cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}",
            f"libcudart={CUDART.path}",
            f"sdpa: flash={torch.backends.cuda.flash_sdp_enabled()} mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()} "
            f"math={torch.backends.cuda.math_sdp_enabled()} cudnn={torch.backends.cuda.cudnn_sdp_enabled()}",
            f"blas={torch.backends.cuda.preferred_blas_library()} tf32_matmul={torch.backends.cuda.matmul.allow_tf32}",
        ]
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.append(f"gpu={p.name} cc={p.major}.{p.minor} mem={p.total_memory / 2**30:.1f}GiB")
        try:
            from torch._inductor import config as ic

            info.append(
                f"inductor: cudagraph_trees={ic.triton.cudagraph_trees} cudagraphs={ic.triton.cudagraphs} "
                f"compile_threads={ic.compile_threads} fx_graph_cache={ic.fx_graph_cache}"
            )
        except Exception as e:
            info.append(f"inductor config: {e!r}")
        try:
            import triton

            info.append(f"triton={triton.__version__}")
        except Exception:  # noqa: S110
            pass
        try:
            import onnxruntime

            info.append(f"onnxruntime={onnxruntime.__version__} providers={onnxruntime.get_available_providers()}")
        except Exception:  # noqa: S110
            pass
        info.append(
            f"cpu_count={os.cpu_count()} affinity={len(os.sched_getaffinity(0))} pid={os.getpid()} argv={sys.argv}"
        )
        keys = (
            "CUDA",
            "TORCH",
            "PYTORCH",
            "TRITON",
            "NCCL",
            "OTEL",
            "PATCH",
            "RUN_",
            "OMP",
            "MKL",
            "TRANSFORMERS",
            "CI",
            "PYTEST",
            "LD_",
            "NVIDIA",
        )
        info.append(
            "env: "
            + " ".join(
                f"{k}={'***' if 'TOKEN' in k or 'HEADERS' in k else v}"
                for k, v in sorted(os.environ.items())
                if k.startswith(keys)
            )
        )
        log("environment:\n    " + "\n    ".join(info))
    except Exception as e:
        log(f"environment logging failed: {e!r}")


def _nvidia_smi():
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        return f"nvidia-smi failed: {e!r}"


# ───────────────────────────── the pytest plugin ─────────────────────────────


def _is_target(nodeid):
    base = nodeid.split("[capdbg_repeat_")[0]
    return any(base.endswith(t) for t in TARGETS)


class CaptureDebugPlugin:
    def __init__(self, config):
        global _trace_mode_cls
        out_dir = os.environ.get("_PATCHED_TESTING_METHODS_OUTPUT_DIR") or os.environ.get("CAPDBG_DIR") or "/tmp"
        os.makedirs(out_dir, exist_ok=True)
        S.fh = open(os.path.join(out_dir, f"capture_debug_{os.getpid()}.log"), "w", encoding="utf-8")
        # CUDA error log management (CUDA >= 12.9): the driver writes the detailed reason of every CUDA error (e.g. which
        # call invalidated a stream capture) to this file. Zero Python / timing cost; read only after a failure. Must be
        # set before the CUDA driver is initialized (torch initializes it lazily, at the first CUDA call of a test).
        global CUDA_LOG_PATH
        CUDA_LOG_PATH = os.path.join(out_dir, f"cuda_driver_errors_{os.getpid()}.log")
        os.environ.setdefault("CUDA_LOG_FILE", CUDA_LOG_PATH)
        CUDA_LOG_PATH = os.environ["CUDA_LOG_FILE"]
        if sys.platform == "linux":
            _install_cuda_logs_callback(out_dir)
        try:
            import torch

            cuda_initialized = torch.cuda.is_initialized()
        except Exception as e:
            cuda_initialized = f"? ({e!r})"
        log(
            f"ci_capture_debug plugin active: mode={PLUGIN_MODE} targets={TARGETS} repeat={REPEAT} "
            f"reset={RESET_BETWEEN_REPEATS} log_file={S.fh.name} CUDA_LOG_FILE={CUDA_LOG_PATH} "
            f"(cuda already initialized: {cuda_initialized})"
        )
        self.heavy_patched = False
        if PLUGIN_MODE == "imports_only":
            # Bisect step: "repeat_only" reproduced the failure (deepseek + whisper originals FAILED), all patched modes
            # hid it. The patched modes import these modules (and so triton) at configure time: only do that here.
            try:
                import torch._inductor.cudagraph_trees  # noqa: F401
                import torch._inductor.runtime.triton_heuristics  # noqa: F401

                log(
                    f"imports_only: imported inductor cudagraph_trees + triton_heuristics; triton in sys.modules="
                    f"{'triton' in sys.modules}"
                )
            except Exception as e:
                log(f"imports_only: import failed: {e!r}")
            return
        if PLUGIN_MODE == "repeat_only":
            # Bisect step: no patch, no GC callback, no faulthandler, no ctypes — only duplicate the targets and dump on
            # failure. Even the MINIMAL (pure-Python) patches seemed to hide the whisper failure.
            return
        if PLUGIN_MODE == "triton_wrapper_only":
            # Bisect step: imports_only did NOT hide deepseek (FAILED). Only the pass-through Python wrapper around
            # `CachingAutotuner.run` (no CUDA call: S.full is never set in this mode).
            _patch_triton_launch()
            log("triton_wrapper_only: CachingAutotuner.run wrapped (pass-through)")
            return
        faulthandler.enable(file=sys.stderr, all_threads=True)
        try:
            import torch

            # No `torch.cuda.init()` here: keep CUDA initialization where the tests normally trigger it.
            if torch.cuda.is_available():
                CUDART.load()
                _trace_mode_cls = _make_trace_mode()
                _patch_cuda_graph()
                _patch_cudagraph_trees()
                _patch_triton_launch()
                gc.callbacks.append(_gc_callback)
        except Exception as e:
            log(f"torch patches failed: {e!r}", stack=True)
        # ORT / transformers patches import onnxruntime and transformers.exporters. Importing them at configure time
        # changes library load order vs a plain run, which may hide the bug: apply them lazily at the first FULL repeat.

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, session, config, items):
        if REPEAT <= 1:
            return
        new_items = []
        for item in items:
            new_items.append(item)
            if _is_target(item.nodeid):
                for k in range(1, REPEAT):
                    cpy = copy.copy(item)
                    cpy._initrequest()
                    cpy._nodeid = f"{item.nodeid}[capdbg_repeat_{k}]"
                    cpy._capdbg_repeat = k
                    new_items.append(cpy)
                log(f"collection: {item.nodeid} will run {REPEAT}x (original + {REPEAT - 1} repeats right after it)")
        items[:] = new_items

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        S.test = item.nodeid
        S.is_target = _is_target(item.nodeid)
        S.repeat_index = getattr(item, "_capdbg_repeat", 0)
        S.full = PLUGIN_MODE == "patched" and S.is_target and S.repeat_index % 2 == 1
        if S.is_target:
            S.trace.clear()
            if S.full and not self.heavy_patched:
                self.heavy_patched = True
                _patch_ort()
                _patch_transformers()
                _log_environment()
            log(
                f"########## TARGET START {item.nodeid} repeat={S.repeat_index} "
                f"mode={'FULL (per-op tracing)' if S.full else 'MINIMAL (pure Python, no CUDA calls)'}"
                + (f" {cuda_mem()}" if S.full else "")
                + f" threads={_thread_names()} gc_count={gc.get_count()} captures_so_far={S.capture_count}"
            )
            if S.repeat_index > 0 and RESET_BETWEEN_REPEATS:
                try:
                    import torch

                    torch._dynamo.reset()
                    log("torch._dynamo.reset() done before repeat")
                except Exception as e:
                    log(f"dynamo reset failed: {e!r}")
        else:
            log(f"test start {item.nodeid} threads={len(threading.enumerate())}", live=False)
        yield
        log(
            f"{'########## TARGET' if S.is_target else 'test'} END {item.nodeid}"
            + (f" {cuda_mem()}" if S.full else "")
            + f" captures_so_far={S.capture_count}",
            live=S.is_target,
        )
        S.test, S.is_target, S.full = None, False, False

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        if call.when != "call" or not report.failed or call.excinfo is None:
            return
        # Full dump for the targets, and for any other test whose error mentions capture (same family of failure).
        if S.is_target or "capture" in str(call.excinfo.value).lower():
            exc = call.excinfo
            log(
                f"!!!!!!!!!! TEST FAILED {item.nodeid}: {exc.value!r}\n"
                + "".join(traceback.format_exception(exc.type, exc.value, exc.tb))
            )
            dump_trace("trace at test failure", n=200)
            dump_threads("threads at test failure")
            # The failure already happened: CUDA queries / imports can no longer hide it.
            _dump_cuda_driver_log("CUDA driver error log at test failure")
            _dump_capture_conflict_stacks("Python stacks of the calls that conflicted with the capture")
            log(f"after failure: peek_error={CUDART.peek_error()} thread_mode={CUDART.thread_capture_mode()}")
            _log_environment()
            log(
                f"modules loaded at failure: onnxruntime={'onnxruntime' in sys.modules} "
                f"openvino={'openvino' in sys.modules} transformers.exporters={'transformers.exporters' in sys.modules}"
            )
            with contextlib.suppress(Exception), open("/proc/self/maps", encoding="utf-8") as f:
                libs = sorted(
                    {
                        line.split()[-1]
                        for line in f
                        if any(k in line for k in ("libcu", "libnv", "cudnn", "onnxruntime"))
                    }
                )
                log("CUDA-related shared libraries loaded (load order unknown, sorted):\n    " + "\n    ".join(libs))
            try:
                import torch

                log("cuda memory summary:\n" + torch.cuda.memory_summary(abbreviated=True))
            except Exception as e:
                log(f"memory_summary failed: {e!r}")
            log("nvidia-smi:\n" + _nvidia_smi())

    def pytest_unconfigure(self, config):
        log(f"session end: total captures={S.capture_count}")
        _dump_cuda_driver_log("CUDA driver error log at session end")
        _dump_capture_conflict_stacks("Python stacks of capture conflicts at session end")
        if _gc_callback in gc.callbacks:
            gc.callbacks.remove(_gc_callback)
        fh, S.fh = S.fh, None
        if fh is not None:
            fh.close()


def register_capture_debug_plugin(config):
    """Single entry point for conftest.py."""
    if os.environ.get("CAPDBG_DISABLE") == "1":
        return
    if config.pluginmanager.has_plugin("ci_capture_debug"):
        return
    config.pluginmanager.register(CaptureDebugPlugin(config), "ci_capture_debug")
