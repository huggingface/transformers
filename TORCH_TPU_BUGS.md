# torch_tpu bugs to report

Found while running the HuggingFace `transformers` model test suite (27 models, fast and slow tests) on
TPU. None of these is filed yet. Each one was reproduced standalone, without transformers, and the
repro and output below are what was actually run and observed.

Observed on `torch_tpu 0.1.1.dev20260923101118`, `libtpu 0.0.49` (TPU v6e, 8 chips on one host),
`torch 2.11.0+cpu`, `torchvision 0.26.0+cpu`, `transformers 5.18.0.dev0`, Python 3.13.5.

For each entry the reporting agent should add a standalone `check_*.sh` in the usual style (exit 0 =
fixed, 1 = still present, 2 = run error) and the full issue text.

---

## Issue: Antialiased upsampling (`interpolate(..., antialias=True)`) is not implemented for TPU

### Context

`transformers`' fast image and video processors resize with torchvision, whose tensor `resize` defaults
to `antialias=True` (`torchvision.transforms.v2.functional.resize`). On TPU that reaches
`aten::_upsample_bicubic2d_aa` / `aten::_upsample_bilinear2d_aa`. In the test suite it fails
`test_can_compile_fast_video_processor` for InternVL and SmolVLM.

### Current behavior

```python
import torch
import torch_tpu
import torch.nn.functional as F

x = torch.rand(1, 3, 64, 64)
for mode, antialias in [("bicubic", False), ("bilinear", True), ("bicubic", True)]:
    F.interpolate(x.to("tpu"), size=(32, 32), mode=mode, antialias=antialias).cpu()
```

```
bicubic  antialias=False: ok, max difference vs CPU 2.38e-07
bilinear antialias=True:  RuntimeError: operator 'aten::_upsample_bilinear2d_aa.out' is not implemented for TPU. Please file a feature request
bicubic  antialias=True:  RuntimeError: operator 'aten::_upsample_bicubic2d_aa.out' is not implemented for TPU. Please file a feature request
```

The non-antialiased kernels exist and match CPU, so the gap is the two `_aa` variants.

### Why this matters

Antialiased resizing is the default for tensors in torchvision's v2 transforms, so any image or video
preprocessing that runs on the device fails, and the error names an ATen operator the user never called.

### Desired behavior

`_upsample_bilinear2d_aa` and `_upsample_bicubic2d_aa` (forward, and backward for training-time
augmentation) work on TPU and match CPU.

---

## Issue: `torch.compile(mode="reduce-overhead" | "max-autotune")` fails on TPU tensors

### Context

`torch.compile`'s `mode` is a standard argument that user code and libraries pass without knowing the
device. `transformers`' `CompileConfig` defaults to `mode="reduce-overhead"`, and the test
`test_torch_compile_for_training` calls `torch.compile(model, fullgraph=True, mode="reduce-overhead")`.
It fails on TPU (`LlamaModelTest::test_torch_compile_for_training`).

### Current behavior

```python
import torch
import torch_tpu

def f(x):
    return (x * 2).sin().sum()

torch.compile(f, mode="reduce-overhead")(torch.randn(8, device="tpu"))
```

```
torch._dynamo.exc.BackendCompilerFailed: backend='_default_backend_selector' raised:
TypeError: Unexpected keyword arguments: {'mode': 'reduce-overhead'}
```

| call, on a TPU tensor | result |
|---|---|
| `torch.compile(f)` | ok |
| `torch.compile(f, mode="default")` | ok |
| `torch.compile(f, options={"trace.enabled": False})` | ok |
| `torch.compile(f, mode="reduce-overhead")` | `TypeError: Unexpected keyword arguments: {'mode': 'reduce-overhead'}` |
| `torch.compile(f, mode="max-autotune")` | `TypeError: Unexpected keyword arguments: {'mode': 'max-autotune'}` |
| `torch.compile(f, backend="tpu", mode="reduce-overhead")` | same `TypeError`, from `backend='tpu'` |

The same calls on CPU tensors (with `torch_tpu` imported) all work.

torch_tpu replaces `torch.compile` with a wrapper (`torch_tpu/_loader.py`, `_default_tpu_compile`) that
defaults the backend to `_default_backend_selector`, which picks the TPU backend when an input is on TPU
and forwards all keyword arguments to it. The TPU backend itself rejects `mode`, as the explicit
`backend="tpu"` row shows.

### Why this matters

Code that compiles with a mode, which is common, fails on TPU as soon as its inputs are on the device.

### Desired behavior

The TPU backend accepts `mode`. It could honour the modes that have a TPU meaning, or ignore the ones
that don't (`reduce-overhead` is about CUDA graphs), ideally with a warning. Rejecting a standard
argument is the one behaviour that breaks callers.

---

## Issue: `torch.compile(backend="inductor")` fails on TPU: `register_custom_kernel()` gets bytes, expects str

### Context

`inductor` is `torch.compile`'s default backend, and code that names it explicitly (for example
`transformers`' `CompileConfig(backend="inductor")`) reaches it even with torch_tpu's default-backend
wrapper in place. The transformers branch avoids it by compiling with the backend the model picks for
the device, so this shows up in no test failure. The failure is still a plain bug.

### Current behavior

```python
import torch
import torch_tpu

torch.compile(lambda x: x + 1, backend="inductor")(torch.randn(4, device="tpu")).cpu()
```

```
TypeError: register_custom_kernel(): incompatible function arguments. The following argument types are supported:
    1. (name: str, kernel_key: str, *, serialized_mlir_module: str) -> None
Invoked with: 'pallas_fused_add_626395d0', 'pallas_fused_add_626395d0_4', b'ML\xefR\rStableHLO_v1.18.0\x00...'
```

The call site is `torch_tpu/_internal/pallas/pallas.py:713`:

```python
tpu_torch_pallas.register_custom_kernel(
    self.name,
    mlir_fingerprint,
    serialized_mlir_module=lowered.mlir_module_serialized,
)
```

`lowered.mlir_module_serialized` is `bytes`, while the binding declares `serialized_mlir_module: str`.
The kernel is generated and lowered successfully; only the hand-off to the runtime fails.

### Why this matters

Every Inductor compile of a TPU graph fails at the first generated Pallas kernel, and the error points
into a binding signature rather than at anything the user did.

### Desired behavior

The binding accepts the serialized module as `bytes` (it is a binary MLIR bytecode blob, so `bytes` is
the right type), or the caller converts it. Inductor-compiled functions then run on TPU.

---

## Issue: Single-host multi-process `tpu_dist` cannot be started with standard PyTorch launchers

### Context

`transformers`' tensor-parallel and FSDP tests start one worker per rank with
`torch.multiprocessing.spawn`, set `MASTER_ADDR`/`MASTER_PORT`, bind the rank's device with
`set_device(rank)`, and call `dist.init_process_group(backend, rank=rank, world_size=2)`. That is the
standard recipe on CUDA/XPU/HPU. On TPU every one of those tests fails in the ranks with:

```
RuntimeError: missing required environment variables for distributed training:
TORCH_TPU_SLICEBUILDER_ADDRESSES, TORCH_TPU_TOPOLOGY; please run the program via torchrun or similar
tools so that the environment is set up properly
```

These tests also cannot open a device the pytest parent holds (filed separately, #4166). The failures
below are independent of that: the parent in these repros never touches the device.

### Current behavior

Four separate problems, each reproduced on its own:

**1. The required environment has no public API, and torchrun does not provide it.** The variables are
set only by `torch_tpu._internal.distributed.launchers.environment.set_tpu_launch_env()`, a private
module, which has to run in the parent before spawning (its docstring says "Nothing upstream populates
these, so TorchTPU has to own the step"). Under plain `torchrun --nproc-per-node 2 script.py`, the
ranks have no `TORCH_TPU_*` variables, and each rank logs

```
PjrtBackend::GetClient failed to initialize: FAILED_PRECONDITION: missing required environment variables for distributed training: TORCH_TPU_SLICEBUILDER_ADDRESSES, TORCH_TPU_TOPOLOGY; ...
```

and then aborts (next point). The error's advice to use torchrun therefore does not work.

**2. A rank that cannot initialise aborts instead of raising.** With `mp.spawn` and no launch
environment, each rank dies on its first device use with

```
F0000 device_rt.cc:295] Check failed: client != nullptr PjRtClient is null after initialization.
RAW: Raising signal 6 with default behavior
```

so the parent only sees a rank killed by SIGABRT. In the transformers tests the ranks got the
`RuntimeError` above instead, so the same missing setup sometimes raises and sometimes aborts,
depending on the path into the runtime.

**3. Only 1, 4 or 8 ranks are accepted on v6e.** `set_tpu_launch_env(nproc_per_node=2)` raises
`RuntimeError: No TPU topology found for count: 2`. The v6e map in `_internal/utils/hardware.py` has
only `{1: "1,1,1", 4: "2,2,1", 8: "2,4,1"}`, so a 2-process job, the most common test size, cannot
run on a v6e host.

**4. `set_device(local_rank)` raises.** Each rank sees its own chip as `tpu:0`, so the usual
per-rank binding fails for every rank but 0:

```
ValueError: Cannot set TPU device to index 1, current process is bound to device index 0.
Changing the active TPU device within a process is not supported.
```

### What works

```python
import os
import torch, torch_tpu
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, world_size):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29541",
                      RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size))
    dist.init_process_group("tpu_dist", rank=rank, world_size=world_size)   # no set_device
    t = torch.ones(2, device="tpu") * (rank + 1)
    dist.all_reduce(t)
    print(rank, t.cpu().tolist(), t.device)
    dist.destroy_process_group()

if __name__ == "__main__":
    from torch_tpu._internal.distributed.launchers import environment
    environment.set_tpu_launch_env(nproc_per_node=4)                        # private API
    mp.spawn(worker, args=(4,), nprocs=4)
```

prints `[10.0, 10.0] on tpu:0` on all four ranks. It needs the private API, `RANK`/`LOCAL_RANK`/
`WORLD_SIZE` in the environment on top of the `init_process_group` arguments, a supported rank count,
and no `set_device`.

### Why this matters

Frameworks and test suites start multi-process jobs with `torch.multiprocessing.spawn` or `torchrun`
and bind devices with `set_device(local_rank)`. With the current behaviour none of that works on TPU:
the setup step is private, the error message points to a launcher that does not help, failures kill the
process, and the smallest multi-rank size is rejected on v6e.

### Desired behavior

1. A public way to prepare the launch environment (for example `torch_tpu.distributed.prepare_launch_env()`),
   or a `torchrun`-compatible entry point that does it. Alternatively, derive the variables inside
   `init_process_group` from `MASTER_ADDR`/`MASTER_PORT`/`WORLD_SIZE` when they are missing.
2. The missing-environment case raises a Python exception in every path, never `Check failed`.
3. 2-rank sub-slices on v6e, if the hardware allows it; otherwise a clear error that lists the
   supported counts.
4. `torch.tpu.set_device(local_rank)` accepts the index of the chip the process is bound to (or is a
   no-op in one-chip-per-process mode) rather than raising, so device-agnostic code keeps working.

### Out of scope

The device collision when the parent already holds the chips (#4166).
