# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""
Standalone probes for the backend gaps the model tests hit on this device.

Each probe is the smallest thing that reproduces one gap, with no transformers involved, so it can
be handed to the backend team as-is. Each is also the retirement condition for whatever workaround
the gap forced: when a probe stops reporting the gap, the workaround it justifies can go.

    python3 utils/tpu_ci/backend_gaps.py                 # run them all
    python3 utils/tpu_ci/backend_gaps.py --only cholesky # print one, to paste into a bug report
    python3 utils/tpu_ci/backend_gaps.py --show cholesky

Exit code is 0 when every probe passes, i.e. when there is nothing left to work around.

Probes run in a subprocess on purpose: the backend answers some of these by aborting the process
rather than raising, which a parent cannot catch.

A gap only belongs here once it reproduces standalone. Execution is deferred, so the Python frame
on top when the process aborts is whatever forced materialisation and not necessarily the operator
at fault -- a traceback alone is not enough to name one.
"""

import argparse
import subprocess
import sys


DEVICE = "tpu"

# name -> (what is broken, body of the probe). Each body must raise or abort while the gap is
# present, and return normally once it is fixed.
PROBES = {
    "cholesky": (
        "torch.linalg.cholesky_ex is missing, so resizing token embeddings fails: the new rows are "
        "drawn from a MultivariateNormal fitted to the old ones, and its positive-definite check "
        "factorises the covariance.",
        """
covariance = torch.eye(4, device=DEVICE) * 2.0
torch.linalg.cholesky_ex(covariance)
""",
    ),
    "ctc_loss": (
        "torch.nn.functional.ctc_loss aborts the process instead of raising, so a CTC model takes "
        "the whole pytest run down with it and its other results are lost too.",
        """
logits = torch.randn(2, 8, 5, device=DEVICE)
# Flattened 1-D targets and an explicit blank, which is how the real callers reach it.
log_probs = torch.nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
targets = torch.randint(1, 5, (6,), dtype=torch.long, device=DEVICE)
input_lengths = torch.full((2,), 8, dtype=torch.long, device=DEVICE)
target_lengths = torch.full((2,), 3, dtype=torch.long, device=DEVICE)
torch.nn.functional.ctc_loss(
    log_probs, targets, input_lengths, target_lengths, blank=0, reduction="sum", zero_infinity=False
)
""",
    ),
    "compile": (
        "torch.compile does not work with the default inductor backend, so anything that compiles "
        "without asking the model which backend suits its device fails. The backend this device "
        "registers with TorchDynamo does work. The error moves around between inductor internals, "
        "so the probe reports whatever it hits rather than matching on one message.",
        """
def f(x):
    return x + 1

compiled = torch.compile(f, backend="inductor")
compiled(torch.randn(4, device=DEVICE))
""",
    ),
    "weight_tying": (
        "Moving a module to the device replaces its parameters rather than swapping their data, so "
        "parameters that were tied to one another come back as two separate tensors. "
        "`nn.Module._apply` only keeps them tied when `torch._has_compatible_shallow_copy_type` "
        "holds for the moved tensor, and it does not here.",
        """
import torch.nn as nn

embedding = nn.Embedding(5, 4)
head = nn.Linear(4, 5, bias=False)
head.weight = embedding.weight  # the tied lm_head every causal LM has
model = nn.Sequential(embedding, head).to(DEVICE)
if embedding.weight is not head.weight:
    raise RuntimeError("the move untied the two parameters")
""",
    ),
    "boolean_mask_indexing": (
        "Indexing trailing dimensions with a multi-dimensional boolean mask -- `x[:, :, mask]` -- "
        "tries to broadcast the mask instead of selecting with it, and raises a shape error. "
        "`masked_select` with the same mask works, and a mask covering every dimension works too.",
        """
x = torch.randn(2, 3, 7, 7, device=DEVICE)
mask = torch.zeros(7, 7, dtype=torch.bool, device=DEVICE)
mask[0, 0] = True
selected = x[:, :, mask]
if selected.shape != (2, 3, 1):
    raise RuntimeError(f"selected {tuple(selected.shape)}, expected (2, 3, 1)")
""",
    ),
    "interpolate_same_size": (
        "torch.nn.functional.interpolate in 'linear' mode aborts the process when the size asked "
        "for is the size the input already has. Resizing to any other size is fine, as are the "
        "other modes, so the models that hit it are the ones running at their native resolution.",
        """
x = torch.randn(1, 8, 27, device=DEVICE)
torch.nn.functional.interpolate(x, size=27, mode="linear").cpu()
""",
    ),
    "attention_precision": (
        "Scaled dot product attention computes in bfloat16 whatever "
        "`torch.set_float32_matmul_precision` asks for, while a plain matmul honours it. A float32 "
        "model's attention therefore disagrees with its eager equivalent at bfloat16 accuracy, and "
        "the tests that compare the two need bfloat16 tolerances in every dtype.",
        """
torch.set_float32_matmul_precision("highest")
torch.manual_seed(0)
query, key, value = (torch.randn(2, 4, 16, 32) for _ in range(3))
reference = torch.nn.functional.scaled_dot_product_attention(query.double(), key.double(), value.double())
attention = torch.nn.functional.scaled_dot_product_attention(
    *(tensor.to(DEVICE) for tensor in (query, key, value))
)
error = ((attention.cpu().double() - reference).abs().mean() / reference.abs().mean()).item()
# float32 rounding lands around 1e-7 here, bfloat16 rounding around 1e-3.
if error > 1e-5:
    raise RuntimeError(f"float32 attention carries {error:.1e} relative error")
""",
    ),
    "device_index": (
        "`Tensor.to(<int>)` ignores the index and lands on device 0. An integer-keyed device map -- "
        "which is what `accelerate` builds -- therefore places every submodule on the same chip "
        "instead of spreading them. Needs more than one chip to be visible to show up.",
        """
last = torch.tpu.device_count() - 1
moved = torch.zeros(2).to(last)
if moved.device.index != last:
    raise RuntimeError(f"to({last}) returned a tensor on {moved.device}")
""",
    ),
    "visible_devices": (
        "TPU_VISIBLE_DEVICES restricts the chips a process opens but not what "
        "`torch.tpu.device_count()` reports, so a run pinned to one chip still looks like a "
        "multi-chip run to every test that gates on the device count.",
        """
import os
import subprocess as sp

child = sp.run(
    [sys.executable, "-c", "import torch, torch_tpu; print(torch.tpu.device_count())"],
    capture_output=True,
    text=True,
    env={**os.environ, "TPU_VISIBLE_DEVICES": "0"},
)
count = int(child.stdout.strip().splitlines()[-1])
if count != 1:
    raise RuntimeError(f"one chip made visible, device_count() reports {count}")
""",
    ),
    "device_sharing": (
        "A second process cannot open a device this one holds, and the runtime aborts rather than "
        "raising. Tests that re-run themselves in a subprocess -- run_test_using_subprocess -- "
        "therefore always fail once the parent has touched the device.",
        """
import subprocess as sp

# Touch the device here first, so the child meets a device that is already held.
torch.zeros(2, device=DEVICE).sum().item()
child = sp.run(
    [sys.executable, "-c", "import torch, torch_tpu; torch.zeros(2, device='tpu').sum().item()"],
    capture_output=True,
)
if child.returncode != 0:
    raise RuntimeError(f"child could not use the device (exit {child.returncode})")
""",
    ),
}

PREAMBLE = """\
import sys
import torch
import torch_tpu  # noqa: F401

DEVICE = "{device}"
"""


def script_for(name: str) -> str:
    return PREAMBLE.format(device=DEVICE) + PROBES[name][1]


def run_probe(name: str) -> bool:
    """True when the gap is gone. Runs in a subprocess so an abort is survivable."""
    result = subprocess.run([sys.executable, "-c", script_for(name)], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  {name}: fixed")
        return True
    # The runtime keeps logging after it fails, so take its first complaint rather than the last line.
    interesting = ("Error", "error:", "Check failed", "not implemented", "NotImplemented", "Aborted")
    detail = next(
        (line.strip() for line in result.stderr.splitlines() if any(k in line for k in interesting)),
        result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "",
    )
    print(f"  {name}: still present (exit {result.returncode}) -- {detail[:150]}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=sorted(PROBES), help="run just this probe")
    parser.add_argument("--show", choices=sorted(PROBES), help="print a probe's script and exit")
    args = parser.parse_args()

    if args.show:
        print(script_for(args.show))
        return

    names = [args.only] if args.only else sorted(PROBES)
    print(f"Probing {len(names)} known backend gap(s) on {DEVICE!r}:")
    results = {}
    for name in names:
        print(f"- {PROBES[name][0]}")
        results[name] = run_probe(name)

    remaining = [name for name, fixed in results.items() if not fixed]
    if remaining:
        print(f"\n{len(remaining)} gap(s) still present: {', '.join(remaining)}")
        raise SystemExit(1)
    print("\nEvery probe passed: the workarounds these justify can be removed.")


if __name__ == "__main__":
    main()
