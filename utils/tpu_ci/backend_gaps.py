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
    "interpolate": (
        "torch.nn.functional.interpolate aborts the process instead of raising. Any model whose "
        "forward interpolates -- relative position embeddings, for one -- takes the whole pytest "
        "run down with it, so every other result for that model is lost too.",
        """
x = torch.randn(1, 4, 8, device=DEVICE)
torch.nn.functional.interpolate(x, size=16, mode="linear")
""",
    ),
    "compile": (
        "torch.compile fails in inductor with a bare NotImplementedError out of dtype_to_str, so "
        "every compiled-forward test fails.",
        """
def f(x):
    return x + 1

compiled = torch.compile(f, backend="inductor")
compiled(torch.randn(4, device=DEVICE))
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
    # The last non-empty line is the exception, or the runtime's complaint when it aborted.
    detail = next((line for line in reversed(result.stderr.splitlines()) if line.strip()), "")
    print(f"  {name}: still present (exit {result.returncode}) {detail[:120]}")
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
