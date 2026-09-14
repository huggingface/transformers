# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Persistent worker behind the ``td`` launcher.

One of these runs per rank, started once by :mod:`transformers.distributed.runner`
and kept alive across jobs. It parks on a `TCPStore` key until the launcher submits
a script, imports that script fresh, calls its ``main()``, votes on the outcome, and
goes back to waiting. Keeping the process group up between jobs is the whole point:
NCCL init and CUDA context creation happen once, not on every iteration.

It is executed by ``torch.distributed.run`` as a plain script, not imported, so it
deliberately avoids package-relative imports.
"""

import gc
import importlib.util
import os
import pathlib
import sys
import time
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist


def purge(workspace: str):
    """Drop the previous job's modules so the next import re-executes them.

    Without this, editing a job script (or anything it imports from the workspace)
    between submissions would have no effect: the stale module object stays cached
    in ``sys.modules`` for the life of the daemon.
    """
    torch._dynamo.reset()
    for name, module in list(sys.modules.items()):
        # __main__ is this file; dropping it breaks the loop running right now.
        if name in ("__main__", "__job__"):
            continue
        file = getattr(module, "__file__", None)
        if name == "transformers" or name.startswith("transformers."):
            del sys.modules[name]
        elif file and str(pathlib.Path(file).resolve()).startswith(workspace):
            del sys.modules[name]


def fresh_import(path: str):
    spec = importlib.util.spec_from_file_location("__job__", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(path: str):
    """Execute a job script.

    Importing it already runs its module body, so a plain top-level script is a
    valid job with nothing to declare — the body *is* the work. ``main()`` is
    called afterwards when the script defines one, which is what you want as soon
    as the script grows imports or helpers that should not re-run per call.
    """
    module = fresh_import(path)
    entry = getattr(module, "main", None)
    if callable(entry):
        entry()


def reclaim() -> tuple[float, float]:
    """Give the previous job's GPU memory back between runs.

    Dropping the job module makes its tensors garbage, but two things stop that
    from showing up as free memory on its own: reference cycles wait for a gc
    pass, and the caching allocator holds freed blocks rather than returning them
    to the driver. A long-lived pool therefore looks permanently full and the next
    job fragments against a cache it cannot use. Neither is automatic, so do both.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gib = 1024**3
    return torch.cuda.memory_allocated() / gib, torch.cuda.memory_reserved() / gib


def quiet_unless_rank0(rank: int, log_dir: str):
    """Send every rank but 0 to its own file, leaving the console to rank 0.

    Four ranks interleaving into one stream is unreadable, and torchrun's own
    filtering (``--local-ranks-filter``) only works through ``--tee``, which stamps
    every line with a ``[default0]:`` prefix that cannot be turned off. Doing it
    here keeps the console clean and still keeps each rank's output on disk.

    dup2 on the file descriptors rather than rebinding ``sys.stdout``, so that
    NCCL and the rest of the C++ side follow the redirect too.
    """
    if rank == 0:
        return
    os.makedirs(log_dir, exist_ok=True)
    sink = open(os.path.join(log_dir, f"rank{rank}.log"), "w", buffering=1)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(sink.fileno(), sys.stdout.fileno())
    os.dup2(sink.fileno(), sys.stderr.fileno())


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    workspace = str(pathlib.Path(os.environ.get("WORKSPACE", os.getcwd())).resolve())
    log_dir = os.environ.get("TD_LOG_DIR", os.path.join(workspace, ".td"))
    quiet_unless_rank0(rank, log_dir)
    torch.cuda.set_device(local_rank)

    # Exactly one control store: rank 0 owns the server, everyone else connects to
    # it. Building it twice makes rank 0 bind CTRL_PORT a second time (EADDRINUSE),
    # which kills the server socket and takes every other rank down with it.
    store = dist.TCPStore(
        os.environ["MASTER_ADDR"],
        int(os.environ.get("CTRL_PORT", 29600)),
        is_master=(rank == 0),
        wait_for_workers=False,
        timeout=timedelta(days=7),
    )

    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    ctrl = dist.new_group(backend="gloo")  # control plane, never NCCL

    # Announce readiness only once the loop below is actually reachable, so the
    # launcher cannot connect to a server that is a millisecond away from crashing.
    if rank == 0:
        print(f"[ready] world={world}", flush=True)
        store.set("ready", str(world))

    n = 0
    while True:
        script = store.get(f"job:{n}").decode()  # blocks until a client submits
        n += 1
        if script == "__exit__":
            break
        if script == "__skip__":
            # A client reserved this slot and then died before naming a script.
            # Without the placeholder it writes on the way out, every rank would
            # sit on this key forever and the pool would accept nothing further.
            continue
        if rank == 0:
            print(f"[job {n - 1}] {script}", flush=True)
        ok = 1
        started = time.monotonic()
        try:
            purge(workspace)
            run_script(script)
        except Exception:
            traceback.print_exc()
            sys.stderr.flush()
            ok = 0
        finally:
            elapsed = time.monotonic() - started
            live, reserved = reclaim()
        sys.stdout.flush()

        # One slot per rank rather than a sum, so rank 0 can name who failed —
        # their traceback went to their own file and is not on the console.
        votes = torch.zeros(world, dtype=torch.int32)
        votes[rank] = ok
        dist.all_reduce(votes, group=ctrl)  # vote over gloo
        if rank == 0:
            failed = [i for i, v in enumerate(votes.tolist()) if not v]
            result = "ok" if not failed else "fail"
            print(
                f"[job {n - 1}] finished: {result} in {elapsed:.1f}s "
                f"(gpu0 {live:.1f} GiB live / {reserved:.1f} GiB reserved)",
                flush=True,
            )
            if [i for i in failed if i != 0]:
                where = ", ".join(f".td/rank{i}.log" for i in failed if i != 0)
                print(f"[job {n - 1}] ranks {failed} failed; see {where}", flush=True)
            # Duration rides along with the verdict: rank 0 is the only rank that
            # timed the whole job, and the client cannot measure it itself without
            # counting its own poll interval and startup.
            store.set(f"done:{n - 1}", f"{result} {elapsed:.3f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
