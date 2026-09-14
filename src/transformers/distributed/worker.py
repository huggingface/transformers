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

import importlib.util
import os
import pathlib
import sys
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
    module = fresh_import(path)
    entry = getattr(module, "main", None)
    if not callable(entry):
        raise AttributeError(f"{path} defines no main(); the job runner calls main() with no arguments")
    entry()


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
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

    workspace = str(pathlib.Path(os.environ.get("WORKSPACE", os.getcwd())).resolve())

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
        if rank == 0:
            print(f"[job {n - 1}] {script}", flush=True)
        ok = 1
        try:
            purge(workspace)
            run_script(script)
        except Exception:
            traceback.print_exc()
            sys.stderr.flush()
            ok = 0
        sys.stdout.flush()

        votes = torch.tensor([ok])
        dist.all_reduce(votes, group=ctrl)  # vote over gloo
        if rank == 0:
            result = "ok" if votes.item() == world else "fail"
            print(f"[job {n - 1}] finished: {result}", flush=True)
            store.set(f"done:{n - 1}", result)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
