# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Runs test functions on distributed worker processes that are kept alive across tests.

Spawning the workers (and importing torch in each of them) takes longer than most tests, so `run_distributed` starts
them once per (world_size, backend) and pytest(-xdist worker) process, and sends every test function to them.
"""

import os
import sys
import time
import traceback
from datetime import timedelta
from multiprocessing.connection import wait

from transformers import set_seed
from transformers.distributed.utils import get_distributed_backend
from transformers.testing_utils import cleanup, is_torch_available


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.distributed.distributed_c10d import GroupMember, _world


# a collective that waits longer than this raises, e.g. when another rank skipped it
_COLLECTIVE_TIMEOUT = timedelta(minutes=5)
# longer than `_COLLECTIVE_TIMEOUT`, so a stuck collective reports its own error first
_TEST_TIMEOUT = timedelta(minutes=10)


def _get_distributed_device_type():
    device_type = torch._C._get_accelerator().type
    return "cpu" if device_type == "mps" else device_type


def _get_distributed_backend():
    return get_distributed_backend(_get_distributed_device_type()) or "gloo"


def _get_rank_device(rank):
    device_type = _get_distributed_device_type()
    if device_type == "cpu":
        return torch.device("cpu")
    return torch.device(device_type, rank)


def _set_rank_device(rank, backend=None):
    # gloo ranks run on CPU, and there may be fewer devices than ranks
    if _get_distributed_device_type() != "cpu" and backend != "gloo":
        torch.accelerator.set_device_index(rank)


def _worker_main(rank, world_size, master_port, backend, pipe):
    """Entry point of a worker process: joins the process group as `rank`, then runs every test function received on
    `pipe` until pytest exits."""
    # pytest captures our output under the test running when it is written, and a worker may be killed (after
    # another one failed) before it flushes a block-buffered stdout
    sys.stdout.reconfigure(line_buffering=True)
    os.environ.update(WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK=str(rank))
    # some backends, e.g. tpu, require the rank to be set before initializing the process group
    _set_rank_device(rank, backend)
    store = dist.TCPStore("localhost", master_port, world_size=None, is_master=False)
    dist.init_process_group(backend, store=store, rank=rank, world_size=world_size, timeout=_COLLECTIVE_TIMEOUT)

    while True:
        try:
            func, args, kwargs = pipe.recv()
        except EOFError:  # pytest exited
            return
        # free the previous test's models, compiled graphs and device mesh groups, and reset the RNG
        cleanup(_get_distributed_device_type(), gc_collect=True)
        for group in list(_world.pg_map):
            if group is not GroupMember.WORLD:
                dist.destroy_process_group(group)
        set_seed(42)
        try:
            func(rank, *args, **kwargs)
            dist.barrier()
            error = None
        except BaseException:
            error = f"Rank {rank} failed:\n{traceback.format_exc()}"
        pipe.send(error)


class WorkerGroup:
    """The `world_size` worker processes that pytest sends test functions to, kept alive across tests."""

    # one running group per (world_size, backend)
    _running: dict[tuple[int, str], "WorkerGroup"] = {}

    def __init__(self, world_size: int, backend: str):
        ctx = mp.get_context("spawn")
        # held for as long as the workers live, so another xdist worker cannot claim its port
        self.store = dist.TCPStore("localhost", 0, world_size=None, is_master=True, wait_for_workers=False)
        self.pipes, self.processes = [], []
        for rank in range(world_size):
            pipe, worker_pipe = ctx.Pipe()
            # daemon: multiprocessing kills the workers when pytest exits
            process = ctx.Process(
                target=_worker_main, args=(rank, world_size, self.store.port, backend, worker_pipe), daemon=True
            )
            process.start()
            # the worker now holds the only copy of its end, so `pipe` reads EOF once the worker dies
            worker_pipe.close()
            self.pipes.append(pipe)
            self.processes.append(process)

    def run_test(self, func, *args, **kwargs):
        """Run `func(rank, *args, **kwargs)` on every worker, raising the first error, death or timeout."""
        for pipe in self.pipes:
            pipe.send((func, args, kwargs))
        pending = list(self.pipes)
        deadline = time.monotonic() + _TEST_TIMEOUT.total_seconds()
        while pending:
            ready = wait(pending, timeout=max(0, deadline - time.monotonic()))
            if not ready:
                raise TimeoutError(f"The distributed workers did not finish within {_TEST_TIMEOUT}")
            for pipe in ready:
                pending.remove(pipe)
                try:
                    error = pipe.recv()
                except EOFError:
                    raise RuntimeError("A distributed test worker died, see its output above") from None
                if error is not None:
                    raise RuntimeError(error)

    def kill(self):
        """Kill the workers, e.g. when some may be stuck in a collective."""
        for process in self.processes:
            process.kill()

    @classmethod
    def get_or_start(cls, world_size: int, backend: str) -> "WorkerGroup":
        if (world_size, backend) not in cls._running:
            cls._running[world_size, backend] = cls(world_size, backend)
        return cls._running[world_size, backend]


def run_distributed(world_size: int, backend: str | None = None):
    backend = backend or _get_distributed_backend()

    def decorator(func):
        def wrapper(*args, **kwargs):
            workers = WorkerGroup.get_or_start(world_size, backend)
            try:
                workers.run_test(func, *args, **kwargs)
            except BaseException:
                # the next `get_or_start` starts new workers
                del WorkerGroup._running[world_size, backend]
                workers.kill()
                raise

        return wrapper

    return decorator
