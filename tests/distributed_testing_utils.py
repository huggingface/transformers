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
"""Runs test functions on distributed ranks that are kept alive across tests.

Spawning the ranks (and importing torch in each of them) takes longer than most tests, so `run_distributed` starts
them once per (world_size, backend) and pytest(-xdist worker) process, and sends every test function to them.
"""

import atexit
import multiprocessing
import os
import queue
import sys
import time
import traceback
from datetime import timedelta

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


def _persistent_rank(rank, world_size, store_host, store_port, backend, tasks, results):
    """A rank kept alive across tests: it initializes the process group once, then runs every test function it receives."""
    os.environ.update(
        WORLD_SIZE=str(world_size),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        MASTER_ADDR=store_host,
        MASTER_PORT=str(store_port),
    )
    # some backends, e.g. tpu, require the rank to be set before initializing the process group
    _set_rank_device(rank, backend)
    store = dist.TCPStore(store_host, store_port, world_size=None, is_master=False)
    dist.init_process_group(
        backend=backend, store=store, rank=rank, world_size=world_size, timeout=_COLLECTIVE_TIMEOUT
    )
    results.put(None)

    while True:
        try:
            func, args, kwargs = tasks.get(timeout=5)
        except queue.Empty:
            # exit if pytest died without killing us
            if not multiprocessing.parent_process().is_alive():
                return
            continue
        # so a test's random numbers do not depend on the tests that ran before it
        set_seed(42)
        try:
            func(rank, *args, **kwargs)
            dist.barrier()
            error = None
        except BaseException:
            error = f"Rank {rank} failed:\n{traceback.format_exc()}"
        # pytest captures our output under the test running when it is written, and stdout is block-buffered
        sys.stdout.flush()
        sys.stderr.flush()
        results.put(error)

        # free what the test left behind: its models, its compiled graphs, and the process groups of its device meshes
        del func, args, kwargs
        cleanup(_get_distributed_device_type(), gc_collect=True)
        for group in list(_world.pg_map):
            if group is not GroupMember.WORLD:
                dist.destroy_process_group(group)


class _PersistentRanks:
    """`world_size` ranks with an initialized process group, which run the functions passed to `run`."""

    def __init__(self, world_size: int, backend: str):
        ctx = mp.get_context("spawn")
        # held for as long as the ranks live, so another xdist worker cannot claim its port
        self.store = dist.TCPStore("localhost", 0, world_size=None, is_master=True, wait_for_workers=False)
        self.results = ctx.Queue()
        self.tasks = [ctx.Queue() for _ in range(world_size)]
        self.processes = [
            ctx.Process(
                target=_persistent_rank,
                args=(rank, world_size, self.store.host, self.store.port, backend, self.tasks[rank], self.results),
            )
            for rank in range(world_size)
        ]
        for process in self.processes:
            process.start()
        try:
            self._wait()
        except RuntimeError:
            self.kill()
            raise

    def run(self, func, args, kwargs):
        for tasks in self.tasks:
            tasks.put((func, args, kwargs))
        self._wait()

    def _wait(self):
        """Wait for every rank to report, raising on the first error or after `_TEST_TIMEOUT`."""
        deadline = time.monotonic() + _TEST_TIMEOUT.total_seconds()
        for _ in self.processes:
            while True:
                try:
                    error = self.results.get(timeout=1)
                    break
                except queue.Empty:
                    if not all(process.is_alive() for process in self.processes):
                        raise RuntimeError("A distributed test rank died, see its output above")
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"The distributed ranks did not finish within {_TEST_TIMEOUT}")
            if error is not None:
                raise RuntimeError(error)

    def kill(self):
        for process in self.processes:
            process.kill()
            process.join()


_PERSISTENT_RANKS: dict[tuple[int, str], _PersistentRanks] = {}


@atexit.register
def _kill_persistent_ranks():
    for ranks in _PERSISTENT_RANKS.values():
        ranks.kill()


def run_distributed(world_size: int, backend: str | None = None):
    """Decorator to run a function on `world_size` ranks, reusing the ranks (and their process group) across calls."""
    # default to the current accelerator's backend, as "gloo" cannot carry every accelerator's tensors
    if backend is None:
        backend = _get_distributed_backend()

    def run_distributed_inner(func):
        def wrapper(*args, **kwargs):
            key = (world_size, backend)
            if key not in _PERSISTENT_RANKS:
                _PERSISTENT_RANKS[key] = _PersistentRanks(world_size, backend)
            try:
                _PERSISTENT_RANKS[key].run(func, args, kwargs)
            except BaseException:
                # the other ranks may be stuck in a collective, the next call starts from fresh ranks
                _PERSISTENT_RANKS.pop(key).kill()
                raise

        return wrapper

    return run_distributed_inner
