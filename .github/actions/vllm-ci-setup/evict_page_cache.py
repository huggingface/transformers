"""pytest plugin of the vLLM CI (`vllm-ci-caller.yml`): evict the page cache of the HF cache files before each test.

Before starting, vLLM's CPU worker checks that the KV cache fits in the container's free memory, computed as
`memory.max - memory.current` of its cgroup. `memory.current` includes the page cache of every model file read
earlier in the job: clean and reclaimable, but counted, so after a few large models the check fails ("Available
memory on node 0 (... GiB) ... is less than requested memory for kv"). `posix_fadvise(POSIX_FADV_DONTNEED)` drops
these pages without privileges (`/proc/sys/vm/drop_caches` and `memory.reclaim` are read-only in the CI container).

Loaded with `PYTEST_PLUGINS=evict_page_cache` (see the `evict-page-cache` input of the `vllm-ci-setup` action).
"""

import os
import time


CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def _cgroup_memory_gib():
    try:
        with open("/sys/fs/cgroup/memory.current", encoding="utf-8") as f:
            return int(f.read()) / 2**30
    except (OSError, ValueError):
        return float("nan")


def evict_page_cache():
    """Drop the page cache of all the files under `CACHE_DIR` (best effort)."""
    for root, _, files in os.walk(CACHE_DIR):
        for name in files:
            try:
                fd = os.open(os.path.join(root, name), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
            finally:
                os.close(fd)


def pytest_runtest_setup(item):
    before = _cgroup_memory_gib()
    start = time.monotonic()
    evict_page_cache()
    print(
        f"\n[evict_page_cache] cgroup memory: {before:.1f} -> {_cgroup_memory_gib():.1f} GiB "
        f"({time.monotonic() - start:.1f}s)",
        flush=True,
    )
