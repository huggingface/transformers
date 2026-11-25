"""pytest plugin of the vLLM CI (`vllm-ci-caller.yml`): after each test, evict the page cache of its models' files.

Before starting, vLLM's CPU worker checks that the KV cache fits in the container's free memory, computed as
`memory.max - memory.current` of its cgroup. `memory.current` includes the page cache of every model file read
earlier in the job: clean and reclaimable, but counted, so after a few large models the check fails ("Available
memory on node 0 (... GiB) ... is less than requested memory for kv"). `posix_fadvise(POSIX_FADV_DONTNEED)` drops
these pages without privileges (`/proc/sys/vm/drop_caches` and `memory.reclaim` are read-only in the CI container).

Only the HF cache directories of the models in the test's parameters (Hub repo ids, or objects with a `name`) are
evicted: walking the whole shared cache takes far too long.

Loaded with `PYTEST_PLUGINS=evict_page_cache` (see the `evict-page-cache` input of the `vllm-ci-setup` action).
"""

import os
import re
import time


HUB_DIR = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
REPO_ID = re.compile(r"[\w.-]+/[\w.-]+")


def _cgroup_memory_gib():
    try:
        with open("/sys/fs/cgroup/memory.current", encoding="utf-8") as f:
            return int(f.read()) / 2**30
    except (OSError, ValueError):
        return float("nan")


def _repo_dirs(item):
    """The HF cache directories of the Hub models in the test's parameters."""
    dirs = set()
    callspec = getattr(item, "callspec", None)
    for value in callspec.params.values() if callspec else ():
        name = value if isinstance(value, str) else getattr(value, "name", None)
        if isinstance(name, str) and REPO_ID.fullmatch(name):
            path = os.path.join(HUB_DIR, "models--" + name.replace("/", "--"))
            if os.path.isdir(path):
                dirs.add(path)
    return dirs


def _evict(directory):
    for root, _, files in os.walk(directory):
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


def pytest_runtest_teardown(item, nextitem):
    dirs = _repo_dirs(item)
    if not dirs:
        return
    before = _cgroup_memory_gib()
    start = time.monotonic()
    for directory in sorted(dirs):
        _evict(directory)
    names = ", ".join(os.path.basename(d) for d in sorted(dirs))
    print(
        f"\n[evict_page_cache] {names}: cgroup memory {before:.1f} -> {_cgroup_memory_gib():.1f} GiB "
        f"({time.monotonic() - start:.1f}s)",
        flush=True,
    )
