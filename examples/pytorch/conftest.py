# Copyright 2020 The HuggingFace Team. All rights reserved.
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

# tests directory-specific settings - this file is run automatically
# by pytest before any tests are run

import os
import sys
import threading
import time
import warnings
from os.path import abspath, dirname, join


# allow having multiple repository checkouts and not needing to remember to rerun
# `pip install -e '.[dev]'` when switching between checkouts and running tests.
git_repo_path = abspath(join(dirname(dirname(dirname(__file__))), "src"))
sys.path.insert(1, git_repo_path)


# silence FutureWarning warnings in tests since often we can't act on them until
# they become normal warnings - i.e. the tests still need to test the current functionality
warnings.simplefilter(action="ignore", category=FutureWarning)


def pytest_addoption(parser):
    from transformers.testing_utils import pytest_addoption_shared

    pytest_addoption_shared(parser)


def pytest_terminal_summary(terminalreporter):
    from transformers.testing_utils import pytest_terminal_summary_main

    make_reports = terminalreporter.config.getoption("--make-reports")
    if make_reports:
        pytest_terminal_summary_main(terminalreporter, id=make_reports)


# ──────────────────────────────────────────────────────────────────────────────
# Memory monitoring — Linux only (CI runner has /proc)
# Logs system, controller, worker, and subprocess memory to stdout (flushed
# immediately) so the full picture appears in the GitHub Actions job log.
# ──────────────────────────────────────────────────────────────────────────────

_MEM_ENABLED = sys.platform.startswith("linux")
_t0 = time.monotonic()
_is_worker = False
_worker_id = "ctrl"
_worker_pids = {}  # {workerid: pid}  — controller only
_ctrl_stop = threading.Event()
_worker_stop = threading.Event()


def _mlog(msg):
    t = time.monotonic() - _t0
    line = f"[MEM t={t:8.2f}s] {msg}\n"
    # sys.__stdout__ bypasses pytest's per-worker stdout capture so lines
    # appear in the CI log even when the test passes.
    try:
        sys.__stdout__.write(line)
        sys.__stdout__.flush()
    except Exception:  # noqa: S110
        pass


# ── /proc helpers ─────────────────────────────────────────────────────────────


def _sys_mem():
    """(used_mb, total_mb) from /proc/meminfo."""
    fields = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            fields[k.strip()] = int(v.split()[0])
    used = (
        fields["MemTotal"]
        - fields["MemFree"]
        - fields.get("Buffers", 0)
        - fields.get("Cached", 0)
        - fields.get("SReclaimable", 0)
    )
    return used / 1024, fields["MemTotal"] / 1024


def _rss(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:  # noqa: S110
        pass
    return 0.0


def _pss(pid):
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) / 1024
    except Exception:  # noqa: S110
        pass
    return 0.0


def _children(ppid):
    result = []
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/status") as f:
                    for line in f:
                        if line.startswith("PPid:"):
                            if int(line.split()[1]) == ppid:
                                result.append(int(entry))
                            break
            except Exception:  # noqa: S110
                pass
    except Exception:  # noqa: S110
        pass
    return result


def _tree(pid):
    """(rss_mb, pss_mb) summed over pid + all descendants."""
    rss = _rss(pid)
    pss = _pss(pid)
    for child in _children(pid):
        cr, cp = _tree(child)
        rss += cr
        pss += cp
    return rss, pss


# ── background monitor loops ──────────────────────────────────────────────────


def _ctrl_monitor_loop():
    my_pid = os.getpid()
    while not _ctrl_stop.is_set():
        try:
            # Pick up any worker PIDs written to /tmp that we haven't seen yet
            for gw in ("gw0", "gw1", "gw2", "gw3"):
                if gw not in _worker_pids:
                    pid_file = f"/tmp/mem_worker_{gw}.pid"
                    if os.path.exists(pid_file):
                        try:
                            with open(pid_file) as f:
                                _worker_pids[gw] = int(f.read().strip())
                        except Exception:  # noqa: S110
                            pass
            sys_used, sys_total = _sys_mem()
            ctrl_rss, ctrl_pss = _tree(my_pid)
            parts = [
                f"SYS={sys_used:.0f}/{sys_total:.0f}MB",
                f"CTRL(pid={my_pid}) RSS={ctrl_rss:.0f} PSS={ctrl_pss:.0f}MB",
            ]
            for wid, wpid in sorted(_worker_pids.items()):
                wr, wp = _tree(wpid)
                nch = len(_children(wpid))
                parts.append(f"{wid}(pid={wpid}) RSS={wr:.0f} PSS={wp:.0f}MB nch={nch}")
            _mlog("[CTRL] " + " | ".join(parts))
        except Exception as exc:
            _mlog(f"[CTRL] monitor error: {exc}")
        _ctrl_stop.wait(5)


def _worker_monitor_loop(wid, pid):
    while not _worker_stop.is_set():
        try:
            rss, pss = _tree(pid)
            nch = len(_children(pid))
            sys_used, _ = _sys_mem()
            _mlog(f"[{wid}] RSS+subtree={rss:.0f} PSS+subtree={pss:.0f}MB nchildren={nch} SYS={sys_used:.0f}MB")
        except Exception as exc:
            _mlog(f"[{wid}] monitor error: {exc}")
        _worker_stop.wait(2)


# ── wav2vec2 detailed checkpoints ─────────────────────────────────────────────


def _checkpoint(label):
    try:
        rss, pss = _tree(os.getpid())
        sys_used, _ = _sys_mem()
        _mlog(f"[{_worker_id}] CKPT {label} | RSS+subtree={rss:.0f} PSS+subtree={pss:.0f}MB | SYS={sys_used:.0f}MB")
    except Exception:  # noqa: S110
        pass


def _patch_wav2vec2(item):
    """Apply memory-checkpoint monkeypatches for the wav2vec2 pretraining test."""
    restores = []

    try:
        import datasets as _ds

        _orig = _ds.load_dataset

        def _p(*a, **kw):
            _checkpoint("load_dataset START")
            r = _orig(*a, **kw)
            _checkpoint("load_dataset END")
            return r

        _ds.load_dataset = _p
        restores.append(lambda o=_orig: setattr(_ds, "load_dataset", o))
    except Exception:  # noqa: S110
        pass

    try:
        from datasets import DatasetDict as _DD

        _orig_map = _DD.map
        _orig_filter = _DD.filter

        def _p_map(self, fn, *a, **kw):
            _checkpoint(f"DatasetDict.map START num_proc={kw.get('num_proc')}")
            r = _orig_map(self, fn, *a, **kw)
            _checkpoint(f"DatasetDict.map END num_proc={kw.get('num_proc')}")
            return r

        def _p_filter(self, fn, *a, **kw):
            _checkpoint(f"DatasetDict.filter START num_proc={kw.get('num_proc')}")
            r = _orig_filter(self, fn, *a, **kw)
            _checkpoint(f"DatasetDict.filter END num_proc={kw.get('num_proc')}")
            return r

        _DD.map = _p_map
        _DD.filter = _p_filter
        restores.append(lambda o=_orig_map: setattr(_DD, "map", o))
        restores.append(lambda o=_orig_filter: setattr(_DD, "filter", o))
    except Exception:  # noqa: S110
        pass

    try:
        from transformers import Wav2Vec2ForPreTraining as _W2V

        _orig_init = _W2V.__init__

        def _p_init(self, *a, **kw):
            _checkpoint("Wav2Vec2ForPreTraining.__init__ START")
            _orig_init(self, *a, **kw)
            _checkpoint("Wav2Vec2ForPreTraining.__init__ END")

        _W2V.__init__ = _p_init
        restores.append(lambda o=_orig_init: setattr(_W2V, "__init__", o))
    except Exception:  # noqa: S110
        pass

    try:
        from accelerate import Accelerator as _Acc

        _orig_prepare = _Acc.prepare
        _orig_backward = _Acc.backward
        _counters = [0, 0]  # [prepare_n, backward_n]

        def _p_prepare(self, *a, **kw):
            _counters[0] += 1
            _checkpoint(f"Accelerator.prepare #{_counters[0]} START")
            r = _orig_prepare(self, *a, **kw)
            _checkpoint(f"Accelerator.prepare #{_counters[0]} END")
            return r

        def _p_backward(self, loss, **kw):
            _counters[1] += 1
            _checkpoint(f"Accelerator.backward #{_counters[1]} START")
            r = _orig_backward(self, loss, **kw)
            _checkpoint(f"Accelerator.backward #{_counters[1]} END")
            return r

        _Acc.prepare = _p_prepare
        _Acc.backward = _p_backward
        restores.append(lambda o=_orig_prepare: setattr(_Acc, "prepare", o))
        restores.append(lambda o=_orig_backward: setattr(_Acc, "backward", o))
    except Exception:  # noqa: S110
        pass

    try:
        import torch

        _orig_step = torch.optim.AdamW.step
        _step_n = [0]

        def _p_step(self, *a, **kw):
            _step_n[0] += 1
            _checkpoint(f"AdamW.step #{_step_n[0]} START")
            r = _orig_step(self, *a, **kw)
            _checkpoint(f"AdamW.step #{_step_n[0]} END")
            return r

        torch.optim.AdamW.step = _p_step
        restores.append(lambda o=_orig_step: setattr(torch.optim.AdamW, "step", o))
    except Exception:  # noqa: S110
        pass

    def _cleanup():
        for fn in restores:
            try:
                fn()
            except Exception:  # noqa: S110
                pass
        _checkpoint("wav2vec2 patches cleaned up")

    item.addfinalizer(_cleanup)
    _checkpoint("wav2vec2 patches applied")


# ── pytest hooks ──────────────────────────────────────────────────────────────


def pytest_configure(config):
    global _is_worker, _worker_id
    if not _MEM_ENABLED:
        return
    if hasattr(config, "workerinput"):
        _is_worker = True
        _worker_id = config.workerinput.get("workerid", "gw?")
        # Publish PID so the controller monitor can read it
        try:
            with open(f"/tmp/mem_worker_{_worker_id}.pid", "w") as f:
                f.write(str(os.getpid()))
        except Exception:  # noqa: S110
            pass
        _mlog(f"[{_worker_id}] Worker process started pid={os.getpid()}")
        threading.Thread(
            target=_worker_monitor_loop,
            args=(_worker_id, os.getpid()),
            daemon=True,
            name="mem-worker",
        ).start()
    else:
        _mlog(f"[CTRL] Controller started pid={os.getpid()}")
        threading.Thread(target=_ctrl_monitor_loop, daemon=True, name="mem-ctrl").start()


def pytest_testnodeready(node):
    """Controller: read worker PID from /tmp file written by the worker."""
    if not _MEM_ENABLED:
        return
    try:
        wid = node.workerid
        pid_file = f"/tmp/mem_worker_{wid}.pid"
        # Wait briefly for the worker to write its PID file
        for _ in range(20):
            if os.path.exists(pid_file):
                break
            time.sleep(0.1)
        with open(pid_file) as f:
            pid = int(f.read().strip())
        _worker_pids[wid] = pid
        _mlog(f"[CTRL] Worker {wid} ready pid={pid}")
    except Exception as exc:
        _mlog(f"[CTRL] Could not get pid for {getattr(node, 'workerid', '?')}: {exc}")


def pytest_runtest_setup(item):
    """Worker: log memory at test start; apply wav2vec2 patches if needed."""
    if not _MEM_ENABLED or not _is_worker:
        return
    rss, pss = _tree(os.getpid())
    sys_used, _ = _sys_mem()
    _mlog(f"[{_worker_id}] SETUP {item.nodeid} | RSS+subtree={rss:.0f} PSS+subtree={pss:.0f}MB | SYS={sys_used:.0f}MB")
    if "wav2vec2_pretraining" in item.name:
        _patch_wav2vec2(item)


def pytest_runtest_teardown(item, nextitem):
    """Worker: log memory after test completes."""
    if not _MEM_ENABLED or not _is_worker:
        return
    rss, pss = _tree(os.getpid())
    sys_used, _ = _sys_mem()
    _mlog(
        f"[{_worker_id}] TEARDOWN {item.nodeid} | RSS+subtree={rss:.0f} PSS+subtree={pss:.0f}MB | SYS={sys_used:.0f}MB"
    )


def pytest_runtest_logstart(nodeid, location):
    """Controller: log system memory when a worker starts a test."""
    if not _MEM_ENABLED or _is_worker:
        return
    try:
        sys_used, _ = _sys_mem()
        _mlog(f"[CTRL] LOGSTART {nodeid} | SYS={sys_used:.0f}MB")
    except Exception:  # noqa: S110
        pass


def pytest_runtest_logreport(report):
    """Controller: log system memory when a worker finishes a test phase."""
    if not _MEM_ENABLED or _is_worker or report.when not in ("call", "setup"):
        return
    if report.outcome == "passed":
        return  # only log failures/errors for call/setup to reduce noise
    try:
        sys_used, _ = _sys_mem()
        _mlog(f"[CTRL] {report.when.upper()} {report.nodeid} outcome={report.outcome} | SYS={sys_used:.0f}MB")
    except Exception:  # noqa: S110
        pass


def pytest_sessionfinish(session, exitstatus):
    if not _MEM_ENABLED:
        return
    _ctrl_stop.set()
    _worker_stop.set()
    label = f"worker:{_worker_id}" if _is_worker else "CTRL"
    _mlog(f"[{label}] Session finished, monitors stopped")
