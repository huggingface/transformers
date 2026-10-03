#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

# TEMPORARY: simple 2-minute loop to test whether cancel signals reach this process.
import os
import signal
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

print(f"[DEBUG] Python PID={os.getpid()}, PGID={os.getpgrp()}", flush=True)
try:
    _ppid = os.getppid()
    print(f"[DEBUG] PPID={_ppid}", flush=True)
    with open(f"/proc/{_ppid}/cmdline", "rb") as _f:
        _parent_cmdline = _f.read().replace(b"\x00", b" ").decode(errors="replace").strip()
    print(f"[DEBUG] parent cmdline: {_parent_cmdline!r}", flush=True)
except Exception as _e:
    print(f"[DEBUG] parent info unavailable: {_e}", flush=True)

print(f"[DEBUG] GITHUB_RUN_ID={os.environ.get('GITHUB_RUN_ID')!r}", flush=True)
print(f"[DEBUG] GITHUB_JOB={os.environ.get('GITHUB_JOB')!r}", flush=True)


def _sigterm_handler(signum, frame):
    print(f"[SIGNAL] SIGTERM received! signum={signum}", flush=True)
    sys.exit(1)


def _sigint_handler(signum, frame):
    print(f"[SIGNAL] SIGINT received! signum={signum}", flush=True)
    sys.exit(1)


def _sighup_handler(signum, frame):
    print(f"[SIGNAL] SIGHUP received! signum={signum}", flush=True)
    sys.exit(1)


signal.signal(signal.SIGTERM, _sigterm_handler)
signal.signal(signal.SIGINT, _sigint_handler)
signal.signal(signal.SIGHUP, _sighup_handler)
print("[DEBUG] Signal handlers registered: SIGTERM, SIGINT, SIGHUP", flush=True)

print("[DEBUG] Starting 2-minute loop (printing every 1 s) ...", flush=True)
for i in range(120):
    print(f"[LOOP] tick {i + 1}/120", flush=True)
    time.sleep(1)

print("[DEBUG] Loop finished normally after 2 minutes.", flush=True)
sys.exit(0)
