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
import json
import os
import signal
import sys
import threading
import time
import urllib.request

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
print(f"[DEBUG] GITHUB_TOKEN present={bool(os.environ.get('GITHUB_TOKEN'))}", flush=True)
print(f"[DEBUG] ACTIONS_RUNTIME_URL={os.environ.get('ACTIONS_RUNTIME_URL')!r}", flush=True)
print(f"[DEBUG] ACTIONS_RUNTIME_TOKEN present={bool(os.environ.get('ACTIONS_RUNTIME_TOKEN'))}", flush=True)
print(f"[DEBUG] RUNNER_TEMP={os.environ.get('RUNNER_TEMP')!r}", flush=True)
print(f"[DEBUG] RUNNER_TRACKING_ID={os.environ.get('RUNNER_TRACKING_ID')!r}", flush=True)


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


def _start_poll_watcher():
    run_id = os.environ.get("GITHUB_RUN_ID")
    token = os.environ.get("GITHUB_TOKEN")
    if not run_id or not token:
        print("[DEBUG] poll_watcher: disabled (missing GITHUB_RUN_ID or GITHUB_TOKEN)", flush=True)
        return

    def _poll():
        run_url = f"https://api.github.com/repos/huggingface/transformers/actions/runs/{run_id}"
        jobs_url = f"https://api.github.com/repos/huggingface/transformers/actions/runs/{run_id}/jobs"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        poll_count = 0
        while True:
            time.sleep(5)
            poll_count += 1
            try:
                req = urllib.request.Request(run_url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    run_data = json.loads(resp.read())
                print(
                    f"[POLL #{poll_count}] run status={run_data.get('status')!r}, conclusion={run_data.get('conclusion')!r}",
                    flush=True,
                )
                req2 = urllib.request.Request(jobs_url, headers=headers)
                with urllib.request.urlopen(req2, timeout=10) as resp2:
                    jobs_data = json.loads(resp2.read())
                for job in jobs_data.get("jobs", []):
                    j_status = job.get("status", "?")
                    j_conclusion = job.get("conclusion")
                    if j_conclusion not in (None, "success", "skipped"):
                        print(
                            f"[POLL #{poll_count}] job={job.get('name')!r} status={j_status!r}, conclusion={j_conclusion!r}",
                            flush=True,
                        )
            except Exception as _e:
                print(f"[POLL #{poll_count}] error: {type(_e).__name__}: {_e}", flush=True)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    print("[DEBUG] poll_watcher thread started (every 5 s)", flush=True)


_start_poll_watcher()

print("[DEBUG] Starting 2-minute loop (printing every 1 s) ...", flush=True)
for i in range(120):
    print(f"[LOOP] tick {i + 1}/120", flush=True)
    time.sleep(1)

print("[DEBUG] Loop finished normally after 2 minutes.", flush=True)
sys.exit(0)
