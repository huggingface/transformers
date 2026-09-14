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
"""``td`` — submit a script to a persistent multi-node worker pool.

``torchrun`` pays NCCL init and CUDA context setup on every invocation, which
dominates the loop when you are iterating on a sharding plan. ``td`` starts the
ranks once as a daemon and then submits scripts to it over a `TCPStore`, so the
second run onwards is just the script.

    td train.py                                   # infer nodes/GPUs from Slurm
    td train.py --nodes a,b --gpus 8,8            # or say it explicitly
    td --down                                     # let the daemon finish and exit
    td --kill                                     # hard stop every rank

The submitted script must define ``main()``; it is called on every rank with the
process group already initialized.
"""

import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from datetime import timedelta
from typing import Annotated

import typer


PORT, CTRL = 29500, 29600
STARTUP_TIMEOUT = 300


def default_nodes() -> list[str]:
    """Slurm's allocation if there is one, otherwise just this host."""
    nodelist = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    if not nodelist and os.environ.get("SLURM_JOB_ID"):
        out = subprocess.run(
            ["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"],
            capture_output=True,
            text=True,
        ).stdout
        for field in out.split():
            if field.startswith("NodeList=") and field != "NodeList=(null)":
                nodelist = field.split("=", 1)[1]
                break
    if nodelist:
        hosts = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist], capture_output=True, text=True
        ).stdout.split()
        if hosts:
            return hosts
    return [socket.gethostname()]


def default_gpus(n_nodes: int) -> list[int]:
    """Assume a homogeneous allocation: same visible GPU count on every node."""
    per_node = os.environ.get("SLURM_GPUS_ON_NODE")
    if per_node:
        count = int(per_node)
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            count = len([d for d in visible.split(",") if d.strip()])
        else:
            import torch

            count = torch.cuda.device_count()
    if count < 1:
        raise typer.BadParameter("could not detect any GPU; pass --gpus explicitly")
    return [count] * n_nodes


class Cluster:
    """The set of nodes a daemon is (or would be) running on."""

    SSH = [
        "ssh",
        "-n",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=10m",
        "-o",
        "ControlPath=~/.ssh/cm-%r@%h:%p",
    ]

    def __init__(self, nodes: list[str], gpus: list[int]):
        if len(nodes) != len(gpus):
            raise typer.BadParameter("--nodes and --gpus must have the same length")
        self.nodes, self.gpus = nodes, gpus
        self.ips = [self.resolve(h) for h in nodes]
        self.head, self.head_ip = nodes[0], self.ips[0]
        self.cwd = os.getcwd()
        self.log_dir = os.path.join(self.cwd, ".td")
        self.head_log = os.path.join(self.log_dir, f"{self.head}.log")

    @staticmethod
    def resolve(host: str) -> str:
        ip = socket.gethostbyname(host)
        if ip.startswith("127."):
            raise typer.BadParameter(f"{host} resolves to {ip} (loopback) — pass its private IP instead")
        return ip

    @staticmethod
    def is_local(ip: str) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            try:
                s.bind((ip, 0))
                return True
            except OSError:
                return False

    def run_on(self, host: str, ip: str, command: str, **kwargs):
        argv = ["bash", "-c", command] if self.is_local(ip) else self.SSH + [host, command]
        return subprocess.run(argv, **kwargs)

    # -- daemon state ----------------------------------------------------------

    def listening(self) -> bool:
        with socket.socket() as s:
            s.settimeout(1.0)
            return s.connect_ex((self.head_ip, CTRL)) == 0

    def store(self, timeout=timedelta(days=7)):
        import torch.distributed as dist

        return dist.TCPStore(self.head_ip, CTRL, is_master=False, timeout=timeout)

    def ready(self) -> bool:
        """Up *and* parked in the job loop — not merely holding an open socket.

        An open port only means rank 0 got as far as binding it, which it does well
        before the process group is usable (and just before it dies, if it is going
        to). The worker sets ``ready`` once it can actually accept a job.
        """
        if not self.listening():
            return False
        try:
            return self.store(timedelta(seconds=2)).check(["ready"])
        except Exception:
            return False

    def crashed(self) -> str | None:
        """Name of a node whose launcher died, so startup can fail fast."""
        for host in self.nodes:
            try:
                with open(os.path.join(self.log_dir, f"{host}.log"), errors="replace") as f:
                    blob = f.read()
            except OSError:
                continue
            if "ChildFailedError" in blob or "Traceback (most recent call last)" in blob:
                return host
        return None

    # -- lifecycle -------------------------------------------------------------

    def kill(self):
        # Anchored on how up() launches, and bracketed so the patterns cannot match
        # the killing shell's own argv — a bare `-f worker.py` matches the command
        # line running it, so pkill -9 kills itself before reaching the workers, and
        # would also shoot down any unrelated shell that happens to mention it.
        command = (
            r"pkill -9 -f '[-]m torch\.distributed\.run --nnodes='; "
            r"pkill -9 -f '[d]istributed/worker\.py'; true"
        )
        for host, ip in zip(self.nodes, self.ips):
            print(f"  killing {host}", flush=True)
            self.run_on(host, ip, command)

    @staticmethod
    def forwarded_env() -> str:
        """Carry the submitting shell's HF/Transformers settings onto every node.

        A remote node's non-interactive shell does not source the user's profile, so
        without this the head rank resolves ``HF_HOME`` one way and the other nodes
        silently fall back to their own defaults — every rank then downloads to a
        different cache. Tokens are deliberately not forwarded: they would land in
        the command line and be visible in ``ps``. ``huggingface_hub`` picks those
        up from the token file under a now-consistent ``HF_HOME`` instead.
        """
        keep = [
            (k, v)
            for k, v in sorted(os.environ.items())
            if (k.startswith(("HF_", "HUGGINGFACE_", "TRANSFORMERS_")) and "TOKEN" not in k)
        ]
        return " ".join(f"{k}={shlex.quote(v)}" for k, v in keep)

    def up(self):
        os.makedirs(self.log_dir, exist_ok=True)
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")

        # PYTHONUNBUFFERED is what makes the logs live: worker stdout is a file, so
        # without it Python block-buffers ~8KB and the follower sees nothing for
        # minutes. TORCH_NCCL_ENABLE_MONITORING=0 stops the NCCL heartbeat thread
        # from polling torchrun's rendezvous store forever; once that store closes
        # it emits a full C++ stack every second per rank and buries the real logs.
        env = (
            f"CTRL_PORT={CTRL} WORKSPACE={self.cwd} TORCH_NCCL_ASYNC_ERROR_HANDLING=1 "
            f"TORCH_NCCL_ENABLE_MONITORING=0 PYTHONUNBUFFERED=1 {self.forwarded_env()}"
        )

        # Truncate up front (cwd is shared) so crashed() cannot trip over the
        # previous run's traceback before each node's own redirect clears the file.
        for host in self.nodes:
            open(os.path.join(self.log_dir, f"{host}.log"), "w").close()

        started = []
        try:
            for i, (host, ip, n_gpu) in enumerate(zip(self.nodes, self.ips, self.gpus)):
                log = os.path.join(self.log_dir, f"{host}.log")
                inner = (
                    f"cd {self.cwd} && exec env {env} {sys.executable} -u -m torch.distributed.run "
                    f"--nnodes={len(self.nodes)} --nproc-per-node={n_gpu} --node-rank={i} "
                    f"--master-addr={self.head_ip} --master-port={PORT} {worker}"
                )
                # `&` binds to the whole `a && b` list, so bash forks a subshell for
                # it that keeps ssh's stdout/stderr open for as long as the daemon
                # lives. ssh then never sees EOF and this call blocks forever. Group
                # it and redirect the whole unit instead.
                command = (
                    f"( mkdir -p {self.log_dir} && exec setsid bash -c {shlex.quote(inner)} ) "
                    f"> {log} 2>&1 < /dev/null &"
                )
                where = "local" if self.is_local(ip) else "ssh"
                print(f"  {host} ({where}) node_rank={i}", flush=True)
                self.run_on(host, ip, command, check=True, timeout=60)
                started.append((host, ip))
        except Exception:
            print("launch failed, cleaning up", flush=True)
            self.kill()
            raise


class Follower(threading.Thread):
    """Stream the head node's log to stdout while a job runs.

    Deliberately not ``tail -F``: tail polls on its own clock, so stopping it right
    after a job finishes swallows the last second of output, including the job's
    final lines. This drains to EOF once more on :meth:`close`, after the result is
    already known, so nothing written before the job ended can be lost.
    """

    def __init__(self, path: str, offset: int | None = None):
        super().__init__(daemon=True)
        self.path = path
        if offset is None:
            offset = os.path.getsize(path) if os.path.exists(path) else 0
        self.offset = offset
        self.stopped = threading.Event()

    def drain(self):
        try:
            if os.path.getsize(self.path) < self.offset:
                self.offset = 0  # a relaunch truncated it
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                data = f.read()
        except OSError:
            return
        if data:
            self.offset += len(data)
            sys.stdout.buffer.write(data)
            sys.stdout.flush()

    def run(self):
        while not self.stopped.wait(0.2):
            self.drain()

    def close(self):
        self.stopped.set()
        self.join(timeout=2)
        self.drain()


def td(
    script: Annotated[str | None, typer.Argument(help="Python file defining main(); run on every rank.")] = None,
    nodes: Annotated[
        str | None, typer.Option(help="Comma-separated hostnames or IPs. Defaults to the Slurm allocation.")
    ] = None,
    gpus: Annotated[
        str | None, typer.Option(help="Comma-separated GPU count per node. Defaults to the visible devices.")
    ] = None,
    down: Annotated[bool, typer.Option(help="Ask the daemon to exit once it is idle.")] = False,
    kill: Annotated[bool, typer.Option(help="Hard-stop every rank on every node.")] = False,
):
    """Run a script on a persistent multi-node worker pool, starting it if needed."""
    node_list = nodes.split(",") if nodes else default_nodes()
    gpu_list = [int(x) for x in gpus.split(",")] if gpus else default_gpus(len(node_list))
    cluster = Cluster(node_list, gpu_list)

    if kill:
        cluster.kill()
        return

    if down:
        if cluster.listening():
            store = cluster.store()
            store.set(f"job:{store.add('next', 1) - 1}", "__exit__")
            print("stopping")
        else:
            print("not running")
        return

    if script is None:
        raise typer.BadParameter("need a script (or --down / --kill)")
    if not os.path.exists(script):
        raise typer.BadParameter(f"no such script: {script}")

    # One follower for the whole run, started before anything can write to the log.
    # Stopping and restarting it around startup lets startup failures scroll by
    # unseen, which is exactly when you most want the output.
    follower = None
    try:
        if not cluster.ready():
            print("daemon down, starting...", flush=True)
            cluster.up()
            follower = Follower(cluster.head_log, offset=0)
            follower.start()
            for i in range(STARTUP_TIMEOUT):
                if cluster.ready():
                    break
                dead = cluster.crashed()
                if dead:
                    raise typer.Exit(_fail(follower, f"daemon failed to start on {dead}; see .td/{dead}.log"))
                if i and i % 15 == 0:
                    print(f"  ...waiting for daemon ({i}s)", flush=True)
                time.sleep(1)
            else:
                raise typer.Exit(_fail(follower, f"daemon never came up; see .td/{cluster.head}.log"))
            print("--- daemon up ---", flush=True)
        else:
            follower = Follower(cluster.head_log)
            follower.start()

        store = cluster.store()
        job = store.add("next", 1) - 1
        store.set(f"job:{job}", os.path.abspath(script))
        try:
            result = store.get(f"done:{job}").decode()
        except KeyboardInterrupt:
            raise typer.Exit(_fail(follower, "detached (job still running)"))
        except Exception as e:
            raise typer.Exit(_fail(follower, f"lost the daemon while job {job} was running: {e}"))

        follower.close()
        follower = None
        print(f"[job {job}] {result}")
        raise typer.Exit(0 if result == "ok" else 1)
    finally:
        if follower is not None:
            follower.close()


def _fail(follower: Follower | None, message: str) -> int:
    if follower is not None:
        follower.close()
    print(f"\n{message}", file=sys.stderr, flush=True)
    return 1


def main():
    typer.run(td)


if __name__ == "__main__":
    main()
