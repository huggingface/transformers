# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Two independent launchers on one host, with one visible GPU per simulated node."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch, require_torch_multi_gpu


if is_torch_available():
    import torch


_CUDA_CASES = (
    "ddp_checkpoint",
    "ddp_reentrant_static",
    "ddp_unused",
    "ddp_change_plan",
    "fsdp_checkpoint",
    "tp_checkpoint",
    "trainer_checkpoint",
    "trainer_unused_checkpoint",
    "native_trainer_tp",
    "native_trainer_fsdp",
    "pipeline_llama",
    "pipeline_qwen3_5",
    "pipeline_trainer",
    "pipeline_trainer_bf16",
    "continuous_tp",
    "continuous_pp",
    "adapter_gemma3n",
    "adapter_recurrent_gemma",
    "adapter_opt",
)


def _run_two_node_case(directory, case, device_type):
    """Retain per-node output, fail on launcher errors, and check actual NCCL data transports."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    processes = []
    handles = []
    try:
        for node_rank in range(2):
            environment = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(node_rank) if device_type == "cuda" else "",
                "TRANSFORMERS_TEST_DEVICE": device_type,
                "OMP_NUM_THREADS": "1",
                "HF_HUB_OFFLINE": "1",
                "PYTHONPATH": str(repository / "src") + os.pathsep + str(repository),
                "NCCL_NET": "Socket",
                "NCCL_P2P_DISABLE": "1",
                "NCCL_SHM_DISABLE": "1",
                "NCCL_NVLS_ENABLE": "0",
                "NCCL_DEBUG": "INFO",
                "NCCL_DEBUG_SUBSYS": "INIT,NET,GRAPH",
                "NCCL_DEBUG_FILE": str(directory / f"nccl-node{node_rank}-%h-%p.log"),
            }
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=2",
                "--nproc-per-node=1",
                f"--node-rank={node_rank}",
                "--master-addr=127.0.0.1",
                f"--master-port={port}",
                "--rdzv-backend=static",
                "--max-restarts=0",
                "--monitor-interval=1",
                str(Path(__file__).resolve()),
                "--case",
                case,
                "--device-type",
                device_type,
                "--output-dir",
                str(directory),
            ]
            handle = (directory / f"node{node_rank}.log").open("w")
            handles.append(handle)
            processes.append(
                subprocess.Popen(
                    command, env=environment, cwd=repository, stdout=handle, stderr=handle, start_new_session=True
                )
            )
        deadline = time.monotonic() + 300
        for node_rank, process in enumerate(processes):
            result = process.wait(timeout=max(1, deadline - time.monotonic()))
            if result:
                raise AssertionError((directory / f"node{node_rank}.log").read_text()[-12000:])
        reports = [json.loads((directory / f"rank{rank}.json").read_text()) for rank in range(2)]
        for rank, report in enumerate(reports):
            assert report["status"] == "passed", report
            assert report["rank"] == rank and report["group_rank"] == rank, report
            assert report["local_rank"] == 0 and report["local_world_size"] == 1, report
            assert report["world_size"] == 2, report
            if device_type == "cuda":
                assert report["visible_gpu_count"] == 1 and report["current_device"] == 0, report
                logs = list(directory.glob(f"nccl-node{rank}-*.log"))
                assert logs, f"Missing NCCL logs for node {rank}"
                transport = "\n".join(path.read_text() for path in logs)
                assert "via NET/Socket" in transport, "No NCCL data channel used Socket"
                assert "via P2P" not in transport and "via SHM" not in transport, "A local transport remained active"
        if device_type == "cuda":
            assert reports[0]["gpu_uuid"] != reports[1]["gpu_uuid"], "Both simulated nodes used the same GPU"
        return reports
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
        for handle in handles:
            handle.close()


def _worker(case, device_type, directory):
    from tests.test_layer_execution_distributed import _distributed_worker, _native_trainer_worker
    from tests.test_layer_execution_pipeline import (
        _adapter_pipeline_worker,
        _continuous_worker,
        _pipeline_worker,
        _trainer_pipeline_worker,
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    report = {
        "case": case,
        "rank": rank,
        "local_rank": local_rank,
        "group_rank": int(os.environ["GROUP_RANK"]),
        "world_size": int(os.environ["WORLD_SIZE"]),
        "local_world_size": int(os.environ["LOCAL_WORLD_SIZE"]),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "status": "running",
    }
    output = Path(directory) / f"rank{rank}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        if device_type == "cuda":
            torch.cuda.set_device(local_rank)
            report.update(
                visible_gpu_count=torch.cuda.device_count(),
                current_device=torch.cuda.current_device(),
                gpu_uuid=str(torch.cuda.get_device_properties(local_rank).uuid),
            )
        store = str(Path(directory) / "store")
        options = {"local_rank": local_rank, "init_method": "env://"}
        if case.startswith("native_trainer_"):
            _native_trainer_worker(rank, store, case.removeprefix("native_trainer_"), **options)
        elif case.startswith("pipeline_trainer"):
            _trainer_pipeline_worker(rank, store, device_type, mixed_precision=case.endswith("bf16"), **options)
        elif case.startswith("pipeline_"):
            _pipeline_worker(rank, store, case.removeprefix("pipeline_"), True, True, device_type, **options)
        elif case.startswith("continuous_"):
            _continuous_worker(rank, store, case.removeprefix("continuous_"), device_type, **options)
        elif case.startswith("adapter_"):
            _adapter_pipeline_worker(rank, store, case.removeprefix("adapter_"), device_type, **options)
        else:
            _distributed_worker(rank, store, case, device_type, None, **options)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")


@require_torch
@unittest.skipUnless(
    is_torch_available() and torch.distributed.is_available() and torch.distributed.is_gloo_available(),
    "Requires Gloo",
)
class LayerExecutionTwoNodeCPUTest(unittest.TestCase):
    def test_independent_launchers(self):
        with tempfile.TemporaryDirectory() as directory:
            _run_two_node_case(directory, "ddp_checkpoint", "cpu")


@require_torch_multi_gpu
@unittest.skipUnless(
    is_torch_available() and torch.distributed.is_available() and torch.distributed.is_nccl_available(),
    "Requires NCCL",
)
class LayerExecutionTwoNodeCUDATest(unittest.TestCase):
    @parameterized.expand([(case,) for case in _CUDA_CASES])
    def test_independent_launchers(self, case):
        with tempfile.TemporaryDirectory() as directory:
            _run_two_node_case(directory, case, "cuda")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=_CUDA_CASES, required=True)
    parser.add_argument("--device-type", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    _worker(args.case, args.device_type, args.output_dir)
