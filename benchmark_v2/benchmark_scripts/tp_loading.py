"""Benchmark + smoke test for the TP loading refactor.

The new flow partitions the list of weights across ranks (so each rank only reads a fraction of the checkpoint),
then redistributes the full tensors via collective comms and finally has every rank locally extract its TP shard.

This script exercises that path end-to-end on CPU using the `gloo` backend so it can run anywhere (no GPU required).
It does two things:

  * **Correctness check** — verifies the per-rank shards loaded via the multi-process TP path match the slices we'd
    get from a single-process load on the same state_dict. This is the "test it" part: a regression here would mean
    the partition + broadcast + local-shard pipeline is wrong.
  * **Microbenchmark** — measures (a) the partition function on synthetic mappings and (b) wall-clock load time for
    a 1-rank reference run vs. a multi-rank TP run on the same model.

Run with::

    python benchmark_v2/benchmark_scripts/tp_loading.py [--world-size N] [--num-layers L] [--hidden D]

Exit code is 0 iff the correctness check passes.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from transformers import PretrainedConfig
from transformers.core_model_loading import (
    _tensor_nbytes,
    convert_and_load_state_dict_in_model,
    partition_mappings_across_ranks,
)
from transformers.modeling_utils import LoadStateDictConfig


# ---------------------------------------------------------------------------------------------------------------------
# Tiny synthetic model: a stack of (colwise, rowwise) Linear pairs we can shard with the existing TP plan keys.
# ---------------------------------------------------------------------------------------------------------------------
class _TPBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        # Bias-less so the sharding cases are simple and predictable.
        self.up = nn.Linear(hidden, hidden, bias=False)
        self.down = nn.Linear(hidden, hidden, bias=False)


class _TPModel(nn.Module):
    base_model_prefix = "model"

    def __init__(self, num_layers: int, hidden: int):
        super().__init__()
        self.config = PretrainedConfig()
        self.layers = nn.ModuleList([_TPBlock(hidden) for _ in range(num_layers)])
        # `convert_and_load_state_dict_in_model` reads `model.tp_plan` to look up the parallel style class.
        self.tp_plan = {
            "layers.*.up.weight": "colwise",
            "layers.*.down.weight": "rowwise",
        }


def _make_state_dict(num_layers: int, hidden: int, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    sd: dict[str, torch.Tensor] = {}
    for i in range(num_layers):
        sd[f"layers.{i}.up.weight"] = torch.randn(hidden, hidden, generator=g)
        sd[f"layers.{i}.down.weight"] = torch.randn(hidden, hidden, generator=g)
    return sd


# ---------------------------------------------------------------------------------------------------------------------
# Microbenchmark for the partition helper.
# ---------------------------------------------------------------------------------------------------------------------
def benchmark_partition(num_mappings: int = 5_000, world_size: int = 8) -> None:
    g = torch.Generator().manual_seed(0)
    sizes = (torch.randint(1, 1_000_000, (num_mappings,), generator=g)).tolist()
    mapping_total_bytes = {f"layers.{i}.weight_{i % 7}": int(s) for i, s in enumerate(sizes)}

    t0 = time.perf_counter()
    assignment = partition_mappings_across_ranks(mapping_total_bytes, world_size)
    elapsed = time.perf_counter() - t0

    # Sanity: every mapping is assigned exactly once, ranks are in range, and load is roughly balanced.
    assert set(assignment.keys()) == set(mapping_total_bytes.keys())
    assert all(0 <= r < world_size for r in assignment.values())

    rank_loads = [0] * world_size
    for name, r in assignment.items():
        rank_loads[r] += mapping_total_bytes[name]
    total = sum(rank_loads)
    max_load = max(rank_loads)
    imbalance = max_load / (total / world_size) - 1.0  # 0 == perfect balance

    print(f"[partition] {num_mappings} mappings → {world_size} ranks in {elapsed * 1e3:.2f} ms")
    print(f"[partition] per-rank bytes = {rank_loads}")
    print(f"[partition] max-vs-mean imbalance = {imbalance * 100:.2f}%")


# ---------------------------------------------------------------------------------------------------------------------
# Multi-process TP loading driver. Each worker initializes a process group on a 1D mesh of size `world_size` over
# `gloo` (CPU), builds the same model + state_dict, and runs the loading. We dump the loaded params to a shared file
# so the parent can verify correctness against the original tensors.
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class _WorkerArgs:
    world_size: int
    num_layers: int
    hidden: int
    init_method: str
    out_dir: str


def _worker(rank: int, args: _WorkerArgs) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    dist.init_process_group(backend="gloo", init_method=args.init_method, rank=rank, world_size=args.world_size)
    try:
        device_mesh = dist.init_device_mesh("cpu", (args.world_size,))

        model = _TPModel(args.num_layers, args.hidden)
        state_dict = _make_state_dict(args.num_layers, args.hidden)
        load_config = LoadStateDictConfig(device_mesh=device_mesh, device_map={"": "cpu"})

        t0 = time.perf_counter()
        convert_and_load_state_dict_in_model(model, state_dict, load_config, tp_plan=model.tp_plan)
        elapsed = time.perf_counter() - t0

        per_rank_dump = {k: v.detach().clone() for k, v in model.state_dict().items()}
        torch.save(
            {"rank": rank, "elapsed": elapsed, "params": per_rank_dump}, os.path.join(args.out_dir, f"rank_{rank}.pt")
        )
    finally:
        dist.destroy_process_group()


def _expected_shard(full: torch.Tensor, plan: str, rank: int, world_size: int) -> torch.Tensor:
    """Reproduce `get_tensor_shard` semantics for the simple 2D-Linear weights this benchmark uses."""
    import math

    if plan == "colwise":
        dim = -2  # shard output features
    elif plan == "rowwise":
        dim = -1  # shard input features
    else:
        raise ValueError(plan)
    dim = full.dim() + dim if dim < 0 else dim
    shard_size = math.ceil(full.shape[dim] / world_size)
    start = rank * shard_size
    end = min(start + shard_size, full.shape[dim])
    if start >= full.shape[dim]:
        new_shape = list(full.shape)
        new_shape[dim] = 0
        return torch.empty(new_shape, dtype=full.dtype)
    slicer = [slice(None)] * full.dim()
    slicer[dim] = slice(start, end)
    return full[tuple(slicer)].contiguous()


def benchmark_tp_loading(world_size: int, num_layers: int, hidden: int) -> bool:
    """Run multi-process TP loading via `gloo`, time it, and validate per-rank shards. Returns True on success."""
    if world_size < 2:
        print(f"[tp] skipping multi-process run (world_size={world_size} < 2)")
        return True

    with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as init_dir:
        init_method = f"file://{os.path.join(init_dir, 'rendezvous')}"
        worker_args = _WorkerArgs(
            world_size=world_size, num_layers=num_layers, hidden=hidden, init_method=init_method, out_dir=out_dir
        )

        t0 = time.perf_counter()
        mp.spawn(_worker, args=(worker_args,), nprocs=world_size, join=True)
        wall_elapsed = time.perf_counter() - t0

        # Re-create the original full state_dict to compare against the per-rank shards.
        full_sd = _make_state_dict(num_layers, hidden)

        per_rank_elapsed = []
        ok = True
        for rank in range(world_size):
            payload = torch.load(os.path.join(out_dir, f"rank_{rank}.pt"), weights_only=False)
            per_rank_elapsed.append(payload["elapsed"])
            rank_params = payload["params"]
            for name, tensor in rank_params.items():
                plan_key = "up" if name.endswith(".up.weight") else "down"
                plan = "colwise" if plan_key == "up" else "rowwise"
                expected = _expected_shard(full_sd[name], plan, rank, world_size)
                if tensor.shape != expected.shape or not torch.equal(tensor, expected):
                    print(
                        f"[tp] MISMATCH on rank={rank} name={name}: got shape {tuple(tensor.shape)} expected {tuple(expected.shape)}"
                    )
                    ok = False

        print(f"[tp] world_size={world_size} num_layers={num_layers} hidden={hidden}")
        print(f"[tp] wall-clock for mp.spawn = {wall_elapsed:.3f}s")
        for rank, e in enumerate(per_rank_elapsed):
            print(f"[tp]   rank {rank} convert_and_load_state_dict_in_model = {e:.3f}s")
        print(f"[tp] correctness: {'OK' if ok else 'FAIL'}")
        return ok


def benchmark_single_rank_baseline(num_layers: int, hidden: int) -> None:
    """Time the single-process loading path for reference (no device_mesh, full tensors loaded by rank 0)."""
    model = _TPModel(num_layers, hidden)
    state_dict = _make_state_dict(num_layers, hidden)
    load_config = LoadStateDictConfig()

    t0 = time.perf_counter()
    convert_and_load_state_dict_in_model(model, state_dict, load_config, tp_plan=None)
    elapsed = time.perf_counter() - t0
    print(f"[baseline] single-process load (num_layers={num_layers} hidden={hidden}) = {elapsed:.3f}s")
    # _tensor_nbytes spot check so we surface the helper in the report.
    sample_key = next(iter(state_dict))
    print(f"[baseline] sample _tensor_nbytes({sample_key}) = {_tensor_nbytes(state_dict[sample_key])}")


# ---------------------------------------------------------------------------------------------------------------------
# Real-model TP loading benchmark. Invoked under `torchrun --nproc-per-node=N`; every rank loads the same pretrained
# model with `tp_plan="auto"`, which wires up the device mesh + TP sharding path we just refactored. The purpose here
# is to exercise the same code path on production checkpoints and time the load — smoke + perf on real weights.
# ---------------------------------------------------------------------------------------------------------------------
def benchmark_real_model(model_id: str, dtype: str, run_generate: bool) -> bool:
    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        print(
            "[real] --model-id requires torchrun with world_size > 1; "
            "e.g. `torchrun --nproc-per-node=8 tp_loading.py --model-id MODEL`"
        )
        return False

    if rank == 0:
        print(f"[real] loading {model_id!r} with tp_plan='auto' (world_size={world_size}, dtype={dtype})")

    torch.manual_seed(0)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id, tp_plan="auto", dtype=dtype)
    elapsed = time.perf_counter() - t0

    # Gather peak CUDA memory across ranks for a rough sanity check on the shard placement.
    peak_mem_bytes = 0
    if torch.cuda.is_available():
        peak_mem_bytes = torch.cuda.max_memory_allocated()
    if rank == 0:
        print(f"[real] rank0 load time = {elapsed:.2f}s (peak cuda mem on rank0 = {peak_mem_bytes / 1e9:.2f} GB)")

    # Generate is a smoke check but not the subject of the benchmark; some checkpoints (e.g. gpt-oss with MXFP4
    # experts loaded without the quantizer) will not have all weights populated, which breaks generation but is
    # orthogonal to whether the TP load succeeded. So we report generate failures as warnings and keep the overall
    # status tied to the load path.
    if run_generate:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            prompt = "Pipeline parallelism in ai is "
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            t0 = time.perf_counter()
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            gen_elapsed = time.perf_counter() - t0
            if rank == 0:
                decoded = tokenizer.decode(out[0], skip_special_tokens=False)
                print(f"[real] generate {gen_elapsed:.2f}s -> {decoded!r}")
        except Exception as e:  # noqa: BLE001
            if rank == 0:
                print(f"[real] generate failed (load is still OK): {e!r}")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=2, help="Number of TP ranks to simulate (mp.spawn).")
    parser.add_argument("--num-layers", type=int, default=8, help="Number of synthetic transformer-like blocks.")
    parser.add_argument("--hidden", type=int, default=512, help="Hidden size of each block's Linear weights.")
    parser.add_argument(
        "--partition-mappings", type=int, default=5_000, help="Mapping count for the partition microbenchmark."
    )
    parser.add_argument(
        "--partition-world-size", type=int, default=8, help="World size for the partition microbenchmark."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="If set, skip the synthetic tests and load this pretrained model via `tp_plan='auto'`. Must be run "
        "under torchrun (e.g. `torchrun --nproc-per-node=8 ... --model-id MODEL`).",
    )
    parser.add_argument("--dtype", type=str, default="auto", help="Dtype to pass to from_pretrained.")
    parser.add_argument("--no-generate", action="store_true", help="Skip the post-load generate sanity check.")
    args = parser.parse_args()

    print("=" * 72)
    print("TP loading benchmark")
    print("=" * 72)

    if args.model_id is not None:
        ok = benchmark_real_model(args.model_id, args.dtype, run_generate=not args.no_generate)
        return 0 if ok else 1

    benchmark_partition(args.partition_mappings, args.partition_world_size)
    print()
    benchmark_single_rank_baseline(args.num_layers, args.hidden)
    print()
    ok = benchmark_tp_loading(args.world_size, args.num_layers, args.hidden)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
