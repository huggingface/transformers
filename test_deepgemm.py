"""Smoke-test the three DeepGEMM experts dispatches with synthetic experts.

Each test builds a synthetic experts module with the right weight dtypes / SF formats and
runs the kernel forward, checking the output is finite and shaped correctly.

Coverage:
  1. DSv3-style: FP8 weights (`float8_e4m3fn`) + float32 SF — Hopper SM90+
  2. DSv4-style: FP4 weights (`int8`-packed e2m1) + UE8M0 SF — Blackwell SM100+
  3. Mega MoE:   same as DSv4 but with EP dispatch + combine inside the kernel — SM100+
                 + distributed (uses `transform_weights_for_mega_moe` for the layout)

Usage:
    # Single GPU (DSv3 + DSv4):
    python test_deepgemm_integration.py

    # Mega MoE (≥2 ranks):
    torchrun --nproc_per_node=2 test_deepgemm_integration.py
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from transformers.integrations.deepgemm import (
    _load_deepgemm_kernel,
    deepgemm_fp8_fp4_experts_forward,
    deepgemm_fp8_fp4_megamoe_experts_forward,
)


_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max
_UE8M0_SF_DTYPE = torch.float8_e8m0fnu


def _round_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Round a positive float tensor to the nearest power of 2 representable as UE8M0."""
    return torch.pow(2.0, torch.ceil(torch.log2(x.clamp(min=torch.finfo(torch.float32).tiny)))).to(_UE8M0_SF_DTYPE)


def _make_fp8_experts(num_experts: int, hidden_size: int, intermediate_size: int, ue8m0_sf: bool, device: torch.device) -> SimpleNamespace:
    """Synthetic FP8 experts.

    DeepGEMM picks the SF recipe per-arch based on the SF dtype (see
    `get_default_recipe` in `csrc/utils/layout.hpp`):

      * SM90 + float SF       → recipe (1, 128, 128): block-quantized SF for B,
                                shape `(E, N/128, K/128)`.
      * SM100 + float SF      → recipe (1, 128, 128): same block-quantized
                                shape; kernel broadcasts → packs UE8M0
                                internally (DSv3 path, "legacy" on Blackwell).
      * SM100 + UE8M0 SF      → recipe (1, 1, 128): per-row SF for B, shape
                                `(E, N, K/128)`. This is the DSv4-FP8 path.

    `ue8m0_sf=False` exercises the float-SF (DSv3) path; `ue8m0_sf=True`
    exercises the per-row UE8M0 (DSv4-FP8) path.
    """
    block_k = 128
    # Per-row when UE8M0 (gran_mn=1), block-128 when float SF (gran_mn=128).
    block_n = 1 if ue8m0_sf else 128

    def _alloc(e: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Random bf16 → clamp to FP8 range → cast. Values are nonsense but byte-valid.
        w_fp32 = (torch.randn(e, n, k, device=device) * 0.1).clamp(-_FP8_MAX, _FP8_MAX)
        w_fp8 = w_fp32.to(_FP8_DTYPE)
        sf_n = -(-n // block_n)  # ceil-div
        sf_k = -(-k // block_k)
        sf = (torch.rand(e, sf_n, sf_k, device=device) * 0.05 + 0.001).to(torch.float32)
        if ue8m0_sf:
            sf = _round_to_ue8m0(sf)
        return w_fp8, sf

    gate_up, gate_up_sf = _alloc(num_experts, 2 * intermediate_size, hidden_size)
    down, down_sf = _alloc(num_experts, hidden_size, intermediate_size)
    return SimpleNamespace(
        num_experts=num_experts,
        has_gate=True,
        has_bias=False,
        is_transposed=False,
        # block_size matches the actual SF granularity:
        #   (128, 128) for the DSv3 (float-SF) block-quantized path,
        #   (1, 128)   for the DSv4-FP8 (UE8M0-SF) per-row path.
        block_size=(block_n, block_k),
        activation_scheme="dynamic",
        config=SimpleNamespace(hidden_act="silu"),
        gate_up_proj=gate_up,
        gate_up_proj_scale_inv=gate_up_sf,
        down_proj=down,
        down_proj_scale_inv=down_sf,
        _apply_gate=lambda x: F.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1],
        act_fn=F.silu,
    )


def _make_fp4_experts(num_experts: int, hidden_size: int, intermediate_size: int, device: torch.device) -> SimpleNamespace:
    """Synthetic FP4 experts (`int8`-packed e2m1, K dim halved; UE8M0 SF, gran_k=32)."""

    def _alloc(e: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Any int8 byte pattern is a valid FP4-packed (2 e2m1 nibbles per byte).
        w = torch.randint(low=-128, high=128, size=(e, n, k // 2), dtype=torch.int8, device=device)
        # Random positive scales → round to UE8M0 (any e8m0 byte is a power-of-2 or special).
        sf = (torch.rand(e, n, k // 32, device=device) * 0.05 + 0.001).to(torch.float32)
        sf = _round_to_ue8m0(sf)
        return w, sf

    gate_up, gate_up_sf = _alloc(num_experts, 2 * intermediate_size, hidden_size)
    down, down_sf = _alloc(num_experts, hidden_size, intermediate_size)
    return SimpleNamespace(
        num_experts=num_experts,
        has_gate=True,
        has_bias=False,
        is_transposed=False,
        block_size=None,  # FP4 ignores block_size — kernel infers SF granularity from dtype.
        activation_scheme="dynamic",
        config=SimpleNamespace(hidden_act="silu"),
        gate_up_proj=gate_up,
        gate_up_proj_scale_inv=gate_up_sf,
        down_proj=down,
        down_proj_scale_inv=down_sf,
        _apply_gate=lambda x: F.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1],
        act_fn=F.silu,
    )


def _random_routing(num_tokens: int, top_k: int, num_experts: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
    w = torch.rand(num_tokens, top_k, dtype=torch.float32, device=device)
    return idx, w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def _check_output(out: torch.Tensor, expected_shape: tuple[int, ...], label: str) -> None:
    assert out.shape == expected_shape, f"[{label}] shape mismatch: {tuple(out.shape)} vs {expected_shape}"
    assert torch.isfinite(out).all(), f"[{label}] output has non-finite values"
    print(f"[{label}] PASS  out: {tuple(out.shape)} dtype={out.dtype}")


# ── Tests ────────────────────────────────────────────────────────────────────────


def test_dsv3_fp8(device: torch.device) -> None:
    label = "DSv3 (FP8 + float SF)"
    if torch.cuda.get_device_capability(device)[0] < 9:
        print(f"[{label}] SKIP: needs SM90+ (Hopper)")
        return
    T, H, I, E, K = 256, 1024, 512, 16, 4
    experts = _make_fp8_experts(E, H, I, ue8m0_sf=False, device=device)
    hidden = torch.randn(T, H, dtype=torch.bfloat16, device=device) * 0.1
    idx, w = _random_routing(T, K, E, device)
    out = deepgemm_fp8_fp4_experts_forward(experts, hidden, idx, w.to(torch.bfloat16))
    _check_output(out, (T, H), label)


def test_dsv4_fp8(device: torch.device) -> None:
    label = "DSv4-FP8 (FP8 + UE8M0 SF)"
    if torch.cuda.get_device_capability(device)[0] < 10:
        print(f"[{label}] SKIP: needs SM100+ (Blackwell) for UE8M0 SF dispatch")
        return
    T, H, I, E, K = 256, 1024, 512, 16, 4
    experts = _make_fp8_experts(E, H, I, ue8m0_sf=True, device=device)
    hidden = torch.randn(T, H, dtype=torch.bfloat16, device=device) * 0.1
    idx, w = _random_routing(T, K, E, device)
    out = deepgemm_fp8_fp4_experts_forward(experts, hidden, idx, w.to(torch.bfloat16))
    _check_output(out, (T, H), label)


def test_dsv4_fp4(device: torch.device) -> None:
    label = "DSv4 (FP4 + UE8M0 SF)"
    if torch.cuda.get_device_capability(device)[0] < 10:
        print(f"[{label}] SKIP: needs SM100+ (Blackwell)")
        return
    T, H, I, E, K = 256, 1024, 512, 16, 4
    experts = _make_fp4_experts(E, H, I, device)
    hidden = torch.randn(T, H, dtype=torch.bfloat16, device=device) * 0.1
    idx, w = _random_routing(T, K, E, device)
    out = deepgemm_fp8_fp4_experts_forward(experts, hidden, idx, w.to(torch.bfloat16))
    _check_output(out, (T, H), label)


def test_megamoe(device: torch.device, world_size: int, rank: int) -> None:
    label = "Mega MoE (FP8 act × FP4 weight, fused EP)"
    if torch.cuda.get_device_capability(device)[0] < 10:
        if rank == 0:
            print(f"[{label}] SKIP: needs SM100+ (Blackwell)")
        return
    if world_size < 2:
        if rank == 0:
            print(f"[{label}] SKIP: needs >=2 ranks (run with `torchrun --nproc_per_node=2`)")
        return

    deepgemm = _load_deepgemm_kernel()
    T_local, H, I, K = 64, 1024, 512, 4
    E_global = 16
    E_local = E_global // world_size

    # Build raw FP4 experts on this rank's slice, then transform to the kernel's layout.
    raw = _make_fp4_experts(E_local, H, I, device)
    gate_up_t, gate_up_sf_t = deepgemm.transform_weights_for_mega_moe(
        raw.gate_up_proj, raw.gate_up_proj_scale_inv, is_l1=True
    )
    down_t, down_sf_t = deepgemm.transform_weights_for_mega_moe(
        raw.down_proj, raw.down_proj_scale_inv, is_l1=False
    )

    experts = SimpleNamespace(
        gate_up_proj=gate_up_t,
        gate_up_proj_scale_inv=gate_up_sf_t,
        down_proj=down_t,
        down_proj_scale_inv=down_sf_t,
        symm_buffer=None,  # lazily allocated on first call
        config=SimpleNamespace(),  # no swiglu_limit → kernel runs unclamped
    )

    hidden = torch.randn(T_local, H, dtype=torch.bfloat16, device=device) * 0.1
    # Mega MoE expects GLOBAL expert ids (no per-rank remap); -1 marks skipped slots.
    idx = torch.randint(0, E_global, (T_local, K), dtype=torch.int32, device=device)
    w = torch.rand(T_local, K, dtype=torch.float32, device=device)
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    out = deepgemm_fp8_fp4_megamoe_experts_forward(
        experts, hidden, idx, w.to(torch.bfloat16), process_group=dist.group.WORLD
    )
    if rank == 0:
        _check_output(out, (T_local, H), label)


# ── Entrypoint ───────────────────────────────────────────────────────────────────


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")

    if rank == 0:
        print(f"device cap: SM{''.join(str(x) for x in torch.cuda.get_device_capability(device))}, "
              f"world_size={world_size}\n")

    # Single-GPU paths run on rank 0 only (ranks > 0 only participate in Mega MoE).
    failures: list[tuple[str, BaseException]] = []
    if rank == 0:
        for fn in (test_dsv3_fp8, test_dsv4_fp8, test_dsv4_fp4):
            try:
                fn(device)
            except BaseException as exc:
                failures.append((fn.__name__, exc))
                print(f"[{fn.__name__}] FAIL — {type(exc).__name__}: {exc}")

    if world_size > 1:
        dist.barrier()
        try:
            test_megamoe(device, world_size, rank)
        except BaseException as exc:
            if rank == 0:
                failures.append(("test_megamoe", exc))
                print(f"[test_megamoe] FAIL — {type(exc).__name__}: {exc}")
        dist.destroy_process_group()

    if rank == 0:
        if failures:
            print(f"\n=== {len(failures)} test(s) failed ===")
            for name, exc in failures:
                print(f"  - {name}: {type(exc).__name__}: {exc}")
            raise SystemExit(1)
        print("\n=== all tests passed ===")


if __name__ == "__main__":
    main()
