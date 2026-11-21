# Local Metal decode benchmark log

Goal: >= 60 tok/s sustained batch-1 decode, quantized, Apple Metal (M3 Max 30-core GPU, 96 GB).

Protocol (unless noted): `bench_experts_offload.py` (generate + streamer timing, no hand loop), greedy, batch 1,
4 different chat prompts x 128 new tokens each, median ms/token over all decode steps -> tok/s.
"Sustained" = a 1024-token generation's tok/s over the last 512 tokens (`--sustained`).
Prefill reported separately (1024-token prompt, one forward).

Model: Qwen3-30B-A3B (48 layers, 128 experts, top-8), experts Metal affine 4-bit g64 (`ArthurZ/Qwen3-30B-A3B-metal-experts-4bit`,
transformers format), dense layers bf16 unless noted.

| # | Date | Engine / config | Decode tok/s | Sustained tok/s | Prefill tok/s | Notes |
|---|---|---|---|---|---|---|
| 1 | 2026-11-20 | transformers eager, MPS, 4-bit Metal experts via offload store at cache 1.0, sdpa (`profile_offload.py ArthurZ/Qwen3-30B-A3B-metal-experts-4bit 1.0 sdpa`) | 7.17 | - | - | 139 ms/token: GPU drain ~88 ms (~1.6 ms/layer, ~10x over bandwidth), host ~54 ms, read waits 0.6 ms |
| 2 | 2026-11-20 | llama.cpp (brew, Metal) Qwen3-30B-A3B Q4_K_M, `llama-bench -p 512 -n 128,1024 -fa 1 -r 3 -ngl 99` | 53.10 (tg128) | 49.94 (tg1024) | 608 (pp512) | engine reference on this Mac |
| 3 | 2026-11-20 | transformers eager, MPS, 4-bit Metal experts, all on device (`device_map="mps"`, no offload), sdpa (`bench_experts_offload.py --configs full --new-tokens 64 --prompt-tokens 512`) | 31.32 | - | 372 | offload layer at 100% residency costs 4x (7.17): skip it when everything fits |
| 4 | 2026-11-20 | #3 + every nn.Linear but the router -> Metal affine 8-bit g64 (`--quantize-dense 8`) | 54.58 | - | 772 | dense bytes/token 2.4 -> 1.2 GB; ppl 2.2486 vs bf16-dense 2.2607 (quality kept) |
| 5 | 2026-11-20 | #3 + every nn.Linear but the router -> Metal affine 4-bit g64 (`--quantize-dense 4`) | 61.69 (62.13 rerun) | 56.80 | 787 (891 at 1024) | ppl 2.3651 (+4.6% vs 2.2607): quality NOT kept; decode slows with context (sdpa) |
| 6 | 2026-11-20 | #5 with metal-flash-sdpa attention (`--attn kernels-community/metal-flash-sdpa`) | 60.73 | 57.15 | 892 | no sustained gain over sdpa (56.80) |
| 7 | 2026-11-20 | always-on 4-bit g32, min-max (`--quantize-dense 4g32`) | 60.78 | 56.20 | 893 | ppl 2.2939 vs 2.2459 (+2.1%) |
| 8 | 2026-11-20 | quality only (`quality_check.py ... 4c 4g32c`): per-group MSE clipping search | - | - | - | ppl: 4c 2.3043 (+1.8%), **4g32c 2.2759 (+0.5%)** vs bf16 always-on 2.2642; same bytes as 4g32 |
| 9 | 2026-11-20 | #7 with always-on 4g32c + `generate(cache_implementation="static")` + torch.compile on MPS (`full+compile`) | ~~35.42~~ | ~~33.99~~ | ~~371~~ | INVALID: ran while mlx-lm install/downloads saturated the CPU; and generate never compiles on MPS (0 frames compiled, see #10) |
| 10 | 2026-11-20 | `diag_compile_mps.py` (4g32c): eager dynamic / eager static / static + generate's compile | 60.88 / 60.55 / 60.53 | - | - | generate's auto-compile is CUDA-only: 0 frames compiled on MPS; static cache costs nothing |
| 11 | 2026-11-20 | `diag_compile_mps.py`: explicit `torch.compile(model.forward)` + static cache, before fake ops | 60.52 (default) / 62.86 (max-autotune-no-cudagraphs) | - | - | 61 graph breaks/step: hub Metal ops (`gather_qmm`, `quantized_matmul`) had no fake impl |
| 12 | 2026-11-20 | + fake impls for the Metal ops (`_register_fake_kernels`): 0 graph breaks, 2 graphs | 88.30 (median step) | - | - | eager same run 57.75 |
| **13** | 2026-11-20 | **public API** `verify_generate.py`: 4-bit experts + always-on 4g32c, `generate(cache_implementation="static", compile_config=CompileConfig() w/ _compile_all_devices=True)`, Metal quantizer `is_compileable=True` | **86.17** (mean, 4 prompts x 128) | **79.47** (mean, tokens 512-1024) | - | greedy == eager 50/128 (bf16 fusion numerics); ppl compiled 2.2747 vs eager 2.2799 (no loss) |

**Qwen3-30B-A3B (stepping stone, not the goal):** 79.47 tok/s sustained on Metal, quantized, through `generate`, quality kept (ppl +0.5% vs bf16 always-on, compile -0.2%). **The goal is Qwen3.8-Flash-Next: not met yet.**
Reproduce: `PYTHONPATH=src python verify_generate.py ArthurZ/Qwen3-30B-A3B-metal-experts-4bit 4g32c` (M3 Max 30-core GPU, 96 GB, torch 2.15.0, kernels 0.18.2).
| 14 | 2026-11-20 | regression check after the NaN fix: MetalExperts `torch.where` sentinel mask | 54.42 | 49.59 | - | where() in MetalExperts broke compiled decode (152 extra aten::copy_/token); reverted to the weight product, NaN-safety kept by zero-initialized pools |
| 15 | 2026-11-20 | #13 config, current code (`verify_generate.py ... 4g32c`) | **88.26** | **81.98** | - | Qwen3-30B-A3B stepping stone; greedy == eager 27/128 (bf16 fusion numerics) |
| 16 | 2026-11-20 | `bench_generate_overhead.py`: StaticCache `cumulative_length` created on the device instead of `.to(device)` (was a sync per layer on MPS) | step 12.27 -> 11.67 ms | - | - | per-generate overhead 70.9 -> 53.4 ms (10-token prompt, 1 new token) |

## Qwen3.8-Flash-Next on M3 Max (synthetic weights at the real dims, `bench_qwen38_synthetic.py`)

Real config (48 layers: 36 gated-delta-net + 12 full attention with QSA indexer, hidden 2560, top-10 routing, vocab
248k), random weights, 64 experts instead of 512 (same work per token), 4-bit experts g64 + 4-bit dense g32, static
cache + compile, 256 greedy tokens. Weights generated once to `~/.cache/qwen38-synthetic-64e-all4bit.safetensors`.

| # | Date | Change | Decode tok/s (median) | ms/token | Notes |
|---|---|---|---|---|---|
| Q1 | 2026-11-20 | first run (scales float32: per-call cast of the expert scale banks) | 20.0 | 50.9 | build artefact |
| Q2 | 2026-11-20 | bf16 scales like a loaded checkpoint | 29.6 | 34.5 | GPU busy 99.8% (Metal System Trace) |
| Q3 | 2026-11-20 | thin linears (min dim <= 640) kept bf16 | 30.3 | 33.7 | +1.2 GB/token bf16 offsets the slow MLX thin qmv |
| Q4 | 2026-11-20 | all dense 4-bit + own batch-1 `qmv4` for out_features <= 1024 (`integrations/metal_qmv.py`) | **39.2** | 26.1 | MLX qmv: 4x10240 in 51-105 us |
| Q5 | 2026-11-20 | own gather qmv for experts (scalar, then 4-row vectorized) | 37.3 / 33.4 | | reverted: MLX `gather_qmm` faster; 4-row kernel starves thin matrices |

Stub ablations of Q4 (`diag_qwen38_ablate.py`, 25.0 ms/token): experts 7.97 ms (1.33 GB, 167 GB/s), linear_attn proj
6.67 ms (1.32 GB, 198 GB/s), hyper-connections 2.93 ms (0.40 GB, 137 GB/s), self_attn proj+indexer 2.85 ms (0.46 GB),
lm_head 1.77 ms (0.40 GB, 225 GB/s), shared expert 1.22 ms (0.15 GB); all matmuls 21.9 ms for 4.06 GB, the rest 3.1 ms.
Measured bandwidth: 250 GB/s read (`x.sum()`), so 4-bit single-token decode tops out near 4.06 GB / 250 GB/s + 3.1 ms
= 19.3 ms = **52 tok/s**. 100 tok/s needs <= ~1.7 GB/token, or several tokens per forward (speculative / MTP).
| Q6 | 2026-11-21 | vectorized `qmv4` (uint4 + float4) for thin linears; same for experts | 38.1 / 38.4 | | noise: thin matrices are latency-bound, MLX `gather_qmm` already at par |
| Q7 | 2026-11-21 | fuse sibling projections sharing an input (180 groups: linear_attn qkv+z+b+a, hyper down+inject, shared gate+up) | 39.6 | 25.7 | noise; stubbing all 216 tiny dispatches only saves 0.94 ms |
| Q8 | 2026-11-21 | speculative-verify proxy: decode step over a batch of k | k=4: 58.3 ms, k=8: 90.3 ms | | expert union grows with k: 100 tok/s would need 5.8 (k=4) / 9.0 (k=8) accepted per step |
| Q9 | 2026-11-21 | experts decode: MLX gate/up + own down projection with the routing sum in one kernel (`experts_down_sum`) | 41.2 | 24.7 | MLX down at 176 GB/s (short rows), ours 217 GB/s |
| Q10 | 2026-11-21 | short-side batch-1 `qmv4` (16/32 lanes per row, uint4 loads) for linears with a side <= 1024 | 42.5 | 24.0 | hyper up 22.3 -> 13.5 us, inject 42.2 -> 7.4 us |
| Q11 | 2026-11-21 | `fuse_sibling_linears`: projections sharing an input as one `MetalLinear` (192 groups) | 43.2 | 23.6 | small dispatches are latency-bound |
| Q12 | 2026-11-21 | gated delta rule advances the cache's float32 state in place (no clone, no copy back) | 44.4 | 22.5 | 12 MB/layer of state traffic gone |
| Q13 | 2026-11-21 | + attention q/k/v/indexer fusion, 4 threads per state column, casts outside the custom ops | **45.4** (256 tok), 45.0 (512 tok) | 22.4 | greedy == unfused on a tiny model |

Stub ablations of Q10 (24.15 ms): experts 6.81 ms, linear_attn proj 6.64, hyper-connections 3.08, self_attn+indexer 2.91,
lm_head 1.82, shared expert 1.14, rest 3.46.
| Q14 | 2026-11-21 | bf16 recurrent state (float32 compute, one rounding per token: 5.5e-3 rel. error, flat over 2048 tokens) | 46.6 | 21.8 | half the state traffic |
| Q15 | 2026-11-21 | every batch-1 4-bit linear through `qmv4` (large ones too) | **48.1** | 21.2 | MLX at par alone, slower in the model |
| Q16 | 2026-11-21 | tried: eager attention instead of sdpa / own gathered gate-up / 64-lane expert down / fused hyper-connection kernels | 42.3 / 46.9 / 48.1 / 47.6 | | kept only the 64-lane down kernel (239 vs 192 GB/s alone, neutral in the model) |

Fine stub ablation of Q15 (`diag_qwen38_ablate_fine.py`, 20.80 ms): linear_attn in_proj 3.56 (at peak), experts gate/up 3.66
(241 GB/s), experts down ~2.95 (150 GB/s; its stub also drops gate/up: 6.61 measured), linear_attn out_proj 1.77, lm_head 1.60
(250 GB/s), hyper up 1.58, hyper down+inject 1.47 (130 GB/s), attention qkv+indexer 1.27, sdpa 0.98, shared expert 0.63+0.49,
gated delta rule 0.53, conv 0.23. Per-dispatch Metal attribution is not available here: GPU counters are a time series
(default set: RT unit) and `torch.mps.profiler(wait_until_completed)` does not sync custom / hub Metal kernels.
