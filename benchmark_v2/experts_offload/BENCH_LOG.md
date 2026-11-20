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
