# Inkling-2-Mini decode hill-climb (MPS, transformers generate, static cache, torch.compile)

Targets / constraints: c1 decode tok/s; keep quality (exact checkpoint numerics; the 4-bit dense path is an opt-in,
labelled lossy trade); context must stay stable past the 1024-token sliding window. Memory: 96 GB unified, model ~10 GB.

## Log (before this file; single runs, GPU contention noted, so treat small deltas as noise)
| step | tok/s | verdict |
|---|---:|---|
| PR #51050 eager, bf16-dequantized experts | (46 GiB) | baseline |
| native NVFP4/MXFP8 experts (#51062), exact dense | 29.6 | kept: 9.9 vs 46.2 GiB, logits equal |
| + Metal short convs (hub routing) | 29.9 | kept, ~noise |
| + fused decode attention | 30.2 | kept |
| + fused block attention residual | 32.9 | kept |
| + fused message gate | 34.2 | kept (exact config best) |
| opt-in 4-bit dense + lm_head (bench hack) | 48.3 | kept as labelled lossy option |
| grouped 6 attention projections (one qmv4) | 48.0 | REJECTED: inside noise, dispatches were not the cost |
| inductor asserts off | 28.4 vs 29.9 | REJECTED: host time overlaps GPU here |
| cpp_wrapper | n/a | REJECTED: 437k-line C++ never finished compiling (~50 min) |
| plan path: stored experts + {"*": MetalConfig(4)} | 32.1 -> 48.6 | fixed MetalConfig.is_compileable (compile was silently off) |
| + optimized fp4 expert kernels | (in 48.6) | kept |
| + fused router | 37.8 (GPU 88% busy) | kept (kernel 33 vs 62 us), number void |
| + faster attn_res / 1-dispatch message gate / fused short-conv step | 55.0 | kept |
| + fused attention inputs | 55.4 (GPU 99% busy) | kept, needs a clean A/B |
| 900-token generation | 56.3 | stable |
| DSpark speculation (7 drafts) | 18.2 vs 33.8 greedy | REJECTED: acc 2.4 incl. bonus, T=8 verify ~4x T=1 |
| >1024-token static generation | crash | FIXED: hybrid_sliding mask mapping (bug since Inkling #47347) |

## Experiments (from here: control / candidate / control, >= 6 samples, medians, AC + idle GPU)

### 2026-11-22 baselines (AC, GPU idle, 6 samples x 128 tokens, medians)
| run | tok/s |
|---|---:|
| base-4bit | 55.54 |
| base-exact | 38.13 |
| base-4bit | 55.75 |
Run-to-run noise ~0.4%. Exact = 73% of its 52 tok/s ceiling, 4-bit = 59% of 95.
Also fixed: safetensors MPS dlpack OnceLock/GIL deadlock (upstream since #767) that hung threaded exact loads.

### E1 HC_GROUP: q / gate_attn / relative-states / sink projections as one matmul (control-candidate-control, both paths)
- attempt 1 VOID: base-4bit 55.70, grp-4bit 27.00, base-4bit 24.53 (control halved too: environment changed mid-queue), then
  the queue died silently in run 4 (exact control, after compile). No thermal/power warning recorded. Rerun.
- attempt 2: base-4bit 31.88 | grp-4bit 41.09 | base-4bit 41.48 ; base-exact 28.65 | grp-exact 27.86 | base-exact 25.41.
  Controls far below the morning baselines (55.6 / 38.1) and swinging 30%: environment degraded (no thermal/power flag;
  the GPU busy check before each run passed). VERDICT: REJECTED (does not beat both neighbouring controls on either
  path; also rejected once before). The rest of the hill-climb waits for a stable machine (baseline A/A within ~2%).

### Ablation of the current best (4-bit plan, stable machine: A/A 55.82 / 55.63)
| removed | ms/token saved (of 17.74) |
|---|---:|
| experts | 9.46 |
| 4-bit dense matvecs (~370 qmv4 + lm_head) | 7.37 (floor ~3.2: biggest gap) |
| message gate | 1.46 |
| attention | 0.93 |
| router | 0.79 |
| attention residual | 0.67 |
| short conv | 0.32 |
Next: E1 rerun on a stable machine (grouping is the lever for the latency-bound thin matvecs).
- E1 rerun 1: base 55.42 | grp 57.63 | base 34.62 (VOID: samples decay 38.0 -> 31.5 within the run = thermal throttling
  after ~2 back-to-back runs). Harness now voids runs whose last sample is >5% below the first; queue cools 180 s between runs.
- E1 rerun 2 (180 s cooldown before each run): base 50.89 | grp 57.49 | base 55.71, no run throttled, outputs coherent.
  VERDICT: ACCEPTED (+3.2% over the better control; repeats the earlier clean 57.63). Grouping the 4 projections
  that read hidden_states into one qmv4 cuts the latency-bound thin matvecs (sinks 24 rows, relative states 384).
  Next: productize (fuse at load for MetalLinear), then E2 = the same for the exact bf16 path.
- prod-4bit (fusion at load in the Metal quantizer): 57.50 (matches the accepted 57.49). Best = 57.5 tok/s.

### E2 HC_GROUP on the exact bf16 path (concatenated weights, one F.linear)
