# Inkling-2-Mini decode ceilings (M3 Max 30-core, 96 GB unified)

Measured bandwidth: ~290 GB/s (lm_head matvec reached 294-300 GB/s; `x.sum` 285 GB/s). Compute: not the limit at batch 1.

Bytes per decoded token (48 layer evaluations: layers 0-5, 6-17 x 3 loop passes, 18-23):

| component | exact | opt-in 4-bit dense |
|---|---:|---:|
| experts, native (44 NVFP4 evals x 35.8 MB + 4 MXFP8 x 65.7 MB) | 1.84 GB | 1.84 GB |
| dense (attention, gates, convs, message gates, fp32 router) | 2.80 GB | 0.92 GB |
| lm_head (201024 x 2304) | 0.93 GB | 0.26 GB |
| KV read at 1K context | ~0.2 GB | ~0.2 GB |
| **decode ceiling (ceiling.py)** | **52 / 50 / 46 tok/s** at 0 / 1K / 4K | **95 / 90 / 76 tok/s** |

Speculation (bundled DSpark drafter, 7 drafts, measured acceptance 2.4 incl. bonus): 39 tok/s exact, 46 tok/s 4-bit dense,
i.e. below plain decode: verifying 8 tokens reads the union of their experts (~3-5x the expert bytes).
A target of 150 tok/s is above every ceiling of this checkpoint on this machine without new lossy trades.
Inputs: ceiling-exact.json, ceiling-dense4.json (`python3 ~/.claude/skills/inference-optimisation/scripts/ceiling.py <json>`).
