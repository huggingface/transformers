# GGUF inference benches (Apple Silicon)

Standalone scripts used during the GgufLinear → llama.cpp parity work. They
are not run by CI — drop them into your test process manually when you want
to reproduce a number from PR #45977.

All scripts assume:

- Apple Silicon (MPS) + `kernels` installed + `gguf` installed.
- `TRANSFORMERS_GGUF_METAL_KERNELS_SO=<path-to-_gguf_dequant_*.abi3.so>`
  if you want to test a locally-built kernel build instead of the Hub one
  (`ArthurZ/gguf-kernels`).
- A cached GGUF model. Examples use Qwen1.5-MoE-A2.7B Q4_K_M
  (`gdax/Qwen1.5-MoE-A2.7B_gguf` on the Hub) and pull the matching
  tokenizer from `Qwen/Qwen1.5-MoE-A2.7B`.

## End-to-end benches

| Script | What it measures |
|---|---|
| `bench_generate.py` | `model.generate()` under `torch.compile`, vs gguf_bmm vs eager-per-expert MoE paths. Use the `--gguf` / `--repo` / `--n` / `--compile` flags. |
| `bench_fast_greedy_decode.py` | The `fast_greedy_decode` helper from `integrations.gguf_linear` (matches llama.cpp's `tg64` throughput on M3 Max). |
| `bench_generate_batch_gguf.py` | `generate_batch` (continuous batching) on a GGUF model. Accepts `<max_new_tokens> <attn_impl>` positional args. |
| `bench_generate_batch_reference.py` | `generate_batch` on the PR #45974 reference model (Qwen2.5-0.5B-Instruct fp16) to validate the `kernels-community/metal-flash-sdpa` wiring independently. |

## Micro-benchmarks

| Script | What it measures |
|---|---|
| `microbench_mul_mat_vec_q4_K.py` | Pure-GPU steady-state cost of `mul_mat_vec_q4_K_f32` at Qwen attention shape (22 µs / call). |
| `microbench_multi_matvec.py` | `gguf_mul_mat_vec_multi_f32` vs 3 separate matvec calls (39 % win on QKV). |
| `microbench_kv_update.py` | `Cache.update` cost (raw indexed assignment vs the full Python wrapper). |

## Correctness checks

| Script | What it checks |
|---|---|
| `correctness_decode_op.py` | `gguf_moe_decode_f32` end-to-end logits match the per-projection fallback to fp32 precision (max diff ~ 8e-6 on Qwen1.5-MoE-A2.7B Q4_K_M). |
