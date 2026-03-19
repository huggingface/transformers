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
"""
Benchmark prefill and decode throughput for `transformers serve`.

Tests:
- pp (prefill): sends a large prompt with max_tokens=1. Measures TTFT ≈ pure prefill time.
  Default sizes: 256, 1024 tokens.
- tg (decode): sends a 512-token prompt (--tg-prefill) and generates many tokens.
  Measures decode throughput after subtracting TTFT. Default sizes: 128, 512 tokens.

Modes:
- bench: greedy decoding (do_sample=False, temp=0). Deterministic, best for reproducible numbers.
- chat: sampling (do_sample=True, temp=0.7). Simulates real chat usage.

Recommended benchmarks:

    # HF model — greedy
    python tests/cli/benchmark_serve.py --model Qwen/Qwen2.5-7B-Instruct --mode bench

    # HF model — sampling (simulates real chat)
    python tests/cli/benchmark_serve.py --model Qwen/Qwen2.5-7B-Instruct --mode chat

    # GGUF model — greedy
    python tests/cli/benchmark_serve.py \\
        --model "Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-fp16-00001-of-00004.gguf --processor Qwen/Qwen2.5-7B-Instruct" \\
        --mode bench

    # GGUF model — sampling
    python tests/cli/benchmark_serve.py \\
        --model "Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-fp16-00001-of-00004.gguf --processor Qwen/Qwen2.5-7B-Instruct" \\
        --mode chat

    # Against an existing server
    python tests/cli/benchmark_serve.py --url http://localhost:8000 --processor Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import os
import statistics
import time

# Force single GPU — must be set before any CUDA initialization
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import requests

from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "Sphinx of black quartz, judge my vow. "
) * 200

_TG_PREFILL_DEFAULT = 512


def make_prompt(tokenizer, num_tokens: int) -> str:
    """Build a prompt string that tokenizes to exactly `num_tokens` tokens."""
    token_ids = tokenizer.encode(_FILLER, add_special_tokens=False)
    if len(token_ids) < num_tokens:
        repeats = (num_tokens // len(token_ids)) + 1
        token_ids = (token_ids * repeats)[:num_tokens]
    else:
        token_ids = token_ids[:num_tokens]
    return tokenizer.decode(token_ids)


def wait_for_server(base_url: str, timeout: int = 120) -> bool:
    """Poll GET /health until 200 or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/health", timeout=2).status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    return False


def streaming_chat_completion(
    base_url: str, messages: list, max_tokens: int, seed: int,
    do_sample: bool = False,
) -> dict:
    """Send a streaming chat completion request. Returns {total, ttft, completion_tokens, text}."""
    gen_cfg = {"max_new_tokens": max_tokens, "min_new_tokens": max_tokens, "do_sample": do_sample}
    if do_sample:
        gen_cfg["temperature"] = 0.7

    payload = {
        "messages": messages,
        "stream": True,
        "seed": seed,
        "generation_config": json.dumps(gen_cfg),
    }

    t_start = time.perf_counter()
    t_first_token = None
    completion_tokens = None
    text_chunks = []

    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, stream=True, timeout=300)
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[len("data: "):]
        if data_str.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices", [])
        if not choices:
            continue

        content = choices[0].get("delta", {}).get("content")
        if content is not None and content != "":
            text_chunks.append(content)
            if t_first_token is None:
                t_first_token = time.perf_counter()

        if chunk.get("usage"):
            completion_tokens = chunk["usage"].get("completion_tokens")

        if choices[0].get("finish_reason") is not None:
            break

    t_end = time.perf_counter()

    return {
        "total": t_end - t_start,
        "ttft": (t_first_token - t_start) if t_first_token else None,
        "completion_tokens": completion_tokens,
        "text": "".join(text_chunks),
    }


def streaming_response(
    base_url: str, messages: list, max_tokens: int, seed: int,
    do_sample: bool = False,
) -> dict:
    """Send a streaming responses API request. Returns {total, ttft, completion_tokens, text}."""
    gen_cfg = {"max_new_tokens": max_tokens, "min_new_tokens": max_tokens, "do_sample": do_sample}
    if do_sample:
        gen_cfg["temperature"] = 0.7

    # Convert messages to Responses API input format
    input_messages = messages
    payload = {
        "input": input_messages,
        "stream": True,
        "seed": seed,
        "generation_config": json.dumps(gen_cfg),
    }

    t_start = time.perf_counter()
    t_first_token = None
    text_chunks = []

    resp = requests.post(f"{base_url}/v1/responses", json=payload, stream=True, timeout=300)
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        try:
            chunk = json.loads(line[len("data: "):])
        except json.JSONDecodeError:
            continue

        etype = chunk.get("type")
        if etype == "response.output_text.delta":
            delta = chunk.get("delta", "")
            if delta:
                text_chunks.append(delta)
                if t_first_token is None:
                    t_first_token = time.perf_counter()
        elif etype == "response.completed":
            break

    t_end = time.perf_counter()
    text = "".join(text_chunks)

    return {
        "total": t_end - t_start,
        "ttft": (t_first_token - t_start) if t_first_token else None,
        "completion_tokens": len(text_chunks),  # approximate — one chunk per streamer push
        "text": text,
    }


def streaming_request(
    base_url: str, messages: list, max_tokens: int, seed: int,
    do_sample: bool = False,
    endpoint: str = "chat",
) -> dict:
    """Dispatch to chat completions or responses API based on endpoint."""
    kw = dict(base_url=base_url, messages=messages, max_tokens=max_tokens,
              seed=seed, do_sample=do_sample)
    if endpoint == "responses":
        return streaming_response(**kw)
    return streaming_chat_completion(**kw)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def bench_pp(
    base_url: str, tokenizer, pp: int, warmup: int, iterations: int, seed: int,
    do_sample: bool = False, endpoint: str = "chat",
) -> dict:
    """Prefill benchmark: large prompt, max_tokens=1. Measures TTFT ≈ pure prefill time."""
    prompt = make_prompt(tokenizer, pp)
    messages = [{"role": "user", "content": prompt}]
    kw = {"do_sample": do_sample, "endpoint": endpoint}

    for _ in range(warmup):
        streaming_request(base_url, messages, max_tokens=1, seed=seed, **kw)

    ttfts = []
    for _ in range(iterations):
        r = streaming_request(base_url, messages, max_tokens=1, seed=seed, **kw)
        if r["ttft"] is not None:
            ttfts.append(r["ttft"])

    ttft = statistics.median(ttfts) if ttfts else None
    tok_s = pp / ttft if ttft and ttft > 0 else None

    return {"test": f"pp{pp}", "tokens": pp, "tok_s": tok_s, "time": ttft}


def bench_tg(
    base_url: str, tokenizer, tg: int, warmup: int, iterations: int, seed: int,
    tg_prefill: int = 512, do_sample: bool = False, endpoint: str = "chat",
) -> dict:
    """Decode benchmark: generate `tg` tokens after a `tg_prefill`-token prompt."""
    prompt = make_prompt(tokenizer, tg_prefill)
    messages = [{"role": "user", "content": prompt}]
    kw = {"do_sample": do_sample, "endpoint": endpoint}

    for _ in range(warmup):
        streaming_request(base_url, messages, max_tokens=tg, seed=seed, **kw)

    decode_times = []
    token_counts = []
    last_text = ""
    for _ in range(iterations):
        r = streaming_request(base_url, messages, max_tokens=tg, seed=seed, **kw)
        if r["ttft"] is not None:
            decode_times.append(r["total"] - r["ttft"])
            token_counts.append(r["completion_tokens"] if r["completion_tokens"] is not None else tg)
            last_text = r["text"]

    if decode_times:
        dt = statistics.median(decode_times)
        toks = statistics.median(token_counts)
        tok_s = toks / dt if dt > 0 else None
    else:
        dt = None
        toks = tg
        tok_s = None

    return {"test": f"tg{tg}", "tokens": int(toks), "tok_s": tok_s, "time": dt, "text": last_text}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def format_duration(seconds) -> str:
    if seconds is None:
        return "N/A"
    if seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.2f}s"


def format_throughput(value) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


_PREVIEW_WIDTH = 120


def truncate_preview(text: str, width: int = _PREVIEW_WIDTH) -> str:
    """Single-line preview of generated text."""
    if not text:
        return ""
    line = text.replace("\n", " ").strip()
    if len(line) > width:
        return line[:width - 1] + "\u2026"
    return line


def print_table(rows: list[dict], title: str = "", reference_texts: dict | None = None, is_reference: bool = False) -> None:
    """Print results in a bordered table.

    Args:
        reference_texts: dict mapping test name (e.g. "tg128") to reference text.
            When provided, decode rows show REF/MATCH/MISMATCH.
        is_reference: if True, this is the reference table — show REF instead of MATCH.
    """
    if not rows:
        return

    has_text = any(row.get("text") for row in rows)
    has_ref = reference_texts is not None and has_text

    headers = ["test", "tokens", "tok/s", "time"]
    align = ["<", ">", ">", ">"]
    if has_ref:
        headers.append("ref")
        align.append("<")
    if has_text:
        headers.append("preview")
        align.append("<")

    formatted_rows = []
    for row in rows:
        cells = [
            row["test"],
            str(row["tokens"]),
            format_throughput(row["tok_s"]),
            format_duration(row["time"]),
        ]
        text = row.get("text", "")
        if has_ref:
            ref_text = reference_texts.get(row["test"])
            if not text:
                cells.append("")
            elif ref_text is None:
                cells.append("")
            elif is_reference:
                cells.append("REF")
            elif text == ref_text:
                cells.append("MATCH")
            else:
                cells.append("MISMATCH")
        if has_text:
            cells.append(truncate_preview(text))
        formatted_rows.append(cells)

    widths = [max(len(h), *(len(r[i]) for r in formatted_rows)) for i, h in enumerate(headers)]

    def pad(text, width, a):
        return text.ljust(width) if a == "<" else text.rjust(width)

    def make_row(cells):
        return "| " + " | ".join(pad(c, w, a) for c, w, a in zip(cells, widths, align)) + " |"

    def make_sep(char="-"):
        return "+" + "+".join(char * (w + 2) for w in widths) + "+"

    print()
    if title:
        print(title)
    print(make_sep("-"))
    print(make_row(headers))
    print(make_sep("="))
    for r in formatted_rows:
        print(make_row(r))
    print(make_sep("-"))
    print()


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------


def start_server(
    model: str, port: int, processor: str | None = None, attn_implementation: str | None = None,
    compile: bool = False,
):
    """Start a transformers serve instance. Returns the Serve object."""
    from transformers.cli.serve_refactored import Serve

    kwargs = {"force_model": model, "port": port, "non_blocking": True, "log_level": "warning"}
    if processor:
        kwargs["processor"] = processor
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if compile:
        kwargs["compile"] = True
    return Serve(**kwargs)


def parse_model_spec(spec: str) -> dict:
    """Parse 'model_id' or 'model_id --processor tokenizer_id'.

    Returns {"model": str, "processor": str | None, "tokenizer": str}
    """
    parts = spec.split()
    model = parts[0]
    processor = None
    for i, p in enumerate(parts):
        if p == "--processor" and i + 1 < len(parts):
            processor = parts[i + 1]
    tokenizer_id = processor if processor else model
    return {"model": model, "processor": processor, "tokenizer": tokenizer_id}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark transformers serve (prefill & decode separately)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python benchmark_serve.py --model Qwen/Qwen2.5-7B-Instruct
  python benchmark_serve.py --model "org/model-GGUF/file.gguf --processor org/model"
  python benchmark_serve.py --url http://localhost:8000 --processor Qwen/Qwen2.5-7B-Instruct
""",
    )
    parser.add_argument("--model", type=str, action="append", dest="models",
                        help="Model spec (repeatable). For GGUF: 'gguf_id --processor tokenizer_id'")
    parser.add_argument("--processor", type=str, default=None,
                        help="Processor/tokenizer ID for --url mode (default: derived from model)")
    parser.add_argument("--port", type=int, default=8642, help="Server port")
    parser.add_argument("--url", type=str, default=None,
                        help="Connect to existing server (skip start/stop)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations (minimum 1)")
    parser.add_argument("--iterations", type=int, default=3, help="Measurement iterations")
    parser.add_argument("--pp", type=int, nargs="+", default=[256, 1024], help="Prefill token counts")
    parser.add_argument("--tg", type=int, nargs="+", default=[128, 512], help="Decode token counts")
    parser.add_argument("--tg-prefill", type=int, default=_TG_PREFILL_DEFAULT,
                        help="Prefill size for decode tests (default: 512)")
    parser.add_argument("--attn-impl", type=str, nargs="+", default=["sdpa", "eager", "flash_attention_2"],
                        help="Attention implementations to benchmark (default: sdpa eager flash_attention_2)")
    parser.add_argument("--compile", action="store_true",
                        help="Enable static cache + torch.compile on the server for faster decode")
    parser.add_argument("--mode", type=str, choices=["bench", "chat"], default="bench",
                        help="bench: greedy (temp=0). chat: sampling (do_sample=True, temp=0.7)")
    parser.add_argument("--endpoint", type=str, choices=["chat", "responses"], default="responses",
                        help="API endpoint to benchmark (default: responses = /v1/responses)")
    parser.add_argument("--seed", type=int, default=42, help="Torch seed")
    args = parser.parse_args()

    args.warmup = max(args.warmup, 1)
    do_sample = args.mode == "chat"
    mode_str = "chat (do_sample=True, temp=0.7)" if do_sample else "bench (greedy, temp=0)"
    endpoint = args.endpoint
    endpoint_path = "/v1/responses" if endpoint == "responses" else "/v1/chat/completions"

    if args.url:
        # Against an existing server
        base_url = args.url.rstrip("/")
        tokenizer_id = args.processor or (args.models[0] if args.models else "Qwen/Qwen2.5-7B-Instruct")
        print(f"Using server at {base_url}, endpoint={endpoint_path}, mode={mode_str}")
        print(f"Loading tokenizer from {tokenizer_id}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

        rows = []
        for pp in args.pp:
            print(f"  pp{pp}")
            rows.append(bench_pp(base_url, tokenizer, pp, args.warmup, args.iterations, args.seed, do_sample=do_sample, endpoint=endpoint))
        for tg in args.tg:
            print(f"  tg{tg}")
            rows.append(bench_tg(base_url, tokenizer, tg, args.warmup, args.iterations, args.seed, tg_prefill=args.tg_prefill, do_sample=do_sample, endpoint=endpoint))
        print_table(rows)

    else:
        # Start a fresh server per model, benchmark, stop
        if not args.models:
            args.models = ["Qwen/Qwen2.5-7B-Instruct"]

        for model_str in args.models:
            spec = parse_model_spec(model_str)
            # Reference texts from the first attn impl (bench mode only)
            reference_texts = None

            for attn_impl in args.attn_impl:
                print(f"\nStarting server for {spec['model']} (attn={attn_impl})...")
                try:
                    server = start_server(spec["model"], args.port, spec["processor"], attn_implementation=attn_impl,
                                          compile=args.compile)
                except Exception as e:
                    print(f"  ERROR: Failed to start server with attn={attn_impl}: {e}. Skipping.")
                    continue

                base_url = f"http://localhost:{args.port}"
                if not wait_for_server(base_url):
                    print("  ERROR: Server did not become ready. Skipping.")
                    server.kill_server()
                    continue

                tokenizer = AutoTokenizer.from_pretrained(spec["tokenizer"])

                # Warmup (always dynamic cache — static cache compiles shapes, so a short warmup would break longer requests)
                streaming_request(base_url, [{"role": "user", "content": "hi"}], max_tokens=5, seed=args.seed, endpoint=endpoint)

                rows = []
                for pp in args.pp:
                    print(f"  pp{pp}")
                    rows.append(bench_pp(base_url, tokenizer, pp, args.warmup, args.iterations, args.seed, do_sample=do_sample, endpoint=endpoint))
                for tg in args.tg:
                    print(f"  tg{tg}")
                    rows.append(bench_tg(base_url, tokenizer, tg, args.warmup, args.iterations, args.seed, tg_prefill=args.tg_prefill, do_sample=do_sample, endpoint=endpoint))

                server.kill_server()

                # Build reference from first attn impl in greedy mode
                if not do_sample and reference_texts is None:
                    reference_texts = {
                        row["test"]: row["text"] for row in rows if row.get("text")
                    }
                    # Pass reference_texts so the first impl shows "REF" in the ref column
                    print_table(rows, title=f"{spec['model']} | attn={attn_impl} ({mode_str})",
                                reference_texts=reference_texts if len(args.attn_impl) > 1 else None,
                                is_reference=True)
                else:
                    print_table(rows, title=f"{spec['model']} | attn={attn_impl} ({mode_str})",
                                reference_texts=reference_texts if not do_sample else None)

            # Summary: check for mismatches across attn impls (bench mode only)
            if not do_sample and reference_texts and len(args.attn_impl) > 1:
                print_reference_summary(reference_texts, args.attn_impl[0])


def print_reference_summary(reference_texts: dict[str, str], ref_impl: str) -> None:
    """Print a summary noting that outputs are compared against the reference implementation."""
    print(f"Reference comparison: all decode outputs compared against '{ref_impl}'.")
    print(f"  MATCH    = identical text (greedy decoding is deterministic)")
    print(f"  MISMATCH = text differs (FP divergence across attention kernels — check preview for correctness)")
    print()


if __name__ == "__main__":
    main()
