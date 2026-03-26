"""
Load test for `transformers serve` — measures throughput and latency under concurrent requests.

Unlike benchmark_serve.py (single-user perf), this tests server capacity:
- How many tokens/sec can the server sustain under load?
- What's the latency distribution (p50/p90/p99) as concurrency increases?
- Does the server stay stable under pressure?

Each --max-concurrency value sends that many requests simultaneously.

Examples:
    # Sweep concurrency levels
    python tests/cli/benchmark_serve_load.py --model Qwen/Qwen2.5-7B-Instruct \\
        --max-concurrency 1 4 8 32 --continuous-batching

    # 500 concurrent non-streaming requests
    python tests/cli/benchmark_serve_load.py --model Qwen/Qwen2.5-7B-Instruct \\
        --max-concurrency 500 --continuous-batching --no-stream

    # Against an existing server
    python tests/cli/benchmark_serve_load.py --url http://localhost:8000 \\
        --processor Qwen/Qwen2.5-7B-Instruct --max-concurrency 1 4 8
"""

import argparse
import asyncio
import json
import os
import random
import statistics
import time


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import aiohttp

from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

_FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "Sphinx of black quartz, judge my vow. "
) * 200


def make_prompt(tokenizer, num_tokens: int) -> str:
    token_ids = tokenizer.encode(_FILLER, add_special_tokens=False)[:num_tokens]
    return tokenizer.decode(token_ids)


def make_prompts(tokenizer, num_requests: int, prompt_tokens: int, variance: float = 0.2) -> list[str]:
    """Generate a list of prompts with some length variance to simulate realistic traffic."""
    prompts = []
    for _ in range(num_requests):
        # Vary prompt length by ±variance around the target
        length = max(10, int(prompt_tokens * (1.0 + random.uniform(-variance, variance))))
        prompts.append(make_prompt(tokenizer, length))
    return prompts


# ---------------------------------------------------------------------------
# Request sender
# ---------------------------------------------------------------------------


async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt: str,
    max_new_tokens: int,
    seed: int,
    endpoint: str = "responses",
    model: str | None = None,
    stream: bool = True,
) -> dict:
    """Send a single request and collect timing metrics."""
    gen_cfg = {"max_new_tokens": max_new_tokens, "do_sample": False}

    if endpoint == "responses":
        url = f"{base_url}/v1/responses"
        payload = {
            "model": model,
            "input": [{"role": "user", "content": prompt}],
            "stream": stream,
            "seed": seed,
            "generation_config": json.dumps(gen_cfg),
        }
    else:
        url = f"{base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "seed": seed,
            "generation_config": json.dumps(gen_cfg),
        }

    t_start = time.perf_counter()
    t_first_token = None
    token_times = []
    text_chunks = []
    non_streaming_tokens = 0
    error = None

    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status != 200:
                error = f"HTTP {resp.status}: {await resp.text()}"
                return _make_result(t_start, error=error)

            if not stream:
                # Non-streaming: single JSON response — get token count from usage
                body = await resp.json()
                t_first_token = time.perf_counter()
                if endpoint == "responses":
                    output_tokens = body.get("usage", {}).get("output_tokens", 0)
                    for item in body.get("output", []):
                        if item.get("type") == "message":
                            for part in item.get("content", []):
                                if part.get("type") == "output_text":
                                    text_chunks.append(part.get("text", ""))
                else:
                    output_tokens = body.get("usage", {}).get("completion_tokens", 0)
                    for choice in body.get("choices", []):
                        content = choice.get("message", {}).get("content", "")
                        if content:
                            text_chunks.append(content)
                # Use server-reported token count instead of len(text_chunks)
                non_streaming_tokens = output_tokens
                token_times.append(t_first_token)
            else:
                # Streaming: parse SSE events
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Extract token content based on endpoint format
                    has_content = False
                    if endpoint == "responses":
                        if chunk.get("type") == "response.output_text.delta":
                            delta = chunk.get("delta", "")
                            if delta:
                                text_chunks.append(delta)
                                has_content = True
                        elif chunk.get("type") == "response.completed":
                            break
                    else:
                        choices = chunk.get("choices", [])
                        if choices:
                            content = choices[0].get("delta", {}).get("content")
                            if content is not None and content != "":
                                text_chunks.append(content)
                                has_content = True
                            if choices[0].get("finish_reason") is not None:
                                break

                    if has_content:
                        now = time.perf_counter()
                        token_times.append(now)
                        if t_first_token is None:
                            t_first_token = now

    except asyncio.TimeoutError:
        error = "timeout"
    except Exception as e:
        error = str(e)

    output_token_count = non_streaming_tokens if not stream else None
    return _make_result(t_start, t_first_token, token_times, text_chunks, error, output_token_count=output_token_count)


def _make_result(t_start, t_first_token=None, token_times=None, text_chunks=None, error=None, output_token_count=None):
    t_end = time.perf_counter()
    token_times = token_times or []
    text_chunks = text_chunks or []

    # Inter-token latencies
    itl = []
    for i in range(1, len(token_times)):
        itl.append(token_times[i] - token_times[i - 1])

    return {
        "e2e": t_end - t_start,
        "ttft": (t_first_token - t_start) if t_first_token else None,
        "tpot": statistics.mean(itl) if itl else None,  # time per output token
        "itl": itl,
        "output_tokens": output_token_count if output_token_count is not None else len(text_chunks),
        "text": "".join(text_chunks),
        "error": error,
    }


# ---------------------------------------------------------------------------
# Load generators
# ---------------------------------------------------------------------------


async def run_concurrency_test(
    base_url: str,
    prompts: list[str],
    max_new_tokens: int,
    seed: int,
    endpoint: str,
    model: str | None = None,
    stream: bool = True,
) -> list[dict]:
    """Send all prompts concurrently and collect results."""
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [send_request(session, base_url, p, max_new_tokens, seed, endpoint, model=model, stream=stream) for p in prompts]
        results = await asyncio.gather(*tasks)

    return list(results)



# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(results: list[dict], duration: float) -> dict:
    """Compute aggregate metrics from individual request results."""
    successful = [r for r in results if r["error"] is None]
    failed = [r for r in results if r["error"] is not None]

    if not successful:
        return {"error": "all requests failed", "failures": len(failed)}

    total_output_tokens = sum(r["output_tokens"] for r in successful)

    e2e_latencies = [r["e2e"] for r in successful]
    ttfts = [r["ttft"] for r in successful if r["ttft"] is not None]
    tpots = [r["tpot"] for r in successful if r["tpot"] is not None]

    # Flatten all inter-token latencies
    all_itl = []
    for r in successful:
        all_itl.extend(r["itl"])

    def percentiles(values):
        if not values:
            return {}
        values = sorted(values)
        n = len(values)
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p90": values[int(n * 0.9)],
            "p99": values[min(int(n * 0.99), n - 1)],
            "min": values[0],
            "max": values[-1],
        }

    return {
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "duration": duration,
        "total_output_tokens": total_output_tokens,
        "throughput_req_per_sec": len(successful) / duration,
        "throughput_tok_per_sec": total_output_tokens / duration,
        "e2e_latency": percentiles(e2e_latencies),
        "ttft": percentiles(ttfts),
        "tpot": percentiles(tpots),
        "itl": percentiles(all_itl),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def format_ms(seconds):
    if seconds is None:
        return "N/A"
    return f"{seconds * 1000:.1f}ms"


def print_metrics(metrics: dict, label: str):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    if "error" in metrics:
        print(f"  ERROR: {metrics['error']}")
        return

    print(f"  Requests:    {metrics['successful']} ok / {metrics['failed']} failed / {metrics['total_requests']} total")
    print(f"  Duration:    {metrics['duration']:.1f}s")
    print(f"  Throughput:  {metrics['throughput_req_per_sec']:.2f} req/s, {metrics['throughput_tok_per_sec']:.1f} tok/s")
    print(f"  Tokens:      {metrics['total_output_tokens']} total output")
    print()

    headers = ["metric", "mean", "median", "p90", "p99", "min", "max"]
    rows = []
    for name in ["e2e_latency", "ttft", "tpot", "itl"]:
        p = metrics.get(name, {})
        if not p:
            continue
        rows.append([
            name.upper().replace("_", " "),
            format_ms(p.get("mean")),
            format_ms(p.get("median")),
            format_ms(p.get("p90")),
            format_ms(p.get("p99")),
            format_ms(p.get("min")),
            format_ms(p.get("max")),
        ])

    if rows:
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        fmt = "  " + " | ".join(f"{{:<{w}}}" for w in widths)
        sep = "  " + "-+-".join("-" * w for w in widths)
        print(fmt.format(*headers))
        print(sep)
        for row in rows:
            print(fmt.format(*row))
    print()


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------


def wait_for_server(base_url: str, timeout: int = 120) -> bool:
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/health", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def start_server(model: str, port: int, compile: bool = False, continuous_batching: bool = False,
                  attn_implementation: str | None = None):
    from transformers.cli.serve_refactored import Serve

    kwargs = {"force_model": model, "port": port, "non_blocking": True, "log_level": "warning"}
    if compile:
        kwargs["compile"] = True
    if continuous_batching:
        kwargs["continuous_batching"] = True
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return Serve(**kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main(args):
    base_url = args.url if args.url else f"http://localhost:{args.port}"
    server = None

    # Default to flash_attention_3 when using continuous batching
    if args.continuous_batching and args.attn_impl is None:
        args.attn_impl = "flash_attention_3"

    if not args.url:
        print(f"Starting server for {args.model}...")
        server = start_server(args.model, args.port, compile=args.compile, continuous_batching=args.continuous_batching,
                              attn_implementation=args.attn_impl)
        if not wait_for_server(base_url):
            print("ERROR: Server did not start")
            if server:
                server.kill_server()
            return
        print("Server ready.")

    tokenizer_id = args.processor or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    num_requests = max(args.max_concurrency)
    prompts = make_prompts(tokenizer, num_requests, args.prompt_tokens, variance=args.prompt_variance)
    print(f"Generated {num_requests} prompts (~{args.prompt_tokens} tokens each, ±{int(args.prompt_variance*100)}%)")
    print(f"Max new tokens per request: {args.max_new_tokens}")
    print(f"Endpoint: /v1/{args.endpoint} ({'streaming' if args.stream else 'non-streaming'})")

    # Warmup — small batch to warm CUDA graphs and JIT kernels
    warmup_size = min(16, num_requests)
    warmup_prompts = prompts[:warmup_size]
    print(f"Warming up ({args.warmup}x {warmup_size} requests)...")
    for _ in range(args.warmup):
        await run_concurrency_test(
            base_url, warmup_prompts, args.max_new_tokens, args.seed, args.endpoint, model=args.model, stream=args.stream,
        )
    print("Warmup done.")

    # Run tests — one round per concurrency level
    for concurrency in args.max_concurrency:
        test_prompts = prompts[:concurrency]
        label = f"{concurrency} concurrent requests"
        print(f"\nRunning: {label}")
        t0 = time.perf_counter()
        results = await run_concurrency_test(
            base_url, test_prompts, args.max_new_tokens, args.seed, args.endpoint, model=args.model, stream=args.stream,
        )
        duration = time.perf_counter() - t0
        metrics = compute_metrics(results, duration)
        print_metrics(metrics, label)

    if server:
        server.kill_server()


def main():
    parser = argparse.ArgumentParser(
        description="Load test for transformers serve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--processor", type=str, default=None)
    parser.add_argument("--url", type=str, default=None, help="Existing server URL (skip start/stop)")
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument("--compile", action="store_true", help="Enable --compile on the server")
    parser.add_argument("--continuous-batching", action="store_true", help="Enable continuous batching on the server")
    parser.add_argument("--attn-impl", type=str, default=None, help="Attention implementation (e.g. flash_attention_3)")
    parser.add_argument("--endpoint", type=str, choices=["chat", "responses"], default="responses")
    parser.add_argument("--no-stream", action="store_true", help="Use non-streaming requests")

    # Load parameters
    parser.add_argument("--max-concurrency", type=int, nargs="+", default=[1, 2, 4],
                        help="Number of concurrent requests to send (default: 1 2 4)")

    # Prompt parameters
    parser.add_argument("--prompt-tokens", type=int, default=256, help="Target prompt length in tokens (default: 256)")
    parser.add_argument("--prompt-variance", type=float, default=0.2,
                        help="Prompt length variance as fraction (default: 0.2 = ±20%%)")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max tokens to generate per request (default: 128)")

    parser.add_argument("--warmup", type=int, default=2, help="Warmup requests (default: 2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.stream = not args.no_stream

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
