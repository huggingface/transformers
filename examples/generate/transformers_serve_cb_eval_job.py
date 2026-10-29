# /// script
# dependencies = [
#     "inspect-ai",
#     "inspect-evals",
#     "huggingface-hub",
#     "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main",
#     "openai>=2.26.0",
#     "torchvision",
#     "kernels",
# ]
# ///
import argparse
import subprocess
import sys
import time

from inspect_ai import eval
from inspect_ai.log import bundle_log_dir


def wait_for_server_up(server_process, port=8000, timeout=600):
    start_time = time.time()

    import urllib.error
    import urllib.request

    while time.time() - start_time < timeout:
        try:
            req = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            if req.status == 200:
                elapsed = time.time() - start_time
                print("\n" + "=" * 70)
                print(f"✓ Server is ready! (took {elapsed:.1f} seconds)")
                print("=" * 70)
                sys.stdout.flush()
                break
        except (
            urllib.error.URLError,
            ConnectionRefusedError,
            TimeoutError,
            OSError,
        ):
            elapsed = time.time() - start_time
            print(f"[{elapsed:.0f}s] Still waiting...", flush=True)
            time.sleep(5)
    else:
        print("\n" + "=" * 70)
        print("✗ Server failed to start within timeout")
        print("=" * 70)
        sys.stdout.flush()
        if server_process:
            server_process.terminate()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run transformers serve with continuous batching and evaluate with inspect-ai"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to serve (e.g., meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--no-continuous-batching",
        action="store_true",
        help="Disable continuous batching (enabled by default)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the transformers serve server (default: 8000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of evaluation samples to run (default: 10)",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=10,
        help="Maximum concurrent connections for evaluation (default: 10)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Temperature for generation (default: 0)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./gpqa-diamond-transformers-serve",
        help="Directory to save evaluation logs (default: ./gpqa-diamond-transformers-serve)",
    )
    parser.add_argument(
        "--output-space",
        type=str,
        default="SaylorTwift/transformers-CB",
        help="Output directory for bundled logs (will be prefixed with 'hf/', default: SaylorTwift/transformers-CB)",
    )

    # Continuous batching configuration arguments
    parser.add_argument(
        "--cb-block-size",
        type=int,
        default=256,
        help="KV cache block size in tokens for continuous batching (default: 256)",
    )
    parser.add_argument(
        "--cb-num-blocks",
        type=int,
        default=None,
        help="Number of KV cache blocks for continuous batching (default: auto-calculated)",
    )
    parser.add_argument(
        "--cb-max-batch-tokens",
        type=int,
        default=None,
        help="Maximum tokens per batch for continuous batching (default: auto-calculated)",
    )
    parser.add_argument(
        "--cb-max-memory-percent",
        type=float,
        default=0.8,
        help="Maximum GPU memory percentage to use for KV cache (0.0-1.0, default: 0.8)",
    )
    parser.add_argument(
        "--cb-use-cuda-graph",
        action=argparse.BooleanOptionalAction,
        help="Enable CUDA graphs for continuous batching performance",
    )

    args = parser.parse_args()

    server_process = None

    # Build transformers serve command
    serve_cmd = [
        "transformers",
        "serve",
        args.model,
    ]

    # Add continuous batching if not disabled
    if not args.no_continuous_batching:
        serve_cmd.append("--continuous-batching")

        # Add continuous batching configuration arguments
        serve_cmd.extend(["--cb-block-size", str(args.cb_block_size)])

        if args.cb_num_blocks is not None:
            serve_cmd.extend(["--cb-num-blocks", str(args.cb_num_blocks)])

        if args.cb_max_batch_tokens is not None:
            serve_cmd.extend(["--cb-max-batch-tokens", str(args.cb_max_batch_tokens)])

        serve_cmd.extend(["--cb-max-memory-percent", str(args.cb_max_memory_percent)])

        if args.cb_use_cuda_graph is True:
            serve_cmd.append("--cb-use-cuda-graph")
        elif args.cb_use_cuda_graph is False:
            serve_cmd.append("--no-cb-use-cuda-graph")

    # Always use sdpa attention implementation
    serve_cmd.extend(["--attn-implementation", "kernels-community/flash-attn2"])
    serve_cmd.extend(["--port", str(args.port)])

    print("Starting transformers serve with continuous batching...")
    print(f"Model: {args.model}")
    if not args.no_continuous_batching:
        print(f"CB Block Size: {args.cb_block_size}")
        print(f"CB Num Blocks: {args.cb_num_blocks if args.cb_num_blocks else 'auto'}")
        print(f"CB Max Batch Tokens: {args.cb_max_batch_tokens if args.cb_max_batch_tokens else 'auto'}")
        print(f"CB Max Memory: {args.cb_max_memory_percent * 100}%")
        print(f"CB CUDA Graph: {args.cb_use_cuda_graph}")
    print(f"Temperature: {args.temperature}")
    print(f"Command: {' '.join(serve_cmd)}")
    print("=" * 70)
    print("SERVER OUTPUT:")
    print("=" * 70)

    # Start server with output going directly to stdout/stderr
    server_process = subprocess.Popen(serve_cmd, stdout=None, stderr=None)

    wait_for_server_up(server_process, port=args.port, timeout=600)

    eval(
        "hf/Idavidrein/gpqa/diamond",
        model=f"openai-api/transformers-serve/{args.model}",
        log_dir=args.log_dir,
        model_base_url=f"http://localhost:{args.port}/v1",
        display="plain",
        limit=args.limit,
        model_args={"stream": False},
        temperature=args.temperature,
        max_connections=args.max_connections,
        max_tokens=2048,
    )

    bundle_log_dir(args.log_dir, output_dir=f"hf/{args.output_space}", overwrite=True)


if __name__ == "__main__":
    main()
