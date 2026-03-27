# /// script
# dependencies = [
#     "inspect-ai",
#     "inspect-evals",
#     "huggingface-hub",
#     "transformers[serving] @ git+https://github.com/NathanHB/transformers.git@fix-continuous-batching-json-response",
#     "openai>=2.26.0",
#     "torchvision",
# ]
# ///
import argparse
import shlex
import subprocess
import sys
import time
from inspect_ai import eval
from inspect_ai.log import bundle_log_dir
from inspect_evals.gpqa import gpqa_diamond


def wait_for_server_up(server_process, timeout=600):
    start_time = time.time()

    import urllib.request
    import urllib.error

    while time.time() - start_time < timeout:
        try:
            req = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
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
        "--limit",
        type=int,
        default=5,
        help="Number of evaluation samples to run (default: 5)",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=2,
        help="Maximum concurrent connections for evaluation (default: 2)",
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

    args = parser.parse_args()

    server_process = None

    # Build transformers serve command
    serve_cmd = [
        "transformers",
        "serve",
    ]

    # Add continuous batching if not disabled
    if not args.no_continuous_batching:
        serve_cmd.append("--continuous-batching")

    # Always use sdpa attention implementation
    serve_cmd.extend(["--attn-implementation", "sdpa"])

    print("Starting transformers serve with continuous batching...")
    print(f"Model: {args.model}")
    print(f"Command: {' '.join(serve_cmd)}")
    print("=" * 70)
    print("SERVER OUTPUT:")
    print("=" * 70)

    # Start server with output going directly to stdout/stderr
    server_process = subprocess.Popen(serve_cmd, stdout=None, stderr=None)

    wait_for_server_up(server_process, timeout=600)

    eval(
        gpqa_diamond,
        model=f"openai-api/transformers-serve/{args.model}",
        log_dir=args.log_dir,
        model_base_url="http://localhost:8000/v1",
        display="plain",
        limit=args.limit,
        model_args=dict(stream=False),
        max_connections=args.max_connections,
        max_tokens=2048,
    )

    bundle_log_dir(args.log_dir, output_dir=f"hf/{args.output_space}", overwrite=True)


if __name__ == "__main__":
    main()
