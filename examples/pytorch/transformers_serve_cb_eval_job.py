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
import subprocess
import sys
import time
from inspect_ai import eval
from inspect_ai.log import bundle_log_dir
from inspect_evals.gpqa import gpqa_diamond


def main():
    server_process = None
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    model = "Qwen/Qwen3.5-0.8B"
    log_dir = "./gpqa-diamond-transformers-serve"
    # 1. Start server in background
    print("Starting transformers serve with continuous batching...")
    print(f"Model: {model}")
    print("=" * 70)
    print("SERVER OUTPUT:")
    print("=" * 70)

    # Start server with output going directly to stdout/stderr
    server_process = subprocess.Popen(
        [
            "transformers",
            "serve",
            "--continuous-batching",
            "--attn-implementation",
            "sdpa",
        ],
        # Don't capture output - let it go to stdout/stderr directly
        stdout=None,
        stderr=None,
    )

    max_wait = 600  # 10 minutes
    start_time = time.time()

    import urllib.request
    import urllib.error

    while time.time() - start_time < max_wait:
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

        # 3. Send test prompt

    eval(
        gpqa_diamond,
        model=f"openai-api/transformers-serve/{model}",
        log_dir=log_dir,
        model_base_url="http://localhost:8000/v1",
        display="plain",
        limit=5,
        model_args=dict(stream=False),
        max_connections=2,
        max_tokens=2048,
    )

    bundle_log_dir(log_dir, output_dir="hf/SaylorTwift/transformers-CB", overwrite=True)

if __name__ == "__main__":
    main()
