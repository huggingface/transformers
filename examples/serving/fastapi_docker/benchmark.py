#!/usr/bin/env python3
"""
Performance benchmarking script for FastAPI Transformers inference.
Measures baseline vs optimized performance to demonstrate improvements.
"""

import json
import os
import subprocess
import sys
import time
from statistics import mean, stdev

import requests


def run_server(port, env_vars=None):
    """Start a FastAPI server with given environment variables."""
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
        cwd=os.path.dirname(__file__),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def wait_for_server(port, timeout=30):
    """Wait for server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def benchmark_endpoint(port, inputs, num_runs=5):
    """Benchmark the predict endpoint with given inputs."""
    times = []

    for _ in range(num_runs):
        payload = {"inputs": inputs}
        start_time = time.time()

        try:
            response = requests.post(f"http://127.0.0.1:{port}/predict", json=payload, timeout=30)
            end_time = time.time()

            if response.status_code == 200:
                times.append(end_time - start_time)
            else:
                print(f"Request failed with status {response.status_code}")
                return None
        except Exception as e:
            print(f"Request failed: {e}")
            return None

    return {"mean": mean(times), "stdev": stdev(times) if len(times) > 1 else 0, "times": times}


def run_benchmark():
    """Run complete benchmark suite."""
    print("🚀 Starting FastAPI Transformers Performance Benchmark")
    print("=" * 60)

    # Test configurations
    configs = [
        {"name": "Baseline", "port": 8020, "env": {}},
        {"name": "Optimized Threads (8)", "port": 8021, "env": {"NUM_THREADS": "8"}},
        {"name": "Torch Compile", "port": 8022, "env": {"TORCH_COMPILE": "1"}},
        {"name": "Combined Optimizations", "port": 8023, "env": {"NUM_THREADS": "8", "TORCH_COMPILE": "1"}},
    ]

    # Test inputs - various batch sizes
    test_cases = [
        {"name": "Single Input", "inputs": ["I love this product!"]},
        {
            "name": "Small Batch (4)",
            "inputs": ["I love this!", "This is terrible.", "Amazing quality!", "Not worth the money."],
        },
        {"name": "Medium Batch (16)", "inputs": ["Sample text for classification."] * 16},
        {"name": "Large Batch (32)", "inputs": ["Sample text for classification."] * 32},
    ]

    results = {}

    for config in configs:
        print(f"\n📊 Testing configuration: {config['name']}")
        print("-" * 40)

        # Start server
        proc = run_server(config["port"], config["env"])

        try:
            if not wait_for_server(config["port"]):
                print(f"❌ Server failed to start on port {config['port']}")
                continue

            print("✅ Server started successfully")
            time.sleep(2)  # Allow server to stabilize

            config_results = {}

            for test_case in test_cases:
                print(f"  Testing {test_case['name']}...")

                result = benchmark_endpoint(config["port"], test_case["inputs"], num_runs=3)

                if result:
                    config_results[test_case["name"]] = result
                    print(f"    Mean: {result['mean']:.3f}s ± {result['stdev']:.3f}s")
                else:
                    print("    ❌ Failed")

            results[config["name"]] = config_results

        finally:
            # Clean up server
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            time.sleep(1)

    # Calculate improvements
    print("\n🎯 Performance Summary")
    print("=" * 60)

    baseline_key = "Baseline"
    if baseline_key in results:
        baseline = results[baseline_key]

        for config_name, config_results in results.items():
            if config_name == baseline_key:
                continue

            print(f"\n{config_name} vs Baseline:")

            for test_name in baseline.keys():
                if test_name in config_results:
                    baseline_time = baseline[test_name]["mean"]
                    optimized_time = config_results[test_name]["mean"]

                    improvement = ((baseline_time - optimized_time) / baseline_time) * 100

                    print(f"  {test_name}: {improvement:+.1f}% ({baseline_time:.3f}s → {optimized_time:.3f}s)")

    # Save detailed results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n💾 Detailed results saved to benchmark_results.json")
    print("\n✨ Benchmark complete!")


if __name__ == "__main__":
    # Install requests if needed
    try:
        import requests
    except ImportError:
        print("Installing requests...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        import requests

    run_benchmark()
