#!/usr/bin/env python3
"""
Benchmark script to compare performance between PromptDepthAnythingImageProcessor (slow)
and PromptDepthAnythingImageProcessorFast (fast).
"""

import time

import numpy as np
import torch
from PIL import Image


try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from transformers import (
    PromptDepthAnythingImageProcessor,
    PromptDepthAnythingImageProcessorFast,
)


def create_test_images(batch_size=8, image_size=(384, 512), num_channels=3):
    """Create test images and prompt depths for benchmarking."""
    images = []
    prompt_depths = []

    for _ in range(batch_size):
        # Create random RGB image
        image = np.random.randint(0, 255, (*image_size, num_channels), dtype=np.uint8)
        images.append(Image.fromarray(image))

        # Create random depth map (smaller than image)
        depth_h, depth_w = image_size[0] // 2, image_size[1] // 2
        prompt_depth = np.random.random((depth_h, depth_w)).astype(np.float32) * 1000  # simulate mm values
        prompt_depths.append(prompt_depth)

    return images, prompt_depths


def benchmark_processor(processor, images, prompt_depths, num_iterations=10, device="cpu"):
    """Benchmark a single processor."""
    times = []

    # Move processor to device if it's the fast one
    if hasattr(processor, "device"):
        processor.device = device

    # Warmup runs
    for _ in range(3):
        try:
            _ = processor(images, prompt_depth=prompt_depths, return_tensors="pt")
        except Exception as e:
            print(f"Warmup failed: {e}")
            return None, None

    # Actual benchmark
    for i in range(num_iterations):
        start_time = time.time()

        try:
            outputs = processor(images, prompt_depth=prompt_depths, return_tensors="pt")

            # Move tensors to specified device if needed
            if device != "cpu" and torch.cuda.is_available():
                if hasattr(outputs, "pixel_values"):
                    outputs.pixel_values = outputs.pixel_values.to(device)
                if hasattr(outputs, "prompt_depth"):
                    outputs.prompt_depth = outputs.prompt_depth.to(device)
                # Synchronize to ensure all operations are complete
                torch.cuda.synchronize()

        except Exception as e:
            print(f"Iteration {i} failed: {e}")
            return None, None

        end_time = time.time()
        times.append(end_time - start_time)

    return times, outputs


def run_benchmark():
    """Run the full benchmark comparing slow and fast processors."""
    print("🚀 PromptDepthAnything Image Processor Benchmark")
    print("=" * 50)

    # Test configurations
    batch_sizes = [1, 4, 8, 16]
    image_sizes = [(384, 384), (512, 512), (768, 768)]
    devices = ["cpu"]

    # Add GPU if available
    if torch.cuda.is_available():
        devices.append("cuda")

    results = {}

    for device in devices:
        print(f"\n📱 Testing on device: {device}")
        print("-" * 30)

        results[device] = {}

        for batch_size in batch_sizes:
            for image_size in image_sizes:
                print(f"\n🔧 Config: batch_size={batch_size}, image_size={image_size}")

                # Create test data
                images, prompt_depths = create_test_images(batch_size, image_size)

                # Initialize processors
                try:
                    slow_processor = PromptDepthAnythingImageProcessor()
                    fast_processor = PromptDepthAnythingImageProcessorFast()
                except Exception as e:
                    print(f"❌ Failed to initialize processors: {e}")
                    continue

                # Benchmark slow processor
                print("⏳ Benchmarking slow processor...")
                slow_times, slow_outputs = benchmark_processor(slow_processor, images, prompt_depths, device=device)

                if slow_times is None:
                    print("❌ Slow processor benchmark failed")
                    continue

                # Benchmark fast processor
                print("⚡ Benchmarking fast processor...")
                fast_times, fast_outputs = benchmark_processor(fast_processor, images, prompt_depths, device=device)

                if fast_times is None:
                    print("❌ Fast processor benchmark failed")
                    continue

                # Calculate statistics
                slow_mean = np.mean(slow_times)
                slow_std = np.std(slow_times)
                fast_mean = np.mean(fast_times)
                fast_std = np.std(fast_times)
                speedup = slow_mean / fast_mean

                # Store results
                config_key = f"{batch_size}x{image_size[0]}x{image_size[1]}"
                results[device][config_key] = {
                    "slow_mean": slow_mean,
                    "slow_std": slow_std,
                    "fast_mean": fast_mean,
                    "fast_std": fast_std,
                    "speedup": speedup,
                }

                # Print results
                print("📊 Results:")
                print(f"   Slow: {slow_mean:.4f}s ± {slow_std:.4f}s")
                print(f"   Fast: {fast_mean:.4f}s ± {fast_std:.4f}s")
                print(f"   Speedup: {speedup:.2f}x")

                # Verify outputs are equivalent (shapes and approximate values)
                #
                # Output verification checks that the fast and slow processors produce functionally
                # equivalent results by comparing:
                # 1. Tensor shapes (must be identical)
                # 2. Pixel values within tolerance (atol=1e-1, rtol=1e-3 - same as transformers tests)
                # 3. Prompt depth values within tolerance (if present)
                #
                # PASSED = Both processors produce numerically equivalent outputs within acceptable tolerances
                # FAILED = Outputs differ beyond acceptable tolerances, indicating a potential implementation issue
                try:
                    if slow_outputs and fast_outputs:
                        slow_pixel_values = slow_outputs.pixel_values
                        fast_pixel_values = fast_outputs.pixel_values.to(slow_pixel_values.dtype)

                        shape_match = slow_pixel_values.shape == fast_pixel_values.shape
                        # Use the same tolerance as the actual tests (atol=1e-1, rtol=1e-3)
                        values_close = torch.allclose(slow_pixel_values, fast_pixel_values, rtol=1e-3, atol=1e-1)

                        if hasattr(slow_outputs, "prompt_depth") and hasattr(fast_outputs, "prompt_depth"):
                            slow_prompt_depth = slow_outputs.prompt_depth
                            fast_prompt_depth = fast_outputs.prompt_depth.to(slow_prompt_depth.dtype)

                            depth_shape_match = slow_prompt_depth.shape == fast_prompt_depth.shape
                            depth_values_close = torch.allclose(
                                slow_prompt_depth,
                                fast_prompt_depth,
                                rtol=1e-3,
                                atol=1e-5,
                            )
                        else:
                            depth_shape_match = depth_values_close = True

                        if shape_match and values_close and depth_shape_match and depth_values_close:
                            print("✅ Output verification: PASSED")
                            print(
                                "   (Shape checked ✓, pixel value equality checked ✓, depth value equality checked ✓)"
                            )
                        else:
                            print("❌ Output verification: FAILED")
                            print(
                                "   (One or more checks failed: shape, pixel values, or depth values differ beyond tolerances)"
                            )
                            print(f"   Shape match: {shape_match}")
                            print(f"   Values close: {values_close}")
                            print(f"   Depth shape match: {depth_shape_match}")
                            print(f"   Depth values close: {depth_values_close}")

                            # Add detailed analysis when values are not close
                            if not values_close:
                                diff = torch.abs(slow_pixel_values - fast_pixel_values)
                                max_diff = diff.max().item()
                                mean_diff = diff.mean().item()
                                print("   📊 Pixel value differences:")
                                print(f"      Max absolute difference: {max_diff:.8f}")
                                print(f"      Mean absolute difference: {mean_diff:.8f}")
                                print(
                                    f"      Slow range: [{slow_pixel_values.min():.6f}, {slow_pixel_values.max():.6f}]"
                                )
                                print(
                                    f"      Fast range: [{fast_pixel_values.min():.6f}, {fast_pixel_values.max():.6f}]"
                                )

                                # Test different tolerances to understand what would work
                                tolerances = [1e-2, 5e-2, 1e-1, 2e-1]
                                for tol in tolerances:
                                    is_close = torch.allclose(
                                        slow_pixel_values,
                                        fast_pixel_values,
                                        rtol=tol,
                                        atol=tol,
                                    )
                                    print(f"      Would pass with rtol={tol}, atol={tol}: {is_close}")

                                # Find location of max difference
                                max_diff_idx = torch.unravel_index(torch.argmax(diff), diff.shape)
                                print(f"      Max diff location: {max_diff_idx}")
                                print(f"      Slow value at max diff: {slow_pixel_values[max_diff_idx]:.8f}")
                                print(f"      Fast value at max diff: {fast_pixel_values[max_diff_idx]:.8f}")

                            if not depth_values_close and hasattr(slow_outputs, "prompt_depth"):
                                depth_diff = torch.abs(slow_prompt_depth - fast_prompt_depth)
                                print("   📊 Prompt depth differences:")
                                print(f"      Max absolute difference: {depth_diff.max().item():.8f}")
                                print(f"      Mean absolute difference: {depth_diff.mean().item():.8f}")
                except Exception as e:
                    print(f"⚠️  Output verification failed: {e}")

    # Generate summary
    print("\n" + "=" * 50)
    print("📈 BENCHMARK SUMMARY")
    print("=" * 50)

    for device, device_results in results.items():
        print(f"\n🖥️  {device.upper()}:")
        print("-" * 20)

        if not device_results:
            print("   No results available")
            continue

        speedups = [r["speedup"] for r in device_results.values()]
        avg_speedup = np.mean(speedups)
        min_speedup = np.min(speedups)
        max_speedup = np.max(speedups)

        print(f"   Average speedup: {avg_speedup:.2f}x")
        print(f"   Min speedup: {min_speedup:.2f}x")
        print(f"   Max speedup: {max_speedup:.2f}x")

        print("\n   Detailed results:")
        for config, stats in device_results.items():
            print(f"     {config}: {stats['speedup']:.2f}x speedup")

    # Create visualization if matplotlib is available
    if HAS_MATPLOTLIB:
        try:
            create_visualization(results)
        except Exception as e:
            print(f"\n⚠️  Visualization failed: {e}")
    else:
        print("\n📊 Matplotlib not available, skipping visualization")


def create_visualization(results):
    """Create a bar chart visualization of the benchmark results."""
    if not HAS_MATPLOTLIB:
        print("\n📊 Matplotlib not available, skipping visualization")
        return

    print("\n📊 Creating visualization...")

    for device, device_results in results.items():
        if not device_results:
            continue

        configs = list(device_results.keys())
        speedups = [device_results[config]["speedup"] for config in configs]

        # Calculate average speedup
        avg_speedup = np.mean(speedups)

        # Add average to the end of the data
        all_configs = configs + ["Average"]
        all_speedups = speedups + [avg_speedup]

        # Create colors - steelblue for individual configs, orange for average
        colors = ["steelblue"] * len(configs) + ["orange"]
        alphas = [0.7] * len(configs) + [0.8]

        plt.figure(figsize=(12, 6))

        # Create bars individually to handle different colors and alphas
        bars = []
        for i, (config, speedup, color, alpha) in enumerate(zip(all_configs, all_speedups, colors, alphas)):
            bar = plt.bar(i, speedup, color=color, alpha=alpha)
            bars.extend(bar)

        # Add value labels on bars
        for i, (bar, speedup) in enumerate(zip(bars, all_speedups)):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{speedup:.2f}x",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.xlabel("Configuration (batch_size x height x width)")
        plt.ylabel("Speedup (times faster)")
        plt.title(f"PromptDepthAnything Fast vs Slow Image Processor Speedup - {device.upper()}")
        plt.xticks(range(len(all_configs)), all_configs, rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        # Save plot
        filename = f"prompt_depth_anything_benchmark_{device}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"   Saved plot: {filename}")
        plt.close()


if __name__ == "__main__":
    run_benchmark()
