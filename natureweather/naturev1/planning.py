# Copyright 2026 Nathan. Apache-2.0.
"""
Measure what a run will actually cost before committing to it.

Everything here is measured on your hardware rather than assumed. Published throughput figures for a card
tell you very little about a model whose cost is dominated by gathers and cross-attention rather than by
dense matmuls, so the only honest estimate comes from running a few steps and extrapolating.

:func:`autotune_batch_size` finds the largest batch that fits, which is how a large card earns its money:
VRAM converts into batch size, batch size converts into throughput, and with gradient checkpointing you
can trade a third more compute for most of the activation memory and buy more batch still.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


def device_report(device: torch.device | str | None = None) -> dict:
    """What the training device actually is, and how much memory it has."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    report = {"device": str(device), "type": device.type}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        report.update(
            name=properties.name,
            total_vram_gb=properties.total_memory / 1024**3,
            multiprocessors=properties.multi_processor_count,
            capability=f"{properties.major}.{properties.minor}",
            bf16_supported=torch.cuda.is_bf16_supported(),
        )
    return report


@dataclass
class Benchmark:
    """The outcome of timing real steps."""

    batch_size: int
    seconds_per_step: float
    samples_per_second: float
    peak_vram_gb: float
    gradient_checkpointing: bool

    def __repr__(self) -> str:
        return (
            f"Benchmark(batch={self.batch_size}, {self.seconds_per_step*1000:.0f} ms/step, "
            f"{self.samples_per_second:.2f} samples/s, {self.peak_vram_gb:.1f} GB peak"
            f"{', checkpointed' if self.gradient_checkpointing else ''})"
        )


def benchmark_steps(
    step_fn, batch_size: int, warmup: int = 3, iterations: int = 10,
    device: torch.device | str | None = None, gradient_checkpointing: bool = False,
) -> Benchmark:
    """
    Time ``step_fn()`` properly: warm up first, synchronise, and record peak memory.

    Warmup matters more than usual here -- the first call builds and caches the grid correspondences,
    which is a one-off cost that would otherwise be smeared across the estimate and make the run look
    far slower than it is.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    for _ in range(warmup):
        step_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    for _ in range(iterations):
        step_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) / iterations

    peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    return Benchmark(batch_size, elapsed, batch_size / elapsed, peak, gradient_checkpointing)


def autotune_batch_size(
    make_step, start: int = 1, limit: int = 4096, target_fraction: float = 0.85,
    device: torch.device | str | None = None,
) -> int:
    """
    Find the largest batch that fits, by doubling until it does not and then backing off.

    Args:
        make_step: ``batch_size -> callable`` building a step function for that batch.
        target_fraction: how much of VRAM to aim for. Leaving ~15% free covers allocator fragmentation
            and the transient spike when the optimizer state is first built, which is the usual reason a
            run that "fits" dies twenty minutes in.

    Returns:
        The largest batch size that ran cleanly.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        return start
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    best, size = start, start

    while size <= limit:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            step = make_step(size)
            step()
            torch.cuda.synchronize()
            used = torch.cuda.max_memory_allocated(device) / 1024**3
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        if used > total * target_fraction:
            best = size
            break
        best, size = size, size * 2
    torch.cuda.empty_cache()
    return best


def training_plan(
    samples_per_second: float,
    corpus_samples: int,
    epochs: float = 1.0,
    watts: float = 600.0,
    electricity_per_kwh: float = 0.15,
    cloud_per_hour: float | None = None,
) -> dict:
    """
    Turn a measured throughput into wall time and money.

    Args:
        samples_per_second: from :func:`benchmark_steps`, on your hardware.
        corpus_samples: samples in one pass -- 92,040 for the WeatherBench 2 six-hourly store.
        epochs: passes over the corpus. Weather models are usually trained for a handful.
        watts: board power under load. Quoted TDP is a reasonable stand-in.
        electricity_per_kwh: your rate. The default is a rough US average.
        cloud_per_hour: if renting instead, the hourly rate, for comparison.

    Returns:
        Hours, days, energy and cost, with the assumptions echoed back so the number can be checked.
    """
    total_samples = corpus_samples * epochs
    seconds = total_samples / max(samples_per_second, 1e-9)
    hours = seconds / 3600
    kwh = watts / 1000 * hours
    plan = {
        "corpus_samples": corpus_samples,
        "epochs": epochs,
        "total_samples": int(total_samples),
        "samples_per_second": samples_per_second,
        "hours": hours,
        "days": hours / 24,
        "energy_kwh": kwh,
        "electricity_cost": kwh * electricity_per_kwh,
        "assumptions": {"watts": watts, "electricity_per_kwh": electricity_per_kwh},
    }
    if cloud_per_hour:
        plan["cloud_cost"] = hours * cloud_per_hour
        plan["cloud_per_hour"] = cloud_per_hour
    return plan


def format_plan(plan: dict) -> str:
    """One-screen summary of a training plan."""
    lines = [
        f"  corpus          {plan['corpus_samples']:,} samples x {plan['epochs']:g} epochs "
        f"= {plan['total_samples']:,}",
        f"  throughput      {plan['samples_per_second']:.2f} samples/s (measured)",
        f"  wall time       {plan['hours']:,.1f} hours = {plan['days']:,.1f} days",
        f"  energy          {plan['energy_kwh']:,.0f} kWh at {plan['assumptions']['watts']:.0f} W",
        f"  electricity     ${plan['electricity_cost']:,.2f} at "
        f"${plan['assumptions']['electricity_per_kwh']:.2f}/kWh",
    ]
    if "cloud_cost" in plan:
        lines.append(f"  cloud equivalent ${plan['cloud_cost']:,.2f} at ${plan['cloud_per_hour']:.2f}/hr")
    return "\n".join(lines)
