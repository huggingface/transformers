import torch

from scripts.benchmark_scheduler import _build_html_report


def test_build_html_report_basic():
    results = [
        {
            "name": "Baseline (no scheduler)",
            "avg_latency_ms": 10.0,
            "min_latency_ms": 9.0,
            "max_latency_ms": 11.0,
            "tokens_per_sec": 1000.0,
            "overhead_pct": 0.0,
        },
        {
            "name": "mode=force",
            "avg_latency_ms": 11.0,
            "min_latency_ms": 10.0,
            "max_latency_ms": 12.0,
            "tokens_per_sec": 900.0,
            "overhead_pct": 10.0,
        },
    ]

    html = _build_html_report(
        model_name="dummy-model",
        device=torch.device("cpu"),
        max_new_tokens=16,
        batch_size=2,
        do_sample=False,
        warmup=1,
        runs=3,
        results=results,
    )

    # Basic sanity checks
    assert "<html" in html.lower()
    assert "dummy-model" in html
    assert "Baseline (no scheduler)" in html
    assert "mode=force" in html
