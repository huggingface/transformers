import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark_v2.framework.benchmark_runner import get_git_revision
from benchmark_v2.framework.compare import compare, format_report, load_scenarios


BENCHMARK_V2_DIR = Path(__file__).parent.parent


def write_results(path: Path, scenarios: dict[str, list[float]], failed: tuple[str, ...] = ()) -> Path:
    """Write a results file in the format saved by `run_benchmarks.py`, with the given e2e latencies per scenario."""
    entries = {}
    for i, (name, e2e) in enumerate({**scenarios, **{name: [] for name in failed}}.items()):
        entries[f"hash{i}"] = {
            "metadata": {"model_id": "model", "success": name not in failed},
            "measurements": {
                "e2e_latency": e2e,
                "time_to_first_token": [x / 100 for x in e2e],
                "inter_token_latency": [x / 200 for x in e2e],
                "shape_and_decoded_outputs": [],
                "gpu_metrics": None,
                "timestamps": None,
            },
            "config": {"name": name, "batch_size": 1, "num_tokens_to_generate": 128},
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def verdicts(baseline: Path | list[Path], candidate: Path | list[Path], threshold: float = 1.0) -> dict[str, str]:
    base, _ = load_scenarios(baseline if isinstance(baseline, list) else [baseline])
    cand, _ = load_scenarios(candidate if isinstance(candidate, list) else [candidate])
    return {f"{c.scenario}/{c.metric}": c.verdict for c in compare(base, cand, threshold)}


@pytest.mark.parametrize(
    "factor, expected",
    [(1.005, "ok"), (0.995, "ok"), (1.05, "regression"), (0.95, "improvement")],
)
def test_latency_change_against_threshold(tmp_path, factor, expected):
    base = write_results(tmp_path / "base/m_benchmark_1.json", {"s": [4.0, 4.0, 4.0]})
    cand = write_results(tmp_path / "cand/m_benchmark_1.json", {"s": [4.0 * factor] * 3})
    result = verdicts(base, cand)
    assert result["s/e2e_latency"] == expected
    # Throughput moves the other way, but a slower run is still a regression
    assert result["s/throughput"] == expected


def test_directory_of_runs_uses_median_of_run_medians(tmp_path):
    for run, e2e in enumerate([4.0, 4.0, 9.0]):  # one outlier run
        write_results(tmp_path / f"base/run{run}/m/m_benchmark_{run}.json", {"s": [e2e] * 3})
    scenarios, _ = load_scenarios([tmp_path / "base"])
    assert scenarios["s"].per_run_medians["e2e_latency"] == [4.0, 4.0, 9.0]
    assert scenarios["s"].median("e2e_latency") == 4.0
    assert scenarios["s"].between_run_cov_pct("e2e_latency") > 1.0


def test_missing_and_failed_scenarios_are_reported_not_compared(tmp_path):
    base = write_results(tmp_path / "base/m_benchmark_1.json", {"a": [1.0, 1.0], "b": [1.0, 1.0]})
    cand = write_results(tmp_path / "cand/m_benchmark_1.json", {"a": [1.0, 1.0], "c": [1.0, 1.0]}, failed=("b",))
    baseline, _ = load_scenarios([base])
    candidate, cand_failed = load_scenarios([cand])
    comparisons = compare(baseline, candidate)
    assert {c.scenario for c in comparisons} == {"a"}
    assert cand_failed == ["b"]
    report = format_report(comparisons, baseline, candidate, {"candidate": cand_failed}, 1.0)
    assert "Only in baseline:\n  b" in report
    assert "Only in candidate:\n  c" in report
    assert "Failed in candidate:\n  b" in report


def test_noise_above_threshold_is_flagged(tmp_path):
    base = write_results(tmp_path / "base/m_benchmark_1.json", {"s": [3.0, 4.0, 5.0]})
    cand = write_results(tmp_path / "cand/m_benchmark_1.json", {"s": [3.0, 4.0, 5.0]})
    baseline, _ = load_scenarios([base])
    candidate, _ = load_scenarios([cand])
    report = format_report(compare(baseline, candidate), baseline, candidate, {}, 1.0)
    assert "noisy" in report


@pytest.mark.parametrize("factor, exit_code", [(1.0, 0), (1.10, 1)])
def test_cli_exit_code(tmp_path, factor, exit_code):
    base = write_results(tmp_path / "base/m_benchmark_1.json", {"s": [2.0, 2.0]})
    cand = write_results(tmp_path / "cand/m_benchmark_1.json", {"s": [2.0 * factor] * 2})
    out_json = tmp_path / "comparison.json"
    result = subprocess.run(
        [
            sys.executable,
            "compare_results.py",
            "--baseline",
            str(base),
            "--candidate",
            str(cand),
            "--json",
            str(out_json),
        ],
        cwd=BENCHMARK_V2_DIR,
        capture_output=True,
        text=True,
    )
    assert result.returncode == exit_code, result.stderr
    assert len(json.loads(out_json.read_text(encoding="utf-8"))) == 4


def test_git_revision_is_a_commit_or_unknown():
    revision = get_git_revision()
    assert revision == "unknown" or (len(revision) == 40 and all(c in "0123456789abcdef" for c in revision))
