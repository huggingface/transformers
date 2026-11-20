"""Compare benchmark_v2 results against a baseline to detect regressions.

A side (baseline or candidate) is one or more result files written by `run_benchmarks.py`. With several files
(repeated runs), each scenario's metric is the median of the per-run medians, and the spread between runs is reported.
"""

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path


# metric name -> (key in the saved measurements, unit, higher is better)
METRICS = {
    "e2e_latency": ("e2e_latency", "s", False),
    "time_to_first_token": ("time_to_first_token", "s", False),
    "inter_token_latency": ("inter_token_latency", "s", False),
    "throughput": (None, "tok/s", True),  # derived from e2e_latency
}


def _cov_pct(values: list[float]) -> float:
    """Coefficient of variation in percent (0 for fewer than two values)."""
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    return 100 * statistics.stdev(values) / mean if mean else 0.0


@dataclass
class ScenarioRuns:
    """One scenario's measurements across one or more runs."""

    name: str
    per_run_medians: dict[str, list[float]] = field(default_factory=dict)
    within_run_cov_pct: list[float] = field(default_factory=list)  # e2e latency spread inside each run

    def median(self, metric: str) -> float:
        return statistics.median(self.per_run_medians[metric])

    def between_run_cov_pct(self, metric: str) -> float:
        return _cov_pct(self.per_run_medians[metric])


def find_result_files(path: str | Path) -> list[Path]:
    """A results JSON file, or every `*_benchmark_*.json` under a directory."""
    path = Path(path)
    if path.is_file():
        return [path]
    files = sorted(path.rglob("*_benchmark_*.json"))
    if not files:
        raise FileNotFoundError(f"No benchmark result files found under {path}")
    return files


def load_scenarios(paths: list[str | Path]) -> tuple[dict[str, ScenarioRuns], list[str]]:
    """Load successful scenarios from result files, keyed by scenario name. Also returns failed scenario names."""
    scenarios: dict[str, ScenarioRuns] = {}
    failed = set()
    for path in paths:
        for file in find_result_files(path):
            for entry in json.loads(file.read_text(encoding="utf-8")).values():
                config, measurements = entry["config"], entry["measurements"]
                name = config["name"]
                e2e = measurements.get("e2e_latency") or []
                if not entry["metadata"].get("success", True) or not e2e:
                    failed.add(name)
                    continue
                runs = scenarios.setdefault(name, ScenarioRuns(name))
                tokens = config["batch_size"] * config["num_tokens_to_generate"]
                values = {key: measurements[src] for key, (src, _, _) in METRICS.items() if src is not None}
                values["throughput"] = [tokens / x for x in e2e]
                for metric, vals in values.items():
                    runs.per_run_medians.setdefault(metric, []).append(statistics.median(vals))
                runs.within_run_cov_pct.append(_cov_pct(e2e))
    return scenarios, sorted(failed - set(scenarios))


@dataclass
class MetricComparison:
    scenario: str
    metric: str
    unit: str
    baseline: float
    candidate: float
    change_pct: float  # signed: positive means the candidate is worse
    noise_pct: float  # largest observed spread (within-run e2e or between-run), either side
    verdict: str  # "regression", "improvement" or "ok"


def compare(
    baseline: dict[str, ScenarioRuns], candidate: dict[str, ScenarioRuns], threshold_pct: float = 1.0
) -> list[MetricComparison]:
    """Compare every scenario present on both sides. A metric is a regression (or improvement) when the candidate is
    worse (or better) than the baseline by more than `threshold_pct` percent."""
    comparisons = []
    for name in sorted(baseline.keys() & candidate.keys()):
        base, cand = baseline[name], candidate[name]
        for metric, (_, unit, higher_is_better) in METRICS.items():
            b, c = base.median(metric), cand.median(metric)
            change = 100 * (c - b) / b if b else 0.0
            worse_pct = -change if higher_is_better else change
            noise = max(
                [*base.within_run_cov_pct, *cand.within_run_cov_pct]
                + [base.between_run_cov_pct(metric), cand.between_run_cov_pct(metric)]
            )
            if worse_pct > threshold_pct:
                verdict = "regression"
            elif worse_pct < -threshold_pct:
                verdict = "improvement"
            else:
                verdict = "ok"
            comparisons.append(MetricComparison(name, metric, unit, b, c, worse_pct, noise, verdict))
    return comparisons


def format_report(
    comparisons: list[MetricComparison],
    baseline: dict[str, ScenarioRuns],
    candidate: dict[str, ScenarioRuns],
    failed: dict[str, list[str]],
    threshold_pct: float,
) -> str:
    lines = [f"Threshold: {threshold_pct}% (positive change = candidate is worse)", ""]
    scenario = None
    for comp in comparisons:
        if comp.scenario != scenario:
            scenario = comp.scenario
            lines.append(scenario)
        flag = {"regression": "  <-- REGRESSION", "improvement": "  <-- improvement"}.get(comp.verdict, "")
        if comp.noise_pct >= threshold_pct:
            flag += f"  (noisy: {comp.noise_pct:.2f}% spread >= threshold)"
        lines.append(
            f"  {comp.metric:<20} {comp.baseline:>12.5g} -> {comp.candidate:<12.5g} {comp.unit:<6}"
            f" {comp.change_pct:+7.2f}%  noise {comp.noise_pct:5.2f}%{flag}"
        )
    for label, names in (
        ("Only in baseline", sorted(baseline.keys() - candidate.keys())),
        ("Only in candidate", sorted(candidate.keys() - baseline.keys())),
        ("Failed in baseline", failed.get("baseline", [])),
        ("Failed in candidate", failed.get("candidate", [])),
    ):
        if names:
            lines += ["", f"{label}:"] + [f"  {n}" for n in names]
    regressions = sum(c.verdict == "regression" for c in comparisons)
    improvements = sum(c.verdict == "improvement" for c in comparisons)
    lines += ["", f"{regressions} regression(s), {improvements} improvement(s), {len(comparisons)} metric(s) compared"]
    return "\n".join(lines)
