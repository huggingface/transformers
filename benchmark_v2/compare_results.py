#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Compare benchmark results against a baseline and flag regressions.

    python compare_results.py --baseline results_main/ --candidate results_pr/ --threshold 1.0

Each side is a results JSON file or a directory of them (e.g. several repeated runs). Exits with status 1 if any
metric regressed by more than the threshold, so it can gate CI.
"""

import argparse
import json
import sys
from dataclasses import asdict

from framework.compare import compare, format_report, load_scenarios


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", nargs="+", required=True, help="Baseline result file(s) or directory(ies)")
    parser.add_argument("--candidate", nargs="+", required=True, help="Candidate result file(s) or directory(ies)")
    parser.add_argument(
        "--threshold", type=float, default=1.0, help="Change (in %%) above which a metric is a regression"
    )
    parser.add_argument("--json", type=str, help="Also write the comparison to this JSON file")
    args = parser.parse_args()

    baseline, baseline_failed = load_scenarios(args.baseline)
    candidate, candidate_failed = load_scenarios(args.candidate)
    comparisons = compare(baseline, candidate, args.threshold)
    print(
        format_report(
            comparisons,
            baseline,
            candidate,
            {"baseline": baseline_failed, "candidate": candidate_failed},
            args.threshold,
        )
    )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in comparisons], f, indent=2)
    sys.exit(1 if any(c.verdict == "regression" for c in comparisons) else 0)
