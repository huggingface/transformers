#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Inc. team.
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

"""Run the model linter over the files a PR touches and write the findings to JSON.

Runs in the untrusted `pull_request` job, so it only reads the working tree and writes a file. The
companion `post_mlinter_review.py` runs separately with write access and turns this JSON into a review.
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ref", default="origin/main", help="Base ref to diff against.")
    parser.add_argument("--output", type=Path, default=Path("mlinter-findings.json"))
    args = parser.parse_args()

    try:
        from mlinter import (
            DEFAULT_ENABLED_TRF_RULES,
            TRF_RULE_SPECS,
            __version__,
            analyze_file,
            get_changed_modeling_files,
        )
    except ImportError:
        print("transformers-mlinter is not installed; nothing to do.", file=sys.stderr)
        return 0

    changed = sorted(get_changed_modeling_files(args.base_ref))
    findings = []
    for path in changed:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            violations = analyze_file(path, text, enabled_rules=set(DEFAULT_ENABLED_TRF_RULES))
        except Exception as exc:  # a parse error is the PR's problem, not the linter's
            print(f"skipped {path}: {exc}", file=sys.stderr)
            continue
        for violation in violations:
            findings.append(
                {
                    "path": str(violation.file_path),
                    "line": violation.line_number,
                    "rule": violation.rule_id,
                    # The rule id is already the first token of the message; the review renders it
                    # separately, so strip it here rather than in the poster.
                    "message": violation.message.split(": ", 1)[-1],
                }
            )

    rules_used = sorted({finding["rule"] for finding in findings if finding["rule"]})
    payload = {
        "mlinter_version": __version__,
        "changed_files": [str(path) for path in changed],
        "findings": findings,
        "rules": {
            rule: {
                "description": TRF_RULE_SPECS[rule]["description"],
                "why_bad": TRF_RULE_SPECS[rule]["explanation"]["why_bad"],
                "diff": TRF_RULE_SPECS[rule]["explanation"]["diff"],
            }
            for rule in rules_used
            if rule in TRF_RULE_SPECS
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"{len(findings)} finding(s) across {len(changed)} changed model file(s) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
