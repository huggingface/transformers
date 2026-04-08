#!/usr/bin/env python
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

"""
Aggregate multiple failure report JSON files into a single file.

This script reads all JSON files from a directory and combines them
into a single JSON array.
"""

import argparse
import json
import sys
from pathlib import Path


def aggregate_failures(input_dir, output_file):
    """
    Aggregate failure reports from multiple JSON files.

    Args:
        input_dir: Directory containing failure report JSON files
        output_file: Path to output aggregated JSON file

    Returns:
        Number of failures aggregated
    """
    failures = []
    input_path = Path(input_dir)

    if input_path.exists() and input_path.is_dir():
        for failure_file in input_path.glob("*.json"):
            try:
                with open(failure_file) as f:
                    failure_data = json.load(f)
                    failures.append(failure_data)
            except Exception as e:
                print(f"Error reading {failure_file}: {e}", file=sys.stderr)

    # Write aggregated failures
    with open(output_file, "w") as f:
        json.dump(failures, f, indent=2)

    print(f"Aggregated {len(failures)} failure(s) from {input_dir} to {output_file}")
    return len(failures)


def main():
    parser = argparse.ArgumentParser(description="Aggregate failure report JSON files")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing failure report JSON files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output file path for aggregated JSON",
    )

    args = parser.parse_args()

    aggregate_failures(args.input_dir, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
