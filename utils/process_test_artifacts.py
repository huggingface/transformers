# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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

This helper computes the "ideal" number of nodes to use in circle CI.
For each job, we compute this parameter and pass it to the `generated_config.yaml`.
"""

import json
import math
import os


MAX_PARALLEL_NODES = 8  # TODO create a mapping!
AVERAGE_TESTS_PER_NODES = 5


def count_lines(filepath):
    """Count the number of lines in a file."""
    try:
        with open(filepath, "r") as f:
            return len(f.read().split("\n"))
    except FileNotFoundError:
        return 0


def compute_parallel_nodes(line_count, max_tests_per_node=10):
    """Compute the number of parallel nodes required."""
    num_nodes = math.ceil(line_count / AVERAGE_TESTS_PER_NODES)
    if line_count < 4:
        return 1
    return min(MAX_PARALLEL_NODES, num_nodes)


def process_artifacts(input_file, output_file):
    # Read the JSON data from the input file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Process items and build the new JSON structure
    transformed_data = {}
    for item in data.get("items", []):
        if "test_list" in item["path"]:
            key = os.path.splitext(os.path.basename(item["path"]))[0]
            transformed_data[key] = item["url"]
            parallel_key = key.split("_test")[0] + "_parallelism"
            file_path = os.path.join("test_preparation", f"{key}.txt")
            line_count = count_lines(file_path)
            transformed_data[parallel_key] = compute_parallel_nodes(line_count)

    # Remove the "generated_config" key if it exists
    if "generated_config" in transformed_data:
        del transformed_data["generated_config"]

    # Write the transformed data to the output file
    with open(output_file, "w") as f:
        json.dump(transformed_data, f, indent=2)


if __name__ == "__main__":
    input_file = "test_preparation/artifacts.json"
    output_file = "test_preparation/transformed_artifacts.json"
    process_artifacts(input_file, output_file)
