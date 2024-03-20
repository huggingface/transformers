# Copyright 2024 The HuggingFace Team and the AllenNLP authors. All rights reserved.
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
Benchmark for models' `from_pretrained` method
"""
import argparse
import json

from benchmark_utils_generic import BenchMark, SpeedBenchMark

import transformers


class FromPretrainedBenchMark(BenchMark):
    def _target(self, model_class, repo_id):
        self._run_buffer["config"]["target_kwargs"]["model_class"] = model_class
        self._run_buffer["config"]["target_kwargs"]["repo_id"] = repo_id

        def target():
            _ = getattr(transformers, model_class).from_pretrained(repo_id)

        return target


class FromPretrainedSpeedBenchMark(SpeedBenchMark, FromPretrainedBenchMark):
    pass


if __name__ == "__main__":
    from transformers import AutoModel

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", default=None, type=str, required=False, help="Path to a prepared run file or a previously run output file."
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to the output file where the run's info. will be saved."
    )
    args = parser.parse_args()

    if args.config_path is None:
        init_kwargs = {}

        repo_id = "bert-base-uncased"
        run_kwargs = {
            "measure_kwargs": {"number": 2, "repeat": 3},
            "target_kwargs": {"model_class": "AutoModel", "repo_id": repo_id},
            "inputs_kwargs": [{}],
            "report_kwargs": {"output_path": "benchmark_report.json"},
        }
        run_configs = [run_kwargs]
    else:
        with open(args.config_path) as fp:
            config = json.load(fp)
            init_kwargs = config["init_kwargs"]
            run_configs = [run["config"] for run in config["runs"]]

    benchmark = FromPretrainedSpeedBenchMark(**init_kwargs)

    for run_config in run_configs:
        run_config["report_kwargs"]["output_path"] = args.output_path
        result = benchmark.run(**run_config)
