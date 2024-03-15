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
import json

from benchmark_utils_generic import BenchMark, SpeedBenchMark


class FromPretrainedBenchMark(BenchMark):
    def _target(self, model_class, repo_id):
        self._buffer["target_kwargs"]["model_class"] = model_class
        self._buffer["target_kwargs"]["repo_id"] = repo_id

        def target():
            _ = model_class.from_pretrained(repo_id)

        return target


class FromPretrainedSpeedBenchMark(SpeedBenchMark, FromPretrainedBenchMark):
    pass


if __name__ == "__main__":
    from transformers import AutoModel

    repo_id = "bert-base-uncased"

    benchmark = FromPretrainedSpeedBenchMark()

    run_kwargs = {
        "measure_kwargs": {"number": 2, "repeat": 3},
        "target_kwargs": {"model_class": AutoModel, "repo_id": repo_id},
        "inputs_kwargs": [{}],
        "report_kwargs": {"output_path": "benchmark_report.json"},
    }
    result = benchmark.run(**run_kwargs)
    print(json.dumps(result, indent=4))
