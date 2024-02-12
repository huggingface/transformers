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
Benchmark
"""
import timeit


class BenchMark:

    def measure(self, func, **measure_kwargs):
        raise NotImplementedError

    def target(self, **target_kwargs):
        raise NotImplementedError

    def inputs(self, **inputs_kwargs):
        return {}

    def _convert_to_json(self, report):
        if isinstance(report, list):
            return [self._convert_to_json(x) for x in report]
        if isinstance(report, dict):
            return {k: self._convert_to_json(v) for k, v in report.items()}
        if isinstance(report, type):
            return report.__name__
        return report

    def report(self, result, run_info, only_result=False, output_path=None):

        report = {"result": result}
        if not only_result:
            report["run_kwargs"] = run_info

        report = self._convert_to_json(report)

        return report

    def run(self, measure_kwargs, target_kwargs, inputs_kwargs, report_kwargs):
        target = self.target(**target_kwargs)

        all_inputs_kwargs = [inputs_kwargs] if isinstance(inputs_kwargs, dict) else inputs_kwargs
        results = []
        for _inputs_kwargs in all_inputs_kwargs:

            inputs = self.inputs(**_inputs_kwargs)
            result = self.measure(target, **measure_kwargs)(**inputs)
            results.append(result)

        run_info = {
            "measure_kwargs": measure_kwargs,
            "target_kwargs": target_kwargs,
            "inputs_kwargs": inputs_kwargs,
            "report_kwargs": report_kwargs,
        }

        if isinstance(inputs_kwargs, dict):
            results = results[0]

        return self.report(results, run_info, **report_kwargs)


class SpeedBenchMark(BenchMark):

    def measure(self, func, number=3, repeat=1):

        def wrapper():

            # as written in https://docs.python.org/2/library/timeit.html#timeit.Timer.repeat, min should be taken rather than the average
            runtimes = timeit.repeat(
                func,
                repeat=repeat,
                number=number,
            )

            return {"time": min(runtimes) / number}

        return wrapper
