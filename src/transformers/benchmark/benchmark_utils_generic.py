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
Benchmark utilities - module containing base classes for benchmarking
"""
import json
import os.path
import timeit


class BenchMark:
    """Base class specifying the methods to be implemented in benchmark subclasses.

    All the methods except `run` are designed to be private: only the `run` method should be used by an end user.
    """

    def __init__(self, *arg, **kwargs):
        self._buffer = {"init_kwargs": {}, "runs": []}
        self._run_buffer = {
            "config": {
                "inputs_kwargs": {},
                "target_kwargs": {},
                "measure_kwargs": {},
                "report_kwargs": {},
            },
            "result": None,
        }

    def _reset_run_buffer(self):
        self._run_buffer = {
            "config": {
                "inputs_kwargs": {},
                "target_kwargs": {},
                "measure_kwargs": {},
                "report_kwargs": {},
            },
            "result": None,
        }

    def _measure(self, func, **measure_kwargs):
        """Return a callable that, when called, will return some measurement results for the argument `func`.

        See `SpeedBenchMark` for an example implementation.
        """
        raise NotImplementedError

    def _target(self, **target_kwargs):
        """Return a callable against which we would like to perform benchmark.

        See `FromPretrainedBenchMark` and `CacheBenchMark` for example implementations.
        """
        raise NotImplementedError

    def _inputs(self, **inputs_kwargs):
        return {}

    def _convert_to_json(self, report):
        if isinstance(report, list):
            return [self._convert_to_json(x) for x in report]
        if isinstance(report, dict):
            return {k: self._convert_to_json(v) for k, v in report.items()}
        if isinstance(report, type):
            return report.__name__
        return report

    def _report(self, result, output_path=None, only_result=False):
        self._run_buffer["config"]["report_kwargs"]["output_path"] = output_path
        self._run_buffer["config"]["report_kwargs"]["only_result"] = only_result

        self._run_buffer["result"] = result
        self._buffer["runs"].append(self._run_buffer)

        report = {"result": result}
        if not only_result:
            report = self._buffer["runs"][-1]

        complete_report = self._convert_to_json(self._buffer)
        if output_path is not None:

            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            output_path = os.path.join(output_path, "benchmark_report.json")

            with open(output_path, "w", encoding="UTF-8") as fp:
                json.dump(complete_report, fp, ensure_ascii=False, indent=4)

        report = self._convert_to_json(report)

        return report

    def run(self, measure_kwargs=None, target_kwargs=None, inputs_kwargs=None, report_kwargs=None):
        self._reset_run_buffer()

        if measure_kwargs is None:
            measure_kwargs = {}
        if target_kwargs is None:
            target_kwargs = {}
        if inputs_kwargs is None:
            inputs_kwargs = {}
        if report_kwargs is None:
            report_kwargs = {}
        if measure_kwargs is None:
            measure_kwargs = {}

        target = self._target(**target_kwargs)

        all_inputs_kwargs = [inputs_kwargs] if isinstance(inputs_kwargs, dict) else inputs_kwargs
        results = []
        for _inputs_kwargs in all_inputs_kwargs:
            inputs = self._inputs(**_inputs_kwargs)
            result = self._measure(target, **measure_kwargs)(**inputs)
            results.append(result)

        if isinstance(inputs_kwargs, dict):
            results = results[0]

        return self._report(results, **report_kwargs)


class SpeedBenchMark(BenchMark):
    """A simple class used to benchmark the running time of a callable."""

    def _measure(self, func, number=3, repeat=1):
        self._run_buffer["config"]["measure_kwargs"]["number"] = number
        self._run_buffer["config"]["measure_kwargs"]["repeat"] = repeat

        def wrapper(*args, **kwargs):
            # as written in https://docs.python.org/2/library/timeit.html#timeit.Timer.repeat, min should be taken rather than the average

            def _func():
                func(*args, **kwargs)

            runtimes = timeit.repeat(
                _func,
                repeat=repeat,
                number=number,
            )

            return {"time": min(runtimes) / number}

        return wrapper
