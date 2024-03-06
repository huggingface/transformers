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
import json
import timeit


# def save_args(func):
#
#     info = inspect.signature(func)
#
#     def wrapper(*args, **kwargs):
#         self = args[0]
#
#         if not hasattr(self, "init_kwargs"):
#
#             parameters = info.parameters
#
#             pos_parameters = []
#             keyword_parameters = {}
#
#             for parameter in parameters.values():
#                 parameter_name = parameter.name
#                 if parameter.default == inspect._empty:
#                     pos_parameters.append(parameter_name)
#                 else:
#                     keyword_parameters[parameter_name] = parameter.default
#
#             parameter_names = pos_parameters + list(keyword_parameters.keys())
#
#             arguments = {}
#
#             if len(args) > 0:
#                 for parameter_name, arg_value in zip(pos_parameters[:len(args)], args):
#                     if parameter_name == "self":
#                         continue
#                     arguments[parameter_name] = arg_value
#
#             # sorted
#             for parameter_name in parameter_names:
#                 if parameter_name == "self":
#                     continue
#                 elif parameter_name in arguments:
#                     continue
#                 if parameter_name in kwargs:
#                     arguments[parameter_name] = kwargs[parameter_name]
#                 else:
#                     arguments[parameter_name] = keyword_parameters[parameter_name]
#
#             self.init_kwargs = arguments
#
#         func(*args, **kwargs)
#
#     return wrapper


class BenchMark:
    def __init__(self, *arg, **kwargs):
        pass

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
            report["init_kwargs"] = getattr(self, "init_kwargs", {})
            report["run_kwargs"] = run_info

        report = self._convert_to_json(report)
        if output_path is not None:
            with open(output_path, "w", encoding="UTF-8") as fp:
                json.dump(report, fp, ensure_ascii=False, indent=4)

        return report

    def run(self, measure_kwargs=None, target_kwargs=None, inputs_kwargs=None, report_kwargs=None):
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
