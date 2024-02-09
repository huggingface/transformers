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

    def run(self, measure_kwargs, prep_kwargs, input_kwargs):

        return self.measure(self.prepare_target(**prep_kwargs), **measure_kwargs)(**input_kwargs)

    def report(self):
        pass


class FromPretrainedBenchMark(BenchMark):

    def prepare_target(self, model_class, repo_id):

        def target():
            _ = model_class.from_pretrained(repo_id)

        return target

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


if __name__ == "__main__":

    from transformers import AutoModel
    repo_id = "bert-base-uncased"

    benchmakr = FromPretrainedBenchMark()

    run_kwargs = {
        "measure_kwargs": {"number": 2, "repeat": 3},
        "prep_kwargs": {"model_class": AutoModel, "repo_id": repo_id},
        "input_kwargs": {},
    }
    result = benchmakr.run(**run_kwargs)
    print(result)
