# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Benchmarking the library on inference and training """

from transformers import HfArgumentParser, PyTorchBenchmark, PyTorchBenchmarkArguments


def main():
    parser = HfArgumentParser(PyTorchBenchmarkArguments)
    benchmark_args = parser.parse_args_into_dataclasses()[0]
    benchmark = PyTorchBenchmark(args=benchmark_args)
    benchmark.run()


if __name__ == "__main__":
    main()
