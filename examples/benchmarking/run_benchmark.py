# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
    try:
        benchmark_args = parser.parse_args_into_dataclasses()[0]
    except ValueError as e:
        arg_error_msg = "Arg --no_{0} is no longer used, please use --no-{0} instead."
        begin_error_msg = " ".join(str(e).split(" ")[:-1])
        full_error_msg = ""
        depreciated_args = eval(str(e).split(" ")[-1])
        wrong_args = []
        for arg in depreciated_args:
            # arg[2:] removes '--'
            if arg[2:] in PyTorchBenchmarkArguments.deprecated_args:
                # arg[5:] removes '--no_'
                full_error_msg += arg_error_msg.format(arg[5:])
            else:
                wrong_args.append(arg)
        if len(wrong_args) > 0:
            full_error_msg = full_error_msg + begin_error_msg + str(wrong_args)
        raise ValueError(full_error_msg)

    benchmark = PyTorchBenchmark(args=benchmark_args)
    benchmark.run()


if __name__ == "__main__":
    main()
