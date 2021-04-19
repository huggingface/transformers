#!/usr/bin/env python
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import argparse
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--mask-tokens", type=int, required=True)
    parser.add_argument("--group-files", type=int, default=50)
    args = parser.parse_args()
    input_files = glob.glob(os.path.join(args.input_files, "*", "*"))

    os.makedirs(args.output_dir, exist_ok=True)

    output_index = 0
    out_files_factor = 5
    file_index_size = len(str(len(input_files) * out_files_factor))
    for offset in range(0, len(input_files), args.group_files):
        grouped_files = ','.join(input_files[offset:offset+args.group_files])
        out_range_to = len(input_files[offset:offset+args.group_files]) * out_files_factor
        output_files = [os.path.join(args.output_dir, ("wiki_{:0"+str(file_index_size)+"d}.tfrecord").format(output_index)) for output_index in range(output_index, output_index + out_range_to)]
        grouped_output_files = ','.join(output_files)
        subprocess.run(
            "python3 third_party/create_pretraining_data.py "
            f"--input-file {grouped_files} "
            f"--output-file {grouped_output_files} "
            f"--sequence-length {args.sequence_length} "
            f"--mask-tokens {args.mask_tokens} "
            "--duplication-factor 5 "
            "--do-lower-case "
            "--model bert-base-uncased",
            stderr=subprocess.STDOUT,
            shell=True)
        output_index += out_range_to
