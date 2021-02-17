#!/usr/bin/env python
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from pathlib import Path

import fire


def minify(src_dir: str, dest_dir: str, n: int):
    """Write first n lines of each file f in src_dir to dest_dir/f """
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)
    for path in src_dir.iterdir():
        new = [x.rstrip() for x in list(path.open().readlines())][:n]
        dest_path = dest_dir.joinpath(path.name)
        print(dest_path)
        dest_path.open("w").write("\n".join(new))


if __name__ == "__main__":
    fire.Fire(minify)
