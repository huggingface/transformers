# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
Utility that checks the supports of 3rd party libraries are listed in the documentation file. Currently, this includes:
- flash attention support
- SDPA support

Use from the root of the repo with (as used in `make repo-consistency`):

```bash
python utils/check_support_list.py
```

It has no auto-fix mode.
"""

import os
from glob import glob


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_doctest_list.py
REPO_PATH = "."


def check_flash_support_list():
    with open(os.path.join(REPO_PATH, "docs/source/en/perf_infer_gpu_one.md"), "r") as f:
        doctext = f.read()

        doctext = doctext.split("FlashAttention-2 is currently supported for the following architectures:")[1]
        doctext = doctext.split("You can request to add FlashAttention-2 support")[0]

    patterns = glob(os.path.join(REPO_PATH, "src/transformers/models/**/modeling_*.py"))
    patterns_tf = glob(os.path.join(REPO_PATH, "src/transformers/models/**/modeling_tf_*.py"))
    patterns_flax = glob(os.path.join(REPO_PATH, "src/transformers/models/**/modeling_flax_*.py"))
    patterns = list(set(patterns) - set(patterns_tf) - set(patterns_flax))
    archs_supporting_fa2 = []
    for filename in patterns:
        with open(filename, "r") as f:
            text = f.read()

            if "_supports_flash_attn_2 = True" in text:
                model_name = os.path.basename(filename).replace(".py", "").replace("modeling_", "")
                archs_supporting_fa2.append(model_name)

    for arch in archs_supporting_fa2:
        if arch not in doctext:
            raise ValueError(
                f"{arch} should be in listed in the flash attention documentation but is not. Please update the documentation."
            )


def check_sdpa_support_list():
    with open(os.path.join(REPO_PATH, "docs/source/en/perf_infer_gpu_one.md"), "r") as f:
        doctext = f.read()

        doctext = doctext.split(
            "For now, Transformers supports SDPA inference and training for the following architectures:"
        )[1]
        doctext = doctext.split("Note that FlashAttention can only be used for models using the")[0]
        doctext = doctext.lower()

    patterns = glob(os.path.join(REPO_PATH, "src/transformers/models/**/modeling_*.py"))
    patterns_tf = glob(os.path.join(REPO_PATH, "src/transformers/models/**/modeling_tf_*.py"))
    patterns_flax = glob(os.path.join(REPO_PATH, "src/transformers/models/**/modeling_flax_*.py"))
    patterns = list(set(patterns) - set(patterns_tf) - set(patterns_flax))
    archs_supporting_sdpa = []
    for filename in patterns:
        with open(filename, "r") as f:
            text = f.read()

            if "_supports_sdpa = True" in text:
                model_name = os.path.basename(filename).replace(".py", "").replace("modeling_", "")
                archs_supporting_sdpa.append(model_name)

    for arch in archs_supporting_sdpa:
        if not any(term in doctext for term in [arch, arch.replace("_", "-"), arch.replace("_", " ")]):
            raise ValueError(
                f"{arch} should be in listed in the SDPA documentation but is not. Please update the documentation."
            )


if __name__ == "__main__":
    check_flash_support_list()
    check_sdpa_support_list()
