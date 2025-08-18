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

from .dependency_versions_table import deps
from .utils.versions import require_version, require_version_core


# define which module versions we always want to check at run time
# (usually the ones defined in `install_requires` in setup.py)
#
# order specific notes:
# - tqdm must be checked before tokenizers

pkgs_to_check_at_runtime = [
    "python",
    "tqdm",
    "regex",
    "requests",
    "packaging",
    "filelock",
    "numpy",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
    "accelerate",
    "pyyaml",
]

for pkg in pkgs_to_check_at_runtime:
    if pkg in deps:
        if pkg == "tokenizers":
            # must be loaded here, or else tqdm check may fail
            from .utils import is_tokenizers_available

            if not is_tokenizers_available():
                continue  # not required, check version only if installed
        elif pkg == "accelerate":
            # must be loaded here, or else tqdm check may fail
            from .utils import is_accelerate_available

            # Maybe switch to is_torch_available in the future here so that Accelerate is hard dep of
            # Transformers with PyTorch
            if not is_accelerate_available():
                continue  # not required, check version only if installed

        require_version_core(deps[pkg])
    else:
        raise ValueError(f"can't find {pkg} in {deps.keys()}, check dependency_versions_table.py")


def dep_version_check(pkg, hint=None):
    require_version(deps[pkg], hint)
