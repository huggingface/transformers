# Copyright 2025 The HuggingFace Team. All rights reserved.
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
Shared utilities for the serving layer.
"""

from __future__ import annotations


class _StreamError:
    """Sentinel to signal an error from the generate thread."""

    def __init__(self, msg: str):
        self.msg = msg


def set_torch_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)


def reset_torch_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
