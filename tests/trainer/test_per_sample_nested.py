# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import torch

from transformers.trainer_pt_utils import nested_truncate


def test_nested_truncate_tuple_of_lists_truncates_at_sample_level():
    """
    Regression test for tuple[list[Tensor], ...] structures.

    When truncating the last batch, truncation must happen at the
    sample/list level and must not truncate tensor dimensions or
    drop tuple elements.
    """
    remainder = 1

    labels = (
        [torch.randn(6, 4), torch.randn(3, 4)],
        [torch.randint(0, 5, (6,)), torch.randint(0, 5, (3,))],
    )

    truncated = nested_truncate(labels, remainder)

    assert isinstance(truncated, tuple)
    assert len(truncated) == 2

    assert isinstance(truncated[0], list)
    assert isinstance(truncated[1], list)

    assert len(truncated[0]) == remainder
    assert len(truncated[1]) == remainder
