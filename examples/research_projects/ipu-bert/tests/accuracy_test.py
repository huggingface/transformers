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

import torch
from bert_model import accuracy, accuracy_masked


def test_accuracy():
    pred = torch.tensor(
        [[1.0, 2.0, 5.0],
         [5.0, 2.0, 1.0],
         [2.0, 5.0, 1.0],
         [2.0, 1.0, 5.0]]
    )
    # test all right = 100% accuracy
    labels = torch.tensor([2, 0, 1, 2])
    assert accuracy(pred, labels) == 1.0

    # test all wrong = 0% accuracy
    labels = torch.tensor([0, 1, 0, 0])
    assert accuracy(pred, labels) == 0.0

    # test 50% right
    labels = torch.tensor([2, 1, 1, 0])
    assert accuracy(pred, labels) == 0.5


def test_accuracy_masked():
    ignore_token = -100
    # prediction tensor dimensions:
    #   [bs, seq_len, vocab_size]
    pred = torch.tensor(
        [
            [[1.0, 2.0, 5.0],
             [5.0, 2.0, 1.0]],
            [[2.0, 5.0, 1.0],
             [2.0, 1.0, 5.0]]
        ]
    )
    # label tensor dimensions:
    #   [bs, seq_len]
    labels = torch.tensor(
        [[2, 0], [1, 2]]
    )

    # No mask with 100% correct
    assert accuracy_masked(pred, labels, ignore_token) == 1.0

    # No mask with 0% correct
    labels = torch.tensor(
        [[1, 2], [0, 1]]
    )
    assert accuracy_masked(pred, labels, ignore_token) == 0.0

    # with 1 mask token per sequence with 100% correct
    labels = torch.tensor(
        [[ignore_token, 0], [1, ignore_token]]
    )
    assert accuracy_masked(pred, labels, ignore_token) == 1.0

    # with 1 mask token per sequence with 0% correct
    labels = torch.tensor(
        [[ignore_token, 2], [0, ignore_token]]
    )
    assert accuracy_masked(pred, labels, ignore_token) == 0.0

    # with 1 mask token per sequence with 50% correct
    labels = torch.tensor(
        [[ignore_token, 2], [1, ignore_token]]
    )
    assert accuracy_masked(pred, labels, ignore_token) == 0.5

    # with only mask tokens should be nan
    labels = torch.tensor(
        [[ignore_token, ignore_token], [ignore_token, ignore_token]]
    )
    assert accuracy_masked(pred, labels, ignore_token).isnan()
