# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import unittest

from transformers.testing_utils import is_torch_available, require_torch


if is_torch_available():
    import torch

    from transformers.integrations.heterogeneity import ReturnEntry, get_skip_replacement


@require_torch
class TestSkipReplacement(unittest.TestCase):
    def test_get_skip_replacement_returns_none(self):
        replacement_factory = get_skip_replacement(torch.nn.Linear, None)

        self.assertIsNone(replacement_factory()(torch.randn(2, 4)))

    def test_get_skip_replacement_transforms_configured_argument(self):
        replacement_factory = get_skip_replacement(
            torch.nn.Linear, ReturnEntry(arg_name="input", transform=lambda x: x * 2)
        )
        module = replacement_factory()
        inputs = torch.randn(2, 4, 64)

        torch.testing.assert_close(module(inputs), inputs * 2)

    def test_get_skip_replacement_raises_for_unknown_return_argument(self):
        with self.assertRaisesRegex(ValueError, "return entry arg names.*missing"):
            get_skip_replacement(torch.nn.Linear, ReturnEntry(arg_name="missing", transform=lambda x: x))

    def test_get_skip_replacement_adds_context_to_transform_error(self):
        def fail_transform(_):
            raise RuntimeError("transform failed")

        replacement_factory = get_skip_replacement(
            torch.nn.Linear, ReturnEntry(arg_name="input", transform=fail_transform)
        )

        with self.assertRaisesRegex(RuntimeError, "failed to apply transform.*argument 'input'.*transform failed"):
            replacement_factory()(torch.randn(2, 4))
