# Copyright 2026 HuggingFace Inc.
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
import importlib
import unittest
from unittest import mock

from transformers.testing_utils import require_torch


@require_torch
class DistributedCheckpointGuardTest(unittest.TestCase):
    def test_guard_implies_hf_storage_is_importable(self):
        # Regression test for https://github.com/huggingface/transformers/issues/49651:
        # `transformers.distributed.checkpoint` imports `hf_storage` at module level whenever the guard is True,
        # so the guard must never be True on a torch version that lacks that module.
        from transformers.distributed.utils import _check_distributed_checkpointing_available

        if _check_distributed_checkpointing_available():
            importlib.import_module("torch.distributed.checkpoint.hf_storage")

    def test_guard_is_false_below_torch_2_8(self):
        from transformers.distributed import utils as distributed_utils

        def fake_ge(version, *args, **kwargs):
            return tuple(int(x) for x in version.split(".")[:2]) <= (2, 7)

        with (
            mock.patch.object(distributed_utils, "is_torch_distributed_available", return_value=True),
            mock.patch.object(distributed_utils, "is_torch_greater_or_equal", side_effect=fake_ge),
        ):
            self.assertFalse(distributed_utils._check_distributed_checkpointing_available())

    def test_modeling_utils_imports(self):
        # `modeling_utils` imports `distributed.checkpoint` unconditionally; it must import on every supported torch.
        importlib.import_module("transformers.distributed.checkpoint")
        importlib.import_module("transformers.modeling_utils")


if __name__ == "__main__":
    unittest.main()
