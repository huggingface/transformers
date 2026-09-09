# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from huggingface_hub.errors import StrictDataclassClassValidationError

from transformers import DeepseekV3Config


class DeepseekV3ConfigValidationTest(unittest.TestCase):
    def test_rejects_invalid_group_topology(self):
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "at least 2 routed experts per group"):
            DeepseekV3Config(n_routed_experts=8, n_group=8, topk_group=1, num_experts_per_tok=2)

        with self.assertRaisesRegex(StrictDataclassClassValidationError, "must be divisible by"):
            DeepseekV3Config(n_routed_experts=10, n_group=4, topk_group=1, num_experts_per_tok=2)

    def test_rejects_routing_topk_beyond_selected_group_capacity(self):
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "cannot exceed the 2 experts available"):
            DeepseekV3Config(n_routed_experts=8, n_group=4, topk_group=1, num_experts_per_tok=5)

        DeepseekV3Config(n_routed_experts=8, n_group=4, topk_group=1, num_experts_per_tok=2)


if __name__ == "__main__":
    unittest.main()
