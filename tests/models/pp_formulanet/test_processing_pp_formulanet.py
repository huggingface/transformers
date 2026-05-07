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

from transformers import PPFormulaNetProcessor
from transformers.testing_utils import require_vision

from ...test_processing_common import ProcessorTesterMixin


# PPFormulaNet is an encoder-decoder VLM that uses pixel_values as encoder inputs.
# It does not consume text input_ids, so processor tests that require input_ids should be skipped.
@require_vision
class PPFormulaNetProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = PPFormulaNetProcessor

    @unittest.skip(reason="PPFormulaNet does not need input_ids")
    def test_model_input_names(self):
        pass

    @unittest.skip(reason="PPFormulaNet does not need input_ids")
    def test_processor_with_multiple_inputs(self):
        pass

    @unittest.skip(reason="PPFormulaNet does not need input_ids")
    def test_structured_kwargs_nested(self):
        pass

    @unittest.skip(reason="PPFormulaNet does not need input_ids")
    def test_structured_kwargs_nested_from_dict(self):
        pass

    @unittest.skip(reason="PPFormulaNet does not need input_ids")
    def test_tokenizer_defaults_preserved_by_kwargs(self):
        pass

    @unittest.skip(reason="PPFormulaNet does not need input_ids")
    def test_unstructured_kwargs(self):
        pass

    @unittest.skip(reason="PPFormulaNet does not need input_ids")
    def test_unstructured_kwargs_batched(self):
        pass

    @unittest.skip(reason="PPFormulaNet does not need input_ids")
    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        pass
