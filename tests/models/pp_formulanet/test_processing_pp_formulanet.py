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

import time
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

    def test_remove_chinese_text_wrapping_strips_cjk_wrapping(self):
        processor = self.get_processor()
        self.assertEqual(processor.remove_chinese_text_wrapping("\\text{中文}"), "中文")
        # `\s*` between `\text` and `{` is allowed.
        self.assertEqual(processor.remove_chinese_text_wrapping("\\text  {中}"), "中")
        # Embedded double quotes are stripped after the match (existing behaviour).
        self.assertEqual(processor.remove_chinese_text_wrapping('\\text{"中"}'), "中")
        # Multiple wrappings: CJK ones get stripped, non-CJK ones left alone.
        self.assertEqual(
            processor.remove_chinese_text_wrapping("\\text{中}A\\text{B}"),
            "中A\\text{B}",
        )

    def test_remove_chinese_text_wrapping_leaves_non_cjk_untouched(self):
        processor = self.get_processor()
        # No CJK in body -> wrapping preserved.
        self.assertEqual(processor.remove_chinese_text_wrapping("\\text{ABC}"), "\\text{ABC}")
        # Empty body -> wrapping preserved.
        self.assertEqual(processor.remove_chinese_text_wrapping("\\text{}"), "\\text{}")
        # Plain LaTeX with no `\text{...}` is untouched.
        self.assertEqual(
            processor.remove_chinese_text_wrapping("a + b = c"),
            "a + b = c",
        )

    def test_remove_chinese_text_wrapping_unclosed_brace_is_linear(self):
        # Regression test for the polynomial-ReDoS in the previous
        # `\\text\s*{([^{}]*[一-鿿]+[^{}]*)}` pattern: the overlapping
        # `[^{}]*` quantifiers around an inner CJK class caused cubic
        # backtracking on `\text{` followed by a long CJK run with no closing
        # brace. On the unpatched code, the call below takes tens of seconds
        # for ~10 KB of CJK input; on the patched code it returns in under a
        # millisecond. The 2-second budget is intentionally loose so this
        # test stays robust on slow CI runners.
        processor = self.get_processor()
        payload = "\\text{" + ("中" * 3200)
        start = time.perf_counter()
        result = processor.remove_chinese_text_wrapping(payload)
        elapsed = time.perf_counter() - start
        self.assertLess(
            elapsed,
            2.0,
            f"remove_chinese_text_wrapping took {elapsed:.3f}s on an unclosed-brace CJK input; "
            "regex may have regressed to the cubic-backtracking pattern.",
        )
        # No closing brace -> no replacement applied.
        self.assertEqual(result, payload)
