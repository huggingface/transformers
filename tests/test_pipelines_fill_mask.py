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

import unittest

from transformers import pipeline
from transformers.testing_utils import require_tf, require_torch, slow

from .test_pipelines_common import MonoInputPipelineCommonMixin


EXPECTED_FILL_MASK_RESULT = [
    [
        {"sequence": "My name is John", "score": 0.00782308354973793, "token": 610, "token_str": " John"},
        {"sequence": "My name is Chris", "score": 0.007475061342120171, "token": 1573, "token_str": " Chris"},
    ],
    [
        {
            "sequence": "The largest city in France is Paris",
            "score": 0.2510891854763031,
            "token": 2201,
            "token_str": " Paris",
        },
        {
            "sequence": "The largest city in France is Lyon",
            "score": 0.21418564021587372,
            "token": 12790,
            "token_str": " Lyon",
        },
    ],
]

EXPECTED_FILL_MASK_TARGET_RESULT = [EXPECTED_FILL_MASK_RESULT[0]]


class FillMaskPipelineTests(MonoInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "fill-mask"
    pipeline_loading_kwargs = {"top_k": 2}
    small_models = ["sshleifer/tiny-distilroberta-base"]  # Models tested without the @slow decorator
    large_models = ["distilroberta-base"]  # Models tested with the @slow decorator
    mandatory_keys = {"sequence", "score", "token"}
    valid_inputs = [
        "My name is <mask>",
        "The largest city in France is <mask>",
    ]
    invalid_inputs = [
        "This is <mask> <mask>"  # More than 1 mask_token in the input is not supported
        "This is"  # No mask_token is not supported
    ]
    expected_check_keys = ["sequence"]

    @require_torch
    def test_torch_fill_mask(self):
        valid_inputs = "My name is <mask>"
        nlp = pipeline(task="fill-mask", model=self.small_models[0])
        outputs = nlp(valid_inputs)
        self.assertIsInstance(outputs, list)

        # This passes
        outputs = nlp(valid_inputs, targets=[" Patrick", " Clara"])
        self.assertIsInstance(outputs, list)

        # This used to fail with `cannot mix args and kwargs`
        outputs = nlp(valid_inputs, something=False)
        self.assertIsInstance(outputs, list)

    @require_torch
    def test_torch_fill_mask_with_targets(self):
        valid_inputs = ["My name is <mask>"]
        valid_targets = [[" Teven", " Patrick", " Clara"], [" Sam"]]
        invalid_targets = [[], [""], ""]
        for model_name in self.small_models:
            nlp = pipeline(task="fill-mask", model=model_name, tokenizer=model_name, framework="pt")
            for targets in valid_targets:
                outputs = nlp(valid_inputs, targets=targets)
                self.assertIsInstance(outputs, list)
                self.assertEqual(len(outputs), len(targets))
            for targets in invalid_targets:
                self.assertRaises(ValueError, nlp, valid_inputs, targets=targets)

    @require_tf
    def test_tf_fill_mask_with_targets(self):
        valid_inputs = ["My name is <mask>"]
        valid_targets = [[" Teven", " Patrick", " Clara"], [" Sam"]]
        invalid_targets = [[], [""], ""]
        for model_name in self.small_models:
            nlp = pipeline(task="fill-mask", model=model_name, tokenizer=model_name, framework="tf")
            for targets in valid_targets:
                outputs = nlp(valid_inputs, targets=targets)
                self.assertIsInstance(outputs, list)
                self.assertEqual(len(outputs), len(targets))
            for targets in invalid_targets:
                self.assertRaises(ValueError, nlp, valid_inputs, targets=targets)

    @require_torch
    @slow
    def test_torch_fill_mask_results(self):
        mandatory_keys = {"sequence", "score", "token"}
        valid_inputs = [
            "My name is <mask>",
            "The largest city in France is <mask>",
        ]
        valid_targets = [" Patrick", " Clara"]
        for model_name in self.large_models:
            nlp = pipeline(
                task="fill-mask",
                model=model_name,
                tokenizer=model_name,
                framework="pt",
                top_k=2,
            )

            mono_result = nlp(valid_inputs[0], targets=valid_targets)
            self.assertIsInstance(mono_result, list)
            self.assertIsInstance(mono_result[0], dict)

            for mandatory_key in mandatory_keys:
                self.assertIn(mandatory_key, mono_result[0])

            multi_result = [nlp(valid_input) for valid_input in valid_inputs]
            self.assertIsInstance(multi_result, list)
            self.assertIsInstance(multi_result[0], (dict, list))

            for result, expected in zip(multi_result, EXPECTED_FILL_MASK_RESULT):
                for r, e in zip(result, expected):
                    self.assertEqual(r["sequence"], e["sequence"])
                    self.assertEqual(r["token_str"], e["token_str"])
                    self.assertEqual(r["token"], e["token"])
                    self.assertAlmostEqual(r["score"], e["score"], places=3)

            if isinstance(multi_result[0], list):
                multi_result = multi_result[0]

            for result in multi_result:
                for key in mandatory_keys:
                    self.assertIn(key, result)

            self.assertRaises(Exception, nlp, [None])

            valid_inputs = valid_inputs[:1]
            mono_result = nlp(valid_inputs[0], targets=valid_targets)
            self.assertIsInstance(mono_result, list)
            self.assertIsInstance(mono_result[0], dict)

            for mandatory_key in mandatory_keys:
                self.assertIn(mandatory_key, mono_result[0])

            multi_result = [nlp(valid_input) for valid_input in valid_inputs]
            self.assertIsInstance(multi_result, list)
            self.assertIsInstance(multi_result[0], (dict, list))

            for result, expected in zip(multi_result, EXPECTED_FILL_MASK_TARGET_RESULT):
                for r, e in zip(result, expected):
                    self.assertEqual(r["sequence"], e["sequence"])
                    self.assertEqual(r["token_str"], e["token_str"])
                    self.assertEqual(r["token"], e["token"])
                    self.assertAlmostEqual(r["score"], e["score"], places=3)

            if isinstance(multi_result[0], list):
                multi_result = multi_result[0]

            for result in multi_result:
                for key in mandatory_keys:
                    self.assertIn(key, result)

            self.assertRaises(Exception, nlp, [None])

    @require_tf
    @slow
    def test_tf_fill_mask_results(self):
        mandatory_keys = {"sequence", "score", "token"}
        valid_inputs = [
            "My name is <mask>",
            "The largest city in France is <mask>",
        ]
        valid_targets = [" Patrick", " Clara"]
        for model_name in self.large_models:
            nlp = pipeline(task="fill-mask", model=model_name, tokenizer=model_name, framework="tf", top_k=2)

            mono_result = nlp(valid_inputs[0], targets=valid_targets)
            self.assertIsInstance(mono_result, list)
            self.assertIsInstance(mono_result[0], dict)

            for mandatory_key in mandatory_keys:
                self.assertIn(mandatory_key, mono_result[0])

            multi_result = [nlp(valid_input) for valid_input in valid_inputs]
            self.assertIsInstance(multi_result, list)
            self.assertIsInstance(multi_result[0], (dict, list))

            for result, expected in zip(multi_result, EXPECTED_FILL_MASK_RESULT):
                for r, e in zip(result, expected):
                    self.assertEqual(r["sequence"], e["sequence"])
                    self.assertEqual(r["token_str"], e["token_str"])
                    self.assertEqual(r["token"], e["token"])
                    self.assertAlmostEqual(r["score"], e["score"], places=3)

            if isinstance(multi_result[0], list):
                multi_result = multi_result[0]

            for result in multi_result:
                for key in mandatory_keys:
                    self.assertIn(key, result)

            self.assertRaises(Exception, nlp, [None])

            valid_inputs = valid_inputs[:1]
            mono_result = nlp(valid_inputs[0], targets=valid_targets)
            self.assertIsInstance(mono_result, list)
            self.assertIsInstance(mono_result[0], dict)

            for mandatory_key in mandatory_keys:
                self.assertIn(mandatory_key, mono_result[0])

            multi_result = [nlp(valid_input) for valid_input in valid_inputs]
            self.assertIsInstance(multi_result, list)
            self.assertIsInstance(multi_result[0], (dict, list))

            for result, expected in zip(multi_result, EXPECTED_FILL_MASK_TARGET_RESULT):
                for r, e in zip(result, expected):
                    self.assertEqual(r["sequence"], e["sequence"])
                    self.assertEqual(r["token_str"], e["token_str"])
                    self.assertEqual(r["token"], e["token"])
                    self.assertAlmostEqual(r["score"], e["score"], places=3)

            if isinstance(multi_result[0], list):
                multi_result = multi_result[0]

            for result in multi_result:
                for key in mandatory_keys:
                    self.assertIn(key, result)

            self.assertRaises(Exception, nlp, [None])
