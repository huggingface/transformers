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

import tempfile
import unittest

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from transformers import (
    AutoProcessor,
    D1OmniFeatureExtractor,
    D1OmniImageProcessor,
    D1OmniProcessor,
    TokenizersBackend,
)
from transformers.testing_utils import require_librosa, require_torch, require_torchvision, require_vision

from .test_modeling_d1_omni import tiny_config


def processor():
    tokens = ["<unk>", "<pad>", "<s>", "<|mask|>"] + [f"<|reserved_{i}|>" for i in range(7, 12)]
    tokens += ["A", "cat", "dog", "state", "true", "false", "yes", "no", "Pick", "low", "high", "Is", "this", "a", "?"]
    backend = Tokenizer(WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = TokenizersBackend(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        additional_special_tokens=tokens[3:9],
    )
    return D1OmniProcessor(D1OmniImageProcessor(), D1OmniFeatureExtractor(), tokenizer, max_length=256)


@require_torch
@require_torchvision
@require_vision
@require_librosa
class D1OmniProcessorTest(unittest.TestCase):
    def test_marker_order_and_truncation(self):
        value = processor()
        inputs = value(
            state="cat " * 1000,
            questions=[
                {"type": "choice", "instructions": "Pick", "criteria": {"cat": "", "dog": ""}},
                {"type": "noul", "instructions": "Is this a cat?"},
                {"type": "score", "instructions": "Pick", "criteria": ["low", "high"]},
            ],
        )
        self.assertLessEqual(inputs.input_ids.shape[1], 256)
        self.assertEqual(inputs.question_type_ids.tolist(), [0, 2, 1])
        self.assertEqual(inputs.marker_mask.sum(-1).tolist(), [2, 2, 2])
        self.assertTrue(
            (
                inputs.input_ids.gather(1, inputs.marker_positions)
                == value.tokenizer.convert_tokens_to_ids("<|mask|>")
            ).all()
        )
        noul = value.tokenizer.decode(inputs.input_ids[1])
        self.assertLess(noul.index("false"), noul.index("true"))

    def test_save_reload(self):
        value = processor()
        question = {"type": "noul", "instructions": "Is this a cat?"}
        expected = value(state="cat", questions=[question])
        with tempfile.TemporaryDirectory() as directory:
            value.save_pretrained(directory)
            loaded = AutoProcessor.from_pretrained(directory)
            self.assertIsInstance(loaded, D1OmniProcessor)
            actual = loaded(state="cat", questions=[question])
        for key in expected:
            self.assertTrue((expected[key] == actual[key]).all())

    def test_audio_preprocessing_and_validation(self):
        value = processor()
        inputs = value(
            audio=np.zeros(200, dtype=np.int16), questions=[{"type": "noul", "instructions": "Is this a cat?"}]
        )
        self.assertEqual(inputs.input_features.shape, (1, 51, 128))
        self.assertEqual(inputs.feature_attention_mask.sum(), 50)
        self.assertTrue(inputs.input_features.isfinite().all())
        with self.assertRaises(ValueError):
            value(audio=np.zeros(8000), sampling_rate=8000, questions=[{"type": "noul"}])
        with self.assertRaises(ValueError):
            value(audio=np.zeros((2, 8000)), questions=[{"type": "noul"}])

    def test_decision_pipeline(self):
        from transformers import D1OmniForDecision, pipeline

        value = processor()
        model = D1OmniForDecision(tiny_config()).eval()
        decide = pipeline("decision", model=model, processor=value, tokenizer=value.tokenizer)
        result = decide(
            {
                "state": "A cat",
                "questions": {
                    "animal": {"type": "choice", "instructions": "Pick", "criteria": {"cat": "", "dog": ""}},
                    "is_cat": {"type": "noul", "instructions": "Is this a cat?"},
                },
            }
        )
        self.assertEqual(set(result), {"animal", "is_cat"})
        self.assertIn(result["animal"]["choice"], ("cat", "dog"))
        self.assertGreaterEqual(result["is_cat"]["noul"], 0)
        self.assertLessEqual(result["is_cat"]["noul"], 1)
