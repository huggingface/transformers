# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from transformers.utils import is_tokenizers_available, is_torch_available

from .test_modeling_gliner2 import _tiny_encoder, _word_tokenizer


@unittest.skipUnless(is_tokenizers_available() and is_torch_available(), "tokenizers and torch are required")
class Gliner2ProcessorTest(unittest.TestCase):
    """Tokenizer-only processor checks.

    ProcessorTesterMixin.setUpClass loads a Hub `model_id` or builds every
    modality component, and its call tests expect `processor(text=...)` to
    return only `model_input_names` tensors. Gliner2Processor requires a schema
    and returns metadata, so the mixin does not apply.
    """

    def test_entity_ids_and_postprocess(self):
        from transformers import Gliner2Config, Gliner2ForSchemaExtraction, Gliner2Processor

        tokenizer = _word_tokenizer()
        processor = Gliner2Processor(tokenizer, word_splitter="whitespace")
        schema = {"entities": {"person": {}}}
        encoded = processor("Ada Lovelace wrote notes", schema=schema, return_tensors="pt")
        self.assertIn("input_ids", encoded)
        self.assertGreater(int(encoded["text_word_mask"].sum()), 0)
        self.assertTrue(bool(encoded["prompt_marker_mask"].any()))
        config = Gliner2Config(encoder_config=_tiny_encoder(vocab_size=tokenizer.vocab_size), max_width=4)
        model = Gliner2ForSchemaExtraction(config).eval()
        model_inputs = {key: value for key, value in encoded.items() if key != "metadata"}
        outputs = model(**model_inputs)
        decoded = processor.post_process_extraction(outputs, encoded["metadata"], threshold=0.5)
        self.assertEqual(len(decoded), 1)
        self.assertIn("entities", decoded[0])
