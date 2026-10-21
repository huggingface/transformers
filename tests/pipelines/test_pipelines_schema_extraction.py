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

from transformers import AutoConfig
from transformers.testing_utils import require_torch
from transformers.utils import is_tokenizers_available, is_torch_available


if is_torch_available():
    from transformers import Gliner2Config, Gliner2ForSchemaExtraction, Gliner2Processor


def _tiny_encoder(vocab_size=128):
    return AutoConfig.for_model(
        "deberta-v2",
        vocab_size=vocab_size,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        relative_attention=True,
        pos_att_type=["p2c", "c2p"],
        type_vocab_size=0,
    )


def _word_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    from transformers import PreTrainedTokenizerFast

    specials = [
        "[PAD]",
        "[UNK]",
        "[SEP_STRUCT]",
        "[SEP_TEXT]",
        "[P]",
        "[C]",
        "[E]",
        "[R]",
        "[L]",
        "[EXAMPLE]",
        "[OUTPUT]",
        "[DESCRIPTION]",
        "(",
        ")",
    ]
    words = ["ada", "lovelace", "wrote", "notes", "person"]
    vocab = {token: index for index, token in enumerate(specials + words)}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")


@require_torch
@unittest.skipUnless(is_tokenizers_available(), "tokenizers is required")
class SchemaExtractionPipelineTests(unittest.TestCase):
    model_mapping = {"schema-extraction": Gliner2ForSchemaExtraction} if is_torch_available() else {}

    def test_pipeline_returns_a_dict(self):
        from transformers import pipeline

        tokenizer = _word_tokenizer()
        processor = Gliner2Processor(tokenizer)
        config = Gliner2Config(encoder_config=_tiny_encoder(vocab_size=max(tokenizer.vocab_size, 32)), max_width=4)
        model = Gliner2ForSchemaExtraction(config).eval()
        extractor = pipeline("schema-extraction", model=model, processor=processor)
        result = extractor("Ada Lovelace wrote notes", schema={"entities": {"person": {}}})
        self.assertIsInstance(result, dict)
        self.assertIn("entities", result)
