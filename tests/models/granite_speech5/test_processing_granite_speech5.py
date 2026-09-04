# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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

from transformers import GraniteSpeech5Processor, ParakeetTokenizer
from transformers.testing_utils import require_torch, require_torchaudio

from ...test_processing_common import ProcessorTesterMixin


@require_torch
@require_torchaudio
class GraniteSpeech5ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = GraniteSpeech5Processor
    text_input_name = "labels"

    @classmethod
    def _setup_tokenizer(cls):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace

        # tiny BPE tokenizer with the CTC blank at id 0, mirroring the real checkpoint's vocabulary layout
        vocab = {"<|blank|>": 0, "<unk>": 1, "l": 2, "o": 3, "w": 4, "e": 5, "r": 6, "lo": 7, "low": 8, "er": 9}
        merges = [("l", "o"), ("lo", "w"), ("e", "r")]
        tokenizer_object = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token="<unk>"))
        tokenizer_object.pre_tokenizer = Whitespace()
        return ParakeetTokenizer(tokenizer_object=tokenizer_object, pad_token="<|blank|>", unk_token="<unk>")
