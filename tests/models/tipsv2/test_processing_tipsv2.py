# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from transformers import AutoProcessor, Tipsv2ImageProcessor, Tipsv2Processor, Tipsv2Tokenizer
from transformers.testing_utils import require_sentencepiece, require_torch, require_vision

from ...test_processing_common import ProcessorTesterMixin


def get_tipsv2_test_sentencepiece_model(tmp_dir):
    import os

    import sentencepiece as spm

    corpus_file = os.path.join(tmp_dir, "corpus.txt")
    with open(corpus_file, "w", encoding="utf-8") as fp:
        fp.write(
            "\n".join(
                [
                    "a cat on a mat",
                    "a dog in the fog",
                    "mixed case text for tipsv2 tokenizer",
                    "lower newer upper older longer string",
                    "padding tokens should use id zero",
                ]
            )
        )

    model_prefix = os.path.join(tmp_dir, "tipsv2_test")
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=64,
        model_type="unigram",
        character_coverage=1.0,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=3,
        hard_vocab_limit=False,
        num_threads=1,
    )
    return f"{model_prefix}.model"


@require_sentencepiece
@require_torch
@require_vision
class Tipsv2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Tipsv2Processor
    image_text_kwargs_max_length = 64
    image_text_kwargs_override_max_length = 32
    image_unstructured_max_length = 48

    @classmethod
    def _setup_tokenizer(cls):
        vocab_file = get_tipsv2_test_sentencepiece_model(cls.tmpdirname)
        return Tipsv2Tokenizer(vocab_file)

    @classmethod
    def _setup_image_processor(cls):
        return Tipsv2ImageProcessor(size={"height": 16, "width": 16})

    def test_processor_call_and_roundtrip(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp_dir:
            vocab_file = get_tipsv2_test_sentencepiece_model(tmp_dir)
            tokenizer = Tipsv2Tokenizer(vocab_file)
            image_processor = Tipsv2ImageProcessor(size={"height": 16, "width": 16})
            processor = Tipsv2Processor(image_processor=image_processor, tokenizer=tokenizer)

            image = Image.fromarray(np.zeros((18, 18, 3), dtype=np.uint8))
            outputs = processor(
                images=image,
                text=["A Cat on a Mat", "A DOG in the Fog"],
                return_tensors="pt",
            )

            self.assertEqual(outputs.input_ids.shape, (2, 64))
            self.assertEqual(outputs.attention_mask.shape, (2, 64))
            self.assertEqual(outputs.pixel_values.shape, (1, 3, 16, 16))
            text_only_outputs = processor(text=["a cat on a mat"], return_tensors="pt")
            self.assertListEqual(outputs.input_ids[0].tolist(), text_only_outputs.input_ids[0].tolist())

            processor.save_pretrained(tmp_dir)
            reloaded = Tipsv2Processor.from_pretrained(tmp_dir)
            auto_reloaded = AutoProcessor.from_pretrained(tmp_dir)

            self.assertIsInstance(reloaded, Tipsv2Processor)
            self.assertIsInstance(auto_reloaded, Tipsv2Processor)
            self.assertEqual(reloaded.tokenizer.pad_token_id, 0)
            self.assertFalse(reloaded.image_processor.do_normalize)
