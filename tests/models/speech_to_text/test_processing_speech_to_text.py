# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import shutil
import tempfile
import unittest
from pathlib import Path
from shutil import copyfile

from transformers import Speech2TextProcessor, Speech2TextTokenizer, SpeechToTextAudioProcessor
from transformers.models.speech_to_text.tokenization_speech_to_text import VOCAB_FILES_NAMES, save_json
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_torch, require_torchaudio

from ...test_processing_common import floats_list


SAMPLE_SP = get_tests_dir("fixtures/test_sentencepiece.model")


@require_torch
@require_torchaudio
@require_sentencepiece
class Speech2TextProcessorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()

        vocab = ["<s>", "<pad>", "</s>", "<unk>", "▁This", "▁is", "▁a", "▁t", "est"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        save_dir = Path(cls.tmpdirname)
        save_json(vocab_tokens, save_dir / VOCAB_FILES_NAMES["vocab_file"])
        if not (save_dir / VOCAB_FILES_NAMES["spm_file"]).exists():
            copyfile(SAMPLE_SP, save_dir / VOCAB_FILES_NAMES["spm_file"])

        tokenizer = Speech2TextTokenizer.from_pretrained(cls.tmpdirname)
        tokenizer.save_pretrained(cls.tmpdirname)

        audio_processor_map = {
            "feature_size": 24,
            "num_mel_bins": 24,
            "padding_value": 0.0,
            "sampling_rate": 16000,
            "return_attention_mask": False,
            "do_normalize": True,
        }
        audio_processor = SpeechToTextAudioProcessor(**audio_processor_map)
        tokenizer = Speech2TextTokenizer.from_pretrained(cls.tmpdirname)
        processor = Speech2TextProcessor(tokenizer=tokenizer, audio_processor=audio_processor)
        processor.save_pretrained(cls.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return Speech2TextTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_audio_processor(self, **kwargs):
        return SpeechToTextAudioProcessor.from_pretrained(self.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        audio_processor = self.get_audio_processor()

        processor = Speech2TextProcessor(tokenizer=tokenizer, audio_processor=audio_processor)

        processor.save_pretrained(self.tmpdirname)
        processor = Speech2TextProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.tokenizer, Speech2TextTokenizer)

        self.assertEqual(processor.audio_processor.to_json_string(), audio_processor.to_json_string())
        self.assertIsInstance(processor.audio_processor, SpeechToTextAudioProcessor)

    def test_save_load_pretrained_additional_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = Speech2TextProcessor(
                tokenizer=self.get_tokenizer(), audio_processor=self.get_audio_processor()
            )
            processor.save_pretrained(tmpdir)

            tokenizer_add_kwargs = Speech2TextTokenizer.from_pretrained(tmpdir, bos_token="(BOS)", eos_token="(EOS)")
            audio_processor_add_kwargs = SpeechToTextAudioProcessor.from_pretrained(
                tmpdir, do_normalize=False, padding_value=1.0
            )

            processor = Speech2TextProcessor.from_pretrained(
                tmpdir, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
            )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, Speech2TextTokenizer)

        self.assertEqual(processor.audio_processor.to_json_string(), audio_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.audio_processor, SpeechToTextAudioProcessor)

    def test_audio_processor(self):
        audio_processor = self.get_audio_processor()
        tokenizer = self.get_tokenizer()

        processor = Speech2TextProcessor(tokenizer=tokenizer, audio_processor=audio_processor)

        raw_speech = floats_list((3, 1000))

        input_feat_extract = audio_processor(raw_speech, return_tensors="np")
        input_processor = processor(raw_speech, return_tensors="np")

        for key in input_feat_extract:
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        audio_processor = self.get_audio_processor()
        tokenizer = self.get_tokenizer()

        processor = Speech2TextProcessor(tokenizer=tokenizer, audio_processor=audio_processor)

        input_str = "This is a test string"

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str)

        for key in encoded_tok:
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_tokenizer_decode(self):
        audio_processor = self.get_audio_processor()
        tokenizer = self.get_tokenizer()

        processor = Speech2TextProcessor(tokenizer=tokenizer, audio_processor=audio_processor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)
