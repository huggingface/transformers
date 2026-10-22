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

import shutil
import tempfile
import unittest

import numpy as np

from transformers import AutoProcessor, CanaryProcessor
from transformers.testing_utils import require_torch


def _get_prompt(source: str, target: str, pnc: bool = True) -> str:
    return (
        "<|startofcontext|><|startoftranscript|><|emo:undefined|>"
        f"<|{source}|><|{target}|>"
        f"{'<|pnc|>' if pnc else '<|nopnc|>'}<|noitn|><|notimestamp|><|nodiarize|>"
    )


@require_torch
class CanaryProcessorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = "nvidia/canary-1b-v2"
        cls.tmpdirname = tempfile.mkdtemp()
        CanaryProcessor.from_pretrained(cls.checkpoint).save_pretrained(cls.tmpdirname)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def get_processor(self):
        return AutoProcessor.from_pretrained(self.tmpdirname)

    def _audio(self, num_samples: int = 16000):
        return np.zeros(num_samples, dtype=np.float32)

    def _decode_prompt(self, processor, inputs, index: int = 0) -> str:
        return processor.tokenizer.decode(inputs["decoder_input_ids"][index], skip_special_tokens=False)

    def test_chat_template_is_loaded(self):
        self.assertIsNotNone(self.get_processor().chat_template)

    def test_apply_transcription_request_transcription(self):
        processor = self.get_processor()
        inputs = processor.apply_transcription_request(audio=self._audio(), source_language="en")
        self.assertIn("input_features", inputs)
        self.assertEqual(self._decode_prompt(processor, inputs), _get_prompt("en", "en"))

    def test_apply_transcription_request_translation(self):
        processor = self.get_processor()
        inputs = processor.apply_transcription_request(audio=self._audio(), source_language="en", target_language="de")
        self.assertEqual(self._decode_prompt(processor, inputs), _get_prompt("en", "de"))

    def test_apply_transcription_request_with_transcription(self):
        processor = self.get_processor()
        audio = [self._audio(), self._audio(24000)]
        transcriptions = ["Hello world.", "A longer transcription so that padding is applied."]

        helper_outputs = processor.apply_transcription_request(
            audio=audio, source_language="en", transcription=transcriptions
        )
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_el},
                        {"type": "text", "source_language": "en", "target_language": "en", "punctuation": True},
                    ],
                },
                {"role": "assistant", "content": text},
            ]
            for audio_el, text in zip(audio, transcriptions)
        ]
        manual_outputs = processor.apply_chat_template(
            conversations, tokenize=True, return_dict=True, processor_kwargs={"output_labels": True}
        )

        self.assertIn("labels", helper_outputs)
        self.assertEqual(set(helper_outputs.keys()), set(manual_outputs.keys()))
        for key in helper_outputs:
            self.assertTrue(helper_outputs[key].equal(manual_outputs[key]))

        # only the transcripts and their EOS are kept, the decoder prompt and the padding are masked
        labels = helper_outputs["labels"]
        for row, text in enumerate(transcriptions):
            kept = processor.tokenizer.decode(labels[row][labels[row] != -100])
            self.assertEqual(kept.strip(), f"{text}<|endoftext|>")

    def test_punctuation_flag(self):
        processor = self.get_processor()
        inputs = processor.apply_transcription_request(audio=self._audio(), source_language="en", punctuation=False)
        self.assertEqual(self._decode_prompt(processor, inputs), _get_prompt("en", "en", pnc=False))

    def test_batch_broadcast_and_per_sample(self):
        processor = self.get_processor()
        inputs = processor.apply_transcription_request(
            audio=[self._audio(), self._audio()], source_language="en", target_language=["en", "es"]
        )
        self.assertEqual(len(inputs["decoder_input_ids"]), 2)
        self.assertEqual(self._decode_prompt(processor, inputs, 0), _get_prompt("en", "en"))
        self.assertEqual(self._decode_prompt(processor, inputs, 1), _get_prompt("en", "es"))

    def test_batch_length_mismatch_raises(self):
        processor = self.get_processor()
        with self.assertRaises(ValueError):
            processor.apply_transcription_request(audio=[self._audio()], source_language=["en", "de"])

    def test_call_output_labels(self):
        processor = self.get_processor()
        outputs = processor(audio=self._audio(), text="hello world", output_labels=True)
        self.assertIn("input_features", outputs)
        self.assertIn("decoder_input_ids", outputs)
        self.assertIn("labels", outputs)
        # the decoder inputs are already right-shifted with respect to `labels`
        self.assertListEqual(outputs["decoder_input_ids"][..., 1:].tolist(), outputs["labels"][..., :-1].tolist())
