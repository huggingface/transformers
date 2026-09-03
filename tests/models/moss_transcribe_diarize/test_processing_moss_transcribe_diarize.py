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

import shutil
import tempfile
import unittest

import torch
from parameterized import parameterized

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    MossTranscribeDiarizeProcessor,
    WhisperFeatureExtractor,
)
from transformers.testing_utils import require_librosa, require_torch

from ...test_processing_common import MODALITY_INPUT_DATA, ProcessorTesterMixin


_CHECKPOINT = "OpenMOSS-Team/MOSS-Transcribe-Diarize"


class MossTranscribeDiarizeProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MossTranscribeDiarizeProcessor
    model_id = _CHECKPOINT
    audio_text_kwargs_max_length = 512
    chat_template_max_length = 512

    @classmethod
    def prepare_processor_dict(cls):
        return {
            "audio_tokens_per_second": 12.5,
            "audio_merge_size": 4,
            "time_marker_every_seconds": 5,
            "enable_time_marker": True,
        }

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)
        if cls.full_tmpdirname is not None:
            shutil.rmtree(cls.full_tmpdirname, ignore_errors=True)

    @require_torch
    def test_can_load_various_tokenizers(self):
        processor = MossTranscribeDiarizeProcessor.from_pretrained(self.tmpdirname)
        tokenizer = AutoTokenizer.from_pretrained(self.tmpdirname)
        self.assertEqual(processor.tokenizer.__class__, tokenizer.__class__)

    @require_torch
    def test_save_load_pretrained_default(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tmpdirname)
        processor = MossTranscribeDiarizeProcessor.from_pretrained(self.tmpdirname)
        feature_extractor = processor.feature_extractor

        processor = MossTranscribeDiarizeProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_pretrained(tmpdir)
            reloaded = MossTranscribeDiarizeProcessor.from_pretrained(tmpdir)

        self.assertEqual(reloaded.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(reloaded.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(reloaded.feature_extractor, WhisperFeatureExtractor)
        self.assertEqual(reloaded.enable_time_marker, processor.enable_time_marker)
        self.assertEqual(reloaded.time_marker_every_seconds, processor.time_marker_every_seconds)

    @require_torch
    def test_chat_template(self):
        processor = AutoProcessor.from_pretrained(self.tmpdirname, trust_remote_code=True)
        expected_prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|audio_start|><|audio_pad|><|audio_end|>\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "path": "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
                    },
                ],
            },
        ]
        formatted_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertEqual(expected_prompt, formatted_prompt)

    @require_torch
    def test_processor_call_with_prepare_audio_inputs(self):
        processor = self.get_processor()
        audio = self.prepare_audio_inputs(batch_size=2)
        text = [
            f"{processor.audio_bos_token}{processor.audio_token}{processor.audio_eos_token}",
            f"{processor.audio_bos_token}{processor.audio_token}{processor.audio_eos_token}",
        ]

        outputs = processor(text=text, audio=audio)

        for key in ("input_ids", "attention_mask", "input_features", "audio_feature_lengths", "audio_chunk_mapping"):
            self.assertIn(key, outputs)
        self.assertEqual(outputs["input_ids"].shape[0], 2)
        self.assertEqual(outputs["input_features"].shape[0], 2)
        self.assertEqual(outputs["audio_chunk_mapping"].tolist(), [0, 1])
        self.assertEqual(outputs["audio_feature_lengths"].tolist(), [1, 1])

    @require_torch
    def test_apply_chat_template_matches_processor_call(self):
        processor = self.get_processor()
        audio = self.prepare_audio_inputs(batch_size=1)[0]
        conversation = [
            {
                "role": "user",
                "content": [{"type": "audio", "audio": audio}],
            }
        ]
        template_outputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
        )
        formatted_prompt = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        manual_outputs = processor(text=formatted_prompt, audio=[audio])

        for key in ("input_ids", "attention_mask", "input_features", "audio_feature_lengths", "audio_chunk_mapping"):
            self.assertIn(key, template_outputs)
            self.assertTrue(template_outputs[key].equal(manual_outputs[key]))

    @require_librosa
    @require_torch
    def test_apply_chat_template_with_audio_url(self):
        processor = self.get_processor()
        audio_url = MODALITY_INPUT_DATA["audio"][0]
        conversation = [
            {
                "role": "user",
                "content": [{"type": "audio", "url": audio_url}],
            }
        ]
        outputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
        )

        self.assertIn("input_features", outputs)
        self.assertIn("audio_chunk_mapping", outputs)
        self.assertGreater(outputs["input_features"].shape[0], 0)
        self.assertEqual(outputs["input_ids"].shape[0], 1)

    @require_torch
    def test_apply_transcription_request(self):
        processor = self.get_processor()
        audio = self.prepare_audio_inputs(batch_size=1)[0]
        prompt = "Transcribe and diarize this clip."

        helper_outputs = processor.apply_transcription_request(audio=[audio], prompt=prompt)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        manual_outputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
        )

        for key in ("input_ids", "attention_mask", "input_features", "audio_feature_lengths", "audio_chunk_mapping"):
            self.assertIn(key, helper_outputs)
            self.assertTrue(helper_outputs[key].equal(manual_outputs[key]))

    @require_librosa
    @require_torch
    def test_apply_transcription_request_with_url(self):
        processor = self.get_processor()
        audio_url = MODALITY_INPUT_DATA["audio"][0]
        outputs = processor.apply_transcription_request(audio=audio_url)

        for key in ("input_ids", "attention_mask", "input_features", "audio_feature_lengths", "audio_chunk_mapping"):
            self.assertIn(key, outputs)
        self.assertEqual(outputs["input_ids"].shape[0], 1)

    def test_feature_extractor_defaults(self):
        self.skipTest("MossTranscribeDiarizeProcessor requires text and audio together.")

    @parameterized.expand([(1, "np"), (1, "pt"), (2, "np"), (2, "pt")])
    def test_apply_chat_template_audio(self, batch_size: int, return_tensors: str):
        self.skipTest(
            "MossTranscribeDiarizeProcessor requires audio in the chat template; "
            "see test_apply_chat_template_matches_processor_call instead."
        )

    def test_apply_chat_template_assistant_mask(self):
        self.skipTest(
            "MossTranscribeDiarizeProcessor requires audio in the chat template; "
            "not compatible with text-only assistant mask tests."
        )

    def test_apply_chat_template_tool_calls_no_content(self):
        self.skipTest(
            "MossTranscribeDiarizeProcessor requires audio in the chat template; "
            "not compatible with text-only tool-call tests."
        )

    @require_torch
    def test_apply_chat_template_batch_with_prepare_audio_inputs(self):
        processor = self.get_processor()
        batch_size = 2
        audios = self.prepare_audio_inputs(batch_size=batch_size)
        conversations = [[{"role": "user", "content": [{"type": "audio", "audio": audio}]}] for audio in audios]
        outputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        self.assertEqual(outputs["input_ids"].shape[0], batch_size)
        self.assertEqual(outputs["input_features"].shape[0], batch_size)
        self.assertEqual(outputs["audio_chunk_mapping"].tolist(), list(range(batch_size)))

    @require_torch
    def test_model_input_names(self):
        processor = self.get_processor()
        text = self.prepare_text_inputs(modalities=["audio"])
        audio = self.prepare_audio_inputs()
        inputs = processor(text=text, audio=audio, return_tensors="pt")
        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    @require_torch
    def test_flat_kwarg_applied_when_modality_dict_lacks_it(self):
        processor = self.get_processor()
        text = self.prepare_text_inputs(modalities=["audio"])
        audio = self.prepare_audio_inputs()
        inputs = processor(text=text, audio=audio, text_kwargs={}, return_tensors="pt")
        for key, value in inputs.items():
            self.assertIsInstance(value, torch.Tensor, msg=f"{key} should be a torch.Tensor")
