# Copyright 2026 Alibaba DAMO Academy and the HuggingFace Inc. team. All rights reserved.
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

import inspect
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from transformers import FunAsrNanoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.audioflamingo3.processing_audioflamingo3 import AudioFlamingo3Processor
from transformers.models.auto.auto_mappings import FEATURE_EXTRACTOR_MAPPING_NAMES, PROCESSOR_MAPPING_NAMES
from transformers.processing_utils import ProcessorMixin
from transformers.testing_utils import execute_subprocess_async, require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

    from transformers.models.fun_asr_nano.modular_fun_asr_nano import (
        FunAsrNanoProcessor as ModularFunAsrNanoProcessor,
    )


class FunAsrNanoProcessorImportTest(unittest.TestCase):
    def test_processing_module_imports_without_torch(self):
        code = """
import sys
sys.modules["torch"] = None
from transformers.models.fun_asr_nano.processing_fun_asr_nano import FunAsrNanoProcessor
print(FunAsrNanoProcessor.__name__)
"""
        env = os.environ.copy()
        env["USE_TORCH"] = "0"
        execute_subprocess_async([sys.executable, "-c", code], env=env, quiet=True)


@require_torch
class FunAsrNanoProcessorTest(unittest.TestCase):
    def test_inherits_audioflamingo3_processor(self):
        self.assertTrue(issubclass(ModularFunAsrNanoProcessor, AudioFlamingo3Processor))

    def test_uses_auto_mappings_for_subprocessors(self):
        self.assertEqual(FEATURE_EXTRACTOR_MAPPING_NAMES["fun_asr_nano"], "FunAsrNanoFeatureExtractor")
        self.assertEqual(PROCESSOR_MAPPING_NAMES["fun_asr_nano"], "FunAsrNanoProcessor")
        self.assertNotIn("feature_extractor_class", ModularFunAsrNanoProcessor.__dict__)
        self.assertNotIn("tokenizer_class", ModularFunAsrNanoProcessor.__dict__)

    def test_constructor_does_not_expose_unused_max_audio_length(self):
        self.assertNotIn("max_audio_len", inspect.signature(ModularFunAsrNanoProcessor.__init__).parameters)
        self.assertNotIn("max_audio_len", inspect.signature(FunAsrNanoProcessor.__init__).parameters)

        tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: 151646)
        with patch(
            "transformers.models.audioflamingo3.processing_audioflamingo3.ProcessorMixin.__init__", return_value=None
        ):
            processor = FunAsrNanoProcessor(object(), tokenizer)

        self.assertEqual(processor.audio_token, "<|object_ref_start|>")
        self.assertEqual(processor.audio_token_id, 151646)
        self.assertFalse(hasattr(processor, "max_audio_len"))

    def test_constructor_defines_fun_asr_nano_default_prompt(self):
        parameter = inspect.signature(FunAsrNanoProcessor.__init__).parameters["default_transcription_prompt"]
        self.assertEqual(parameter.default, "Transcribe the audio:")
        self.assertNotIn("apply_transcription_request", ModularFunAsrNanoProcessor.__dict__)

    def test_modular_processor_reuses_audioflamingo3_call(self):
        self.assertNotIn("__call__", ModularFunAsrNanoProcessor.__dict__)

    def test_audio_token_lengths_match_feature_lengths(self):
        processor = FunAsrNanoProcessor.__new__(FunAsrNanoProcessor)
        lengths = torch.tensor([1, 2, 3, 4])

        self.assertTrue(torch.equal(processor._get_audio_token_length(lengths), lengths))

    def test_process_audio_builds_mask_and_replacements(self):
        class FeatureExtractorStub:
            def __call__(self, audio, **kwargs):
                return BatchFeature(
                    {
                        "input_features": torch.zeros(2, 3, 4),
                        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
                        "feature_lengths": torch.tensor([3, 2]),
                    }
                )

        processor = FunAsrNanoProcessor.__new__(FunAsrNanoProcessor)
        processor.feature_extractor = FeatureExtractorStub()
        processor.audio_token = "<audio>"

        audio_inputs, replacements = processor._process_audio([torch.zeros(4), torch.zeros(3)])

        self.assertNotIn("attention_mask", audio_inputs)
        self.assertIn("input_features_mask", audio_inputs)
        self.assertTrue(torch.equal(audio_inputs["num_audio_tokens"], torch.tensor([3, 2])))
        self.assertEqual(replacements, ["<audio><audio><audio>", "<audio><audio>"])

    def test_process_audio_requires_attention_mask(self):
        class FeatureExtractorStub:
            def __call__(self, audio, **kwargs):
                return BatchFeature(
                    {
                        "input_features": torch.zeros(1, 3, 4),
                        "feature_lengths": torch.tensor([3]),
                    }
                )

        processor = FunAsrNanoProcessor.__new__(FunAsrNanoProcessor)
        processor.feature_extractor = FeatureExtractorStub()

        with self.assertRaisesRegex(ValueError, "attention mask"):
            processor._process_audio([torch.zeros(4)])

    def test_output_labels_use_token_ids_and_mask_audio_and_padding(self):
        processor = FunAsrNanoProcessor.__new__(FunAsrNanoProcessor)
        processor.tokenizer = SimpleNamespace(pad_token_id=0)
        model_inputs = BatchFeature(
            {
                "input_ids": torch.tensor([[10, 151646, 20, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0]]),
                "mm_token_type_ids": torch.tensor([[0, 3, 0, 0]]),
            }
        )

        with patch.object(ProcessorMixin, "__call__", return_value=model_inputs):
            outputs = processor(text="prompt", audio=[torch.zeros(4)], output_labels=True)

        self.assertNotIn("mm_token_type_ids", outputs)
        self.assertTrue(torch.equal(outputs["labels"], torch.tensor([[10, -100, 20, -100]])))

    def test_decode_strip_prefix_preserves_scalar_output(self):
        processor = FunAsrNanoProcessor.__new__(FunAsrNanoProcessor)
        processor.tokenizer = SimpleNamespace(
            decode=lambda *args, **kwargs: 'The transcription of the audio is "hello".'
        )

        decoded = processor.decode([1, 2, 3], strip_prefix=True)

        self.assertEqual(decoded, "hello")

    def test_decode_handles_batches_without_processor_override(self):
        processor = FunAsrNanoProcessor.__new__(FunAsrNanoProcessor)
        processor.tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: ["first", "second"])

        self.assertEqual(processor.decode([[1], [2]]), ["first", "second"])
        self.assertNotIn("batch_decode", ModularFunAsrNanoProcessor.__dict__)
        self.assertIs(ModularFunAsrNanoProcessor.batch_decode, AudioFlamingo3Processor.batch_decode)

    def test_unused_audio_metadata_is_removed_from_model_inputs(self):
        processor = FunAsrNanoProcessor.__new__(FunAsrNanoProcessor)
        self.assertEqual(set(processor.unused_input_names), {"feature_lengths", "num_audio_tokens"})

    def _make_processor(self):
        captured = {}
        processor = FunAsrNanoProcessor.__new__(FunAsrNanoProcessor)
        processor.default_transcription_prompt = "Transcribe the audio:"

        def apply_chat_template(conversations, **kwargs):
            captured["conversations"] = conversations
            captured["kwargs"] = kwargs
            return {"ok": True}

        processor.apply_chat_template = apply_chat_template
        return processor, captured

    def test_apply_transcription_request_single_path(self):
        processor, captured = self._make_processor()

        self.assertEqual(processor.apply_transcription_request(audio="audio.wav"), {"ok": True})

        self.assertEqual(
            captured["conversations"],
            [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcribe the audio:"},
                            {"type": "audio", "path": "audio.wav"},
                        ],
                    }
                ]
            ],
        )
        self.assertTrue(captured["kwargs"]["tokenize"])
        self.assertTrue(captured["kwargs"]["add_generation_prompt"])
        self.assertTrue(captured["kwargs"]["return_dict"])

    def test_apply_transcription_request_batch_paths_and_prompts(self):
        processor, captured = self._make_processor()

        self.assertEqual(
            processor.apply_transcription_request(
                audio=["zh.wav", "en.wav"],
                prompt=["语音转写成中文：", "Transcribe the audio:"],
            ),
            {"ok": True},
        )

        self.assertEqual(
            captured["conversations"],
            [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "语音转写成中文："},
                            {"type": "audio", "path": "zh.wav"},
                        ],
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcribe the audio:"},
                            {"type": "audio", "path": "en.wav"},
                        ],
                    }
                ],
            ],
        )


if __name__ == "__main__":
    unittest.main()
