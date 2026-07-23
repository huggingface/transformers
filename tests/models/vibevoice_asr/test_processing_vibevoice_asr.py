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

import numpy as np
import torch

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    VibeVoiceAcousticTokenizerFeatureExtractor,
    VibeVoiceAsrProcessor,
)
from transformers.testing_utils import require_torch

from ...test_processing_common import ProcessorTesterMixin


class VibeVoiceAsrProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = VibeVoiceAsrProcessor
    audio_input_name = "input_values"
    # Tiny processor created with make_tiny_processor.py from "microsoft/VibeVoice-ASR-HF"
    tiny_model_id = "hf-internal-testing/tiny-processor-vibevoice_asr"

    @classmethod
    @require_torch
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        processor = VibeVoiceAsrProcessor.from_pretrained(cls.tiny_model_id)
        processor.save_pretrained(cls.tmpdirname)

    @require_torch
    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    @require_torch
    def get_feature_extractor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).feature_extractor

    @require_torch
    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @require_torch
    def test_can_load_various_tokenizers(self):
        processor = VibeVoiceAsrProcessor.from_pretrained(self.tiny_model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.tiny_model_id)
        self.assertEqual(processor.tokenizer.__class__, tokenizer.__class__)

    @require_torch
    def test_save_load_pretrained_default(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tiny_model_id)
        processor = VibeVoiceAsrProcessor.from_pretrained(self.tiny_model_id)
        feature_extractor = processor.feature_extractor

        processor = VibeVoiceAsrProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_pretrained(tmpdir)
            reloaded = VibeVoiceAsrProcessor.from_pretrained(tmpdir)

        self.assertEqual(reloaded.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(reloaded.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(reloaded.feature_extractor, VibeVoiceAcousticTokenizerFeatureExtractor)

    @require_torch
    def test_apply_transcription_request_single(self):
        processor = self.get_processor()

        audio_url = (
            "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/f2641_0_throatclearing.wav"
        )
        helper_outputs = processor.apply_transcription_request(audio=audio_url, prompt="About VibeVoice")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "About VibeVoice"},
                    {
                        "type": "audio",
                        "path": audio_url,
                    },
                ],
            }
        ]
        manual_outputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        )

        for key in ("input_ids", "attention_mask", "input_values", "padding_mask"):
            self.assertIn(key, helper_outputs)
            self.assertTrue(helper_outputs[key].equal(manual_outputs[key]))

    # Override: VibeVoice's chat template does not support `continue_final_message`.
    @require_torch
    def _test_apply_chat_template(
        self,
        modality: str,
        batch_size: int,
        return_tensors: str,
        input_name: str,
        processor_name: str,
        input_data: list[str],
    ):
        if return_tensors != "pt":
            self.skipTest("VibeVoiceAsrProcessor only supports PyTorch tensors")
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        if processor_name not in self.processor_class.get_attributes():
            self.skipTest(f"{processor_name} attribute not present in {self.processor_class}")

        # some models have only Fast image processor
        if getattr(processor, processor_name).__class__.__name__.endswith("Fast"):
            return_tensors = "pt"

        batch_messages = [
            [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": "Describe this."}]},
            ]
        ] * batch_size

        # Test that jinja can be applied
        formatted_prompt = processor.apply_chat_template(batch_messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), batch_size)

        # Test that tokenizing with template and directly with `self.tokenizer` gives same output
        formatted_prompt_tokenized = processor.apply_chat_template(
            batch_messages, add_generation_prompt=True, tokenize=True, return_tensors=return_tensors
        )
        add_special_tokens = True
        if processor.tokenizer.bos_token is not None and formatted_prompt[0].startswith(processor.tokenizer.bos_token):
            add_special_tokens = False
        tok_output = processor.tokenizer(
            formatted_prompt, return_tensors=return_tensors, add_special_tokens=add_special_tokens
        )
        expected_output = tok_output.input_ids
        self.assertListEqual(expected_output.tolist(), formatted_prompt_tokenized.tolist())

        # Test that kwargs passed to processor's `__call__` are actually used
        tokenized_prompt_100 = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=return_tensors,
            processor_kwargs={
                "padding": "max_length",
                "truncation": True,
                "max_length": self.chat_template_max_length,
            },
        )
        self.assertEqual(len(tokenized_prompt_100[0]), self.chat_template_max_length)

        # Test that `return_dict=True` returns text related inputs in the dict
        out_dict_text = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
        )
        self.assertTrue(all(key in out_dict_text for key in ["input_ids", "attention_mask"]))
        self.assertEqual(len(out_dict_text["input_ids"]), batch_size)
        self.assertEqual(len(out_dict_text["attention_mask"]), batch_size)

        # Test that with modality URLs and `return_dict=True`, we get modality inputs in the dict
        for idx, url in enumerate(input_data[:batch_size]):
            batch_messages[idx][1]["content"] = [batch_messages[idx][1]["content"][0], {"type": modality, "url": url}]

        out_dict = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
            processor_kwargs={"num_frames": 2},  # by default no more than 2 frames, otherwise too slow
        )
        input_name = getattr(self, input_name)
        self.assertTrue(input_name in out_dict)
        self.assertEqual(len(out_dict["input_ids"]), batch_size)
        self.assertEqual(len(out_dict["attention_mask"]), batch_size)
        self.assertEqual(len(out_dict[input_name]), batch_size)

        return_tensor_to_type = {"pt": torch.Tensor, "np": np.ndarray, None: list}
        for k in out_dict:
            self.assertIsInstance(out_dict[k], return_tensor_to_type[return_tensors])

    @require_torch
    def test_decode_output_formats(self):
        from unittest.mock import patch

        import torch

        processor = self.get_processor()

        # This test is about the processor's ability to parse the model output into structured
        # dicts (return_format="parsed") or plain transcriptions (return_format="transcription_only").
        # We are NOT testing tokenizer decoding here, so it is fine to mock batch_decode.
        # The mock string below is the exact output obtained by decoding the original generated_ids
        # with the full processor (microsoft/VibeVoice-ASR-HF) prior to PR #47213, which switched
        # to a tiny tokenizer that would decode those IDs to garbage and break json.loads().
        generated_ids = torch.tensor([[0]])
        # The decode method calls tokenizer.decode (singular) with skip_special_tokens=True.
        # When called with a 2D tensor (batch), the tokenizer returns a list of strings.
        # extract_speaker_dict then returns list[list[dict]] for a list input.
        # The mock string has special tokens already stripped (skip_special_tokens=True).
        mock_decoded = [
            'assistant\n[{"Start":0,"End":7.56,"Speaker":0,"Content":"Revevoices is a novel framework designed for generating expressive, long-form, multi-speaker conversational audio."}]\n'
        ]

        with patch.object(processor.tokenizer, "decode", return_value=mock_decoded):
            # test parsed output
            dicts = processor.decode(generated_ids, return_format="parsed")
            self.assertIsInstance(dicts, list)
            self.assertIsInstance(dicts[0], list)
            self.assertIsInstance(dicts[0][0], dict)
            self.assertIn("Content", dicts[0][0])
            self.assertIn("Start", dicts[0][0])
            self.assertIn("End", dicts[0][0])
            self.assertIsInstance(dicts[0][0]["Start"], float)
            self.assertIsInstance(dicts[0][0]["End"], float)

            # test transcript only
            transcript = processor.decode(generated_ids, return_format="transcription_only")
            self.assertIsInstance(transcript, list)
            self.assertIsInstance(transcript[0], str)
