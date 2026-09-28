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

import unittest

from parameterized import parameterized

from transformers import OmniASRProcessor
from transformers.testing_utils import require_torch
from transformers.utils.import_utils import is_torch_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch


SAMPLING_RATE = 16000


@require_torch
class OmniASRCTCProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = OmniASRProcessor
    audio_input_name = "input_values"
    text_input_name = "labels"
    model_id = "bezzam/omniasr-ctc-300m-v2"


@require_torch
class OmniASRLLMProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = OmniASRProcessor
    model_id = "bezzam/omniasr-llm-300m-v2"

    @classmethod
    def prepare_processor_dict(cls):
        # The prompt settings (language mapping, prompt token ids, convolution geometry) live in the processor
        # config rather than in its components, so building the processor from components needs them passed.
        processor_dict, _ = OmniASRProcessor.get_processor_dict(cls.model_id)
        return {k: v for k, v in processor_dict.items() if k not in ("feature_extractor", "processor_class")}

    # The decoder prompt is built and left-padded by the processor itself (its length follows the audio), so the
    # text `padding`/`max_length` kwargs these tests check do not shape `input_ids`.
    @parameterized.expand(["images", "videos", "audio"])
    def test_kwargs_overrides_default_subprocessor_kwargs(self, modality):
        self.skipTest("The OmniASR LLM prompt is not padded with the text kwargs.")

    @parameterized.expand(["images", "videos", "audio"])
    def test_structured_kwargs_nested_from_dict(self, modality):
        self.skipTest("The OmniASR LLM prompt is not padded with the text kwargs.")

    @parameterized.expand(["images", "videos", "audio"])
    def test_subprocessor_defaults_preserved_by_kwargs(self, modality):
        self.skipTest("The OmniASR LLM prompt is not padded with the text kwargs.")

    @parameterized.expand(["images", "videos", "audio"])
    def test_unstructured_kwargs(self, modality):
        self.skipTest("The OmniASR LLM prompt is not padded with the text kwargs.")

    # The feature extractor's `attention_mask` (over the raw samples) is returned as `padding_mask`, so that
    # `attention_mask` is free to cover the decoder prompt.
    @parameterized.expand(["text", "images", "videos", "audio"])
    def test_subprocessor_defaults(self, modality):
        if modality != "audio":
            self.skipTest(f"OmniASRProcessor has no {modality} sub-processor.")
        processor = self.get_processor()
        audio = self.prepare_audio_inputs(batch_size=1)
        input_feature_extractor = processor.feature_extractor(
            audio, sampling_rate=SAMPLING_RATE, return_attention_mask=True, return_tensors="pt"
        )
        input_processor = processor(audio, sampling_rate=SAMPLING_RATE)

        torch.testing.assert_close(input_processor["input_values"], input_feature_extractor["input_values"])
        torch.testing.assert_close(input_processor["padding_mask"], input_feature_extractor["attention_mask"])

    def test_output_labels(self):
        processor = self.get_processor()
        text = ["hello world", "hi"]
        # Audio of different lengths, so the prompts are padded too.
        audio = self.prepare_audio_inputs(batch_size=2)
        audio[1] = audio[1][: len(audio[1]) // 2]
        inputs = processor(audio, text=text, language="eng_Latn", sampling_rate=SAMPLING_RATE)

        self.assertIn("labels", inputs)
        input_ids, attention_mask, labels = inputs["input_ids"], inputs["attention_mask"], inputs["labels"]
        self.assertEqual(labels.shape, input_ids.shape)

        # The audio placeholders, the prompt markers and the padding are masked.
        prompt_ids = [processor.audio_token_id, processor.language_token_id, processor.bos_token_id]
        prompt_positions = torch.isin(input_ids, torch.tensor(prompt_ids)) & (attention_mask == 1)
        self.assertTrue(prompt_positions.any())
        self.assertTrue((labels[prompt_positions] == -100).all())
        self.assertTrue((labels[attention_mask == 0] == -100).all())

        eos_token_id = processor.tokenizer.eos_token_id
        for idx, transcript in enumerate(text):
            # The transcript and its EOS close the sequence, right after the prompt's BOS, and are the only labels.
            target = processor.tokenizer(transcript, add_special_tokens=False)["input_ids"] + [eos_token_id]
            kept_positions = labels[idx] != -100
            self.assertListEqual(labels[idx][kept_positions].tolist(), target)
            self.assertListEqual(input_ids[idx, -len(target) :].tolist(), target)
            self.assertEqual(input_ids[idx, -len(target) - 1].item(), processor.bos_token_id)
