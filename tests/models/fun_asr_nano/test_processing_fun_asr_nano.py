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

import unittest

from parameterized import parameterized

from transformers import AutoProcessor, FunAsrNanoProcessor
from transformers.testing_utils import require_librosa, require_torch

from ...test_processing_common import MODALITY_INPUT_DATA, ProcessorTesterMixin


AUDIO_URL = "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"


@require_torch
class FunAsrNanoProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = FunAsrNanoProcessor
    # TODO: swap for `tiny_model_id = "hf-internal-testing/tiny-processor-fun_asr_nano"` once that repo exists.
    # The generic component setup builds a Qwen2 tokenizer with an empty vocab, which tokenizes to nothing.
    model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"

    # Overwrite to skip the numpy cases (keeping as many cases as the parent), as
    # `FunAsrNanoProcessor.__call__` is PyTorch-only like `AudioFlamingo3Processor`.
    @require_librosa
    @parameterized.expand([(1, "np"), (1, "pt"), (2, "np"), (2, "pt")])
    def test_apply_chat_template_audio(self, batch_size: int, return_tensors: str):
        if return_tensors == "np":
            self.skipTest("FunAsrNanoProcessor only supports PyTorch tensors")
        self._test_apply_chat_template(
            "audio", batch_size, return_tensors, "audio_input_name", "feature_extractor", MODALITY_INPUT_DATA["audio"]
        )

    def test_chat_template(self):
        processor = AutoProcessor.from_pretrained(self.tmpdirname)
        expected_prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "语音转写：<|object_ref_start|><|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        messages = [{"role": "user", "content": [{"type": "audio", "path": AUDIO_URL}]}]

        formatted_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        self.assertEqual(expected_prompt, formatted_prompt)

    @require_librosa
    def test_apply_transcription_request_with_language(self):
        processor = AutoProcessor.from_pretrained(self.tmpdirname)

        outputs = processor.apply_transcription_request(audio=AUDIO_URL, language="en", return_tensors="pt")

        for key in ("input_ids", "attention_mask", "input_features", "input_features_mask"):
            self.assertIn(key, outputs)
        # The language is forced by naming it in the instruction that precedes the audio placeholders.
        decoded = processor.tokenizer.decode(outputs["input_ids"][0])
        self.assertIn("语音转写成英文：", decoded)

    @require_librosa
    def test_apply_transcription_request_with_prompt_and_keywords(self):
        processor = AutoProcessor.from_pretrained(self.tmpdirname)
        context = "Vocabulary: Quilter, apostle, gospel."

        outputs = processor.apply_transcription_request(
            audio=AUDIO_URL, language="en", prompt=context, keywords=["Quilter", "apostle"], return_tensors="pt"
        )

        decoded = processor.tokenizer.decode(outputs["input_ids"][0])
        self.assertIn("**上下文信息：**", decoded)
        self.assertIn(context, decoded)
        self.assertIn("热词列表：[Quilter, apostle]", decoded)

    @require_librosa
    def test_apply_transcription_request_batches_per_sample_language(self):
        processor = AutoProcessor.from_pretrained(self.tmpdirname)

        outputs = processor.apply_transcription_request(
            audio=[AUDIO_URL, AUDIO_URL], language=["zh", "en"], return_tensors="pt"
        )

        decoded = [processor.tokenizer.decode(ids) for ids in outputs["input_ids"]]
        self.assertIn("语音转写成中文：", decoded[0])
        self.assertIn("语音转写成英文：", decoded[1])

    def test_apply_transcription_request_rejects_unsupported_language(self):
        processor = AutoProcessor.from_pretrained(self.tmpdirname)

        with self.assertRaisesRegex(ValueError, "Unsupported language"):
            processor.apply_transcription_request(audio=AUDIO_URL, language="French")

    @require_librosa
    def test_audio_placeholder_count_matches_unpadded_feature_frames(self):
        """One audio token is emitted per unpadded feature frame, counted off `input_features_mask`."""
        processor = AutoProcessor.from_pretrained(self.tmpdirname)

        outputs = processor.apply_transcription_request(
            audio=[AUDIO_URL, AUDIO_URL], language="en", return_tensors="pt"
        )

        audio_token_counts = (outputs["input_ids"] == processor.audio_token_id).sum(-1)
        self.assertTrue(audio_token_counts.equal(outputs["input_features_mask"].sum(-1)))

    def test_decode_strips_assistant_framing(self):
        processor = AutoProcessor.from_pretrained(self.tmpdirname)
        text = 'The transcription of the audio is "hello".'
        token_ids = processor.tokenizer(text, return_tensors="pt").input_ids

        self.assertEqual(processor.decode(token_ids[0], strip_prefix=True), "hello")
        self.assertEqual(processor.decode(token_ids[0]), text)

    @require_librosa
    @require_torch
    def test_output_labels(self):
        import torch

        processor = self.get_processor()
        audio = self.prepare_audio_inputs(batch_size=1)[0]
        conversation = [
            [
                {"role": "user", "content": [{"type": "audio", "audio": audio}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hello world."}]},
            ],
        ]

        # No explicit `return_tensors`: `FunAsrNanoProcessorKwargs._defaults` makes it `"pt"`.
        inputs = processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True, processor_kwargs={"output_labels": True}
        )

        self.assertIn("labels", inputs)
        self.assertNotIn("mm_token_type_ids", inputs)
        labels, input_ids = inputs["labels"], inputs["input_ids"]
        self.assertEqual(labels.shape, input_ids.shape)

        # audio placeholder positions are masked out of the loss, text positions are kept
        audio_positions = input_ids == processor.audio_token_id
        self.assertTrue(audio_positions.any())
        self.assertTrue((labels[audio_positions] == -100).all())
        self.assertTrue(
            (
                labels[~audio_positions]
                == torch.where(labels[~audio_positions] == -100, -100, input_ids[~audio_positions])
            ).all()
        )
