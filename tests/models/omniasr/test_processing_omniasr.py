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

from transformers import AutoProcessor, OmniASRProcessor
from transformers.testing_utils import require_librosa, require_torch
from transformers.utils.import_utils import is_torch_available

from ...test_processing_common import MODALITY_INPUT_DATA, ProcessorTesterMixin, url_to_local_path


if is_torch_available():
    import torch


SAMPLING_RATE = 16000
AUDIO_URL = url_to_local_path(
    "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"
)


@require_torch
class OmniASRProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = OmniASRProcessor
    audio_input_name = "input_values"
    model_id = "bezzam/omniasr-llm-300m-v2"

    @classmethod
    def prepare_processor_dict(cls):
        # The convolution geometry that counts the audio placeholders lives in the processor config rather than in
        # its components, so building the processor from components needs it passed.
        processor_dict, _ = OmniASRProcessor.get_processor_dict(cls.model_id)
        return {k: v for k, v in processor_dict.items() if k not in ("feature_extractor", "processor_class")}

    @require_librosa
    @parameterized.expand([(1, "np"), (1, "pt"), (2, "np"), (2, "pt")])
    def test_apply_chat_template_audio(self, batch_size: int, return_tensors: str):
        if return_tensors == "np":
            self.skipTest("OmniASRProcessor only supports PyTorch tensors")
        self._test_apply_chat_template(
            "audio", batch_size, return_tensors, "audio_input_name", "feature_extractor", MODALITY_INPUT_DATA["audio"]
        )

    # Overridden because the processor needs the audio placeholder in `text`: the common test calls the processor
    # with the audio alone.
    @parameterized.expand(["text", "images", "videos", "audio"])
    def test_subprocessor_defaults(self, modality):
        if modality != "audio":
            self.skipTest(f"OmniASRProcessor has no {modality} sub-processor.")
        processor = self.get_processor()
        audio = self.prepare_audio_inputs(batch_size=1)
        input_feature_extractor = processor.feature_extractor(
            audio, sampling_rate=SAMPLING_RATE, return_attention_mask=True, return_tensors="pt"
        )
        input_processor = processor(audio=audio, text=[processor.audio_token], sampling_rate=SAMPLING_RATE)

        torch.testing.assert_close(input_processor["input_values"], input_feature_extractor["input_values"])
        torch.testing.assert_close(input_processor["padding_mask"], input_feature_extractor["padding_mask"])

    def test_chat_template(self):
        processor = AutoProcessor.from_pretrained(self.tmpdirname)
        messages = [
            {
                "role": "user",
                "content": [{"type": "audio", "path": AUDIO_URL}, {"type": "language", "language": "eng_Latn"}],
            }
        ]

        formatted_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        self.assertEqual(formatted_prompt, "<extra_id_1><extra_id_0><|lang:eng_latn|><s>")

    @require_librosa
    def test_apply_transcription_request_with_language(self):
        processor = AutoProcessor.from_pretrained(self.tmpdirname)

        outputs = processor.apply_transcription_request(
            audio=[AUDIO_URL, AUDIO_URL], language=["eng_Latn", "fra_Latn"]
        )

        for key in ("input_ids", "attention_mask", "input_values", "padding_mask"):
            self.assertIn(key, outputs)
        # Each prompt closes with `lid_marker | language | bos`, from which the transcription is decoded.
        tokenizer = processor.tokenizer
        for input_ids, code in zip(outputs["input_ids"], ["eng_latn", "fra_latn"]):
            expected = tokenizer.convert_tokens_to_ids(["<extra_id_0>", f"<|lang:{code}|>", tokenizer.bos_token])
            self.assertListEqual(input_ids[-3:].tolist(), expected)

    def test_iso_639_1_language(self):
        processor = AutoProcessor.from_pretrained(self.tmpdirname)
        messages = [
            {"role": "user", "content": [{"type": "audio", "path": AUDIO_URL}, {"type": "language", "language": "en"}]}
        ]

        formatted_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        self.assertEqual(formatted_prompt, "<extra_id_1><extra_id_0><|lang:eng_latn|><s>")

    @require_librosa
    def test_apply_transcription_request_iso_639_1_language(self):
        processor = AutoProcessor.from_pretrained(self.tmpdirname)

        outputs = processor.apply_transcription_request(audio=AUDIO_URL, language="en")

        tokenizer = processor.tokenizer
        expected = tokenizer.convert_tokens_to_ids(["<extra_id_0>", "<|lang:eng_latn|>", tokenizer.bos_token])
        self.assertListEqual(outputs["input_ids"][0, -3:].tolist(), expected)

    @require_librosa
    def test_output_labels(self):
        processor = self.get_processor()
        text = ["hello world", "hi"]
        # Audio of different lengths, so the prompts are padded too.
        audio = self.prepare_audio_inputs(batch_size=2)
        audio[1] = audio[1][: len(audio[1]) // 2]
        conversations = [
            [
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio": audio_item}, {"type": "language", "language": "eng_Latn"}],
                },
                {"role": "assistant", "content": [{"type": "text", "text": transcript}]},
            ]
            for audio_item, transcript in zip(audio, text)
        ]

        inputs = processor.apply_chat_template(
            conversations, tokenize=True, return_dict=True, processor_kwargs={"output_labels": True}
        )

        self.assertIn("labels", inputs)
        self.assertNotIn("mm_token_type_ids", inputs)
        input_ids, attention_mask, labels = inputs["input_ids"], inputs["attention_mask"], inputs["labels"]
        self.assertEqual(labels.shape, input_ids.shape)
        self.assertTrue((labels[attention_mask == 0] == -100).all())

        tokenizer = processor.tokenizer
        for idx, transcript in enumerate(text):
            # The transcript and its EOS close the sequence, right after the prompt's BOS, and are the only labels.
            target = tokenizer(transcript, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
            self.assertListEqual(labels[idx][labels[idx] != -100].tolist(), target)
            self.assertListEqual(input_ids[idx, -len(target) :].tolist(), target)
            self.assertEqual(input_ids[idx, -len(target) - 1].item(), tokenizer.bos_token_id)

    @require_librosa
    def test_apply_transcription_request_with_transcription(self):
        processor = self.get_processor()
        text = ["hello world", "hi"]
        language = ["eng_Latn", None]
        audio = self.prepare_audio_inputs(batch_size=2)
        audio[1] = audio[1][: len(audio[1]) // 2]

        inputs = processor.apply_transcription_request(audio, language=language, transcription=text)

        # Same as manually writing the transcripts as the assistant turn and asking for labels.
        conversations = []
        for audio_item, lang, transcript in zip(audio, language, text):
            content = [{"type": "audio", "audio": audio_item}]
            if lang is not None:
                content.append({"type": "language", "language": lang})
            conversations.append(
                [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": [{"type": "text", "text": transcript}]},
                ]
            )
        expected = processor.apply_chat_template(
            conversations, tokenize=True, return_dict=True, processor_kwargs={"output_labels": True}
        )

        self.assertIn("labels", inputs)
        for key in ("input_ids", "attention_mask", "labels", "input_values", "padding_mask"):
            torch.testing.assert_close(inputs[key], expected[key])

        with self.assertRaisesRegex(ValueError, "transcription"):
            processor.apply_transcription_request(audio, transcription=text[:1])
