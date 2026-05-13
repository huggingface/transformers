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
import unittest

import numpy as np
from parameterized import parameterized

from transformers import Gemma4Processor
from transformers.testing_utils import get_tests_dir, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    pass

SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_vision
class Gemma4ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Gemma4Processor
    video_unstructured_max_length = 570
    video_text_kwargs_max_length = 570
    video_text_kwargs_override_max_length = 570

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token
        cls.video_token = processor.video_token

    @classmethod
    def _setup_video_processor(cls):
        video_processor_class = cls._get_component_class_from_processor("video_processor")
        gemma4_video_processor_kwargs = {
            "patch_size": 28,
            "max_soft_tokens": 70,
            "pooling_kernel_size": 3,
            "num_frames": 2,
        }
        return video_processor_class(**gemma4_video_processor_kwargs)

    @classmethod
    def _setup_feature_extractor(cls):
        feature_extractor_class = cls._get_component_class_from_processor("feature_extractor")
        gemma4_feature_extractor_kwargs = {}
        return feature_extractor_class(**gemma4_feature_extractor_kwargs)

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        gemma4_image_processor_kwargs = {
            "patch_size": 28,
            "max_soft_tokens": 70,
            "pooling_kernel_size": 3,
        }
        return image_processor_class(**gemma4_image_processor_kwargs)

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        extra_special_tokens = {
            "image_token": "<image_soft_token>",
            "boi_token": "<start_of_image>",
            "eoi_token": "<end_of_image>",
            "audio_token": "<audio_soft_token>",
            "boa_token": "<start_of_audio>",
            "eoa_token": "<end_of_audio>",
        }
        tokenizer = tokenizer_class.from_pretrained(
            SAMPLE_VOCAB, keep_accents=True, extra_special_tokens=extra_special_tokens
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    # Copied from tests.models.llava.test_processing_llava.LlavaProcessorTest.test_get_num_vision_tokens
    def test_get_num_vision_tokens(self):
        "Tests general functionality of the helper used internally in vLLM"

        processor = self.get_processor()

        output = processor._get_num_multimodal_tokens(image_sizes=[(100, 100), (300, 100), (500, 30)])
        self.assertTrue("num_image_tokens" in output)
        self.assertEqual(len(output["num_image_tokens"]), 3)

        self.assertTrue("num_image_patches" in output)
        self.assertEqual(len(output["num_image_patches"]), 3)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    # TODO: raushan or arthur: add the real chat template
    @staticmethod
    def prepare_processor_dict():
        return {
            "chat_template": "{{ bos_token }}\n{%- if messages[0]['role'] == 'system' -%}\n    {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}\n    {%- set loop_messages = messages[1:] -%}\n{%- else -%}\n    {%- set first_user_prefix = \"\" -%}\n    {%- set loop_messages = messages -%}\n{%- endif -%}\n{%- for message in loop_messages -%}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}\n        {{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}\n    {%- endif -%}\n    {%- if (message['role'] == 'assistant') -%}\n        {%- set role = \"model\" -%}\n    {%- else -%}\n        {%- set role = message['role'] -%}\n    {%- endif -%}\n    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else \"\") }}\n    {%- if message['content'] is string -%}\n        {{ message['content'] | trim }}\n    {%- elif message['content'] is iterable -%}\n        {%- for item in message['content'] -%}\n            {%- if item['type'] == 'image' -%}\n                {{ '<start_of_image>' }}\n            {%- elif item['type'] == 'text' -%}\n                {{ item['text'] | trim }}\n            {%- endif -%}\n        {%- endfor -%}\n    {%- else -%}\n        {{ raise_exception(\"Invalid content type\") }}\n    {%- endif -%}\n    {{ '<end_of_turn>\n' }}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{'<start_of_turn>model\n'}}\n{%- endif -%}\n",            "image_seq_length": 3,
        }  # fmt: skip

    # Override as Gemma4 needs images to be an explicitly nested batch
    def prepare_image_inputs(self, batch_size: int | None = None):
        """This function prepares a list of PIL images for testing"""
        images = super().prepare_image_inputs(batch_size)
        if isinstance(images, (list, tuple)):
            images = [[image] for image in images]
        return images

    def test_text_with_image_tokens(self):
        feature_extractor = self.get_component("feature_extractor")
        image_processor = self.get_component("image_processor")
        video_processor = self.get_component("video_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            image_processor=image_processor,
            video_processor=video_processor,
        )
        text_multi_images = f"{processor.boi_token}{processor.boi_token}Dummy text!"
        text_single_image = f"{processor.boi_token}Dummy text!"

        image = self.prepare_image_inputs()

        # We can't be sure what is users intention: if user wants one image per text OR two images for first text and no image for second text
        with self.assertRaises(ValueError):
            _ = processor(text=[text_single_image, text_single_image], images=[image, image], return_tensors="np")

        # The users is expected to be explicit about which image belong to which text by nesting the images list
        out_multiimages = processor(text=text_multi_images, images=[image, image], return_tensors="np")
        out_batch_oneimage = processor(
            text=[text_single_image, text_single_image], images=[[image], [image]], return_tensors="np"
        )
        self.assertListEqual(
            out_batch_oneimage[self.images_input_name].tolist(), out_multiimages[self.images_input_name].tolist()
        )

    def test_special_mm_token_truncation(self):
        """Tests that special vision tokens do not get truncated when `truncation=True` is set."""

        processor = self.get_processor()

        input_str = self.prepare_text_inputs(batch_size=2, modalities="image")
        image_input = self.prepare_image_inputs(batch_size=2)
        _ = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            truncation=None,
            padding=True,
        )

        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                images=image_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=5,
            )

    def test_get_num_multimodal_tokens_matches_processor_call(self):
        "Tests that the helper used internally in vLLM works correctly"

        processor = self.get_processor()
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

        if not hasattr(processor, "_get_num_multimodal_tokens"):
            self.skipTest("Processor doesn't support `_get_num_multimodal_tokens` yet")

        image_sizes = [(100, 100), (300, 100), (500, 30), (213, 167)]

        # Overwritten because Gemma3 needs nested image inputs
        image_inputs = []
        for h, w in image_sizes:
            image_inputs.append([np.random.randint(255, size=(h, w, 3), dtype=np.uint8)])

        text = [f"This is an image {getattr(self, 'image_token', '')}"] * len(image_inputs)
        inputs = processor(
            text=text, images=image_inputs, padding=True, return_mm_token_type_ids=True, return_tensors="pt"
        )

        if "mm_token_type_ids" not in inputs:
            self.skipTest("Processor doesn't support `mm_token_type_ids`")

        num_image_tokens_from_call = inputs.mm_token_type_ids.sum(-1).tolist()
        num_image_tokens_from_helper = processor._get_num_multimodal_tokens(image_sizes=image_sizes)
        self.assertListEqual(num_image_tokens_from_call, num_image_tokens_from_helper["num_image_tokens"])

    @unittest.skip("This test seems to be loading a different video, check for all models and fix")
    def test_apply_chat_template_video_frame_sampling(self):
        pass


class Gemma4AudioTokenCountTest(unittest.TestCase):
    """Regression tests for _compute_audio_num_tokens.

    The original implementation used ceil(duration_ms / 40) which could overshoot
    the actual encoder output length by 1 token for ~50% of audio lengths.
    The fix replicates the exact mel-framing + conv-subsampling arithmetic.
    """

    @staticmethod
    def _encoder_output_length(num_samples: int, sr: int = 16000) -> int:
        """Reference implementation of the encoder's actual output length."""
        frame_length = int(round(sr * 20.0 / 1000.0))
        hop_length = int(round(sr * 10.0 / 1000.0))
        frame_size_for_unfold = frame_length + 1
        pad_left = frame_length // 2
        padded_samples = num_samples + pad_left
        num_mel_frames = (padded_samples - frame_size_for_unfold) // hop_length + 1
        if num_mel_frames <= 0:
            return 0
        t = num_mel_frames
        for _ in range(2):
            t_padded = t + 2
            t = (t_padded - 3) // 2 + 1
        return t

    @staticmethod
    def _compute_tokens(num_samples, sr=16000):
        """Call _compute_audio_num_tokens without constructing a full processor."""

        class _Stub:
            audio_seq_length = 1500

        return Gemma4Processor._compute_audio_num_tokens(_Stub(), np.zeros(num_samples), sr)

    @parameterized.expand(
        [
            ("over_1s_boundary", 16001),
            ("bug_report_194_vs_193", 123521),
            ("over_5s_boundary", 80001),
            ("over_10s_boundary", 160001),
            ("pad_left_effect_1s", 16161),
        ]
    )
    def test_audio_token_count_matches_encoder(self, _name, num_samples):
        """Verify _compute_audio_num_tokens matches the encoder for edge-case lengths."""
        expected = self._encoder_output_length(num_samples)
        actual = self._compute_tokens(num_samples)
        self.assertEqual(actual, expected)

    @parameterized.expand(
        [
            ("1s", 16000, 25),
            ("5s", 80000, 125),
            ("10s", 160000, 250),
            ("30s", 480000, 750),
        ]
    )
    def test_audio_token_count_round_boundaries(self, _name, num_samples, expected_tokens):
        """Verify exact results at round durations."""
        self.assertEqual(self._compute_tokens(num_samples), expected_tokens)

    def test_audio_token_count_short_audio(self):
        """Very short audio that produces zero mel frames should return 0."""
        # With pad_left = 160 and frame_size_for_unfold = 321, anything <= 160 samples => 0 mel frames
        self.assertEqual(self._compute_tokens(160), 0)

    @parameterized.expand(
        [
            # Lengths where the old naive mask would produce +1 extra token
            # after stride-2 conv subsampling.  With sr=16000, hop=160, frame_size=321.
            ("short_boundary", 641),
            ("over_1s", 16001),
            ("over_5s", 80001),
            ("bug_report_length", 123521),
            ("pad_left_effect_1s", 16161),
        ]
    )
    def test_feature_extractor_mask_matches_processor(self, _name, num_samples):
        """Regression: feature extractor mask must agree with processor token count.

        The bug was that ``attention_mask[::hop]`` overcounts real mel frames by +2
        (marks frames as valid even when their window extends into padding).
        After two stride-2 conv blocks this becomes +1 extra token ~50% of the time.
        """
        from transformers import Gemma4AudioFeatureExtractor

        fe = Gemma4AudioFeatureExtractor()

        # Batch with a longer audio to force padding (the trigger for the bug)
        target = np.random.randn(num_samples).astype(np.float32)
        padding_partner = np.random.randn(num_samples + 5000).astype(np.float32)

        features = fe([target, padding_partner], return_tensors="np", padding="longest")
        mask = features["input_features_mask"][0]  # mask for target audio

        # Simulate two stride-2 conv blocks on the mask
        T = len(mask)
        for _ in range(2):
            T_out = (T + 2 - 3) // 2 + 1
            mask = mask[::2][:T_out]
            T = len(mask)

        real_tokens = int(mask.sum())
        expected = self._compute_tokens(num_samples)
        self.assertEqual(real_tokens, expected)
