# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
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
import tempfile
import unittest

import numpy as np
import pytest
from huggingface_hub import hf_hub_download

from transformers import (
    AutoProcessor,
    Qwen2_5OmniProcessor,
    Qwen2Tokenizer,
    WhisperFeatureExtractor,
)
from transformers.testing_utils import require_av, require_librosa, require_torch, require_torchaudio, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import Qwen2VLImageProcessor


@require_vision
@require_torch
@require_torchaudio
class Qwen2_5OmniProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Qwen2_5OmniProcessor

    #  text + audio kwargs testing
    @require_torch
    def test_tokenizer_defaults_preserved_by_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")
        feature_extractor = self.get_component("feature_extractor")
        if hasattr(self, "get_tokenizer"):
            tokenizer = self.get_tokenizer(max_length=800, padding="max_length")
        elif hasattr(self, "get_component"):
            tokenizer = self.get_component("tokenizer", max_length=800, padding="max_length")
        else:
            self.assertTrue(False, "Processor doesn't have get_tokenizer or get_component defined")
        if not tokenizer.pad_token:
            tokenizer.pad_token = "[TEST_PAD]"
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        processor = self.processor_class(
            tokenizer=tokenizer, feature_extractor=feature_extractor, image_processor=image_processor
        )
        self.skip_processor_without_typed_kwargs(processor)
        input_str = "lower newer"
        raw_speech = self.prepare_audio_inputs()
        inputs = processor(text=input_str, audio=raw_speech, return_tensors="pt")
        if "input_ids" in inputs:
            self.assertEqual(len(inputs["input_ids"][0]), 800)
        elif "labels" in inputs:
            self.assertEqual(len(inputs["labels"][0]), 800)

    @require_torch
    @require_vision
    def test_structured_kwargs_audio_nested(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")
        feature_extractor = self.get_component("feature_extractor")
        if hasattr(self, "get_tokenizer"):
            tokenizer = self.get_tokenizer()
        elif hasattr(self, "get_component"):
            tokenizer = self.get_component("tokenizer")
        if not tokenizer.pad_token:
            tokenizer.pad_token = "[TEST_PAD]"
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        processor = self.processor_class(
            tokenizer=tokenizer, feature_extractor=feature_extractor, image_processor=image_processor
        )
        self.skip_processor_without_typed_kwargs(processor)

        input_str = ["lower newer"]
        raw_speech = self.prepare_audio_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "audio_kwargs": {"max_length": 800},
        }

        inputs = processor(text=input_str, audio=raw_speech, **all_kwargs)
        if "input_ids" in inputs:
            self.assertEqual(len(inputs["input_ids"][0]), 2)
        elif "labels" in inputs:
            self.assertEqual(len(inputs["labels"][0]), 2)

    @require_torch
    def test_unstructured_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")
        feature_extractor = self.get_component("feature_extractor")
        if hasattr(self, "get_tokenizer"):
            tokenizer = self.get_tokenizer(max_length=117)
        elif hasattr(self, "get_component"):
            tokenizer = self.get_component("tokenizer", max_length=117)
        if not tokenizer.pad_token:
            tokenizer.pad_token = "[TEST_PAD]"
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        processor = self.processor_class(
            tokenizer=tokenizer, feature_extractor=feature_extractor, image_processor=image_processor
        )
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        raw_speech = self.prepare_audio_inputs()
        inputs = processor(
            text=input_str,
            audio=raw_speech,
            return_tensors="pt",
            padding="max_length",
            max_length=800,
        )

        if "input_ids" in inputs:
            self.assertEqual(len(inputs["input_ids"][0]), 800)
        elif "labels" in inputs:
            self.assertEqual(len(inputs["labels"][0]), 800)

    @require_torch
    def test_doubly_passed_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")
        feature_extractor = self.get_component("feature_extractor")
        if hasattr(self, "get_tokenizer"):
            tokenizer = self.get_tokenizer()
        elif hasattr(self, "get_component"):
            tokenizer = self.get_component("tokenizer")
        if not tokenizer.pad_token:
            tokenizer.pad_token = "[TEST_PAD]"
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        self.processor_class(tokenizer=tokenizer, feature_extractor=feature_extractor, image_processor=image_processor)

    @require_torch
    def test_kwargs_overrides_default_tokenizer_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")
        feature_extractor = self.get_component("feature_extractor")
        if hasattr(self, "get_tokenizer"):
            tokenizer = self.get_tokenizer(max_length=117)
        elif hasattr(self, "get_component"):
            tokenizer = self.get_component("tokenizer", max_length=117)
        if not tokenizer.pad_token:
            tokenizer.pad_token = "[TEST_PAD]"
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        self.processor_class(tokenizer=tokenizer, feature_extractor=feature_extractor, image_processor=image_processor)

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        processor_kwargs = self.prepare_processor_dict()
        processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B", **processor_kwargs)
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def get_feature_extractor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).feature_extractor

    def prepare_processor_dict(self):
        return {
            "chat_template": "{% set audio_count = namespace(value=0) %}{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_bos|><|IMAGE|><|vision_eos|>{% elif content['type'] == 'audio' or 'audio' in content or 'audio_url' in content %}{% set audio_count.value = audio_count.value + 1 %}{% if add_audio_id %}Audio {{ audio_count.value }}: {% endif %}<|audio_bos|><|AUDIO|><|audio_eos|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_bos|><|VIDEO|><|vision_eos|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        }

    def prepare_audio_inputs(self):
        """This function prepares a list of numpy audios."""
        audio_inputs = [np.random.rand(160000) * 2 - 1] * 3  # batch-size=3
        return audio_inputs

    def test_save_load_pretrained_default(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = Qwen2_5OmniProcessor(
            image_processor=image_processor, feature_extractor=feature_extractor, tokenizer=tokenizer
        )
        processor.save_pretrained(self.tmpdirname)
        processor = Qwen2_5OmniProcessor.from_pretrained(self.tmpdirname, use_fast=False)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(processor.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor.tokenizer, Qwen2Tokenizer)
        self.assertIsInstance(processor.image_processor, Qwen2VLImageProcessor)
        self.assertIsInstance(processor.feature_extractor, WhisperFeatureExtractor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = Qwen2_5OmniProcessor(
            image_processor=image_processor, feature_extractor=feature_extractor, tokenizer=tokenizer
        )

        image_input = self.prepare_image_inputs()

        input_image_proc = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, text="dummy", return_tensors="np")

        for key in input_image_proc.keys():
            self.assertAlmostEqual(input_image_proc[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = Qwen2_5OmniProcessor(
            image_processor=image_processor, feature_extractor=feature_extractor, tokenizer=tokenizer
        )

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        audio_input = self.prepare_audio_inputs()
        inputs = processor(text=input_str, images=image_input, audio=audio_input)
        keys = list(inputs.keys())
        self.assertListEqual(
            keys,
            [
                "input_ids",
                "attention_mask",
                "pixel_values",
                "image_grid_thw",
                "feature_attention_mask",
                "input_features",
            ],
        )

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

        # test if it raises when no text is passed
        with pytest.raises(ValueError):
            processor(images=image_input)

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = Qwen2_5OmniProcessor(
            image_processor=image_processor, feature_extractor=feature_extractor, tokenizer=tokenizer
        )

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        video_inputs = self.prepare_video_inputs()
        audio_input = self.prepare_audio_inputs()

        inputs = processor(text=input_str, images=image_input, videos=video_inputs, audio=audio_input)
        self.assertListEqual(sorted(inputs.keys()), sorted(processor.model_input_names))

    def test_image_chat_template_single(self):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is shown in this image?"},
                    ],
                },
            ]
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), 1)

        formatted_prompt_tokenized = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        expected_output = processor.tokenizer(formatted_prompt, return_tensors=None).input_ids
        self.assertListEqual(expected_output, formatted_prompt_tokenized)

        out_dict = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)
        self.assertListEqual(list(out_dict.keys()), ["input_ids", "attention_mask"])

        # Now test the ability to return dict
        messages[0][0]["content"].append(
            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"}
        )
        out_dict = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)
        self.assertTrue(self.images_input_name in out_dict)

        # should always have input_ids and attention_mask
        self.assertEqual(len(out_dict["input_ids"]), 1)
        self.assertEqual(len(out_dict["attention_mask"]), 1)
        self.assertEqual(len(out_dict[self.images_input_name]), 5704)

    def test_image_chat_template_batched(self):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        batched_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is shown in this image?"},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see?"},
                    ],
                },
            ],
        ]

        formatted_prompt = processor.apply_chat_template(batched_messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), 2)

        formatted_prompt_tokenized = processor.apply_chat_template(
            batched_messages, add_generation_prompt=True, tokenize=True, padding=True
        )
        expected_output = processor.tokenizer(
            formatted_prompt, return_tensors=None, padding=True, padding_side="left"
        ).input_ids
        self.assertListEqual(expected_output, formatted_prompt_tokenized)

        out_dict = processor.apply_chat_template(
            batched_messages, add_generation_prompt=True, tokenize=True, return_dict=True, padding=True
        )
        self.assertListEqual(list(out_dict.keys()), ["input_ids", "attention_mask"])

        # Now test the ability to return dict
        batched_messages[0][0]["content"].append(
            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"}
        )
        batched_messages[1][0]["content"].append(
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"}
        )
        out_dict = processor.apply_chat_template(
            batched_messages, add_generation_prompt=True, tokenize=True, return_dict=True, padding=True
        )
        self.assertTrue(self.images_input_name in out_dict)

        # should always have input_ids and attention_mask
        self.assertEqual(len(out_dict["input_ids"]), 2)
        self.assertEqual(len(out_dict["attention_mask"]), 2)
        self.assertEqual(len(out_dict[self.images_input_name]), 7268)

    @require_av
    def test_chat_template_video(self):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        signature = inspect.signature(processor.__call__)
        if "videos" not in {*signature.parameters.keys()} or (
            signature.parameters.get("videos") is not None
            and signature.parameters["videos"].annotation == inspect._empty
        ):
            self.skipTest("Processor doesn't accept videos at input")

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                },
            ]
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), 1)

        formatted_prompt_tokenized = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        expected_output = processor.tokenizer(formatted_prompt, return_tensors=None).input_ids
        self.assertListEqual(expected_output, formatted_prompt_tokenized)

        out_dict = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)
        self.assertListEqual(list(out_dict.keys()), ["input_ids", "attention_mask"])

        # Add video URL for return dict and load with `num_frames` arg
        messages[0][0]["content"][0] = {
            "type": "video",
            "url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_10MB.mp4",
        }
        num_frames = 3
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            num_frames=num_frames,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 9568)

        # Load with `video_fps` arg
        video_fps = 1
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            video_fps=video_fps,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 23920)

        # Load with `video_fps` and `num_frames` args, should raise an error
        with self.assertRaises(ValueError):
            out_dict_with_video = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                video_fps=video_fps,
                num_frames=num_frames,
            )

        # Load without any arg should load the whole video
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 717600)

        # Load video as a list of frames (i.e. images). NOTE: each frame should have same size
        # because we assume they come from one video
        messages[0][0]["content"][0] = {
            "type": "video",
            "url": [
                "https://www.ilankelman.org/stopsigns/australia.jpg",
                "https://www.ilankelman.org/stopsigns/australia.jpg",
            ],
        }
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 5704)

    @require_av
    def test_chat_template_video_custom_sampling(self):
        """
        Tests that models can pass their custom callables to sample video indices.
        """
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        signature = inspect.signature(processor.__call__)
        if "videos" not in {*signature.parameters.keys()} or (
            signature.parameters.get("videos") is not None
            and signature.parameters["videos"].annotation == inspect._empty
        ):
            self.skipTest("Processor doesn't accept videos at input")

        video_file_path = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset"
        )
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "path": video_file_path},
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                },
            ]
        ]

        def dummy_sample_indices_fn(metadata, **fn_kwargs):
            # sample only the first two frame always
            return [0, 1]

        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            sample_indices_fn=dummy_sample_indices_fn,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1196)

    @require_av
    def test_chat_template_video_special_processing(self):
        """
        Tests that models can use their own preprocessing to preprocess conversations.
        """
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        signature = inspect.signature(processor.__call__)
        if "videos" not in {*signature.parameters.keys()} or (
            signature.parameters.get("videos") is not None
            and signature.parameters["videos"].annotation == inspect._empty
        ):
            self.skipTest("Processor doesn't accept videos at input")

        video_file_path = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset"
        )
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "path": video_file_path},
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                },
            ]
        ]

        def _process_messages_for_chat_template(
            conversation,
            batch_images,
            batch_videos,
            batch_video_metadata,
            **chat_template_kwargs,
        ):
            # Let us just always return a dummy prompt
            new_msg = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video"},  # no need to use path, video is loaded already by this moment
                            {"type": "text", "text": "Dummy prompt for preprocess testing"},
                        ],
                    },
                ]
            ]
            return new_msg

        processor._process_messages_for_chat_template = _process_messages_for_chat_template
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)

        # Check with `in` because we don't know how each template formats the prompt with BOS/EOS/etc
        formatted_text = processor.batch_decode(out_dict_with_video["input_ids"], skip_special_tokens=True)[0]
        self.assertTrue("Dummy prompt for preprocess testing" in formatted_text)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 145912)

    @require_librosa
    @require_av
    def test_audio_chat_template_from_video(self):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        signature = inspect.signature(processor.__call__)
        if "videos" not in {*signature.parameters.keys()} or (
            signature.parameters.get("videos") is not None
            and signature.parameters["videos"].annotation == inspect._empty
        ):
            self.skipTest(f"{self.processor_class} does not suport video inputs")

        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")

        video_file_path = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_file_path},
                    {"type": "text", "text": "Which of these animals is making the sound?"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It is a cow."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Tell me all about this animal."},
                ],
            },
        ]

        formatted_prompt = processor.apply_chat_template([messages], add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), 1)  # batch size=1

        out_dict = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="np",
            load_audio_from_video=True,
        )
        self.assertTrue(self.audio_input_name in out_dict)
        self.assertTrue(self.videos_input_name in out_dict)

        # should always have input_ids and attention_mask
        self.assertEqual(len(out_dict["input_ids"]), 1)  # batch-size=1
        self.assertEqual(len(out_dict["attention_mask"]), 1)  # batch-size=1
        self.assertEqual(len(out_dict[self.audio_input_name]), 1)  # 1 audio in the conversation
        self.assertEqual(len(out_dict[self.videos_input_name]), 145912)  # 1 video in the conversation
