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
import shutil
import tempfile
import unittest

import numpy as np
import pytest
from huggingface_hub import hf_hub_download

from transformers import (
    AutoProcessor,
    Qwen2_5OmniProcessor,
    Qwen2TokenizerFast,
    WhisperFeatureExtractor,
)
from transformers.testing_utils import (
    require_av,
    require_librosa,
    require_torch,
    require_torchaudio,
    require_torchvision,
    require_vision,
)
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch

if is_vision_available():
    from transformers import Qwen2VLImageProcessorFast


@require_vision
@require_torch
@require_torchaudio
@require_torchvision
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
        video_processor = self.get_component("video_processor")
        processor = self.processor_class(
            tokenizer=tokenizer,
            video_processor=video_processor,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
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
        video_processor = self.get_component("video_processor")
        processor = self.processor_class(
            tokenizer=tokenizer,
            video_processor=video_processor,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
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
        video_processor = self.get_component("video_processor")
        processor = self.processor_class(
            tokenizer=tokenizer,
            video_processor=video_processor,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
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
        video_processor = self.get_component("video_processor")
        _ = self.processor_class(
            tokenizer=tokenizer,
            video_processor=video_processor,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
        )  # Why delete test? TODO: raushan double check tests after cleaning model

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
        video_processor = self.get_component("video_processor")
        _ = self.processor_class(
            tokenizer=tokenizer,
            video_processor=video_processor,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
        )

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        processor.save_pretrained(cls.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def get_video_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).video_processor

    def get_feature_extractor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).feature_extractor

    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def prepare_audio_inputs(self):
        """This function prepares a list of numpy audios."""
        audio_inputs = [np.random.rand(160000) * 2 - 1] * 3  # batch-size=3
        return audio_inputs

    def test_save_load_pretrained_default(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()
        video_processor = self.get_video_processor()
        processor = self.processor_class(
            tokenizer=tokenizer,
            video_processor=video_processor,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
        )

        processor.save_pretrained(self.tmpdirname)
        processor = Qwen2_5OmniProcessor.from_pretrained(self.tmpdirname, use_fast=True)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(processor.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor.tokenizer, Qwen2TokenizerFast)
        self.assertIsInstance(processor.image_processor, Qwen2VLImageProcessorFast)
        self.assertIsInstance(processor.feature_extractor, WhisperFeatureExtractor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()
        video_processor = self.get_video_processor()
        processor = self.processor_class(
            tokenizer=tokenizer,
            video_processor=video_processor,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
        )

        image_input = self.prepare_image_inputs()

        input_image_proc = image_processor(image_input, return_tensors="pt")
        input_processor = processor(images=image_input, text="dummy", return_tensors="pt")

        for key in input_image_proc:
            self.assertAlmostEqual(input_image_proc[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()
        video_processor = self.get_video_processor()
        processor = self.processor_class(
            tokenizer=tokenizer,
            video_processor=video_processor,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
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
        video_processor = self.get_video_processor()
        processor = self.processor_class(
            tokenizer=tokenizer,
            video_processor=video_processor,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
        )

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        video_inputs = self.prepare_video_inputs()
        audio_input = self.prepare_audio_inputs()

        inputs = processor(text=input_str, images=image_input, videos=video_inputs, audio=audio_input)
        self.assertListEqual(sorted(inputs.keys()), sorted(processor.model_input_names))

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
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        if processor_name not in self.processor_class.attributes:
            self.skipTest(f"{processor_name} attribute not present in {self.processor_class}")

        batch_messages = [
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Describe this."}],
                },
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
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors,
            max_length=100,
        )
        self.assertEqual(len(tokenized_prompt_100[0]), 100)

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
            batch_messages[idx][0]["content"] = [batch_messages[idx][0]["content"][0], {"type": modality, "url": url}]

        out_dict = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
            num_frames=2,  # by default no more than 2 frames, otherwise too slow
        )
        input_name = getattr(self, input_name)
        self.assertTrue(input_name in out_dict)
        self.assertEqual(len(out_dict["input_ids"]), batch_size)
        self.assertEqual(len(out_dict["attention_mask"]), batch_size)

        video_len = 2880 if batch_size == 1 else 5808  # qwen pixels don't scale with bs same way as other models
        mm_len = batch_size * 1564 if modality == "image" else video_len
        self.assertEqual(len(out_dict[input_name]), mm_len)

        return_tensor_to_type = {"pt": torch.Tensor, "np": np.ndarray, None: list}
        for k in out_dict:
            self.assertIsInstance(out_dict[k], return_tensor_to_type[return_tensors])

    @require_av
    def test_apply_chat_template_video_frame_sampling(self):
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
        messages[0][0]["content"].append(
            {
                "type": "video",
                "url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_10MB.mp4",
            }
        )
        num_frames = 3
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            num_frames=num_frames,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 5760)

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
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 14400)

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
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 432000)

        # Load video as a list of frames (i.e. images). NOTE: each frame should have same size
        # because we assume they come from one video
        messages[0][0]["content"][-1] = {
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
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 2904)

    @require_librosa
    @require_av
    @unittest.skip(
        "@raushan: librosa can'r decode this audio in CI runner, fix after adding moviepy or another decoder"
    )
    def test_chat_template_audio_from_video(self):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        signature = inspect.signature(processor.__call__)
        if "videos" not in {*signature.parameters.keys()} or (
            signature.parameters.get("videos") is not None
            and signature.parameters["videos"].annotation == inspect._empty
        ):
            self.skipTest(f"{self.processor_class} does not support video inputs")

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
