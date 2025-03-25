# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import Optional

from huggingface_hub import hf_hub_download

from transformers import AutoProcessor, AutoTokenizer, InternVLProcessor
from transformers.testing_utils import require_av, require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch


if is_vision_available():
    from transformers import GotOcr2ImageProcessor


@require_vision
class InternVLProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = InternVLProcessor
    videos_input_name = "pixel_values"

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        image_processor = GotOcr2ImageProcessor(
            do_resize=True,
            size={"height": 20, "width": 20},
            max_patches=2,
            do_rescale=True,
            rescale_factor=1 / 255,
            do_normalize=True,
            do_center_crop=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            do_convert_rgb=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("yonigozlan/InternVL2_5-1B-MPO-hf", padding_side="left")
        processor_kwargs = self.prepare_processor_dict()
        processor = InternVLProcessor.from_pretrained(
            "yonigozlan/InternVL2_5-1B-MPO-hf",
            image_processor=image_processor,
            tokenizer=tokenizer,
            **processor_kwargs,
        )
        processor.save_pretrained(self.tmpdirname)

    def prepare_processor_dict(self):
        return {"image_seq_length": 10}

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    # Override as InternVLProcessor needs image tokens in prompts
    def prepare_text_inputs(self, batch_size: Optional[int] = None):
        if batch_size is None:
            return "lower newer <image>"

        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")

        if batch_size == 1:
            return ["lower newer <image>"]
        return ["lower newer <image>", "<image> upper older longer string"] + ["<image> lower newer"] * (
            batch_size - 2
        )

    @require_av
    @require_torch
    def test_process_interleaved_images_videos(self):
        processor = self.get_processor()

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
                        },
                        {
                            "type": "image",
                            "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg",
                        },
                        {"type": "text", "text": "What are the differences between these two images?"},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4",
                        },
                        {"type": "text", "text": "What type of shot is the man performing?"},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": "https://llava-vl.github.io/static/images/view.jpg",
                        },
                        {"type": "text", "text": "Write a haiku for this image"},
                    ],
                }
            ],
        ]

        inputs_batched = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

        # Process non batched inputs to check if the pixel_values and input_ids are reconstructed in the correct order when batched together
        images_patches_index = 0
        for i, message in enumerate(messages):
            inputs = processor.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            # We slice with [-inputs["input_ids"].shape[1] :] as the input_ids are left padded
            torch.testing.assert_close(
                inputs["input_ids"][0], inputs_batched["input_ids"][i][-inputs["input_ids"].shape[1] :]
            )
            torch.testing.assert_close(
                inputs["pixel_values"],
                inputs_batched["pixel_values"][
                    images_patches_index : images_patches_index + inputs["pixel_values"].shape[0]
                ],
            )
            images_patches_index += inputs["pixel_values"].shape[0]

    # Override video chat_template tests as InternVLProcessor returns flattened video features
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

        formatted_prompt_tokenized = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors=None
        )
        add_special_tokens = True
        if processor.tokenizer.bos_token is not None and formatted_prompt[0].startswith(processor.tokenizer.bos_token):
            add_special_tokens = False
        expected_output = processor.tokenizer(
            formatted_prompt,
            return_tensors=None,
            add_special_tokens=add_special_tokens,
        ).input_ids
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
        # Difference with common tests, InternVLProcessor returns flattened video features
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), num_frames)

        # Load with `video_fps` arg
        video_fps = 1
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            video_fps=video_fps,
            num_frames=None,  # force to use default num_frames
            sample_indices_fn=None,  # force to use default sampling
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        # Difference with common tests, InternVLProcessor returns flattened video features
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), video_fps * 10)

        # Load with `video_fps` and `num_frames` args, should raise an error
        with self.assertRaises(ValueError):
            out_dict_with_video = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                video_fps=video_fps,
                num_frames=num_frames,
                sample_indices_fn=None,  # force to use default sampling
            )

        # Load without any arg should load the whole video
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            num_frames=None,  # force to use default num_frames
            sample_indices_fn=None,  # force to use default sampling
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        # Difference with common tests, InternVLProcessor returns flattened video features
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 300)

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
            num_frames=None,  # force to use default num_frames
            sample_indices_fn=None,  # force to use default sampling
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        # Difference with common tests, InternVLProcessor returns flattened video features
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 2)

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
                        {
                            "type": "video",
                            "path": video_file_path,
                        },
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                },
            ]
        ]

        def dummmy_sample_indices_fn(metadata, **fn_kwargs):
            # sample only the first two frame always
            return [0, 1]

        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            sample_indices_fn=dummmy_sample_indices_fn,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        # Difference with common tests, InternVLProcessor returns flattened video features
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 2)

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
        # Difference with common tests, InternVLProcessor returns flattened video features, and uses 8 frames by default
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 8)
