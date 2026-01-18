# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
import unittest

import numpy as np

from transformers.testing_utils import require_av, require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import Qwen3VLProcessor

if is_torch_available():
    import torch


@require_vision
@require_torch
@require_torchvision
class Qwen3VLProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Qwen3VLProcessor
    model_id = "Qwen/Qwen3-VL-235B-A22B-Instruct"

    @classmethod
    def _setup_from_pretrained(cls, model_id, **kwargs):
        return super()._setup_from_pretrained(model_id, patch_size=4, max_pixels=56 * 56, min_pixels=28 * 28, **kwargs)

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token

    def test_get_num_vision_tokens(self):
        "Tests general functionality of the helper used internally in vLLM"

        processor = self.get_processor()

        output = processor._get_num_multimodal_tokens(image_sizes=[(100, 100), (300, 100), (500, 30)])
        self.assertTrue("num_image_tokens" in output)
        self.assertEqual(len(output["num_image_tokens"]), 3)

        self.assertTrue("num_image_patches" in output)
        self.assertEqual(len(output["num_image_patches"]), 3)

    def test_model_input_names(self):
        processor = self.get_processor()

        text = self.prepare_text_inputs(modalities=["image", "video"])
        image_input = self.prepare_image_inputs()
        video_inputs = self.prepare_video_inputs()
        inputs_dict = {"text": text, "images": image_input, "videos": video_inputs}
        inputs = processor(**inputs_dict, return_tensors="pt", do_sample_frames=False)

        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    @require_vision
    @require_torch
    @require_torchvision
    def test_multiple_images_per_sample_preserves_batch(self):
        # Build a processor from the small tmp pretrained saved in setUpClass
        processor = self.get_processor()
        # Create two samples: first has 2 images, second has 1 image
        img1 = np.zeros((224, 224, 3), dtype=np.uint8)
        img2 = np.zeros((224, 224, 3), dtype=np.uint8)
        images = [[img1, img2], [img1]]
        text = ["caption one", "caption two"]

        inputs = processor(images=images, text=text, return_tensors="np", padding=True)
        pixel_values = inputs["pixel_values"]

        # Should preserve batch dimension (batch-first) and return an ndarray when tensors='np'
        self.assertIsInstance(pixel_values, np.ndarray)
        self.assertEqual(pixel_values.shape[0], len(images))

    @require_torch
    @require_av
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

        if processor_name not in self.processor_class.get_attributes():
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
            max_frames=2,  # by default no more than 2 frames, otherwise too slow
        )
        input_name = getattr(self, input_name)
        self.assertTrue(input_name in out_dict)
        self.assertEqual(len(out_dict["input_ids"]), batch_size)
        self.assertEqual(len(out_dict["attention_mask"]), batch_size)

        if modality == "video":
            # qwen pixels don't scale with bs same way as other models, calculate expected video token count based on video_grid_thw
            expected_video_token_count = 0
            for thw in out_dict["video_grid_thw"]:
                expected_video_token_count += thw[0] * thw[1] * thw[2]
            mm_len = expected_video_token_count
        else:
            mm_len = batch_size * 192
        self.assertEqual(len(out_dict[input_name]), mm_len)

        return_tensor_to_type = {"pt": torch.Tensor, "np": np.ndarray, None: list}
        for k in out_dict:
            self.assertIsInstance(out_dict[k], return_tensor_to_type[return_tensors])

    @require_av
    @unittest.skip("qwen3_vl can't sample frames from image frames directly, user can use `qwen-vl-utils`")
    def test_apply_chat_template_video_1(self):
        pass

    @require_av
    @unittest.skip("qwen3_vl can't sample frames from image frames directly, user can use `qwen-vl-utils`")
    def test_apply_chat_template_video_2(self):
        pass

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

        # for fast test, set the longest edge to 8192
        processor.video_processor.size["longest_edge"] = 8192

        # Add video URL for return dict and load with `num_frames` arg
        messages[0][0]["content"][0] = {
            "type": "video",
            "url": "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/tiny_video.mp4",
        }
        num_frames = 3
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            num_frames=num_frames,
            fps=None,  # if pass num_frames, fps should be None
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 256)

        # Load with `fps` arg
        fps = 1
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            fps=fps,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 224)

        # Load with `fps` and `num_frames` args, should raise an error
        with self.assertRaises(ValueError):
            out_dict_with_video = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                fps=fps,
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
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 224)

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
            do_sample_frames=False,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 216)

    def test_kwargs_overrides_custom_image_processor_kwargs(self):
        processor = self.get_processor()
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()
        inputs = processor(text=input_str, images=image_input, max_pixels=56 * 56 * 4, return_tensors="pt")
        self.assertEqual(inputs[self.images_input_name].shape[0], 612)
        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertEqual(inputs[self.images_input_name].shape[0], 100)
