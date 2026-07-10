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

import inspect
import unittest

import numpy as np
from parameterized import parameterized

from transformers.testing_utils import require_av, require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin, url_to_local_path


if is_vision_available():
    from transformers import Kimi_K25Processor

if is_torch_available():
    import torch


@require_vision
@require_torch
@require_torchvision
class Kimi_K25ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Kimi_K25Processor
    # Tiny processor created with make_tiny_processor.py from "RaushanTurganbay/kimi2.7-processor"
    tiny_model_id = "hf-internal-testing/tiny-processor-kimi_k25"

    @classmethod
    def _setup_from_pretrained(cls, model_id, **kwargs):
        return super()._setup_from_pretrained(model_id, trust_remote_code=False, **kwargs)

    @classmethod
    def _setup_video_processor(cls):
        # Small spatial size (28×28) and patch sizes keep video tensor allocations minimal.
        video_processor_class = cls._get_component_class_from_processor("video_processor")
        video_processor_kwargs = {
            "size": {"max_height": 28, "max_width": 28},
            "patch_size": 4,
            "temporal_patch_size": 2,
        }
        return video_processor_class(**video_processor_kwargs)

    @classmethod
    def _setup_image_processor(cls):
        # Small spatial size (28×28) and patch size keep image tensor allocations minimal.
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        image_processor_kwargs = {
            "size": {"max_height": 28, "max_width": 28},
            "patch_size": 4,
        }
        return image_processor_class(**image_processor_kwargs)

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token
        cls.video_token = processor.video_token

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
            fps=1,  # by default no more than 1 fps, otherwise too slow
        )
        input_name = getattr(self, input_name)
        self.assertTrue(input_name in out_dict)
        self.assertEqual(len(out_dict["input_ids"]), batch_size)
        self.assertEqual(len(out_dict["attention_mask"]), batch_size)
        if modality == "video":
            expected_video_token_count = 0
            for thw in out_dict["video_grid_thw"]:
                expected_video_token_count += thw[0] * thw[1] * thw[2]
            mm_len = expected_video_token_count
        else:
            mm_len = batch_size * 616
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

        # Add video URL for return dict and load with `num_frames` arg
        messages[0][0]["content"][0] = {
            "type": "video",
            "url": url_to_local_path(
                "https://huggingface.co/datasets/hf-internal-testing/test-videos/resolve/main/tiny_video_320x240.mp4"
            ),
        }
        num_frames = 3
        out_dict_num_frames = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            num_frames=num_frames,
            fps=None,
        )
        self.assertTrue(self.videos_input_name in out_dict_num_frames)
        expected_num_frames_len = sum(thw[0] * thw[1] * thw[2] for thw in out_dict_num_frames["video_grid_thw"])
        self.assertEqual(len(out_dict_num_frames[self.videos_input_name]), expected_num_frames_len)

        # Load with `fps` arg
        fps = 3
        out_dict_fps = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            fps=fps,
        )
        self.assertTrue(self.videos_input_name in out_dict_fps)
        expected_fps_len = sum(thw[0] * thw[1] * thw[2] for thw in out_dict_fps["video_grid_thw"])
        self.assertEqual(len(out_dict_fps[self.videos_input_name]), expected_fps_len)
        # num_frames and fps sampling should produce different token counts
        self.assertNotEqual(expected_num_frames_len, expected_fps_len)

        # Load with `fps` and `num_frames` args, should raise an error
        with self.assertRaises(ValueError):
            processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                fps=fps,
                num_frames=num_frames,
            )

    def test_kwargs_overrides_custom_image_processor_kwargs(self):
        processor = self.get_processor()
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()
        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertEqual(inputs[self.images_input_name].shape[0], 56)
        inputs = processor(
            text=input_str,
            images=image_input,
            size={"max_height": 56 * 56 * 4, "max_width": 56 * 56 * 4},
            return_tensors="pt",
        )
        self.assertEqual(inputs[self.images_input_name].shape[0], 800)

    @unittest.skip("Kimi pops some keys before returning in a processor")
    def test_video_processor_defaults(self):
        pass

    @parameterized.expand([(1, "pt")])
    @unittest.skip("Kimi sampels with FPS by default which is not compatible with this test")
    def test_apply_chat_template_decoded_video(self, batch_size: int, return_tensors: str):
        pass
