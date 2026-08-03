# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
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

import numpy as np
from parameterized import parameterized

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin, url_to_local_path


if is_vision_available():
    from transformers import MiniCPMV4_6Processor

if is_torch_available():
    import torch


@require_vision
@require_torch
@require_torchvision
class MiniCPMV4_6ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MiniCPMV4_6Processor
    model_id = "openbmb/MiniCPM-V-4_6"

    video_text_kwargs_max_length = 600
    video_text_kwargs_override_max_length = 550
    video_unstructured_max_length = 600

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token
        cls.video_token = processor.video_token

    def test_image_processing(self):
        """Test that the processor correctly handles image inputs."""
        processor = self.get_processor()
        text = self.prepare_text_inputs(modalities=["image"])
        image_input = self.prepare_image_inputs()
        inputs = processor(text=text, images=image_input, return_tensors="pt")

        self.assertIn("pixel_values", inputs)
        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertIn("target_sizes", inputs)
        self.assertIsInstance(inputs["pixel_values"], torch.Tensor)
        self.assertEqual(inputs["pixel_values"].shape[0], 1)

    def test_video_processing(self):
        """Test that the processor correctly handles video inputs."""
        processor = self.get_processor()
        text = self.prepare_text_inputs(modalities=["video"])
        video_input = self.prepare_video_inputs()
        inputs = processor(text=text, videos=video_input, do_sample_frames=False, return_tensors="pt")

        self.assertIn("pixel_values_videos", inputs)
        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertIn("target_sizes_videos", inputs)
        self.assertIsInstance(inputs["pixel_values_videos"], torch.Tensor)
        self.assertEqual(inputs["pixel_values_videos"].shape[0], 1)

    def test_text_only_processing(self):
        """Test that the processor works with text-only input (no images)."""
        processor = self.get_processor()
        text = "Hello, how are you?"
        inputs = processor(text=text, return_tensors="pt")

        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertEqual(inputs["input_ids"].ndim, 2)
        self.assertEqual(inputs["attention_mask"].ndim, 2)

    def test_batch_text_only(self):
        """Test batch text-only processing."""
        processor = self.get_processor()
        texts = ["Hello", "World, this is a longer sentence"]
        inputs = processor(text=texts, return_tensors="pt")

        self.assertEqual(inputs["input_ids"].shape[0], 2)
        self.assertEqual(inputs["attention_mask"].shape[0], 2)

    def test_post_process_image_text_to_text(self):
        """Test the post-processing method."""
        processor = self.get_processor()
        generated_ids = torch.tensor([[1, 2, 3, 4, 5]])
        texts = processor.post_process_image_text_to_text(generated_ids)
        self.assertEqual(len(texts), 1)
        self.assertIsInstance(texts[0], str)

    def test_post_process_skip_special_tokens_param(self):
        """Verify skip_special_tokens can be passed as argument without conflict."""
        processor = self.get_processor()
        generated_ids = torch.tensor([[1, 2, 3, 4, 5]])
        texts_skip = processor.post_process_image_text_to_text(generated_ids, skip_special_tokens=True)
        texts_no_skip = processor.post_process_image_text_to_text(generated_ids, skip_special_tokens=False)
        self.assertEqual(len(texts_skip), 1)
        self.assertEqual(len(texts_no_skip), 1)

    def test_use_image_id_kwarg(self):
        """Test that use_image_id is correctly routed through _merge_kwargs."""
        processor = self.get_processor()
        text = f"{self.image_token}Describe."
        image_input = self.prepare_image_inputs()

        inputs_with_id = processor(text=text, images=image_input, use_image_id=True, return_tensors="pt")
        inputs_without_id = processor(text=text, images=image_input, use_image_id=False, return_tensors="pt")

        # With use_image_id=True, input_ids should contain image_id tokens -> different sequences
        self.assertFalse(
            torch.equal(inputs_with_id["input_ids"], inputs_without_id["input_ids"]),
            "use_image_id should produce different input_ids when True vs False",
        )

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
        self.assertEqual(len(out_dict[input_name]), 1)  # always 1 in this model

        return_tensor_to_type = {"pt": torch.Tensor, "np": np.ndarray, None: list}
        for k in out_dict:
            self.assertIsInstance(out_dict[k], return_tensor_to_type[return_tensors])

        # Test continue from final message
        assistant_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "It is the sound of"}],
        }
        for idx, url in enumerate(input_data[:batch_size]):
            batch_messages[idx] = batch_messages[idx] + [assistant_message]
        continue_prompt = processor.apply_chat_template(batch_messages, continue_final_message=True, tokenize=False)
        for prompt in continue_prompt:
            self.assertTrue(prompt.endswith("It is the sound of"))  # no `eos` token at the end

    def test_apply_chat_template_video_frame_sampling(self):
        processor = self.get_processor()

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "url": url_to_local_path(
                                "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/tiny_video.mp4"
                            ),
                        },
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                },
            ]
        ]

        num_frames = 3
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"num_frames": num_frames, "fps": None},
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name][0]), num_frames)

        # Load with `fps` arg
        fps = 10
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"fps": fps, "num_frames": None},
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        # 3 frames are inferred from input video's length and FPS, so can be hardcoded
        self.assertEqual(out_dict_with_video[self.videos_input_name].shape[-1], 129472)

        # When `do_sample_frames=False` no sampling is done and whole video is loaded, even if number of frames is passed
        fps = 10
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            processor_kwargs={
                "do_sample_frames": False,
                "fps": fps,
                "return_tensors": "pt",
            },
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        self.assertEqual(out_dict_with_video[self.videos_input_name].shape[-1], 1424192)

        # Load without any arg should load the whole video
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        self.assertEqual(out_dict_with_video[self.videos_input_name].shape[-1], 129472)

        # Load video as a list of frames (i.e. images).
        # NOTE: each frame should have same size because we assume they come from one video
        messages[0][0]["content"][0] = {
            "type": "video",
            "url": [
                url_to_local_path(
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg"
                )
            ]
            * 2,
        }
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            do_sample_frames=False,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        self.assertEqual(out_dict_with_video[self.videos_input_name].shape[-1], 203392)

    @require_torch
    def test_apply_chat_template_tool_calls_no_content(self):
        # MiniCPM needs different format for tools as per saved jinja template

        processor = self.get_processor()
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the weather?"}],
            },
            {
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {}}}],
            },
        ]

        # Regression test for #45290: tokenize=True used to raise KeyError when "content" was missing
        result = processor.apply_chat_template(messages, tokenize=True)
        self.assertIsInstance(result, torch.Tensor)

    @parameterized.expand([(1, "pt")])
    @unittest.skip("MiniCPM can't sample already decoded videos, have to turn off sampling!")
    def test_apply_chat_template_decoded_video(self, batch_size: int, return_tensors: str):
        pass
