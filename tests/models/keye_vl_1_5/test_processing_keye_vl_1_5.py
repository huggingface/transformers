# Copyright 2025 The Kwai Keye Team and The HuggingFace Inc. team. All rights reserved.
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

import numpy as np
import pytest

from transformers import AutoProcessor, KeyeVL1_5TokenizerFast
from transformers.testing_utils import require_av, require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import KeyeVL1_5Processor, KeyeVL1_5ImageProcessorFast, KeyeVL1_5VideoProcessor, KeyeVL1_5ImageProcessor

if is_torch_available():
    import torch


@require_vision
@require_torch
@require_torchvision
class KeyeVL1_5ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = KeyeVL1_5Processor
    video_token = "<|video_pad|>"
    image_token = "<|image_token|>"
    frame_token = "<|frame|>"
    fast_start = "<|fast_start|>"
    fast_end = "<|fast_end|>"

    """Because the source_url cannot be opened properly, we have replaced it with target_url."""
    source_url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    target_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        processor = KeyeVL1_5Processor.from_pretrained(
            "/mmu_mllm_hdd_2/zangdunju/release/pr", patch_size=4, size={"max_pixels": 56 * 56, "min_pixels": 28 * 28}, trust_remote_code=True
        )

        processor.save_pretrained(cls.tmpdirname)
        cls.image_token = processor.image_token

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def get_video_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).video_processor

    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @require_vision
    def prepare_video_inputs(self, batch_size: Optional[int] = None):
        """This function prepares a list of numpy videos."""
        video_input = [np.random.randint(255, size=(8, 3, 30, 400), dtype=np.uint8)]
        if batch_size is None:
            return video_input
        return [video_input] * batch_size

    # Copied from tests.models.llava.test_processing_llava.LlavaProcessorTest.test_get_num_vision_tokens
    def test_get_num_vision_tokens(self):
        "Tests general functionality of the helper used internally in vLLM"

        processor = self.get_processor()

        output = processor._get_num_multimodal_tokens(image_sizes=[(100, 100), (300, 100), (500, 30)])
        self.assertTrue("num_image_tokens" in output)
        self.assertEqual(len(output["num_image_tokens"]), 3)

        self.assertTrue("num_image_patches" in output)
        self.assertEqual(len(output["num_image_patches"]), 3)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        image_processor = self.get_image_processor()
        video_processor = self.get_video_processor()

        processor = KeyeVL1_5Processor(
            tokenizer=tokenizer, image_processor=image_processor, video_processor=video_processor
        )
        processor.save_pretrained(self.tmpdirname)
        processor = KeyeVL1_5Processor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(processor.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertIsInstance(processor.tokenizer, KeyeVL1_5TokenizerFast)
        self.assertIsInstance(processor.image_processor, KeyeVL1_5ImageProcessor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        video_processor = self.get_video_processor()

        processor = KeyeVL1_5Processor(
            tokenizer=tokenizer, image_processor=image_processor, video_processor=video_processor
        )

        image_input = self.prepare_image_inputs()

        input_image_proc = image_processor(image_input, return_tensors="pt")
        input_processor = processor(images=image_input, text="dummy", return_tensors="pt")

        for key in input_image_proc:
            self.assertAlmostEqual(input_image_proc[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        video_processor = self.get_video_processor()

        processor = KeyeVL1_5Processor(
            tokenizer=tokenizer, image_processor=image_processor, video_processor=video_processor
        )

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(
            list(inputs.keys()),
            ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"],
        )

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

        # test if it raises when no text is passed
        with pytest.raises(TypeError):
            processor(images=image_input)

    def _replace_video_url(
        self,
        batch_messages,
    ):
        source = self.source_url
        target = self.target_url
        for messages in batch_messages:
            for turn in messages:
                if isinstance(turn["content"], str):
                    continue
                for block in turn["content"]:
                    if isinstance(block, dict) and block["type"] == "video" and "url" in block:
                        urls = block["url"]
                        if isinstance(urls, str) and urls == source:
                            block["url"] = target
                        elif isinstance(urls, list):
                            for i in range(len(urls)):
                                if urls[i] == source:
                                    urls[i] = target
        return batch_messages

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
        
        batch_messages = self._replace_video_url(batch_messages)
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
    def test_apply_chat_template_video_frame_sampling(self):
        self.skipTest("Processor is not supported `video frame sampling` now.")

    def test_kwargs_overrides_custom_image_processor_kwargs(self):
        processor = self.get_processor()
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()
        inputs = processor(text=input_str, images=image_input, size={"max_pixels": 56 * 56 * 4, "min_pixels": 28 * 28}, return_tensors="pt")
        self.assertEqual(inputs[self.images_input_name].shape[0], 612)
        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertEqual(inputs[self.images_input_name].shape[0], 100)
    
    def test_model_input_names(self):
        processor = self.get_processor()

        image_input = self.prepare_image_inputs()
        video_inputs = self.prepare_video_inputs()
        audio_inputs = self.prepare_audio_inputs()
        inputs_dict = {"images": image_input, "videos": video_inputs, "audio": audio_inputs}

        call_signature = inspect.signature(processor.__call__)
        input_args = [param.name for param in call_signature.parameters.values()]
        inputs_dict = {k: v for k, v in inputs_dict.items() if k in input_args}
        if "text" in input_args:
            text_list = [self.prepare_text_inputs(modality=modality.rstrip("s")) for modality in inputs_dict]
            text = " ".join(text_list)
            inputs_dict["text"] = text

        inputs = processor(**inputs_dict, return_tensors="pt")

        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    def test_unstructured_kwargs_batched_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=2, modality="video")
        video_input = self.prepare_video_inputs(batch_size=2)
        inputs = processor(
            text=input_str,
            videos=video_input,
            do_sample_frames=False,
            return_tensors="pt",
            do_rescale=True,
            rescale_factor=-1,
            padding="longest",
            max_length=176,
        )

        self.assertLessEqual(inputs[self.videos_input_name][0].mean(), 0)
        self.assertTrue(
            len(inputs[self.text_input_name][0]) == len(inputs[self.text_input_name][1])
            and len(inputs[self.text_input_name][1]) == 176
        )
