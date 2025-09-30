# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Llava-OneVision-1.5 model."""

import copy
import unittest

import requests

from transformers import (
    AutoProcessor,
    LlavaOnevision1_5Config,
    LlavaOnevision1_5ForConditionalGeneration,
    LlavaOnevision1_5Model,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch
else:
    is_torch_greater_or_equal_than_2_0 = False

if is_vision_available():
    from PIL import Image


class LlavaOnevisionVision1_5Text2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        num_channels=3,
        image_size=14,
        seq_length=7,
        ignore_index=-100,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        vision_start_token_id=3,
        image_token_id=151655,
        video_token_id=151656,
        vocab_size=151936,
        is_training=True,
        text_config={
            "attention_bias": False,
            "attention_dropout": 0.0,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "image_token_id": None,
            "initializer_range": 0.02,
            "intermediate_size": 12288,
            "tie_word_embeddings": False,
            "layer_types": ["full_attention" for _ in range(36)],
            "max_position_embeddings": 32768,
            "max_window_layers": 36,
            "model_type": "qwen3",
            "num_attention_heads": 32,
            "num_hidden_layers": 36,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 1000000.0,
            "sliding_window": None,
            "use_cache": True,
            "use_sliding_window": False,
            "video_token_id": None,
            "vocab_size": 151936,
        },
        vision_config={
            "depth": 24,
            "embed_dim": 1024,
            "hidden_act": "gelu",
            "hidden_size": 1024,
            "in_channels": 3,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "layer_norm_eps": 1e-05,
            "model_type": "rice_vit",
            "num_heads": 16,
            "patch_size": 14,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "text_hidden_size": 4096,
        },
    ):
        self.parent = parent
        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size

        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.num_image_tokens = 32
        self.seq_length = seq_length + self.num_image_tokens
        self.ignore_index = ignore_index
        self.pad_token_id = pad_token_id
        self.vision_start_token_id = vision_start_token_id
        self.is_training = is_training

    def get_config(self):
        return LlavaOnevision1_5Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vocab_size=self.vocab_size,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vision_config.patch_size
        temporal_patch_size = config.vision_config.temporal_patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (patch_size**2),
                self.num_channels * (patch_size**2) * temporal_patch_size,
            ]
        )

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.vision_start_token_id] = self.pad_token_id
        input_ids[:, self.num_image_tokens] = self.image_token_id
        input_ids[:, self.num_image_tokens - 1] = self.vision_start_token_id
        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, 1, 1]] * self.batch_size, device=torch_device),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class LlavaOnevision1_5ForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `LlavaOnevision1_5ForConditionalGeneration`.
    """

    all_model_classes = (
        (
            LlavaOnevision1_5Model,
            LlavaOnevision1_5ForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {"image-text-to-text": LlavaOnevision1_5ForConditionalGeneration} if is_torch_available() else {}
    )
    test_pruning = False
    test_head_masking = False
    # MP works but offload doesn't work when the MultiheadAttention is offloaded
    # TODO: One potential solution would be to add to set preload_module_classes = ["Siglip2MultiheadAttentionPoolingHead"]
    # in the dispatch_model function
    test_cpu_offload = False
    test_disk_offload_safetensors = False
    test_disk_offload_bin = False
    _is_composite = True

    def setUp(self):
        self.model_tester = LlavaOnevisionVision1_5Text2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LlavaOnevision1_5Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs through an error with explicit message saying what is wrong
        when number of images don't match number of image tokens in the text.
        Also we need to test multi-image cases when one prompr has multiple image tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            _ = model(**input_dict)  # successful forward with no modifications
            curr_input_dict = copy.deepcopy(input_dict)

            # remove one image but leave the image token in text
            patch_size = config.vision_config.patch_size
            one_img_length = (self.model_tester.image_size**2) // (patch_size**2)
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-one_img_length:, ...]
            curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            input_ids = curr_input_dict["input_ids"][:1]
            pixel_values = curr_input_dict["pixel_values"][:one_img_length]
            image_grid_thw = curr_input_dict["image_grid_thw"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )

            # two images and two image tokens don't raise an error
            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            image_grid_thw = torch.cat([image_grid_thw, image_grid_thw], dim=0)
            _ = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, RiceVisionModel does not support standalone training"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip
    def test_training(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, RiceVisionModel does not support standalone training"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, RiceVisionModel does not support standalone training"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(
        "VLMs need lots of steps to prepare images/mask correctly to get pad-free inputs. Can be tested as part of LLM test"
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass


@require_torch
class LlavaOnevision1_5ForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(
            "Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf", padding_side="left"
        )
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What kind of dog is this?"},
                ],
            }
        ]
        url = "https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/demo_small.jpg"
        self.image = Image.open(requests.get(url, stream=True).raw)

        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_small_model_integration_test(self):
        model = LlavaOnevision1_5ForConditionalGeneration.from_pretrained(
            "Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf", dtype="auto", device_map="auto"
        )

        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[self.image], return_tensors="pt")

        expected_input_ids = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 151652, 151655, 151655]  # fmt: skip
        torch.testing.assert_close(expected_input_ids, inputs.input_ids[0].tolist()[:17])

        expected_pixel_slice = torch.tensor(
            [
                [0.8792, 0.8792, 0.9084],
                [1.1858, 1.1858, 1.2296],
                [1.2004, 1.2004, 1.2150],
                [1.4340, 1.4340, 1.4194],
                [1.3902, 1.4048, 1.4194],
                [1.5216, 1.5362, 1.5362],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        torch.testing.assert_close(expected_pixel_slice, inputs.pixel_values[:6, :3], atol=5e-4, rtol=1e-5)

        # verify generation
        inputs = inputs.to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30)
        EXPECTED_DECODED_TEXT = "system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their"

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_batch(self):
        model = LlavaOnevision1_5ForConditionalGeneration.from_pretrained(
            "Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf", dtype="auto", device_map="auto"
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text, text], images=[self.image, self.image], return_tensors="pt").to(
            torch_device
        )

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their',
            'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their',
        ]  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_expand(self):
        model = LlavaOnevision1_5ForConditionalGeneration.from_pretrained(
            "Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf", dtype="auto", device_map="auto"
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[self.image], return_tensors="pt").to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, num_return_sequences=3)

        EXPECTED_DECODED_TEXT = [
            'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their',
            'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their',
            'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their',
        ]  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_batch_wo_image(self):
        model = LlavaOnevision1_5ForConditionalGeneration.from_pretrained(
            "Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf", dtype="auto", device_map="auto"
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        messages2 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who are you?"},
        ]
        text2 = self.processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text, text2], images=[self.image], padding=True, return_tensors="pt").to(
            torch_device
        )

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their',
            'system\nYou are a helpful assistant.\nuser\nWho are you?\nassistant\nI am an AI language model developed by OpenAI, designed to assist and communicate with users in various tasks and conversations.',
        ]  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_batch_different_resolutions(self):
        model = LlavaOnevision1_5ForConditionalGeneration.from_pretrained(
            "Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf", dtype="auto", device_map="auto"
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        text2 = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        image2 = self.image.resize((224, 224))
        inputs = self.processor(
            text=[text, text2],
            images=[self.image, image2],
            padding=True,
            return_tensors="pt",
        ).to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        expected_decoded_texts = Expectations(
            {
                (None, None): [
                    'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their',
                    'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their',
                ],
                ("cuda", (8, 6)): [
                    'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their',
                    'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their',
                ],
                ("rocm", None): [
                    'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their',
                    'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their',
                ],
            }
        ).get_expectation()  # fmt: skip

        decoded_texts = self.processor.batch_decode(output, skip_special_tokens=True)
        for i, (expected, decoded) in enumerate(zip(expected_decoded_texts, decoded_texts)):
            self.assertEqual(
                decoded,
                expected,
                f"Decoded text {i}:\n{repr(decoded)}\ndoes not match expected decoded text:\n{repr(expected)}",
            )

    @slow
    @require_flash_attn
    @require_torch_gpu
    def test_small_model_integration_test_batch_flashatt2(self):
        model = LlavaOnevision1_5ForConditionalGeneration.from_pretrained(
            "Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text, text], images=[self.image, self.image], return_tensors="pt").to(
            torch_device
        )

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        expected_decoded_text = Expectations({
            ("cuda", None): 'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their',
            ("rocm", (9, 4)): 'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their'
        }).get_expectation()  # fmt: skip

        # Since the test is to generate twice the same text, we just test twice against the expected decoded text
        decoded_texts = self.processor.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(decoded_texts[0], expected_decoded_text)
        self.assertEqual(decoded_texts[1], expected_decoded_text)

    @slow
    @require_flash_attn
    @require_torch_gpu
    def test_small_model_integration_test_batch_wo_image_flashatt2(self):
        model = LlavaOnevision1_5ForConditionalGeneration.from_pretrained(
            "Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        messages2 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who are you?"},
        ]
        text2 = self.processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text, text2], images=[self.image], padding=True, return_tensors="pt").to(
            torch_device
        )

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        # FIXME: The second decoded text in the CUDA expectation seems to be incorrect, it used to be the second text
        # on the ROCm expectation that was the correct one. Either model changed or code is buggy.
        EXPECTED_DECODED_TEXT = Expectations({
            ("cuda", None): [
                'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their',
                'system\nYou are a helpful assistant.\nuser\nWho are you?\nassistant\nI am an AI language model developed by OpenAI, designed to assist and communicate with users in various tasks and conversations.',
            ],
            ("rocm", (9, 4)): [
                'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their',
                'system\nYou are a helpful assistant.\nuser\nWho are you?\nassistant\nI am an AI language model developed by OpenAI, designed to assist and communicate with users in various tasks and conversations.',
            ],
        }).get_expectation()  # fmt: skip

        decoded_text = self.processor.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(decoded_text, EXPECTED_DECODED_TEXT)

    @unittest.skip("Skipping video test as the small model does not handle video inputs yet.")
    def test_small_model_integration_test_with_video(self):
        pass

    @unittest.skip
    def test_training(self):
        pass
