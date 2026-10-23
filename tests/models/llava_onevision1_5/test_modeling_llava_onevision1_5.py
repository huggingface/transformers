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
import tempfile
import unittest

import numpy as np
import requests

from transformers import (
    AutoProcessor,
    LlavaOnevision15Config,
    LlavaOnevision15ForConditionalGeneration,
    LlavaOnevision15Model,
    LlavaOnevision15VisionConfig,
    LlavaOnevision15VisionModel,
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
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch
else:
    is_torch_greater_or_equal_than_2_0 = False

if is_vision_available():
    from PIL import Image


class LlavaOnevision15Text2TextModelTester:
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
        vision_end_token_id=4,
        image_token_id=5,
        video_token_id=6,
        vocab_size=500,
        is_training=True,
        text_config={
            "attention_bias": False,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "hidden_size": 32,
            "head_dim": 8,
            "initializer_range": 0.02,
            "intermediate_size": 37,
            "tie_word_embeddings": False,
            "layer_types": ["full_attention" for _ in range(2)],
            "max_position_embeddings": 512,
            "max_window_layers": 3,
            "model_type": "qwen3",
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 1000000.0,
            "sliding_window": None,
            "use_cache": True,
            "use_sliding_window": False,
            "vocab_size": 500,
        },
        vision_config={
            "depth": 2,
            "embed_dim": 32,
            "hidden_act": "gelu",
            "hidden_size": 32,
            "in_channels": 3,
            "initializer_range": 0.02,
            "intermediate_size": 37,
            "layer_norm_eps": 1e-05,
            "model_type": "llava_onevision1_5_vision",
            "num_heads": 4,
            "patch_size": 14,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "out_hidden_size": 32,
        },
    ):
        self.parent = parent
        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vocab_size = vocab_size
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.hidden_size = text_config["hidden_size"]

        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.num_image_tokens = 32
        self.seq_length = seq_length + self.num_image_tokens
        self.ignore_index = ignore_index
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.is_training = is_training

    def get_config(self):
        return LlavaOnevision15Config(
            text_config={**self.text_config, "pad_token_id": self.pad_token_id},
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
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

        input_ids[:, 0] = self.pad_token_id
        attention_mask[:, 0] = 0

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
class LlavaOnevision15ForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `LlavaOnevision15ForConditionalGeneration`.
    """

    all_model_classes = (
        (
            LlavaOnevision15Model,
            LlavaOnevision15ForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {"image-text-to-text": LlavaOnevision15ForConditionalGeneration} if is_torch_available() else {}
    )
    _is_composite = True

    def setUp(self):
        self.model_tester = LlavaOnevision15Text2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LlavaOnevision15Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_reverse_loading_mapping(self):
        # Legacy checkpoints need prefix changes only for the model with the LM head.
        super().test_reverse_loading_mapping(skip_base_model=True)

    def test_config_instances_and_legacy_tying(self):
        config = self.model_tester.get_config()
        text_dict = config.text_config.to_dict()
        vision_dict = config.vision_config.to_dict()
        original = copy.deepcopy((text_dict, vision_dict))
        restored = LlavaOnevision15Config(text_config=text_dict, vision_config=vision_dict)
        self.assertEqual((text_dict, vision_dict), original)
        from_instances = LlavaOnevision15Config(text_config=config.text_config, vision_config=config.vision_config)
        self.assertIs(from_instances.text_config, config.text_config)
        self.assertIs(from_instances.vision_config, config.vision_config)
        legacy = restored.to_dict()
        legacy["tie_word_embeddings"] = True
        legacy["text_config"]["tie_word_embeddings"] = False
        for saved_version in ("4.58.0.dev0", "5.0.0.dev0"):
            legacy["transformers_version"] = saved_version
            self.assertFalse(LlavaOnevision15Config(**legacy).tie_word_embeddings)

    def test_forward_and_generation_use_1d_positions(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = LlavaOnevision15ForConditionalGeneration(config).to(torch_device).eval()
        position_ids = torch.arange(inputs["input_ids"].shape[1], device=torch_device).unsqueeze(0)
        with torch.no_grad():
            automatic = model(**inputs).logits
            explicit = model(**inputs, position_ids=position_ids).logits
            generation_positions = inputs["attention_mask"].cumsum(-1) - 1
            generation_positions.masked_fill_(inputs["attention_mask"] == 0, 1)
            padded = model(**inputs, position_ids=generation_positions).logits
            generated = model.generate(
                **inputs, max_new_tokens=2, do_sample=False, output_logits=True, return_dict_in_generate=True
            )
            extended = {**inputs, "input_ids": generated.sequences[:, :-1]}
            extended["attention_mask"] = torch.cat(
                (inputs["attention_mask"], torch.ones_like(inputs["input_ids"][:, :1])), dim=-1
            )
            extended["position_ids"] = extended["attention_mask"].cumsum(-1) - 1
            extended["position_ids"].masked_fill_(extended["attention_mask"] == 0, 1)
            uncached = model(**extended, use_cache=False).logits[:, -1]
        torch.testing.assert_close(automatic, explicit)
        torch.testing.assert_close(generated.logits[0], padded[:, -1])
        torch.testing.assert_close(generated.logits[1], uncached, atol=1e-5, rtol=1e-4)

    def test_generate_multiple_sequences_preserves_image_features(self):
        config = self.model_tester.get_config()
        config.vision_config.spatial_merge_size = 2
        model = LlavaOnevision15ForConditionalGeneration(config).to(torch_device).eval()
        input_ids = torch.tensor(
            [[10, config.vision_start_token_id, *([config.image_token_id] * 4), config.vision_end_token_id, 11]],
            device=torch_device,
        )
        inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": floats_tensor([16, 3 * 14 * 14]),
            "image_grid_thw": torch.tensor([[1, 4, 4]], device=torch_device),
        }
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1]
            for generation_kwargs, expansion_size in (
                ({"do_sample": False, "num_beams": 2}, 2),
                ({"do_sample": True, "temperature": 1e-6, "num_return_sequences": 3}, 3),
            ):
                with self.subTest(**generation_kwargs):
                    output = model.generate(
                        **inputs,
                        **generation_kwargs,
                        max_new_tokens=2,
                        output_logits=True,
                        return_dict_in_generate=True,
                    )
                    torch.testing.assert_close(
                        output.logits[0], logits.expand(expansion_size, -1), atol=1e-5, rtol=1e-4
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


@require_torch
class LlavaOnevision15VisionModelTest(unittest.TestCase):
    all_model_classes = (LlavaOnevision15VisionModel,) if is_torch_available() else ()

    def test_standalone_save_load_and_frame_isolation(self):
        config = LlavaOnevision15VisionConfig(
            hidden_size=32, out_hidden_size=32, intermediate_size=48, depth=2, num_heads=4
        )
        model = LlavaOnevision15VisionModel(config).to(torch_device).eval()
        pixels = floats_tensor([40, 3 * 14 * 14])
        # One image and two video frames must attend independently, each with its own class token.
        grid = torch.tensor([[1, 4, 4], [2, 2, 6]], device=torch_device)
        frame_grids = grid[[0, 1, 1]].clone()
        frame_grids[:, 0] = 1
        with torch.no_grad():
            batched = model(pixels, grid).pooler_output
            separate = torch.cat(
                [
                    model(part, frame_grid.unsqueeze(0)).pooler_output
                    for part, frame_grid in zip(pixels.split([16, 12, 12]), frame_grids)
                ]
            )
        torch.testing.assert_close(batched, separate)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded = LlavaOnevision15VisionModel.from_pretrained(directory).to(torch_device).eval()
            with torch.no_grad():
                torch.testing.assert_close(reloaded(pixels, grid).pooler_output, batched)

    def test_class_token_gradients_with_checkpointing(self):
        self._check_class_token_gradients_with_checkpointing(None, torch.float32)

    @require_flash_attn
    @require_torch_gpu
    def test_class_token_gradients_with_flash_attention(self):
        self._check_class_token_gradients_with_checkpointing("flash_attention_2", torch.bfloat16)

    def _check_class_token_gradients_with_checkpointing(self, attn_implementation, dtype):
        config = LlavaOnevision15VisionConfig(
            hidden_size=32, out_hidden_size=32, intermediate_size=48, depth=2, num_heads=4
        )
        model = (
            LlavaOnevision15VisionModel._from_config(config, attn_implementation=attn_implementation, dtype=dtype)
            .to(torch_device)
            .train()
        )
        pixels = floats_tensor([16, 3 * 14 * 14])
        grid = torch.tensor([[1, 4, 4]], device=torch_device)
        model(pixels, grid).pooler_output.square().sum().backward()
        expected = {name: p.grad.clone() for name, p in model.named_parameters()}
        for reentrant in (True, False):
            model.zero_grad(set_to_none=True)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
            model(pixels, grid).pooler_output.square().sum().backward()
            for name, p in model.named_parameters():
                torch.testing.assert_close(p.grad, expected[name])
            self.assertGreater(model.class_embedding.grad.abs().sum().item(), 0)
            self.assertGreater(model.class_pos_emb.grad.abs().sum().item(), 0)


@require_torch
class LlavaOnevision15ForConditionalGenerationIntegrationTest(unittest.TestCase):
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
        model = LlavaOnevision15ForConditionalGeneration.from_pretrained(
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

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        EXPECTED_DECODED_TEXT = "system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their"
        expected_decoded_text = Expectations({
            (None, None): EXPECTED_DECODED_TEXT,
            ("cuda", (8, 0)): "system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, which is known for its friendly and outgoing nature, making it a popular choice as a family pet or",
        }).get_expectation()  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            expected_decoded_text,
        )

    @slow
    def test_small_model_integration_test_batch(self):
        model = LlavaOnevision15ForConditionalGeneration.from_pretrained(
            "Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf", dtype="auto", device_map="auto"
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text, text], images=[self.image, self.image], return_tensors="pt").to(
            torch_device
        )

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)

        EXPECTED_DECODED_TEXT = [
            'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their',
            'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium to large size, short coat, and friendly demeanor. Labradors are known',
        ]  # fmt: skip
        expected_decoded_texts = Expectations(
            {(None, None): EXPECTED_DECODED_TEXT, ("cuda", (8, 0)): [EXPECTED_DECODED_TEXT[0]] * 2}
        ).get_expectation()

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            expected_decoded_texts,
        )

    @slow
    def test_small_model_integration_test_expand(self):
        model = LlavaOnevision15ForConditionalGeneration.from_pretrained(
            "Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf", dtype="auto", device_map="auto"
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[self.image], return_tensors="pt").to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, num_return_sequences=3)

        # Multiple samples can choose different continuations when BF16 logits tie.
        expected_prefix = "system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever"
        decoded_texts = self.processor.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(len(decoded_texts), 3)
        for text in decoded_texts:
            self.assertTrue(text.startswith(expected_prefix), text)

    @slow
    def test_small_model_integration_test_batch_wo_image(self):
        model = LlavaOnevision15ForConditionalGeneration.from_pretrained(
            "Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf", dtype="auto", device_map="auto"
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        messages2 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is one plus one?"},
        ]
        text2 = self.processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text, text2], images=[self.image], padding=True, return_tensors="pt").to(
            torch_device
        )

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)

        EXPECTED_DECODED_TEXT = [
            'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their',
            'system\nYou are a helpful assistant.\nuser\nWhat is one plus one?\nassistant\nOne plus one equals two.',
        ]  # fmt: skip
        expected_decoded_texts = Expectations({
            (None, None): EXPECTED_DECODED_TEXT,
            ("cuda", (8, 0)): [
                "system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, which is known for its friendly and outgoing nature, making it a popular choice as a family pet or",
                EXPECTED_DECODED_TEXT[1],
            ],
        }).get_expectation()  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            expected_decoded_texts,
        )

    @slow
    def test_small_model_integration_test_batch_different_resolutions(self):
        model = LlavaOnevision15ForConditionalGeneration.from_pretrained(
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
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)

        expected_decoded_texts = Expectations(
            {
                (None, None): [
                    'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their',
                    'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Golden Retriever, known for their friendly and tolerant attitudes and intelligence. They are often used as guide dogs, search and',
                ],
                ("cuda", (8, 0)): [
                    'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their',
                    'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Golden Retriever, identifiable by its golden coat and friendly demeanor.',
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
        model = LlavaOnevision15ForConditionalGeneration.from_pretrained(
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
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)

        expected_decoded_text = Expectations({
            ("cuda", None): 'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their',
            ("rocm", (9, 4)): 'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium to large size, short coat, and friendly demeanor. Labradors are known'
        }).get_expectation()  # fmt: skip

        # Since the test is to generate twice the same text, we just test twice against the expected decoded text
        decoded_texts = self.processor.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(decoded_texts[0], expected_decoded_text)
        self.assertEqual(decoded_texts[1], expected_decoded_text)

    @slow
    @require_flash_attn
    @require_torch_gpu
    def test_small_model_integration_test_batch_wo_image_flashatt2(self):
        model = LlavaOnevision15ForConditionalGeneration.from_pretrained(
            "Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        messages2 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is one plus one?"},
        ]
        text2 = self.processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text, text2], images=[self.image], padding=True, return_tensors="pt").to(
            torch_device
        )

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)

        expected_decoded_text = [
            "system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThis appears to be a Labrador Retriever, identifiable by its medium size, short coat, and friendly demeanor. Labradors are known for their",
            "system\nYou are a helpful assistant.\nuser\nWhat is one plus one?\nassistant\nOne plus one equals two.",
        ]

        decoded_text = self.processor.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(decoded_text, expected_decoded_text)

    @slow
    @require_torch_gpu
    def test_small_model_integration_test_with_video(self):
        model = LlavaOnevision15ForConditionalGeneration.from_pretrained(
            "Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf", dtype="auto", device_map="auto"
        )
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "Describe the video."}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        video = np.stack([np.asarray(self.image), np.asarray(self.image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))])
        inputs = self.processor(text=[text], videos=[video], do_sample_frames=False, return_tensors="pt").to(
            torch_device
        )
        self.assertEqual(inputs.video_grid_thw[0, 0].item(), 2)
        with torch.no_grad():
            features = model.get_video_features(
                inputs.pixel_values_videos, inputs.video_grid_thw, return_dict=True
            ).pooler_output
            logits = model(**inputs, logits_to_keep=1).logits[:, -1]
            generated = model.generate(
                **inputs,
                do_sample=False,
                min_new_tokens=2,
                max_new_tokens=2,
                output_logits=True,
                return_dict_in_generate=True,
            )
        self.assertEqual(features[0].shape[0], (inputs.input_ids == model.config.video_token_id).sum().item())
        self.assertTrue(torch.isfinite(features[0]).all().item())
        self.assertTrue(torch.isfinite(logits).all().item())
        torch.testing.assert_close(generated.logits[0], logits.float())
        self.assertEqual(generated.sequences.shape[-1], inputs.input_ids.shape[-1] + 2)
