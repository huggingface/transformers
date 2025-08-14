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
"""Testing suite for the PyTorch Qwen2.5-VL model."""

import copy
import tempfile
import unittest

import requests

from transformers import (
    AutoProcessor,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    cleanup,
    is_flaky,
    require_cv2,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)
from transformers.utils import is_cv2_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
)


if is_cv2_available():
    import cv2

if is_torch_available():
    import torch

else:
    is_torch_greater_or_equal_than_2_0 = False

if is_vision_available():
    from PIL import Image


class Qwen2_5_VLVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        num_channels=3,
        ignore_index=-100,
        image_size=14,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        vision_start_token_id=3,
        image_token_id=4,
        video_token_id=5,
        hidden_act="silu",
        hidden_size=32,
        vocab_size=99,
        intermediate_size=37,
        max_position_embeddings=512,
        max_window_layers=3,
        model_type="qwen2_5_vl",
        num_attention_heads=4,
        num_hidden_layers=4,
        num_key_value_heads=2,
        rope_theta=10000,
        tie_word_embeddings=True,
        is_training=True,
        vision_config={
            "depth": 2,
            "in_chans": 3,
            "hidden_act": "silu",
            "intermediate_size": 32,
            "out_hidden_size": 32,
            "hidden_size": 32,
            "num_heads": 4,
            "patch_size": 14,
            "spatial_patch_size": 14,
            "spatial_merge_size": 1,
            "temporal_patch_size": 2,
        },
        rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.vision_start_token_id = vision_start_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.max_window_layers = max_window_layers
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.vision_config = vision_config
        self.rope_scaling = rope_scaling
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.num_image_tokens = 32
        self.seq_length = seq_length + self.num_image_tokens

    def get_config(self):
        return Qwen2_5_VLConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            vision_config=self.vision_config,
            model_type=self.model_type,
            max_window_layers=self.max_window_layers,
            rope_scaling=self.rope_scaling,
            tie_word_embeddings=self.tie_word_embeddings,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            vision_start_token_id=self.vision_start_token_id,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vocab_size=self.vocab_size,
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
class Qwen2_5_VLModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `Qwen2_5_VLForConditionalGeneration`.
    """

    all_model_classes = (
        (
            Qwen2_5_VLModel,
            Qwen2_5_VLForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = Qwen2_5_VLVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Qwen2_5_VLConfig, has_text_modality=False)

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

    def test_video_forward(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        B = self.model_tester.batch_size
        C = config.vision_config.in_chans
        T = config.vision_config.temporal_patch_size
        P = config.vision_config.patch_size

        input_ids = ids_tensor([B, self.model_tester.seq_length], self.model_tester.vocab_size)

        F = 4
        patch_H = self.model_tester.image_size // P
        patch_W = self.model_tester.image_size // P
        patch_T = F // T
        patches_per_video = patch_T * patch_H * patch_W
        pixel_values_videos = floats_tensor(
            [
                # first dim: batch_size * num_patches
                B * patches_per_video,
                # second dim: in_channels * temporal_patch_size * patch_size^2
                C * T * (P**2),
            ]
        )
        video_grid_thw = torch.tensor([[patch_T, patch_H, patch_W]] * B)

        # sanity check
        assert pixel_values_videos.shape[0] == video_grid_thw.prod(dim=1).sum().item()

        # Insert video token sequence
        input_ids[:, -1] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.video_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.image_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_start_token_id] = self.model_tester.pad_token_id
        input_ids[:, self.model_tester.num_image_tokens] = self.model_tester.video_token_id

        insertion_point = self.model_tester.num_image_tokens

        assert (B * patches_per_video) + insertion_point <= self.model_tester.seq_length
        for b in range(B):
            input_ids[b, insertion_point - 1] = self.model_tester.vision_start_token_id
            input_ids[b, insertion_point : insertion_point + patches_per_video] = self.model_tester.video_token_id

        for model_class in self.all_model_classes:
            second_per_grid_ts = torch.tensor([1.0] * B, device=torch_device)
            model = model_class(config).to(torch_device)
            outputs = model(
                input_ids=input_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )
            self.assertIsNotNone(outputs)

    def attention_mask_padding_matches_padding_free_with_position_ids(
        self, attn_implementation: str, fa_kwargs: bool = False
    ):
        max_new_tokens = 30
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            dummy_input = inputs_dict[model_class.main_input_name]
            if dummy_input.dtype in [torch.float32, torch.float16]:
                dummy_input = dummy_input.to(torch.bfloat16)

            # make sure that all models have enough positions for generation
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = max_new_tokens + dummy_input.shape[1] + 1

            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                if 0 in inputs_dict["attention_mask"][:, -1]:
                    inputs_dict["attention_mask"] = inputs_dict["attention_mask"].flip(1)
                dummy_attention_mask = inputs_dict["attention_mask"]
                inputs_dict["input_ids"][~dummy_attention_mask.bool()] = config.get_text_config().pad_token_id

                model = (
                    model_class.from_pretrained(
                        tmpdirname,
                        dtype=torch.bfloat16,
                        attn_implementation=attn_implementation,
                    )
                    .to(torch_device)
                    .eval()
                )

                # flatten
                padfree_inputs_dict = {
                    "pixel_values": inputs_dict["pixel_values"],
                    "image_grid_thw": inputs_dict["image_grid_thw"],
                    "input_ids": inputs_dict["input_ids"][dummy_attention_mask.bool()].unsqueeze(0),
                }

                # add position_ids
                vision_position_ids, deltas = model.model.get_rope_index(
                    input_ids=inputs_dict["input_ids"],
                    image_grid_thw=inputs_dict["image_grid_thw"],
                    attention_mask=inputs_dict["attention_mask"],
                )  # [3, bs, padded-seq-len]
                vision_padfree_positions = vision_position_ids[:, dummy_attention_mask.bool()].view(
                    3, -1
                )  # [3, bs*padfree-len]
                text_padfree_positions = torch.cat(
                    [torch.arange(length) for length in dummy_attention_mask.sum(1).tolist()]
                )  # [1, bs*padfree-len]
                text_padfree_positions = text_padfree_positions.long().unsqueeze(0).to(torch_device)
                padfree_inputs_dict["position_ids"] = torch.cat([text_padfree_positions, vision_padfree_positions])[
                    :, None, :
                ]

                if fa_kwargs:
                    cu_seq_lens = [0] + dummy_attention_mask.sum(1).tolist()
                    cu_seq_lens = torch.tensor(cu_seq_lens, device=torch_device)
                    max_length = cu_seq_lens.diff().max().item()
                    padfree_inputs_dict.update(
                        {
                            "cu_seq_lens_q": cu_seq_lens.cumsum(-1).to(dtype=torch.int32),
                            "cu_seq_lens_k": cu_seq_lens.cumsum(-1).to(dtype=torch.int32),
                            "max_length_q": max_length,
                            "max_length_k": max_length,
                        }
                    )

                res_padded = model(**inputs_dict, use_cache=False)
                res_padfree = model(**padfree_inputs_dict, use_cache=False)

                logits_padded = res_padded.logits[inputs_dict["attention_mask"].bool()]
                logits_padfree = res_padfree.logits[0]

                # acceptable numerical instability
                tol = torch.finfo(torch.bfloat16).eps
                torch.testing.assert_close(logits_padded, logits_padfree, rtol=tol, atol=tol)

    @unittest.skip(reason="Feedforward chunking is not yet supported")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="CPU offload is not yet supported")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_model_parallelism(self):
        pass

    @unittest.skip(reason="Compile not yet supported because in Qwen2_5_VL models")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Got `CUDA error: misaligned address` with PyTorch 2.0.0.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip(reason="We cannot configure to output a smaller model.")
    def test_model_is_small(self):
        pass

    @is_flaky()  # TODO (joao/raushan): Investigate why this test is flaky on this model
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        super().test_prompt_lookup_decoding_matches_greedy_search()


@require_torch
class Qwen2_5_VLIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
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
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", dtype="auto", device_map="auto"
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
        EXPECTED_DECODED_TEXT = "system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThe dog in the picture appears to be a Labrador Retriever. Labradors are known for their friendly and energetic nature, which is evident in"

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_batch(self):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", dtype="auto", device_map="auto"
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text, text], images=[self.image, self.image], return_tensors="pt").to(
            torch_device
        )

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThe dog in the picture appears to be a Labrador Retriever. Labradors are known for their friendly and energetic nature, which is evident in',
            'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThe dog in the picture appears to be a Labrador Retriever. Labradors are known for their friendly and energetic nature, which is evident in',
        ]  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_expand(self):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", dtype="auto", device_map="auto"
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[self.image], return_tensors="pt").to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, num_return_sequences=3)

        EXPECTED_DECODED_TEXT = [
            'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThe dog in the picture appears to be a Labrador Retriever. Labradors are known for their friendly and energetic nature, which is evident in',
            'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThe dog in the picture appears to be a Labrador Retriever. Labradors are known for their friendly and energetic nature, which is evident in',
            'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThe dog in the picture appears to be a Labrador Retriever. Labradors are known for their friendly and energetic nature, which is evident in',
        ]  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_batch_wo_image(self):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", dtype="auto", device_map="auto"
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
            'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThe dog in the picture appears to be a Labrador Retriever. Labradors are known for their friendly and energetic nature, which is evident in',
            'system\nYou are a helpful assistant.\nuser\nWho are you?\nassistant\n addCriterion',
        ]  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_batch_different_resolutions(self):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", dtype="auto", device_map="auto"
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

        EXPECTED_DECODED_TEXT = [
            "system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThe dog in the picture appears to be a Labrador Retriever. Labradors are known for their friendly and energetic nature, which is evident in",
            "system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\n addCriterion\nThe dog in the picture appears to be a Labrador Retriever. Labradors are known for their friendly and gentle nature, which is",
        ]

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_flash_attn
    @require_torch_gpu
    def test_small_model_integration_test_batch_flashatt2(self):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
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

        EXPECTED_DECODED_TEXT = [
            "system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThe dog in the picture appears to be a Labrador Retriever. Labradors are known for their friendly and intelligent nature, making them popular pets",
            "system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThe dog in the picture appears to be a Labrador Retriever. Labradors are known for their friendly and intelligent nature, making them popular pets",
        ]

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True)[0],
            self.processor.batch_decode(output, skip_special_tokens=True)[1],
        )

    @slow
    @require_flash_attn
    @require_torch_gpu
    def test_small_model_integration_test_batch_wo_image_flashatt2(self):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
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

        EXPECTED_DECODED_TEXT = [
            "system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThe dog in the picture appears to be a Labrador Retriever. Labradors are known for their friendly and intelligent nature, making them popular pets",
            "system\nYou are a helpful assistant.\nuser\nWho are you?\nassistant\nI am Qwen, a large language model created by Alibaba Cloud. I am designed to answer a wide range of questions and provide information on various topics",
        ]

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_cv2
    def test_small_model_integration_test_with_video(self):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", dtype="auto", device_map="auto"
        )

        video_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4"
        messages2 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                    },
                    {"type": "text", "text": "What is shown in this video?"},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(requests.get(video_url).content)
            f.flush()
            cap = cv2.VideoCapture(f.name)

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb).resize((224, 224), Image.BICUBIC))

            cap.release()

        inputs = self.processor(text=[text], videos=[frames], return_tensors="pt").to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            'system\nYou are a helpful assistant.\nuser\nWhat is shown in this video?\nassistant\nThe video shows an indoor tennis court with a person standing on one side, preparing to serve the ball. The individual is dressed in athletic attire, including',
        ]  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )
