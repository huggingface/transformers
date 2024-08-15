# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Qwen2-VL model."""

import gc
import unittest

import requests

from transformers import (
    AutoProcessor,
    Qwen2VLConfig,
    Qwen2VLForConditionalGeneration,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    require_bitsandbytes,
    require_torch,
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


class Qwen2VLVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=8,
        seq_length=7,
        num_channels=3,
        ignore_index=-100,
        image_size=28,
        bos_token_id=0,
        eos_token_id=1,
        vision_start_token_id=151652,
        image_token_id=151655,
        video_token_id=151656,
        hidden_act="silu",
        hidden_size=32,
        vocab_size=152064,
        intermediate_size=37,
        max_position_embeddings=512,
        max_window_layers=3,
        model_type="qwen2_vl",
        num_attention_heads=4,
        num_hidden_layers=3,
        num_key_value_heads=2,
        rope_theta=10000,
        tie_word_embeddings=True,
        is_training=True,
        vision_config={
            "depth": 2,
            "embed_dim": 32,
            "hidden_act": "quick_gelu",
            "hidden_size": 32,
            "mlp_ratio": 4,
            "num_heads": 4,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
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
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = vocab_size

    def get_config(self):
        return Qwen2VLConfig(
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
        vision_seqlen = pixel_values.shape[0] // self.batch_size // (self.vision_config["spatial_merge_size"] ** 2)
        input_ids = ids_tensor([self.batch_size, self.seq_length - 1 + vision_seqlen], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)
        input_ids[:, torch.arange(vision_seqlen, device=torch_device) + 1] = self.image_token_id
        labels = torch.zeros(
            (self.batch_size, self.seq_length - 1 + vision_seqlen), dtype=torch.long, device=torch_device
        )
        patch_size = self.vision_config["patch_size"]
        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor(
                [[1, self.image_size // patch_size, self.image_size // patch_size]] * self.batch_size
            ),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return config, inputs_dict

    def create_and_check_qwen2_vl_model_fp16_forward(
        self, config, input_ids, pixel_values, attention_mask, image_grid_thw
    ):
        model = Qwen2VLForConditionalGeneration(config=config)
        model.to(torch_device)
        model.half()
        model.eval()
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            pixel_values=pixel_values.to(torch.bfloat16),
            return_dict=True,
        )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())

    def create_and_check_qwen2_vl_model_fp16_autocast_forward(
        self, config, input_ids, pixel_values, attention_mask, image_grid_thw
    ):
        config.torch_dtype = torch.float16
        model = Qwen2VLForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
                pixel_values=pixel_values.to(torch.bfloat16),
                return_dict=True,
            )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())


@require_torch
class Qwen2VLModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `Qwen2VLForConditionalGeneration`.
    """

    all_model_classes = (Qwen2VLForConditionalGeneration,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = Qwen2VLVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Qwen2VLConfig, has_text_modality=False)

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

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Feedforward chunking is not yet supported")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="Generate needs input ids")
    def test_inputs_embeds_matches_input_ids_with_generate(self):
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

    @unittest.skip(reason="Compile not yet supported because in Qwen2VL models")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Compile not yet supported because in Qwen2VL models")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Got `CUDA error: misaligned address` with PyTorch 2.0.0.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip(reason="We cannot configure to output a smaller model.")
    def test_model_is_small(self):
        pass


@require_torch
class Qwen2VLIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What kind of dog is this?"},
                ],
            }
        ]
        url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        self.image = Image.open(requests.get(url, stream=True).raw)

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test(self):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            load_in_4bit=True,
        )

        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[self.image], return_tensors="pt")

        expected_input_ids = [
            151644,
            8948,
            198,
            2610,
            525,
            264,
            10950,
            17847,
            13,
            151645,
            198,
            151644,
            872,
            198,
            151652,
            151654,
            151653,
            3838,
            3093,
            315,
            5562,
            374,
            419,
            30,
            151645,
            151644,
            77091,
            198,
        ]
        assert expected_input_ids == inputs.input_ids[0].tolist()

        expected_pixel_slice = torch.tensor(
            [
                [0.8501, 0.8647, 0.8647],
                [1.0106, 1.0106, 1.0252],
                [0.9960, 1.0106, 1.0252],
                [1.0982, 1.1128, 1.1274],
                [1.0836, 1.0982, 1.0982],
                [1.1858, 1.1858, 1.1858],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        assert torch.allclose(expected_pixel_slice, inputs.pixel_values[:6, :3])

        # verify generation
        inputs = inputs.to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30)
        EXPECTED_DECODED_TEXT = "system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?assistant\nThe dog in the picture appears to be a Labrador Retriever or a similar breed. Labradors are known for their friendly and intelligent nature,"

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_batch(self):
        model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", load_in_4bit=True)
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text, text], images=[self.image, self.image], return_tensors="pt").to(
            torch_device
        )

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            "system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?assistant\nThe dog in the picture appears to be a Labrador Retriever or a similar breed. Labradors are known for their friendly and intelligent nature,",
            "system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?assistant\nThe dog in the image appears to be a Labrador Retriever or a similar breed. Labradors are known for their friendly and outgoing nature,",
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
    @require_bitsandbytes
    def test_small_model_integration_test_batch_wo_image(self):
        model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", load_in_4bit=True)
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        messages2 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who are you?"},
        ]
        text2 = self.processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text, text2], images=[self.image], return_tensors="pt").to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            "system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?assistant\nThe dog in the picture appears to be a Labrador Retriever. Labradors are known for their friendly and outgoing personalities, as well as their",
            "system\nYou are a helpful assistant.user\nWho are you?assistant\nI am Qwen, a large language model created by Alibaba Cloud. I am designed to assist with various tasks and answer a wide range of questions to",
        ]

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_batch_different_resolutions(self):
        model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", load_in_4bit=True)
        text, vision_infos = self.processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        messages2 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                        "resized_height": 504,
                        "resized_width": 252,
                    },
                    {"type": "text", "text": "What kind of dog is this?"},
                ],
            }
        ]
        text2, vision_infos2 = self.processor.apply_chat_template(
            messages2, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text, text2], vision_infos=[vision_infos, vision_infos2], return_tensors="pt"
        ).to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            "system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?assistant\nThe dog in the picture appears to be a Labrador Retriever or a similar breed. Labradors are known for their friendly and intelligent nature,",
            "system\nYou are a helpful assistant.\nuser\nWho are you?assistant\nI am a large language model created by Alibaba Cloud. I am called Qwen.",
        ]
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )
