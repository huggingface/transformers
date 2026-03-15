# Copyright 2025 Tencent and The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch PenguinVL model."""

import copy
import gc
import tempfile
import unittest

import numpy as np
import requests
import torch.nn as nn
from parameterized import parameterized
from PIL import Image

from transformers import (
    PenguinVLForConditionalGeneration,
    PenguinVLVisionConfig,
    PenguinVLVisionModel,
    is_torch_available,
)
from transformers.testing_utils import (
    backend_empty_cache,
    require_torch,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    slow,
    torch_device,
)
from transformers.utils import (
    is_torch_bf16_available_on_device,
    is_torch_fp16_available_on_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
    floats_tensor,
    sdpa_kernel,
)


if is_torch_available():
    import torch


def _test_penguin_vision_sdpa_inference(
    self,
    dtype,
    output_attentions,
    enable_kernels,
    atols=None,
    rtols=None,
):
    """Custom SDPA inference test for PenguinVLVisionModel.

    The vision model uses packed sequences (pixel_values has shape
    [total_tokens, channels*patch_size^2]), so the generic padded-batch test
    cannot be used directly.
    """
    if not self.has_attentions:
        self.skipTest(reason="Model architecture does not support attentions")

    if not self.all_model_classes[0]._supports_sdpa:
        self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp32":
        dtype = torch.float32

    if not is_torch_fp16_available_on_device(torch_device) and dtype == torch.float16:
        self.skipTest(f"float16 not supported on {torch_device}")

    if not is_torch_bf16_available_on_device(torch_device) and dtype == torch.bfloat16:
        self.skipTest(f"bfloat16 not supported on {torch_device}")

    if atols is None:
        atols = {
            ("cpu", False, torch.float32): 1e-6,
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-6,
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-6,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-6,
            ("cuda", True, torch.bfloat16): 1e-2,
            ("cuda", True, torch.float16): 5e-3,
        }
    if rtols is None:
        rtols = {
            ("cpu", False, torch.float32): 1e-4,
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-4,
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-4,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-4,
            ("cuda", True, torch.bfloat16): 3e-2,
            ("cuda", True, torch.float16): 5e-3,
        }

    for model_class in self.all_model_classes:
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        set_config_for_less_flaky_test(config)

        model = model_class(config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            model_sdpa = model_class.from_pretrained(tmpdirname, dtype=dtype, attn_implementation="sdpa")
            model_sdpa = model_sdpa.eval().to(torch_device)
            model_eager = model_class.from_pretrained(tmpdirname, dtype=dtype, attn_implementation="eager")
            model_eager = model_eager.eval().to(torch_device)

        set_model_for_less_flaky_test(model_eager)
        set_model_for_less_flaky_test(model_sdpa)

        for batch_size in [7]:
            processed_inputs = {}
            for key in [model.main_input_name] + list(getattr(self, "additional_model_inputs", [])):
                if key in inputs_dict:
                    processed_inputs[key] = inputs_dict[key]

            # Truncate grid_thw and merge_sizes to batch_size images
            for key in ["grid_thw", "merge_sizes"]:
                if key in processed_inputs:
                    value = processed_inputs[key]
                    if value.shape[0] > batch_size:
                        processed_inputs[key] = value[:batch_size].to(torch_device)

            # Adjust pixel_values to exactly match the token count from grid_thw
            target_len = torch.sum(
                processed_inputs["grid_thw"].prod(dim=1) // (processed_inputs["merge_sizes"] ** 2)
            ).item()
            pixel_values = processed_inputs["pixel_values"]
            if pixel_values.size(0) > target_len:
                pixel_values = pixel_values[:target_len]
            processed_inputs["pixel_values"] = pixel_values.to(dtype=dtype, device=torch_device)

            processed_inputs.update(
                {
                    "output_hidden_states": True,
                    "output_attentions": output_attentions,
                }
            )

            with torch.no_grad():
                with sdpa_kernel(
                    enable_flash=enable_kernels,
                    enable_math=True,
                    enable_mem_efficient=enable_kernels,
                ):
                    prepared_inputs = self._prepare_for_class(processed_inputs, model_class)
                    prepared_inputs = {
                        k: v.to(torch_device) if isinstance(v, torch.Tensor) else v for k, v in prepared_inputs.items()
                    }
                    outputs_eager = model_eager(**prepared_inputs)
                    outputs_sdpa = model_sdpa(**prepared_inputs)

            logits_eager = outputs_eager["hidden_states"][-1]
            logits_sdpa = outputs_sdpa["hidden_states"][-1]

            if torch_device in ["cpu", "cuda"]:
                atol = atols[torch_device, enable_kernels, dtype]
                rtol = rtols[torch_device, enable_kernels, dtype]
            else:
                atol = 1e-7
                rtol = 1e-4

            outputs_magnitude = float(
                (torch.max(logits_sdpa.abs().amax(), logits_eager.abs().amax())).detach().to("cpu")
            )
            computed_atol = outputs_magnitude * 3e-2
            if dtype == torch.bfloat16:
                atol = max(atol, computed_atol)

            results = [
                torch.allclose(_logits_sdpa, _logits_eager, atol=atol, rtol=rtol)
                for (_logits_sdpa, _logits_eager) in zip(logits_sdpa, logits_eager)
            ]

            if np.mean(results) < 0.8:
                mean_relative_diff = ((logits_sdpa - logits_eager).abs() / (logits_eager.abs() + 1e-12)).mean()
                raise ValueError(
                    f"mean relative difference for hidden_states: {mean_relative_diff:.3e}, "
                    f"torch atol = {atol}, torch rtol = {rtol}"
                )


class PenguinVLVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        patch_size=2,
        num_channels=3,
        image_size=14,
        is_training=True,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        intermediate_size=37,
        attention_dropout=0.0,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope
        self.seq_length = (self.image_size // self.patch_size) ** 2

    def get_config(self):
        return PenguinVLVisionConfig(
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            intermediate_size=self.intermediate_size,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (patch_size**2),
                self.num_channels * (patch_size**2),
            ]
        )
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        num_patches = self.image_size // config.patch_size
        inputs_dict = {
            "pixel_values": pixel_values,
            "grid_thw": torch.tensor([[1, num_patches, num_patches]] * self.batch_size, device=torch_device),
            "merge_sizes": torch.tensor([1] * self.batch_size, device=torch_device),
        }
        return config, inputs_dict


@require_torch
class PenguinVLVisionModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (PenguinVLVisionModel,) if is_torch_available() else ()
    additional_model_inputs = ["grid_thw", "merge_sizes"]
    test_resize_embeddings = False
    test_cpu_offload = False
    test_disk_offload_safetensors = False
    test_disk_offload_bin = False

    def setUp(self):
        self.model_tester = PenguinVLVisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PenguinVLVisionConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        config._attn_implementation = "eager"

        seq_len = getattr(self.model_tester, "seq_length", None)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            del inputs_dict["output_attentions"]
            config.output_attentions = True
            for k in config.sub_configs:
                getattr(config, k).output_attentions = True

            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0][0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )
            out_len = len(outputs)

            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            self.assertEqual(out_len + 1, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(self_attentions[0][0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(
        self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        if use_attention_mask:
            self.skipTest(reason="PenguinVLVisionModel does not use attention masks")
        _test_penguin_vision_sdpa_inference(self, dtype, output_attentions, enable_kernels)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(copy.deepcopy(config))
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            seq_length = torch.sum(inputs_dict["grid_thw"].prod(dim=1) // (inputs_dict["merge_sizes"] ** 2))
            # The vision encoder processes tokens with a batch dimension of 1 added internally,
            # so captured hidden states have shape [1, seq_length, hidden_size].
            self.assertListEqual(
                list(hidden_states[0].shape),
                [1, seq_length, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            for k in config.sub_configs:
                getattr(config, k).output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for k in config.sub_configs:
            getattr(config, k).output_hidden_states = True

        config.output_hidden_states = True
        config.output_attentions = self.has_attentions

        for k in config.sub_configs:
            getattr(config, k).output_attentions = self.has_attentions

        config._attn_implementation = "eager"

        model_class = self.all_model_classes[0]
        model = model_class._from_config(config, attn_implementation="eager")
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs[0]

        hidden_states = outputs.hidden_states[0]
        hidden_states.retain_grad()

        if self.has_attentions:
            attentions = outputs.attentions[0][0]
            attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(hidden_states.grad)

        if self.has_attentions:
            self.assertIsNotNone(attentions.grad)

    @unittest.skip("DataParallel is not compatible with the packed sequence input format of PenguinVLVisionModel")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip("Vision model requires additional positional inputs (grid_thw and merge_sizes)")
    def test_flash_attn_2_inference_equivalence(self):
        pass

    @unittest.skip("Vision model requires additional positional inputs (grid_thw and merge_sizes)")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    @unittest.skip("Vision model requires additional positional inputs (grid_thw and merge_sizes)")
    def test_flash_attn_kernels_inference_equivalence(self):
        pass


@require_torch
@slow
class PenguinVLIntegrationTest(unittest.TestCase):
    model_id = "tencent/Penguin-VL-8B"

    def setUp(self):
        from transformers import PenguinVLProcessor

        self.processor = PenguinVLProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        self.image = Image.open(requests.get(self.image_url, stream=True).raw)

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def test_small_model_integration_test_single_image(self):
        model = PenguinVLForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=torch_device
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.image},
                    {"type": "text", "text": "Describe the image in one sentence."},
                ],
            }
        ]
        images, frame_types = self.processor.process_vision_info(messages)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=images, text=text, frame_types=frame_types, return_tensors="pt").to(
            torch_device
        )

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        decoded = self.processor.decode(output[0], skip_special_tokens=True)
        EXPECTED_DECODED_TEXT = "user\n\nDescribe the image in one sentence.\nassistant\n<think>\n\n</think>\n\nTwo cats are sleeping on a pink couch next to two remote controls."
        self.assertEqual(decoded, EXPECTED_DECODED_TEXT)

    def test_small_model_integration_test_multi_image(self):
        """Tests that the model can handle prompts with multiple images."""
        model = PenguinVLForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=torch_device
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.image},
                    {"type": "image", "image": self.image.resize((224, 224))},
                    {"type": "text", "text": "Are these two images the same?"},
                ],
            }
        ]
        images, frame_types = self.processor.process_vision_info(messages)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=images, text=text, frame_types=frame_types, return_tensors="pt").to(
            torch_device
        )

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        decoded = self.processor.decode(output[0], skip_special_tokens=True)
        EXPECTED_DECODED_TEXT = "user\n\n\nAre these two images the same?\nassistant\n<think>\n\n</think>\n\nYes, these two images are the same. They both show two cats lying on a pink couch with"
        self.assertEqual(decoded, EXPECTED_DECODED_TEXT)

    def test_small_model_integration_test_video(self):
        """Tests that the model can handle video input (multi-frame clip)."""
        model = PenguinVLForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=torch_device
        )

        # Use the same image duplicated as "video frames"
        frames = [self.image.resize((224, 224))] * 4
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames, "timestamps": [0, 1, 2, 3]},
                    {"type": "text", "text": "Describe what you see in this video."},
                ],
            }
        ]
        images, frame_types = self.processor.process_vision_info(messages)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=images, text=text, frame_types=frame_types, return_tensors="pt").to(
            torch_device
        )

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        decoded = self.processor.decode(output[0], skip_special_tokens=True)
        EXPECTED_DECODED_TEXT = "user\nTime 0s:,Time 1s:,Time 2s:,Time 3s:\nDescribe what you see in this video.\nassistant\n<think>\n\n</think>\n\nThe video features a serene and heartwarming scene of two cats lounging on a bright pink couch"
        self.assertEqual(decoded, EXPECTED_DECODED_TEXT)

    def test_small_model_integration_test_batch(self):
        """Tests batched inference with the same image."""
        model = PenguinVLForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=torch_device
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.image},
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]
        images, frame_types = self.processor.process_vision_info(messages)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            images=images * 2,
            text=[text, text],
            frame_types=frame_types * 2,
            padding=True,
            return_tensors="pt",
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        decoded = self.processor.batch_decode(output, skip_special_tokens=True)
        EXPECTED_DECODED_TEXT = [
            "user\n\nDescribe the image.\nassistant\n<think>\n\n</think>\n\nThe image shows two cats lying on a bright pink surface, likely a couch or bed. Both cats",
            "user\n\nDescribe the image.\nassistant\n<think>\n\n</think>\n\nThe image shows two cats lying on a bright pink surface, likely a couch or bed. Both cats",
        ]
        self.assertEqual(decoded, EXPECTED_DECODED_TEXT)
