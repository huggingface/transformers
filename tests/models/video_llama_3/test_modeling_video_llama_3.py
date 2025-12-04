# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch VideoLLaMA3 model."""

import copy
import gc
import inspect
import tempfile
import unittest

import numpy as np
import pytest
import requests
import torch.nn as nn
from parameterized import parameterized
from PIL import Image

from transformers import (
    AutoProcessor,
    VideoLlama3Config,
    VideoLlama3ForConditionalGeneration,
    VideoLlama3Model,
    VideoLlama3VisionConfig,
    VideoLlama3VisionModel,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
    backend_empty_cache,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    slow,
    torch_device,
)
from transformers.utils import (
    is_torch_bf16_available_on_device,
    is_torch_fp16_available_on_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    sdpa_kernel,
)


if is_torch_available():
    import torch


def _test_encoder_eager_matches_sdpa_inference(
    self,
    dtype,
    output_attentions,
    enable_kernels,
    atols=None,
    rtols=None,
):
    """
    This test is written as a regular function to be able to overload it easily with different tolerances.
    Otherwise, `parameterize.expand` prevents it as it removes the original function from the namespace.
    """
    if not self.has_attentions:
        self.skipTest(reason="Model architecture does not support attentions")

    if not self.all_model_classes[0]._supports_sdpa:
        self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

    # convert shorthand name to torch.dtype
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp32":
        dtype = torch.float32

    if not is_torch_fp16_available_on_device(torch_device) and dtype == torch.float16:
        self.skipTest(f"float16 not supported on {torch_device} (on the specific device currently used)")

    if not is_torch_bf16_available_on_device(torch_device) and dtype == torch.bfloat16:
        self.skipTest(
            f"bfloat16 not supported on {torch_device} (on the specific device currently used, e.g. Nvidia T4 GPU)"
        )

    # Dictionary of tolerances for eager <> sdpa tests. Key = (device, sdpa_kernels_enabled, dtype)
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
            ("cuda", True, torch.bfloat16): 3e-2,  # (different from others)
            ("cuda", True, torch.float16): 5e-3,
        }

    for model_class in self.all_model_classes:
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        set_config_for_less_flaky_test(config)

        model = model_class(config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            model_from_pretrained_kwargs = {
                "pretrained_model_name_or_path": tmpdirname,
                "dtype": dtype,
            }

            if hasattr(config, "use_mask_token") or "use_mask_token" in inspect.signature(model.__init__).parameters:
                model_from_pretrained_kwargs["use_mask_token"] = True

            # TODO: remove this try/except, models should have a shared API
            try:
                model_sdpa = model_class.from_pretrained(**model_from_pretrained_kwargs, attn_implementation="sdpa")
            except ValueError:
                model_sdpa = model_class.from_pretrained(**model_from_pretrained_kwargs)
            model_sdpa = model_sdpa.eval().to(torch_device)

            model_eager = model_class.from_pretrained(**model_from_pretrained_kwargs, attn_implementation="eager")
            model_eager = model_eager.eval().to(torch_device)

        set_model_for_less_flaky_test(model_eager)
        set_model_for_less_flaky_test(model_sdpa)

        # TODO: if we can also check with `batch_size=1` without being flaky?
        for batch_size in [7]:
            input_data_batch_size = batch_size

            processed_inputs = {}
            processed_inputs[model.main_input_name] = inputs_dict[model.main_input_name]

            for key in getattr(self, "additional_model_inputs", []):
                # Some models don't have all `additional_model_inputs`, especially when we
                # craft cases to test model in different settings
                if key in inputs_dict:
                    processed_inputs[key] = inputs_dict[key]

            for key, value in processed_inputs.items():
                if torch.is_floating_point(value):
                    value = value.to(dtype)

                if key == "pixel_values":
                    continue

                # extend value to have at least `input_data_batch_size` elements
                if value.shape[0] < input_data_batch_size:
                    size = (input_data_batch_size - value.shape[0], *value.shape[1:])
                    if key == "grid_thw":
                        extension = torch.randint(high=5, size=size, dtype=value.dtype, device=torch_device)
                    elif key == "merge_sizes":
                        extension = torch.ones(size=size, dtype=value.dtype, device=torch_device)
                    value = torch.cat((value, extension), dim=0).to(torch_device)

                processed_inputs[key] = value[:input_data_batch_size]

            pixel_values = processed_inputs["pixel_values"]
            target_len = torch.sum(processed_inputs["grid_thw"].prod(dim=1) // (processed_inputs["merge_sizes"] ** 2))
            if pixel_values.size(0) < target_len:
                size = (input_data_batch_size - value.shape[0], *value.shape[1:])
                extension = torch.randn(
                    size=(target_len - pixel_values.size(0)), dtype=pixel_values.dtype, device=torch_device
                )
            elif pixel_values.size(0) > target_len:
                pixel_values = pixel_values[:target_len]
            processed_inputs["pixel_values"] = pixel_values

            processed_inputs.update(
                {
                    "output_hidden_states": True,
                    "output_attentions": output_attentions,
                }
            )

            # TODO: test gradients as well (& for FA2 as well!)
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

            key = "hidden_states"

            # TODO: rename logits -> hidden_states
            logits_eager = outputs_eager[key][-1]
            logits_sdpa = outputs_sdpa[key][-1]

            if torch_device in ["cpu", "cuda"]:
                atol = atols[torch_device, enable_kernels, dtype]
                rtol = rtols[torch_device, enable_kernels, dtype]
            elif torch_device == "hpu":
                atol = atols["cuda", enable_kernels, dtype]
                rtol = rtols["cuda", enable_kernels, dtype]
            elif torch_device == "xpu":
                # As of PyTorch 2.5 XPU backend supports only torch.nn.attention.SDPBackend.MATH
                # which is implemented on PyTorch level using aten operators and is
                # device agnostic with respect to implementation of each aten operator.
                atol = atols["cuda", False, dtype]
                rtol = rtols["cuda", False, dtype]
            else:
                atol = 1e-7
                rtol = 1e-4

            # Avoid test flakiness with bf16!
            # bf16 is not good at precision when the magnitude is larger. We have some models like `SiglipVision` with
            # this test passing all the time for fp32/fp16 but flaky with bf16. Furthermore, `llama` and `clip` have
            # this test passing all the time for bf16: it turns out their outputs are of smaller size (0.1 and 1.0)
            # while `siglip` has outputs with maximal values around 3.0/4.0.
            outputs_magnitude = float(
                (torch.max(logits_sdpa.abs().amax(), logits_eager.abs().amax())).detach().to("cpu")
            )
            # The choice of `3e-2` in `outputs_magnitude * 1e-2` might not work if a model has even more larger outputs.
            # (we can try to analyze the `rtol` more closely element-wise in the future and adjust the `rtol` instead of `atol`).
            computed_atol = outputs_magnitude * 3e-2
            if dtype == torch.bfloat16:
                atol = max(atol, computed_atol)

            results = [
                torch.allclose(_logits_sdpa, _logits_eager, atol=atol, rtol=rtol)
                for (_logits_sdpa, _logits_eager) in zip(logits_sdpa, logits_eager)
            ]

            # If 80% batch elements have matched results, it's fine
            if np.mean(results) < 0.8:
                mean_relative_diff = ((logits_sdpa - logits_eager).abs() / (logits_eager.abs() + 1e-12)).mean()
                raise ValueError(
                    f"mean relative difference for {key}: {mean_relative_diff:.3e}, torch atol = {atol}, torch rtol = "
                    f"{rtol}"
                )


class VideoLlama3VisionModelTester:
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
        intermediate_size=37,
        attention_dropout=0.1,
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
        self.intermediate_size = intermediate_size
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope
        self.seq_length = (self.image_size // self.patch_size) ** 2

    def get_config(self):
        return VideoLlama3VisionConfig(
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
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
class VideoLlama3VisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as SIGLIP does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (VideoLlama3VisionModel,) if is_torch_available() else ()
    additional_model_inputs = ["grid_thw", "merge_sizes"]
    test_resize_embeddings = False
    test_head_masking = False
    test_cpu_offload = False
    test_disk_offload_safetensors = False
    test_disk_offload_bin = False

    def setUp(self):
        self.model_tester = VideoLlama3VisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VideoLlama3VisionConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(
        self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        if use_attention_mask:
            self.skipTest(reason="VideoLlama3VisionModel does not use attention mask")
        _test_encoder_eager_matches_sdpa_inference(self, dtype, output_attentions, enable_kernels)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        # force eager attention to support output attentions
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

            # check that output_attentions also work using config
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

            # Check attention is always last and order is fine
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
            self.assertListEqual(
                list(hidden_states[0].shape),
                [seq_length, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
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

        # force eager attention to support output attentions
        config._attn_implementation = "eager"

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class._from_config(config, attn_implementation="eager")
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs[0]

        # Encoder-/Decoder-only models
        hidden_states = outputs.hidden_states[0]
        hidden_states.retain_grad()

        if self.has_attentions:
            attentions = outputs.attentions[0][0]
            attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(hidden_states.grad)

        if self.has_attentions:
            self.assertIsNotNone(attentions.grad)

    @unittest.skip("Vision model requires additional positional inputs (grid_thw and merge_sizes)")
    def test_flash_attn_2_inference_equivalence(self):
        pass

    @unittest.skip("Vision model requires additional positional inputs (grid_thw and merge_sizes)")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    @unittest.skip("Vision model requires additional positional inputs (grid_thw and merge_sizes)")
    def test_flash_attn_kernels_inference_equivalence(self):
        pass


class VideoLlama3VisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        num_channels=3,
        image_size=14,
        is_training=True,
        text_config={
            "attention_dropout": 0.0,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 32,
            "intermediate_size": 37,
            "max_position_embeddings": 512,
            "max_window_layers": 3,
            "model_type": "qwen2",
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 1000000.0,
            "sliding_window": None,
            "tie_word_embeddings": True,
            "vocab_size": 99,
        },
        vision_config={
            "attention_dropout": 0.0,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 32,
            "intermediate_size": 64,
            "layer_norm_eps": 1e-06,
            "model_type": "video_llama_3_vision",
            "num_attention_heads": 4,
            "num_channels": 3,
            "num_hidden_layers": 2,
            "patch_size": 14,
        },
        use_token_compression=True,
        image_token_id=3,
        video_token_id=4,
    ):
        self.parent = parent
        self.hidden_size = text_config["hidden_size"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.patch_size = vision_config["patch_size"]
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.text_config = text_config
        self.vision_config = vision_config
        self.use_token_compression = use_token_compression
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.num_image_tokens = 32
        self.seq_length = seq_length + self.num_image_tokens

    def get_config(self):
        return VideoLlama3Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            use_token_compression=self.use_token_compression,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vision_config.patch_size
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
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        input_ids[:, -1] = config.text_config.pad_token_id
        attention_mask[:, -1] = 0
        input_ids[input_ids == self.video_token_id] = config.text_config.pad_token_id
        input_ids[input_ids == self.image_token_id] = config.text_config.pad_token_id
        input_ids[:, self.num_image_tokens] = self.image_token_id

        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, 1, 1]] * self.batch_size, device=torch_device),
            "image_merge_sizes": torch.tensor([1] * self.batch_size, device=torch_device),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class VideoLlama3ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `VideoLlama3ForConditionalGeneration`.
    """

    all_model_classes = (
        (
            VideoLlama3Model,
            VideoLlama3ForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = {"image-text-to-text": VideoLlama3ForConditionalGeneration}
    test_pruning = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = VideoLlama3VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VideoLlama3Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs through an error with explicit message saying what is wrong
        when number of images don't match number of image tokens in the text.
        Also we need to test multi-image cases when one prompt has multiple image tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            curr_input_dict = copy.deepcopy(input_dict)
            _ = model(**curr_input_dict)  # successfull forward with no modifications

            # remove one image but leave the image token in text
            patch_size = config.vision_config.patch_size
            one_img_length = (self.model_tester.image_size**2) // (patch_size**2)
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-one_img_length:, ...]
            curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][-1:, ...]
            curr_input_dict["image_merge_sizes"] = curr_input_dict["image_merge_sizes"][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            input_ids = curr_input_dict["input_ids"][:1]
            pixel_values = curr_input_dict["pixel_values"][:one_img_length]
            image_grid_thw = curr_input_dict["image_grid_thw"][:1]
            image_merge_sizes = curr_input_dict["image_merge_sizes"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    image_merge_sizes=image_merge_sizes,
                )

            # two images and two image tokens don't raise an error
            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            image_grid_thw = torch.cat([image_grid_thw, image_grid_thw], dim=0)
            image_merge_sizes = torch.cat([image_merge_sizes, image_merge_sizes], dim=0)
            _ = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                image_merge_sizes=image_merge_sizes,
            )

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
                padfree_positions = torch.cat(
                    [torch.arange(length) for length in dummy_attention_mask.sum(1).tolist()]
                )
                padfree_positions = padfree_positions.long().unsqueeze(0).to(torch_device)
                padfree_inputs_dict = {
                    "pixel_values": inputs_dict["pixel_values"],
                    "image_grid_thw": inputs_dict["image_grid_thw"],
                    "image_merge_sizes": inputs_dict["image_merge_sizes"],
                    "input_ids": inputs_dict["input_ids"][dummy_attention_mask.bool()].unsqueeze(0),
                    "position_ids": padfree_positions,
                }

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

                # We need to do simple forward without cache in roder to trigger packed SDPA/FLEX/EAGER path
                res_padded = model(**inputs_dict, use_cache=False)
                res_padfree = model(**padfree_inputs_dict, use_cache=False)

                logits_padded = res_padded.logits[inputs_dict["attention_mask"].bool()]
                logits_padfree = res_padfree.logits[0]

                # acceptable numerical instability
                tol = torch.finfo(torch.bfloat16).eps
                torch.testing.assert_close(logits_padded, logits_padfree, rtol=tol, atol=tol)


@require_torch
@slow
class VideoLlama3IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("lkhl/VideoLLaMA3-2B-Image-HF")
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]
        url = "https://github.com/DAMO-NLP-SG/VideoLLaMA3/raw/refs/heads/main/assets/sora.png"
        self.image = Image.open(requests.get(url, stream=True).raw)

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def test_small_model_integration_test(self):
        model = VideoLlama3ForConditionalGeneration.from_pretrained(
            "lkhl/VideoLLaMA3-2B-Image-HF", dtype=torch.bfloat16, device_map=torch_device
        )

        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[self.image], return_tensors="pt").to(torch_device)

        expected_input_ids = [151644, 872, 198] + [151655] * 10549 + [198, 74785, 279, 2168, 13, 151645, 198, 151644, 77091, 198]  # fmt: skip
        self.assertEqual(expected_input_ids, inputs.input_ids[0].tolist())

        expected_pixel_slice = torch.tensor(
            [
                [-0.8588, -0.9216, -0.9608],
                [-0.9922, -0.9922, -0.9922],
                [-0.9686, -0.9686, -0.9294],
                [-0.9294, -0.9765, -0.9765],
                [-0.9922, -0.9922, -0.9843],
                [-0.6000, -0.4118, -0.3647],
            ],
            dtype=torch.float32,
            device=torch_device,
        )
        torch.testing.assert_close(expected_pixel_slice, inputs.pixel_values[:6, :3], atol=1e-4, rtol=1e-4)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False, repetition_penalty=None)
        # fmt: off
        EXPECTED_DECODED_TEXT = Expectations(
            {
                ("cuda", None): "user\n\nDescribe the image.\nassistant\nThe image captures a vibrant nighttime scene on a bustling city street. A woman in a striking red dress",
                ("xpu", None): "user\n\nDescribe the image.\nassistant\nThe image captures a vibrant night scene in a bustling Japanese city. A woman in a striking red dress",
            }
        ).get_expectation()
        # fmt: on

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_batch(self):
        model = VideoLlama3ForConditionalGeneration.from_pretrained(
            "lkhl/VideoLLaMA3-2B-Image-HF", dtype=torch.bfloat16, device_map=torch_device
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text, text], images=[self.image, self.image], return_tensors="pt").to(
            torch_device
        )

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False, repetition_penalty=None)

        EXPECTED_DECODED_TEXT = [
            "user\n\nDescribe the image.\nassistant\nThe image captures a vibrant nighttime scene on a bustling city street. A woman in a striking red dress",
            "user\n\nDescribe the image.\nassistant\nThe image captures a vibrant nighttime scene on a bustling city street. A woman in a striking red dress",
        ]  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_batch_wo_image(self):
        model = VideoLlama3ForConditionalGeneration.from_pretrained(
            "lkhl/VideoLLaMA3-2B-Image-HF", dtype=torch.bfloat16, device_map=torch_device
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        messages2 = [
            {"role": "user", "content": [{"type": "text", "text": "What is relativity?"}]},
        ]
        text2 = self.processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text, text2], images=[self.image], padding=True, padding_side="left", return_tensors="pt"
        ).to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False, repetition_penalty=None)
        # fmt: off
        EXPECTED_DECODED_TEXT = Expectations(
            {
                ("cuda", None): [
                    "user\n\nDescribe the image.\nassistant\nThe image captures a vibrant nighttime scene on a bustling city street. A woman in a striking red dress",
                    "user\nWhat is relativity?\nassistant\nRelativity is a scientific theory that describes the relationship between space and time. It was first proposed by",
                ],
                ("xpu", None): [
                    "user\n\nDescribe the image.\nassistant\nThe image captures a vibrant night scene in a bustling Japanese city. A woman in a striking red dress",
                    "user\nWhat is relativity?\nassistant\nRelativity is a scientific theory that describes the relationship between space and time. It was first proposed by",
                ],
            }
        ).get_expectation()
        # fmt: on

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_batch_different_resolutions(self):
        model = VideoLlama3ForConditionalGeneration.from_pretrained(
            "lkhl/VideoLLaMA3-2B-Image-HF", dtype=torch.bfloat16, device_map=torch_device
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        text2 = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        image2 = self.image.resize((224, 224))
        inputs = self.processor(
            text=[text, text2], images=[self.image, image2], padding=True, padding_side="left", return_tensors="pt"
        ).to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False, repetition_penalty=None)
        DECODED_TEXT = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_DECODED_TEXT = [
            "user\n\nDescribe the image.\nassistant\nThe image captures a vibrant nighttime scene on a bustling city street. A woman in a striking red dress",
            "user\n\nDescribe the image.\nassistant\nThe image depicts a striking urban scene at night. A person is standing in the center of a wet",
        ]  # fmt: skip

        self.assertEqual(DECODED_TEXT, EXPECTED_DECODED_TEXT)

    @require_flash_attn
    @require_torch_accelerator
    @pytest.mark.flash_attn_test
    def test_small_model_integration_test_batch_flashatt2(self):
        model = VideoLlama3ForConditionalGeneration.from_pretrained(
            "lkhl/VideoLLaMA3-2B-Image-HF",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=torch_device,
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text, text], images=[self.image, self.image], return_tensors="pt").to(
            torch_device
        )

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False, repetition_penalty=None)

        # fmt: off
        EXPECTED_DECODED_TEXTS = Expectations(
            {
                (None, None): ['user\n\nDescribe the image.\nassistant\nThe image captures a vibrant nighttime scene on a bustling city street. A woman in a striking red dress',
                               'user\n\nDescribe the image.\nassistant\nThe image captures a vibrant nighttime scene on a bustling city street. A woman in a striking red dress',
                              ],
                ("xpu", 3): ['user\n\nDescribe the image.\nassistant\nThe image captures a vibrant nighttime scene on a bustling city street. A woman in a striking red dress',
                             'user\n\nDescribe the image.\nassistant\nThe image depicts a vibrant nighttime scene on a bustling city street. A woman in a striking red dress',
                            ],
            }
        )
        # fmt: on
        EXPECTED_DECODED_TEXT = EXPECTED_DECODED_TEXTS.get_expectation()

        DECODED_TEXT = self.processor.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(DECODED_TEXT, EXPECTED_DECODED_TEXT)

    @require_flash_attn
    @require_torch_accelerator
    @pytest.mark.flash_attn_test
    def test_small_model_integration_test_batch_wo_image_flashatt2(self):
        model = VideoLlama3ForConditionalGeneration.from_pretrained(
            "lkhl/VideoLLaMA3-2B-Image-HF",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=torch_device,
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        messages2 = [
            {"role": "user", "content": [{"type": "text", "text": "What is relativity?"}]},
        ]
        text2 = self.processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text, text2], images=[self.image], padding=True, padding_side="left", return_tensors="pt"
        ).to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False, repetition_penalty=None)

        EXPECTED_DECODED_TEXT = [
            'user\n\nDescribe the image.\nassistant\nThe image captures a vibrant nighttime scene on a bustling city street. A woman in a striking red dress',
            'user\nWhat is relativity?\nassistant\nRelativity is a scientific theory that describes the relationship between space and time. It was first proposed by'
        ]  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )
