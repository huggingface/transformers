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
"""Testing suite for the PaddleOCRVL model."""

import copy
import gc
import unittest

import pytest
from parameterized import parameterized

from transformers import (
    AutoProcessor,
    PaddleOCRVLConfig,
    PaddleOCRVLForConditionalGeneration,
    is_torch_available,
)
from transformers.testing_utils import (
    backend_empty_cache,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
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
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch


class PaddleOCRVLVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        seq_length=13,
        num_channels=3,
        image_height=28,
        image_width=28,
        text_config={
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "vocab_size": 103424,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": 32,
            "ignored_index": -100,
            "image_token_id": 100295,
            "intermediate_size": 32,
            "max_position_embeddings": 512,
            "model_type": "paddleocr_vl",
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {"mrope_section": [16, 24, 24], "rope_type": "default", "type": "default"},
            "rope_theta": 500000,
            "tie_word_embeddings": False,
        },
        vision_start_token_id=101305,
        vision_end_token_id=101306,
        image_token_id=100295,
        is_training=True,
        vision_config={
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 144,
            "intermediate_size": 32,
            "layer_norm_eps": 1e-06,
            "model_type": "paddleocr_vl",
            "num_attention_heads": 4,
            "num_channels": 3,
            "num_hidden_layers": 2,
            "pad_token_id": 0,
            "patch_size": 14,
            "spatial_merge_size": 2,
        },
    ):
        self.parent = parent
        self.bos_token_id = text_config["bos_token_id"]
        self.eos_token_id = text_config["eos_token_id"]
        self.pad_token_id = text_config["pad_token_id"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.hidden_size = text_config["hidden_size"]
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.image_token_id = image_token_id
        self.text_config = text_config
        self.vision_config = vision_config
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_height = image_height
        self.image_width = image_width
        self.is_training = is_training
        self.vocab_size = text_config["vocab_size"]
        self.num_image_tokens = 1
        self.seq_length = seq_length + self.num_image_tokens

    def get_config(self):
        return PaddleOCRVLConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            vision_start_token_id=self.vision_start_token_id,
            image_token_id=self.image_token_id,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vision_config.patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size * (self.image_height * self.image_width) // (patch_size**2),
                config.vision_config.num_channels,
                patch_size,
                patch_size,
            ]
        )

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        input_ids[:, :4] = torch.tensor([100273, 2969, 93963, 93919], dtype=input_ids.dtype, device=input_ids.device)
        input_ids[:, 4] = self.vision_start_token_id
        input_ids[:, 5 : 5 + self.num_image_tokens] = self.image_token_id
        input_ids[:, -8] = self.vision_end_token_id
        input_ids[:, -7:] = torch.tensor(
            [93972, 2497, 93963, 23, 92267, 93963, 93919], dtype=input_ids.dtype, device=input_ids.device
        )

        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, 2, 2]] * self.batch_size, device=torch_device),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class PaddleOCRVLModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Model tester for `PaddleOCRVLForConditionalGeneration`.
    """

    all_model_classes = (PaddleOCRVLForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {"image-text-to-text": PaddleOCRVLForConditionalGeneration}
    _is_composite = True

    def setUp(self):
        self.model_tester = PaddleOCRVLVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PaddleOCRVLConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_mismatching_num_image_tokens(self):
        """
        Tests that an explicit error is thrown when the number of image tokens
        doesn't match the number of image placeholders in the text.
        We also test multi-image cases when one prompt has multiple image tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            curr_input_dict = copy.deepcopy(input_dict)  # in-place modifications further
            _ = model(**curr_input_dict)  # successful forward with no modifications

            # remove one image but leave all the image tokens in text
            patch_size = config.vision_config.patch_size
            one_img_length = (self.model_tester.image_height * self.model_tester.image_width) // (patch_size**2)
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-one_img_length:, ...]
            curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][-1:, ...]
            with self.assertRaisesRegex(ValueError, "Image features and image tokens do not match"):
                _ = model(**curr_input_dict)

            # simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            input_ids = curr_input_dict["input_ids"][:1]
            pixel_values = curr_input_dict["pixel_values"][:one_img_length]
            image_grid_thw = curr_input_dict["image_grid_thw"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaisesRegex(ValueError, "Image features and image tokens do not match"):
                _ = model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

            # two images and two image tokens don't raise an error
            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            image_grid_thw = torch.cat([image_grid_thw, image_grid_thw], dim=0)
            _ = model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    # PaddleOCRVL has pixel_values shaped as (bs*patch_len, image_channels, patch_size, patch_size) so we can't slice to batches in generate
    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # We don't want a few model inputs in our model input dictionary for generation tests
        input_keys_to_ignore = [
            # we don't want encoder-decoder models to start from filled decoder ids
            "decoder_input_ids",
            "decoder_attention_mask",
            # we'll set cache use in each test differently
            "use_cache",
            # Ignore labels if it is in the input dict
            "labels",
            # model-specific exceptions should overload/overwrite this function
        ]

        # The diff from the general `prepare_config_and_inputs_for_generate` lies here
        patch_size = config.vision_config.patch_size
        filtered_image_length = (
            batch_size * (self.model_tester.image_height * self.model_tester.image_width) // (patch_size**2)
        )
        filtered_inputs_dict = {
            k: v[:batch_size, ...] if isinstance(v, torch.Tensor) else v
            for k, v in inputs_dict.items()
            if k not in input_keys_to_ignore
        }
        filtered_inputs_dict["pixel_values"] = inputs_dict["pixel_values"][:filtered_image_length]

        # It is important set `eos_token_id` to `None` to avoid early stopping (would break for length-based checks)
        text_gen_config = config.get_text_config(decoder=True)
        if text_gen_config.eos_token_id is not None and text_gen_config.pad_token_id is None:
            text_gen_config.pad_token_id = (
                text_gen_config.eos_token_id
                if isinstance(text_gen_config.eos_token_id, int)
                else text_gen_config.eos_token_id[0]
            )
        text_gen_config.eos_token_id = None
        text_gen_config.forced_eos_token_id = None

        return config, filtered_inputs_dict

    @unittest.skip(reason="PaddleOCRVL does not support.")
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    @unittest.skip(reason="PaddleOCRVL does not support.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="PaddleOCRVL does not support beam search.")
    def test_beam_sample_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="PaddleOCRVL does not support beam search.")
    def test_beam_search_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="PaddleOCRVL does not support beam search.")
    def test_beam_search_generate_dict_output(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="PaddleOCRVL does not support beam search.")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="PaddleOCRVL does not support beam search.")
    def test_beam_sample_generate_dict_output(self):
        pass

    @unittest.skip(reason="PaddleOCRVL needs to apply weight conversions.")
    def test_can_load_from_already_mapped_keys(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="PaddleOCRVL does not support beam search.")
    def test_generate_from_inputs_embeds_1_beam_search(self, _, num_beams):
        pass

    @parameterized.expand([("random",), ("same",)])
    @pytest.mark.generate
    @unittest.skip(reason="PaddleOCRVL does not support assisted decoding.")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="PaddleOCRVL does not support assisted decoding.")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("PaddleOCRVL does not support this test.")
    def test_model_is_small(self):
        pass


@require_torch
@slow
class PaddleOCRVLIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("PaddlePaddle/PaddleOCR-VL")
        self.messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/ocr_demo2.jpg",
                    },
                    {"type": "text", "text": "OCR:"},
                ],
            }
        ]

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def test_small_model_integration_test(self):
        model = (
            PaddleOCRVLForConditionalGeneration.from_pretrained(
                "PaddlePaddle/PaddleOCR-VL",
                dtype="bfloat16",
            )
            .to(torch_device)
            .eval()
        )

        inputs = self.processor.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        expected_input_ids_length = 211
        assert expected_input_ids_length == len(inputs.input_ids[0])

        expected_input_ids = [100273, 2969, 93963, 93919, 101305, 100295, 100295, 100295, 100295, 100295]  # fmt: skip
        assert expected_input_ids == inputs.input_ids[0].tolist()[:10]

        expected_pixel_slice = torch.tensor(
            [
                [1.0000, 1.0000, 1.0000],
                [1.0000, 1.0000, 1.0000],
                [0.9922, 0.9922, 0.9922],
                [1.0000, 1.0000, 1.0000],
                [1.0000, 1.0000, 1.0000],
            ],
            dtype=torch.float32,
            device="cpu",
        )

        assert torch.allclose(expected_pixel_slice, inputs.pixel_values[:5, :, 0, 0], atol=3e-3)

        # verify generation
        inputs = inputs.to(torch_device)
        output = model.generate(**inputs, max_new_tokens=30)
        result = self.processor.decode(output[0][inputs["input_ids"].shape[-1] : -1])

        EXPECTED_DECODED_TEXT = "生甘草"

        self.assertEqual(
            result,
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_batch(self):
        model = (
            PaddleOCRVLForConditionalGeneration.from_pretrained("PaddlePaddle/PaddleOCR-VL", dtype="bfloat16")
            .to(torch_device)
            .eval()
        )

        inputs = self.processor.apply_chat_template(
            [self.messages, self.messages],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)]
        result = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        EXPECTED_DECODED_TEXT = ["生甘草", "生甘草"]

        self.assertEqual(
            result,
            EXPECTED_DECODED_TEXT,
        )

    @require_flash_attn
    @require_torch_accelerator
    @pytest.mark.flash_attn_test
    def test_small_model_integration_test_flashatt2(self):
        model = (
            PaddleOCRVLForConditionalGeneration.from_pretrained(
                "PaddlePaddle/PaddleOCR-VL", dtype="bfloat16", attn_implementation="flash_attention_2"
            )
            .to(torch_device)
            .eval()
        )

        inputs = self.processor.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        expected_input_ids_length = 211
        assert expected_input_ids_length == len(inputs.input_ids[0])

        expected_input_ids = [100273, 2969, 93963, 93919, 101305, 100295, 100295, 100295, 100295, 100295]  # fmt: skip
        assert expected_input_ids == inputs.input_ids[0].tolist()[:10]

        expected_pixel_slice = torch.tensor(
            [
                [1.0000, 1.0000, 1.0000],
                [1.0000, 1.0000, 1.0000],
                [0.9922, 0.9922, 0.9922],
                [1.0000, 1.0000, 1.0000],
                [1.0000, 1.0000, 1.0000],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        assert torch.allclose(expected_pixel_slice, inputs.pixel_values[:5, :, 0, 0], atol=3e-3)

        # verify generation
        inputs = inputs.to(torch_device)
        output = model.generate(**inputs, max_new_tokens=30)
        result = self.processor.decode(output[0][inputs["input_ids"].shape[-1] : -1])

        EXPECTED_DECODED_TEXT = "生甘草"

        self.assertEqual(
            result,
            EXPECTED_DECODED_TEXT,
        )

    @require_flash_attn
    @require_torch_accelerator
    @pytest.mark.flash_attn_test
    def test_small_model_integration_test_batch_flashatt2(self):
        model = (
            PaddleOCRVLForConditionalGeneration.from_pretrained(
                "PaddlePaddle/PaddleOCR-VL", dtype="bfloat16", attn_implementation="flash_attention_2"
            )
            .to(torch_device)
            .eval()
        )

        inputs = self.processor.apply_chat_template(
            [self.messages, self.messages],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)]
        result = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        EXPECTED_DECODED_TEXT = ["生甘草", "生甘草"]

        self.assertEqual(
            result,
            EXPECTED_DECODED_TEXT,
        )
