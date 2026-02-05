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
"""Testing suite for the PPChart2Table model."""

import gc
import unittest

import pytest
from parameterized import parameterized
from PIL import Image

from transformers import (
    AutoProcessor,
    PPChart2TableConfig,
    PPChart2TableForConditionalGeneration,
    is_torch_available,
)
from transformers.testing_utils import (
    backend_empty_cache,
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch


class PPChart2TableVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        seq_length=31,
        num_channels=3,
        image_height=64,
        image_width=64,
        text_config={
            "hidden_size": 32,
            "hidden_act": "silu",
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 32,
            "attention_dropout": 0.0,
            "sliding_window": 32768,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151860,
            "max_position_embeddings": 32768,
            "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
        },
        is_training=False,
        vision_config={
            "depth": 2,
            "embed_dim": 768,
            "hidden_size": 144,
            "img_size": 64,
            "mlp_ratio": 4.0,
            "norm_layer_eps": 1e-6,
            "num_heads": 4,
            "patch_size": 16,
            "qkv_bias": True,
            "use_rel_pos": True,
            "global_attn_indexes": [2, 5, 8, 11],
            "window_size": 14,
            "out_chans": 256,
        },
        bos_token_id=151643,
        eos_token_id=151643,
        im_start_token=151857,
        im_end_token=151858,
        im_patch_token=151859,
    ):
        self.parent = parent
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.hidden_size = text_config["hidden_size"]
        self.im_start_token = im_start_token
        self.im_end_token = im_end_token
        self.im_patch_token = im_patch_token
        self.text_config = text_config
        self.vision_config = vision_config
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_channels = num_channels
        self.image_height = image_height
        self.image_width = image_width
        self.is_training = is_training
        self.vocab_size = text_config["vocab_size"]

    def get_config(self):
        return PPChart2TableConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        pixel_values = torch.randn((1, 3, self.image_height, self.image_width)).to(torch_device)

        num_patch = self.image_height // 16 // 4
        input = (
            [
                151644,
                8948,
                198,
                2610,
                1265,
                1795,
                279,
                11221,
                15516,
                323,
                10339,
                697,
                11253,
                304,
                7716,
                13,
                151645,
                151644,
                872,
                198,
                151857,
            ]
            + [151859] * (num_patch * num_patch)
            + [151858, 198, 14488, 311, 1965, 151645, 151644, 77091, 198]
        )

        input_ids = torch.tensor(input).unsqueeze(0).to(torch_device)

        return config, pixel_values, input_ids

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, input_ids = config_and_inputs
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class PPChart2TableModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (PPChart2TableForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {"image-text-to-text": PPChart2TableForConditionalGeneration}
    _is_composite = True

    def setUp(self):
        self.model_tester = PPChart2TableVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PPChart2TableConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="PPChart2Table does not support.")
    def test_sliding_window_mask(self):
        pass

    @unittest.skip(reason="PPChart2Table does not support.")
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    @unittest.skip(reason="PPChart2Table does not support.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="PPChart2Table does not support beam search.")
    def test_beam_sample_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="PPChart2Table does not support beam search.")
    def test_beam_search_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="PPChart2Table does not support beam search.")
    def test_beam_search_generate_dict_output(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="PPChart2Table does not support beam search.")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="PPChart2Table does not support beam search.")
    def test_beam_sample_generate_dict_output(self):
        pass

    @unittest.skip(reason="PPChart2Table needs to apply weight conversions.")
    def test_can_load_from_already_mapped_keys(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="PPChart2Table does not support beam search.")
    def test_generate_from_inputs_embeds_1_beam_search(self, _, num_beams):
        pass

    @parameterized.expand([("random",), ("same",)])
    @pytest.mark.generate
    @unittest.skip(reason="PPChart2Table does not support assisted decoding.")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="PPChart2Table does not support assisted decoding.")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("PPChart2Table does not support this test.")
    def test_model_is_small(self):
        pass


@require_torch
@slow
class PPChart2TableIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("/workspace/model_weight_torch/PP-Chart2Table")

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def test_small_model_integration_test(self):
        model = PPChart2TableForConditionalGeneration.from_pretrained(
            "/workspace/model_weight_torch/PP-Chart2Table", dtype="float32"
        ).to("cuda")

        image = Image.open(
            "/workspace/PaddleX/paddlex/inference/models/doc_vlm/modeling/chart_parsing_02.png"
        ).convert("RGB")
        inputs = self.processor(images=image).to(model.device)
        breakpoint()
        expected_input_ids_length = 286
        assert expected_input_ids_length == len(inputs.input_ids[0])

        expected_input_ids = [151644, 8948, 198, 2610, 1265, 1795, 279, 11221, 15516, 323]

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

    # def test_small_model_integration_test_batch(self):
    #     model = (
    #         PPChart2TableForConditionalGeneration.from_pretrained("/workspace/model_weight_torch/PP-Chart2Table", dtype="bfloat16")
    #         .to(torch_device)
    #         .eval()
    #     )

    #     image = Image.open("/workspace/PaddleX/paddlex/inference/models/doc_vlm/modeling/chart_parsing_02.png").convert("RGB")
    #     inputs = self.processor(images=image).to(model.device)

    #     output = model.generate(**inputs, max_new_tokens=256)
    #     generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)]
    #     result = self.processor.batch_decode(
    #         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    #     )

    #     EXPECTED_DECODED_TEXT = ["生甘草", "生甘草"]

    #     self.assertEqual(
    #         result,
    #         EXPECTED_DECODED_TEXT,
    #     )

    # @require_flash_attn
    # @require_torch_accelerator
    # @pytest.mark.flash_attn_test
    # def test_small_model_integration_test_flashatt2(self):
    #     model = (
    #         PPChart2TableForConditionalGeneration.from_pretrained(
    #             "/workspace/model_weight_torch/PP-Chart2Table", dtype="bfloat16", attn_implementation="flash_attention_2"
    #         )
    #         .to(torch_device)
    #         .eval()
    #     )

    #     image = Image.open("/workspace/PaddleX/paddlex/inference/models/doc_vlm/modeling/chart_parsing_02.png").convert("RGB")
    #     inputs = self.processor(images=image).to(model.device)

    #     expected_input_ids_length = 211
    #     assert expected_input_ids_length == len(inputs.input_ids[0])

    #     expected_input_ids = [100273, 2969, 93963, 93919, 101305, 100295, 100295, 100295, 100295, 100295]  # fmt: skip
    #     assert expected_input_ids == inputs.input_ids[0].tolist()[:10]

    #     expected_pixel_slice = torch.tensor(
    #         [
    #             [1.0000, 1.0000, 1.0000],
    #             [1.0000, 1.0000, 1.0000],
    #             [0.9922, 0.9922, 0.9922],
    #             [1.0000, 1.0000, 1.0000],
    #             [1.0000, 1.0000, 1.0000],
    #         ],
    #         dtype=torch.float32,
    #         device="cpu",
    #     )
    #     assert torch.allclose(expected_pixel_slice, inputs.pixel_values[:5, :, 0, 0], atol=3e-3)

    #     # verify generation
    #     inputs = inputs.to(torch_device)
    #     output = model.generate(**inputs, max_new_tokens=30)
    #     result = self.processor.decode(output[0][inputs["input_ids"].shape[-1] : -1])

    #     EXPECTED_DECODED_TEXT = "生甘草"

    #     self.assertEqual(
    #         result,
    #         EXPECTED_DECODED_TEXT,
    #     )

    # @require_flash_attn
    # @require_torch_accelerator
    # @pytest.mark.flash_attn_test
    # def test_small_model_integration_test_batch_flashatt2(self):
    #     model = (
    #         PPChart2TableForConditionalGeneration.from_pretrained(
    #             "/workspace/model_weight_torch/PP-Chart2Table", dtype="bfloat16", attn_implementation="flash_attention_2"
    #         )
    #         .to(torch_device)
    #         .eval()
    #     )

    #     image = Image.open("/workspace/PaddleX/paddlex/inference/models/doc_vlm/modeling/chart_parsing_02.png").convert("RGB")
    #     inputs = self.processor(images=image).to(model.device)

    #     # it should not matter whether two images are the same size or not
    #     output = model.generate(**inputs, max_new_tokens=30)
    #     generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)]
    #     result = self.processor.batch_decode(
    #         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    #     )

    #     EXPECTED_DECODED_TEXT = ["生甘草", "生甘草"]

    #     self.assertEqual(
    #         result,
    #         EXPECTED_DECODED_TEXT,
    #     )
