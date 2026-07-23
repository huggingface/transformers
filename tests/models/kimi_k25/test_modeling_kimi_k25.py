# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Kimi2.6 model."""

import unittest

from parameterized import parameterized

from transformers import (
    AutoProcessor,
    DeepseekV3Config,
    Kimi_K25Config,
    Kimi_K25VisionConfig,
    is_torch_available,
    is_vision_available,
)
from transformers.cache_utils import Cache
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...test_modeling_common import (
    floats_tensor,
)
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import Kimi_K25ForConditionalGeneration, Kimi_K25Model


if is_vision_available():
    pass


class Kimi_K25VisionText2TextModelTester(VLMModelTester):
    base_model_class = Kimi_K25Model
    config_class = Kimi_K25Config
    text_config_class = DeepseekV3Config
    vision_config_class = Kimi_K25VisionConfig
    conditional_generation_class = Kimi_K25ForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("video_token_id", 4)
        kwargs.setdefault("image_size", 32)
        kwargs.setdefault("patch_size", 8)
        kwargs.setdefault("num_image_tokens", 16)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("head_dim", 8)
        kwargs.setdefault("num_heads", 4)
        kwargs.setdefault("pos_emb_height", 4)
        kwargs.setdefault("merge_kernel_size", (1, 1))
        kwargs.setdefault("pos_emb_width", 4)
        kwargs.setdefault("pos_emb_time", 1)
        kwargs.setdefault("kv_lora_rank", 16)
        kwargs.setdefault("q_lora_rank", 32)
        kwargs.setdefault("qk_rope_head_dim", 16)
        kwargs.setdefault("v_head_dim", 32)
        kwargs.setdefault("qk_nope_head_dim", 32)
        kwargs.setdefault("attention_probs_dropout_prob", 0.0)

        # MoE fields synced with DeepSeekV3Tester
        # `first_k_dense_replace` enables MoEs after `layer=0`
        kwargs.setdefault("first_k_dense_replace", 1)
        kwargs.setdefault("n_group", 2)
        kwargs.setdefault("topk_group", 1)
        kwargs.setdefault("num_experts_per_tok", 8)
        kwargs.setdefault("n_shared_experts", 1)
        kwargs.setdefault("n_routed_experts", 8)
        kwargs.setdefault("moe_intermediate_size", 16)
        kwargs.setdefault("aux_loss_alpha", 0.001)
        kwargs.setdefault("routed_scaling_factor", 2.5)

        kwargs.setdefault(
            "rope_parameters",
            {
                "rope_type": "default",
                "rope_theta": 10000,
            },
        )
        super().__init__(parent, **kwargs)

        # These can be inferred from existing properties and don't get separate kwargs
        self.projection_hidden_size = self.hidden_size

    def create_pixel_values(self):
        return floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (self.patch_size**2),
                self.num_channels,
                self.patch_size,
                self.patch_size,
            ]
        )

    def place_image_tokens(self, input_ids, config):
        # Place image tokens with vision_start_token_id prefix
        input_ids = input_ids.clone()
        # Clear any accidental special tokens first
        input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        # Place image tokens with vision_start_token_id prefix
        input_ids[:, : self.num_image_tokens] = self.image_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, pixel_values):
        return {
            "image_grid_thw": torch.tensor([[1, 4, 4]] * self.batch_size, device=torch_device),
        }


@require_torch
class Kimi_K25ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Kimi_K25VisionText2TextModelTester

    # Kimi has images shaped as (bs*patch_len, dim) so we can't slice to batches in generate
    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # We don't want a few model inputs in our model input dictionary for generation tests
        input_keys_to_ignore = [
            "decoder_input_ids",
            "decoder_attention_mask",
            "use_cache",
            "labels",
        ]

        # The diff from the general `prepare_config_and_inputs_for_generate` lies here
        patch_size = config.vision_config.patch_size
        filtered_image_length = batch_size * (self.model_tester.image_size**2) // (patch_size**2)
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

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        """Needs to be overridden as deepseek has special MLA cache format (though we don't really use the MLA)"""
        self.assertIsInstance(past_key_values, Cache)

        # (batch, head, seq_length, head_features)
        expected_common_shape = (
            batch_size,
            getattr(config, "num_key_value_heads", config.num_attention_heads),
            seq_length,
        )
        expected_key_shape = expected_common_shape + (config.qk_nope_head_dim + config.qk_rope_head_dim,)
        expected_value_shape = expected_common_shape + (config.v_head_dim,)

        for layer in past_key_values.layers:
            self.assertEqual(layer.keys.shape, expected_key_shape)
            self.assertEqual(layer.values.shape, expected_value_shape)

    def test_reverse_loading_mapping(self):
        super().test_reverse_loading_mapping(skip_base_model=True)

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip("DeepseekV3 backbone is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("DeepseekV3 backbone is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("DeepseekV3 backbone is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("Deepseek-V3 backbone uses MLA so it is not compatible with the standard cache format")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("Deepseek-V3 backbone uses MLA so it is not compatible with the standard cache format")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="SDPA can't dispatch on flash due to unsupported head dims on LM backbone")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Needs to update values in `grid_thw` otherwise it just gets broadcasted")
    def test_mismatching_num_image_tokens(self):
        pass


@slow
@require_torch
class KimiK25IntegrationTest(unittest.TestCase):
    model_id = "hf-internal-testing/kimi-k25-for-integration-test"
    model = None
    processor = None

    @classmethod
    def setUpClass(cls):
        cleanup(torch_device, gc_collect=True)
        cls.model = Kimi_K25ForConditionalGeneration.from_pretrained(cls.model_id, device_map="auto")
        cls.processor = AutoProcessor.from_pretrained(cls.model_id)

        cls.message1 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "What kind of dog is this?"},
                ],
            }
        ]
        cls.message2 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png",
                    },
                    {"type": "text", "text": "What kind of dog is this?"},
                ],
            }
        ]
        cls.message3 = [{"role": "user", "content": "Who would win in a fight - a dinosaur or a cow named Moo Moo?"}]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_logits_text_only(self):
        expectations = Expectations(
            {
                ("cuda", None): [
                    [[ 0.00000000, -0.02641739,  0.01707786, -0.08232787,  0.12016402, -0.15289734, -0.08252842,
                    0.15658118, -0.04173164,  0.10555173],
                    [ 0.00000000, -0.02639876,  0.01706616, -0.08229892,  0.12015337, -0.15289734, -0.08251060,
                    0.15658717, -0.04172155,  0.10556119],
                    [ 0.00000000, -0.02639606,  0.01708876, -0.08232962,  0.12015553, -0.15291165, -0.08253470,
                    0.15658627, -0.04173409,  0.10555199],
                    [ 0.00000000, -0.02638427,  0.01708288, -0.08230965,  0.12014508, -0.15289523, -0.08252528,
                    0.15660310, -0.04173251,  0.10554133],
                    [ 0.00000000, -0.02639318,  0.01706917, -0.08231460,  0.12016111, -0.15290320, -0.08252579,
                    0.15659124, -0.04173930,  0.10554294],
                    [ 0.00000000, -0.02638769,  0.01707737, -0.08231344,  0.12015510, -0.15289856, -0.08252031,
                    0.15659934, -0.04173541,  0.10554094],
                    [ 0.00000000, -0.02639529,  0.01708556, -0.08231322,  0.12014811, -0.15291123, -0.08251978,
                    0.15660490, -0.04173838,  0.10553965],
                    [ 0.00000000, -0.02639412,  0.01707170, -0.08232507,  0.12014882, -0.15289663, -0.08251097,
                    0.15659560, -0.04172803,  0.10553968],
                    [ 0.00000000, -0.02639644,  0.01707644, -0.08233012,  0.12013669, -0.15289611, -0.08249903,
                    0.15660135, -0.04172298,  0.10554458],
                    [ 0.00000000, -0.02639807,  0.01707577, -0.08232461,  0.12014168, -0.15288822, -0.08250540,
                    0.15660320, -0.04172593,  0.10553741]]
                ],
            }
        )  # fmt: skip
        expectations_mean = Expectations({("cuda", None): -0.014419052749872208})

        inputs = self.processor.apply_chat_template(
            self.message3, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        torch.testing.assert_close(logits[:, :10, :10].cpu(), torch.tensor(expectations.get_expectation()))
        torch.testing.assert_close(logits.mean().cpu(), torch.tensor(expectations_mean.get_expectation()))

    def test_model_logits(self):
        expectations = Expectations(
            {
                ("cuda", None): [
                    [[ 0.00000000, -0.02641739,  0.01707786, -0.08232787,  0.12016402, -0.15289734, -0.08252842,
                    0.15658118, -0.04173164,  0.10555173],
                    [ 0.00000000, -0.02639876,  0.01706616, -0.08229892,  0.12015337, -0.15289734, -0.08251060,
                    0.15658717, -0.04172155,  0.10556119],
                    [ 0.00000000, -0.02639606,  0.01708876, -0.08232962,  0.12015553, -0.15291165, -0.08253470,
                    0.15658627, -0.04173409,  0.10555199],
                    [ 0.00000000, -0.02639439,  0.01708424, -0.08231656,  0.12013876, -0.15291429, -0.08251298,
                    0.15660164, -0.04171884,  0.10554679],
                    [ 0.00000000, -0.02638364,  0.01708178, -0.08230615,  0.12013706, -0.15290739, -0.08251728,
                    0.15661213, -0.04172764,  0.10554501],
                    [ 0.00000000, -0.02639332,  0.01708583, -0.08231664,  0.12012445, -0.15291154, -0.08248951,
                    0.15662190, -0.04171957,  0.10554942],
                    [ 0.00000000,  0.08700177,  0.08366316,  0.07953200, -0.16923970,  0.06034253,  0.23461264,
                    0.23159909, -0.08662768, -0.01452735],
                    [ 0.00000000, -0.07244547,  0.02737196,  0.07369756,  0.11849800, -0.01468838, -0.08068197,
                    -0.22196104, -0.10857108,  0.00384381],
                    [ 0.00000000,  0.24396195,  0.07257971,  0.02183190,  0.05994213, -0.09902838, -0.05029112,
                    -0.09634171, -0.11046760,  0.02089387],
                    [ 0.00000000,  0.05029136, -0.03608268,  0.03495589,  0.02199023, -0.06974902,  0.11681847,
                    -0.12890734, -0.07075065,  0.12983331]]
                ],
            }
        )  # fmt: skip
        expectations_mean = Expectations({("cuda", None): 0.007999920286238194})

        inputs = self.processor.apply_chat_template(
            self.message1, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        torch.testing.assert_close(logits[:, :10, :10].cpu(), torch.tensor(expectations.get_expectation()))
        torch.testing.assert_close(logits.mean().cpu(), torch.tensor(expectations_mean.get_expectation()))

    def test_model_logits_batched(self):
        expectations = Expectations(
            {
                ("cuda", None): [
                    [[ 0.00000000, -0.02641735,  0.01707790, -0.08232785,  0.12016401, -0.15289739, -0.08252840,
                    0.15658122, -0.04173165,  0.10555175],
                    [ 0.00000000, -0.02639874,  0.01706615, -0.08229891,  0.12015339, -0.15289740, -0.08251059,
                    0.15658718, -0.04172156,  0.10556121],
                    [ 0.00000000, -0.02639605,  0.01708876, -0.08232963,  0.12015552, -0.15291169, -0.08253470,
                    0.15658626, -0.04173411,  0.10555198],
                    [ 0.00000000, -0.02639438,  0.01708425, -0.08231656,  0.12013876, -0.15291435, -0.08251297,
                    0.15660165, -0.04171887,  0.10554679],
                    [ 0.00000000, -0.02638364,  0.01708182, -0.08230612,  0.12013703, -0.15290746, -0.08251727,
                    0.15661217, -0.04172765,  0.10554502],
                    [ 0.00000000, -0.02639329,  0.01708584, -0.08231664,  0.12012445, -0.15291159, -0.08248951,
                    0.15662192, -0.04171958,  0.10554940],
                    [ 0.00000000,  0.08700177,  0.08366317,  0.07953200, -0.16923971,  0.06034254,  0.23461263,
                    0.23159909, -0.08662771, -0.01452734],
                    [ 0.00000000, -0.07244548,  0.02737197,  0.07369757,  0.11849802, -0.01468838, -0.08068197,
                    -0.22196105, -0.10857107,  0.00384379],
                    [ 0.00000000,  0.24396195,  0.07257971,  0.02183192,  0.05994214, -0.09902842, -0.05029113,
                    -0.09634172, -0.11046763,  0.02089386],
                    [ 0.00000000,  0.05029136, -0.03608267,  0.03495590,  0.02199023, -0.06974902,  0.11681848,
                    -0.12890735, -0.07075066,  0.12983333]],

                    [[ 0.00000000, -0.02641735,  0.01707790, -0.08232785,  0.12016401, -0.15289739, -0.08252840,
                    0.15658122, -0.04173165,  0.10555175],
                    [ 0.00000000, -0.02639874,  0.01706615, -0.08229891,  0.12015339, -0.15289740, -0.08251059,
                    0.15658718, -0.04172156,  0.10556121],
                    [ 0.00000000, -0.02639605,  0.01708876, -0.08232963,  0.12015552, -0.15291169, -0.08253470,
                    0.15658626, -0.04173411,  0.10555198],
                    [ 0.00000000, -0.02639438,  0.01708425, -0.08231656,  0.12013876, -0.15291435, -0.08251297,
                    0.15660165, -0.04171887,  0.10554679],
                    [ 0.00000000, -0.02638364,  0.01708182, -0.08230612,  0.12013703, -0.15290746, -0.08251727,
                    0.15661217, -0.04172765,  0.10554502],
                    [ 0.00000000, -0.02639329,  0.01708584, -0.08231664,  0.12012445, -0.15291159, -0.08248951,
                    0.15662192, -0.04171958,  0.10554940],
                    [ 0.00000000, -0.06525577,  0.08833339,  0.19255474, -0.08825377, -0.13921326,  0.14602648,
                    0.04482564, -0.14889751,  0.11028164],
                    [ 0.00000000, -0.00419279,  0.03873007,  0.31887013, -0.01151188, -0.30564448,  0.16670589,
                    0.19029431, -0.13194472,  0.02135594],
                    [ 0.00000000, -0.05691863,  0.09113241,  0.07550406,  0.09380092, -0.17235518,  0.20939635,
                    0.04311826, -0.04715816, -0.00544562],
                    [ 0.00000000, -0.01738080,  0.05235744,  0.22640616, -0.09752068, -0.16835804,  0.23356910,
                    -0.02671638, -0.13405795, -0.09033854]]
                ],
            }
        )  # fmt: skip
        expectations_mean = Expectations({("cuda", None): 0.007677852641791105})

        inputs = self.processor.apply_chat_template(
            [self.message1, self.message2],
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        torch.testing.assert_close(logits[:, :10, :10].cpu(), torch.tensor(expectations.get_expectation()))
        torch.testing.assert_close(logits.mean().cpu(), torch.tensor(expectations_mean.get_expectation()))
