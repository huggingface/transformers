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
"""Testing suite for the PyTorch Gemma3 model."""

import logging
import tempfile
import unittest

import pytest
from parameterized import parameterized

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma3Config,
    Gemma3TextConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    is_flash_attn_2_available,
    require_deterministic_for_xpu,
    require_flash_attn,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    require_torch_large_accelerator,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...models.gemma.test_modeling_gemma import GemmaModelTester
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        Gemma3ForCausalLM,
        Gemma3ForConditionalGeneration,
        Gemma3ForSequenceClassification,
        Gemma3Model,
        Gemma3Processor,
        Gemma3TextModel,
    )
    from transformers.pytorch_utils import is_torch_greater_or_equal


class Gemma3ModelTester(GemmaModelTester):
    if is_torch_available():
        config_class = Gemma3TextConfig
        model_class = Gemma3TextModel
        for_causal_lm_class = Gemma3ForCausalLM


@require_torch
class Gemma3ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (Gemma3TextModel, Gemma3ForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (Gemma3ForCausalLM,) if is_torch_available() else ()
    test_headmasking = False
    test_pruning = False
    _is_stateful = True
    model_split_percents = [0.5, 0.6]

    def setUp(self):
        self.model_tester = Gemma3ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Gemma3Config, hidden_size=37)

    @unittest.skip("Failing because of unique cache (HybridCache)")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    @parameterized.expand([("random",), ("same",)])
    @pytest.mark.generate
    @unittest.skip("Gemma3 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("Gemma3 has HybridCache which is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @pytest.mark.generate
    @unittest.skip("Gemma3 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("Gemma3 has HybridCache which is not compatible with dola decoding")
    def test_dola_decoding_sample(self):
        pass

    @unittest.skip("Gemma3 has HybridCache and doesn't support continue from past kv")
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip("Gemma3 has HybridCache and doesn't support low_memory generation")
    def test_beam_search_low_memory(self):
        pass

    @unittest.skip("Gemma3 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate(self):
        pass

    @unittest.skip("Gemma3 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("Gemma3 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip("Gemma3 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip("Gemma3 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip("Gemma3 has HybridCache which auto-compiles. Compile and FA2 don't work together.")
    def test_eager_matches_fa2_generate(self):
        pass

    @unittest.skip(
        reason="HybridCache can't be gathered because it is not iterable. Adding a simple iter and dumping `distributed_iterator`"
        " as in Dynamic Cache doesn't work. NOTE: @gante all cache objects would need better compatibility with multi gpu setting"
    )
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip("Gemma3 applies key/query norm which doesn't work with packing")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Gemma3 applies key/query norm which doesn't work with packing")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    def test_generation_beyond_sliding_window_tiny_model(self):
        """Test generation with a tiny randomly initialised model whose input length is larger than the `sliding_window`.
        The model is configured with both `full_attention` and `sliding_attention` layers to make sure the hybrid cache
        and mask slicing logic is covered.
        """
        config = Gemma3TextConfig.from_pretrained("hf-internal-testing/tiny-random-Gemma3ForCausalLM")
        config.attn_implementation = "eager"
        config.layer_types = ["full_attention", "sliding_attention"]
        config.sliding_window = 8
        config.max_position_embeddings = 128
        model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-Gemma3ForCausalLM", config=config
        ).to(torch_device)

        input_len = 10
        input_ids = torch.tensor(
            [
                [42300, 241087, 255445, 81315, 193760, 184471, 67719, 98191, 210651, 124725],
                [102294, 205314, 226646, 62020, 60245, 68025, 251839, 114053, 4695, 175511],
            ],
            device=torch_device,
        )
        attention_mask = torch.ones_like(input_ids).to(torch_device)
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
                use_cache=True,
                cache_implementation="hybrid",
            )
            # 2 generations are needed to trigger https://github.com/huggingface/transformers/issues/39711
            # Since it requires model._cache to have been previously initialized
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5,
                do_sample=False,
                use_cache=True,
                cache_implementation="hybrid",
            )
        generated_sequences = output[:, input_len:].cpu()
        EXPECTED_OUTPUT = torch.tensor([[90109, 90109, 90109, 83191, 83191], [246901, 69832, 69832, 69832, 62288]])
        torch.testing.assert_close(generated_sequences, EXPECTED_OUTPUT)


class Gemma3Vision2TextModelTester:
    def __init__(
        self,
        parent,
        mm_tokens_per_image=2,
        image_token_index=4,
        boi_token_index=5,
        eoi_token_index=6,
        seq_length=25,
        is_training=True,
        vision_config={
            "use_labels": True,
            "image_size": 20,
            "patch_size": 5,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 32,
            "num_key_value_heads": 1,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
        use_cache=False,
    ):
        self.parent = parent
        # `image_token_index` is set to 0 to pass "resize_embeddings" test, do not modify
        self.mm_tokens_per_image = mm_tokens_per_image
        self.image_token_index = image_token_index
        self.boi_token_index = boi_token_index
        self.eoi_token_index = eoi_token_index
        self.llm_tester = Gemma3ModelTester(self.parent)
        self.text_config = self.llm_tester.get_config()
        self.vision_config = vision_config
        self.seq_length = seq_length
        self.pad_token_id = self.text_config.pad_token_id

        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size
        self.num_attention_heads = self.text_config.num_attention_heads
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = vision_config["num_channels"]
        self.image_size = vision_config["image_size"]
        self.encoder_seq_length = seq_length
        self.use_cache = use_cache

    def get_config(self):
        return Gemma3Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_index=self.image_token_index,
            boi_token_index=self.boi_token_index,
            eoi_token_index=self.eoi_token_index,
            mm_tokens_per_image=self.mm_tokens_per_image,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(self.pad_token_id).to(torch_device)

        # set the 3 first tokens to be image, and ensure that no other tokens are image tokens
        # do not change this unless you modified image size or patch size
        input_ids[input_ids == config.image_token_index] = self.pad_token_id
        input_ids[:, :1] = config.image_token_index

        token_type_ids = torch.zeros_like(input_ids)
        token_type_ids[input_ids == config.image_token_index] = 1

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        return config, inputs_dict


@require_torch
class Gemma3Vision2TextModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            Gemma3Model,
            Gemma3ForConditionalGeneration,
            Gemma3ForSequenceClassification,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (Gemma3ForConditionalGeneration,) if is_torch_available() else ()
    test_headmasking = False
    test_pruning = False
    test_missing_keys = False
    _is_stateful = True
    model_split_percents = [0.5, 0.6]

    # MP works but offload doesn't work when the SigLIP MultiheadAttention is offloaded
    # TODO: One potential solution would be to add to set preload_module_classes = ["SiglipMultiheadAttentionPoolingHead"]
    # in the dispatch_model function
    test_cpu_offload = False
    test_disk_offload_safetensors = False
    test_disk_offload_bin = False

    def setUp(self):
        self.model_tester = Gemma3Vision2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Gemma3Config, hidden_size=37)

    def test_bidirectional_image_attention(self):
        """
        Tests that each image can attend to itself bidirectionally. However an image
        cannot attend to future images, even within the same batch.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config._attn_implementation = "eager"
        model = Gemma3Model(config).to(torch_device)

        # First let's pass inputs without change which is one image per text and manipulate
        # `token_type_ids` to make sure bidirectional mask is applied where it has to be
        inputs_dict["token_type_ids"] = torch.zeros_like(inputs_dict["token_type_ids"])
        inputs_dict["token_type_ids"][:, :4] = 1  # unmask first 4 tokens
        with torch.no_grad():
            out = model(**inputs_dict, output_attentions=True)
            # We expect a non-causal mask on first 4 tokens, thus no zeros
            for attention in out.attentions:
                self.assertTrue((attention[..., :4, :4] != 0).all().item())

        # Now when removing `token_type_ids`, we will get simple causal mask
        inputs_dict["token_type_ids"][:, :4] = 0  # mask back first 4 tokens
        with torch.no_grad():
            out = model(**inputs_dict, output_attentions=True)
            # We expect a causal mask on first 4 tokens, thus no zeros
            for attention in out.attentions:
                self.assertFalse((attention[..., :4, :4] != 0).all().item())

        # Let's add two "images" per text, first one spanning 4 tokens and last one 3 tokens
        inputs_dict["token_type_ids"][:, :4] = 1
        inputs_dict["token_type_ids"][:, 7:10] = 1
        with torch.no_grad():
            out = model(**inputs_dict, output_attentions=True)
            for attention in out.attentions:
                self.assertTrue((attention[..., :4, :4] != 0).all().item())
                self.assertTrue((attention[..., 7:10, 7:10] != 0).all().item())

                # We expect a non-causal mask only within same image and no looking ahead to the future
                self.assertTrue((attention[..., :4, 7:10] == 0).all().item())

    @unittest.skip(reason="SiglipVisionModel (vision backbone) does not support standalone training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="SiglipVisionModel (vision backbone) does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="SiglipVisionModel (vision backbone) does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(
        reason="HybridCache can't be gathered because it is not iterable. Adding a simple iter and dumping `distributed_iterator`"
        " as in Dynamic Cache doesn't work. NOTE: @gante all cache objects would need better compatibility with multi gpu setting"
    )
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip("Failing because of unique cache (HybridCache)")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    @parameterized.expand([("random",), ("same",)])
    @pytest.mark.generate
    @unittest.skip("Gemma3 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("Gemma3 has HybridCache which is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @pytest.mark.generate
    @unittest.skip("Gemma3 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("Gemma3 has HybridCache which is not compatible with dola decoding")
    def test_dola_decoding_sample(self):
        pass

    @unittest.skip("Gemma3 has HybridCache and doesn't support continue from past kv")
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip("Gemma3 has HybridCache and doesn't support low_memory generation")
    def test_beam_search_low_memory(self):
        pass

    @unittest.skip("Gemma3 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate(self):
        pass

    @unittest.skip("Gemma3 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("Gemma3 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip("Gemma3 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip("Gemma3 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip("Gemma3 has HybridCache which auto-compiles. Compile and FA2 don't work together.")
    def test_eager_matches_fa2_generate(self):
        pass

    @unittest.skip(
        reason="Siglip (vision backbone) uses the same initialization scheme as the Flax original implementation"
    )
    def test_initialization(self):
        pass

    @unittest.skip("Loading nested configs with overwritten `kwargs` isn't supported yet, FIXME @raushan.")
    def test_load_with_mismatched_shapes(self):
        pass

    @unittest.skip("Loading nested configs with overwritten `kwargs` isn't supported yet, FIXME @raushan.")
    def test_mismatched_shapes_have_properly_initialized_weights(self):
        pass

    def test_automodelforcausallm(self):
        """
        Regression test for #36741/#36917 -- make sure `AutoModelForCausalLM` works with a Gemma3 config, i.e. that
        `AutoModelForCausalLM.from_pretrained` pulls the text config before loading the model
        """
        config = self.model_tester.get_config()
        model = Gemma3ForConditionalGeneration(config)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            for_causal_lm = AutoModelForCausalLM.from_pretrained(tmp_dir)
            self.assertIsInstance(for_causal_lm, Gemma3ForConditionalGeneration)


@slow
@require_torch_accelerator
@require_read_token
class Gemma3IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = Gemma3Processor.from_pretrained("google/gemma-3-4b-it", padding_side="left")

        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        self.messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": url},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @require_deterministic_for_xpu
    def test_model_4b_bf16(self):
        model_id = "google/gemma-3-4b-it"

        model = Gemma3ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(torch_device)

        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nCertainly! \n\nThe image shows a brown and white cow standing on a sandy beach with turquoise water in the background. It looks like a lovely,'],
                ("cuda", 8): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nCertainly! \n\nThe image shows a brown cow standing on a sandy beach with clear turquoise water and a blue sky in the background. It looks like'],
                ("rocm", (9, 5)): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nCertainly! \n\nThe image shows a brown and white cow standing on a sandy beach with turquoise water and a distant coastline in the background. It looks'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_torch_large_accelerator
    @require_deterministic_for_xpu
    def test_model_4b_batch(self):
        model_id = "google/gemma-3-4b-it"

        model = Gemma3ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(torch_device)

        messages_2 = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png",
                    },
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": "Are these images identical?"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            [self.messages, messages_2],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3):
                    [
                        'user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nCertainly! \n\nThe image shows a brown and white cow standing on a sandy beach next to a turquoise ocean. It looks like a very sunny and',
                        'user\nYou are a helpful assistant.\n\n\n\n\n\n\n\n\n\nAre these images identical?\nmodel\nNo, these images are not identical. They depict very different scenes:\n\n*   **Image 1** shows a cow standing on a beach.',
                    ],
                ("cuda", 8):
                    [
                        'user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nCertainly! \n\nThe image shows a brown cow standing on a sandy beach with clear blue water and a blue sky in the background. It looks like',
                        "user\nYou are a helpful assistant.\n\n\n\n\n\n\n\n\n\nAre these images identical?\nmodel\nNo, these images are not identical. \n\nHere's a breakdown of the differences:\n\n*   **Image 1:** Shows a brown"
                    ],
                ("rocm", (9, 5)):
                    [
                        'user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nCertainly! \n\nThe image shows a brown and white cow standing on a sandy beach next to a turquoise ocean. There are some clouds in the blue',
                        'user\nYou are a helpful assistant.\n\n\n\n\n\n\n\n\n\nAre these images identical?\nmodel\nNo, these images are not identical. They depict very different scenes. \n\n*   **Image 1** shows a cow standing on a beach',
                    ],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_torch_large_accelerator
    def test_model_4b_crops(self):
        model_id = "google/gemma-3-4b-it"

        model = Gemma3ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(torch_device)

        crop_config = {
            "images_kwargs": {
                "do_pan_and_scan": True,
                "pan_and_scan_max_num_crops": 448,
                "pan_and_scan_min_crop_size": 32,
                "pan_and_scan_min_ratio_to_activate": 0.3,
            }
        }

        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            **crop_config,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_NUM_IMAGES = 3  # one for the origin image and two crops of images
        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3): ['user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown cow standing on a sandy beach next to a turquoise ocean. There are clouds in the blue sky above.'],
                ("cuda", 7): [],
                ("cuda", 8): ["user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown cow standing on a sandy beach next to a turquoise ocean. There's a bright blue sky with some white clouds in the"],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(len(inputs["pixel_values"]), EXPECTED_NUM_IMAGES)
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_torch_large_accelerator
    @require_deterministic_for_xpu
    def test_model_4b_batch_crops(self):
        model_id = "google/gemma-3-4b-it"

        model = Gemma3ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(torch_device)
        crop_config = {
            "images_kwargs": {
                "do_pan_and_scan": True,
                "pan_and_scan_max_num_crops": 448,
                "pan_and_scan_min_crop_size": 32,
                "pan_and_scan_min_ratio_to_activate": 0.3,
            }
        }
        messages_2 = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png",
                    },
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": "Are these images identical?"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            [self.messages, messages_2],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
            **crop_config,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)
        EXPECTED_NUM_IMAGES = 9  # 3 * (one for the origin image and two crops of images) = 9
        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3): [
                    'user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown cow standing on a sandy beach next to a turquoise ocean. There are clouds in the blue sky above.',
                    'user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nAre these images identical?\nmodel\nNo, the images are not identical. \n\nThe first image shows a cow on a beach, while the second image shows a street scene with a',
                ],
                ("cuda", 7): [],
                ("cuda", 8): [
                    "user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown cow standing on a sandy beach next to a turquoise ocean. There's a bright blue sky with some white clouds in the",
                    'user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nAre these images identical?\nmodel\nNo, the images are not identical. \n\nThe first image shows a cow on a beach, while the second image shows a street scene with a'
                ],
                ("rocm", (9, 5)) : [
                    'user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown cow standing on a sandy beach next to a turquoise ocean. There are clouds in the blue sky above.',
                    'user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nAre these images identical?\nmodel\nNo, the images are not identical. \n\nThe first image shows a cow on a beach, while the second image shows a street scene with a',
                ],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(len(inputs["pixel_values"]), EXPECTED_NUM_IMAGES)
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_torch_large_accelerator
    def test_model_4b_multiimage(self):
        model_id = "google/gemma-3-4b-it"

        model = Gemma3ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(torch_device)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": "What do you see here?"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)
        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3): ["user\nYou are a helpful assistant.\n\n\n\n\n\nWhat do you see here?\nmodel\nOkay, let's break down what I see in this image!\n\nHere's a description of the scene:\n\n*   **Chinese Arch"],
                ("cuda", 7): [],
                ("cuda", 8): ["user\nYou are a helpful assistant.\n\n\n\n\n\nWhat do you see here?\nmodel\nOkay, let's break down what I see in this image:\n\n**Overall Scene:**\n\nIt looks like a street scene in a vibrant,"],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_deterministic_for_xpu
    def test_model_1b_text_only(self):
        model_id = "google/gemma-3-1b-it"

        model = Gemma3ForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        inputs = tokenizer("Write a poem about Machine Learning.", return_tensors="pt").to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3): ['Write a poem about Machine Learning.\n\n---\n\nThe data flows, a river deep,\nWith patterns hidden, secrets sleep.\nA neural net, a watchful eye,\nLearning'],
                ("cuda", 7): ['Write a poem about Machine Learning.\n\n---\n\nThe data flows, a silent stream,\nInto the neural net, a waking dream.\nAlgorithms hum, a coded grace,\n'],
                ("cuda", 8): ['Write a poem about Machine Learning.\n\n---\n\nThe data flows, a silent stream,\nInto the neural net, a waking dream.\nAlgorithms hum, a coded grace,\n'],
                ("rocm", (9, 5)): ['Write a poem about Machine Learning.\n\n---\n\nThe data flows, a river deep,\nWith patterns hidden, secrets sleep.\nA neural net, a watchful eye,\nLearning'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    # TODO: raushan FA2 generates gibberish for no reason, check later
    @require_flash_attn
    @require_torch_large_accelerator
    @pytest.mark.flash_attn_test
    def test_model_4b_flash_attn(self):
        model_id = "google/gemma-3-4b-it"

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).to(torch_device)

        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown and white cow standing on a sandy beach with turquoise water and a distant island in the background. It looks like a sunny day'],
                ("cuda", 7): [],
                ("cuda", 8): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown and white cow standing on a sandy beach with turquoise water and a distant island in the background. It looks like a sunny day'],
                ("rocm", (9, 5)): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown and white cow standing on a sandy beach with a turquoise ocean and a distant island in the background. It looks like a sunny'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    @parameterized.expand([("flash_attention_2",), ("sdpa",), ("eager",)])
    def test_generation_beyond_sliding_window(self, attn_implementation: str):
        """Test that we can correctly generate beyond the sliding window. This is non trivial as
        we need to correctly slice the attention mask in all cases (because we use a HybridCache).
        Outputs for every attention functions should be coherent and identical.
        """
        model_id = "google/gemma-3-1b-it"

        if attn_implementation == "flash_attention_2" and not is_flash_attn_2_available():
            self.skipTest("FlashAttention2 is required for this test.")

        input_text = [
            "This is a nice place. " * 800 + "I really enjoy the scenery,",  # This is larger than 4096 tokens
            "A list of colors: red, blue",  # This will almost all be padding tokens
        ]
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding="left")
        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(torch_device)

        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation=attn_implementation, torch_dtype=torch.float16
        ).to(torch_device)

        # Make sure prefill is larger than sliding window
        input_size = inputs.input_ids.shape[-1]
        self.assertTrue(input_size > model.config.sliding_window)

        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)[:, input_size:]
        output_text = tokenizer.batch_decode(out)

        EXPECTED_COMPLETIONS = [" and I'm going to take a walk.\n\nI really enjoy the scenery, and I'", ", green, yellow, orange, purple, brown, black, white, gray.\n\nI'"]  # fmt: skip
        self.assertEqual(output_text, EXPECTED_COMPLETIONS)

    def test_export_text_only_with_hybrid_cache(self):
        if not is_torch_greater_or_equal("2.6.0"):
            self.skipTest(reason="This test requires torch >= 2.6 to run.")

        from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM

        model_id = "google/gemma-3-1b-it"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        self.assertEqual(model.config.cache_implementation, "hybrid")

        # Export + HybridCache
        model.eval()
        exportable_module = TorchExportableModuleForDecoderOnlyLM(model)
        exported_program = exportable_module.export()
        logging.info(f"\nExported program: {exported_program}")

        # Test generation with the exported model
        prompt = "What is the capital of France?"
        max_new_tokens_to_generate = 20
        # Generate text with the exported model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        export_generated_text = TorchExportableModuleForDecoderOnlyLM.generate(
            exported_program, tokenizer, prompt, max_new_tokens=max_new_tokens_to_generate
        )
        logging.info(f"\nExport generated texts: '{export_generated_text}'")

        input_text = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            eager_outputs = model.generate(
                **input_text,
                max_new_tokens=max_new_tokens_to_generate,
                do_sample=False,  # Use greedy decoding to match the exported model
            )

        eager_generated_text = tokenizer.decode(eager_outputs[0], skip_special_tokens=True)
        logging.info(f"\nEager generated texts: '{eager_generated_text}'")

        self.assertEqual(export_generated_text, eager_generated_text)
