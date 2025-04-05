# coding=utf-8
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
"""Testing suite for the PyTorch Llama4 model."""

import unittest

from transformers import Llama4Config, is_torch_available
from transformers.testing_utils import (
    require_torch,
    require_torch_large_gpu,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        Llama4ForCausalLM,
        Llama4ForConditionalGeneration,
        Llama4Processor,
    )


from parameterized import parameterized

from ...test_modeling_common import floats_tensor


@require_torch
class Llama4ForVisionText2TextModelTester:
    config_class = Llama4Config
    if is_torch_available():
        model_class = Llama4ForConditionalGeneration

    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_index=0,
        projector_hidden_act="gelu",
        seq_length=7,
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
        text_config={
            "seq_length": 7,
            "is_training": True,
            "use_input_mask": True,
            "use_token_type_ids": False,
            "use_labels": True,
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 40,
            "intermediate_size": 128,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 1,
            "torch_dtype": torch.bfloat16,
        },
        is_training=True,
        vision_config={
            "vision_output_dim": 32,
            "hidden_size": 32,
            "model_type": "llama4_vision_model",
            "image_size": 8,
            "patch_size": 2,
            "num_channels": 3,
            "is_training": True,
            "projection_dim": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
            "pixel_shuffle_ratio": 0.5,
            "projector_input_dim": 32,
            "projector_output_dim": 32,
            "torch_dtype": torch.bfloat16,
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.text_config = text_config
        self.vision_config = vision_config
        self.pad_token_id = text_config["pad_token_id"]

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training
        self.batch_size = 3
        self.num_channels = 3
        self.image_size = 336
        self.num_image_tokens = (self.vision_config["image_size"] // self.vision_config["patch_size"]) ** 2
        self.seq_length = seq_length + self.num_image_tokens
        self.encoder_seq_length = self.seq_length

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ],
        )
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        return Llama4Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            torch_dtype=torch.bfloat16,
            image_token_index=self.image_token_index,
        )

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(1).to(device=torch_device)
        input_ids[input_ids == config.image_token_index] = self.pad_token_id
        input_ids[:, : self.num_image_tokens] = config.image_token_index
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class Llama4ForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `Llama4ForConditionalGeneration`.
    """

    all_model_classes = (Llama4ForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"image-text-to-text": Llama4ForConditionalGeneration, "text-generation": Llama4ForCausalLM}
        if is_torch_available()
        else {}
    )
    test_pruning = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = Llama4ForVisionText2TextModelTester(self)
        common_properties = ["image_token_index"]
        self.config_tester = ConfigTester(
            self, config_class=Llama4Config, has_text_modality=False, common_properties=common_properties
        )

    def test_config(self):
        # overwritten from `tests/test_configuration_common.py::ConfigTester` after #36077
        # TODO: avoid overwritten once there is a better fix for #36077
        def check_config_can_be_init_without_params():
            config = self.config_tester.config_class()
            self.config_tester.parent.assertIsNotNone(config)

        self.config_tester.check_config_can_be_init_without_params = check_config_can_be_init_without_params
        self.config_tester.run_common_tests()

    # overwrite inputs_embeds tests because we need to delete "pixel values" for LVLMs
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)

            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values"]

            wte = model.get_input_embeddings()
            inputs["inputs_embeds"] = wte(input_ids)

            with torch.no_grad():
                model(**inputs)

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs through an error with explicit message saying what is wrong
        when number of images doesn't match number of image tokens in the text.
        Also we need to test multi-image cases when one prompt has multiple image tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            _ = model(**input_dict)  # successful forward with no modifications

            # remove one image but leave the image token in text
            input_dict["pixel_values"] = input_dict["pixel_values"][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**input_dict)

            # simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            input_ids = input_dict["input_ids"][:1]
            pixel_values = input_dict["pixel_values"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(input_ids=input_ids, pixel_values=pixel_values)

            # two images and two image tokens don't raise an error
            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            _ = model(input_ids=input_ids, pixel_values=pixel_values)

    @parameterized.expand(
        [
            (-1,),
        ],
    )
    def test_vision_feature_layers(self, vision_feature_layer):
        """
        Test that we can use either one vision feature layer, or a list of
        vision feature layers.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.vision_feature_layer = vision_feature_layer

        num_feature_layers = 1 if isinstance(vision_feature_layer, int) else len(vision_feature_layer)
        hidden_size = config.vision_config.hidden_size
        expected_features = hidden_size * num_feature_layers

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device, dtype=torch.bfloat16)
            # We should have the right number of input features,
            # and should be able to run a forward pass without exploding
            assert model.multi_modal_projector.linear_1.in_features == expected_features
            model(**input_dict)

    @unittest.skip(
        reason="This architecture seems to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecture seems to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seems to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip("FlashAttention only support fp16 and bf16 data type")
    def test_flash_attn_2_fp32_ln(self):
        pass

    @unittest.skip(
        "VLMs need lots of steps to prepare images/mask correctly to get pad-free inputs. Can be tested as part of LLM test"
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(reason="Llama4 uses GQA on all models so the KV cache is a non standard format")
    def test_past_key_values_format(self):
        pass

    @unittest.skip("Failing because of unique cache (HybridCache)")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip("Llama4 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("Llama4 has HybridCache which is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("Llama4 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("Llama4 has HybridCache which is not compatible with dola decoding")
    def test_dola_decoding_sample(self):
        pass

    @unittest.skip("Llama4 has HybridCache and doesn't support continue from past kv")
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip("Llama4 has HybridCache and doesn't support low_memory generation")
    def test_beam_search_low_memory(self):
        pass

    @unittest.skip("Llama4 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate(self):
        pass

    @unittest.skip("Llama4 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("Llama4 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip(
        "Llama4 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support."
    )
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip(
        "Llama4 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support."
    )
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip(
        "Llama4 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support."
    )
    def test_generate_continue_from_inputs_embeds(self):
        pass

    @unittest.skip("Llama4's eager attn/sdpa attn outputs are expected to be different")
    def test_sdpa_equivalence(self):
        pass

    @unittest.skip("Llama4 uses GQA so it is not compatible with the standard cache format")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("Llama4 uses GQA so it is not compatible with the standard cache format")
    def test_generate_compilation_all_outputs(self):
        pass

    @unittest.skip("Llama4 uses GQA so it is not compatible with the standard cache format")
    def test_generate_compile_model_forward(self):
        pass

    @unittest.skip("Llama4 uses GQA so it is not compatible with the standard cache format")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

@slow
@require_torch_large_gpu
class Llama4IntegrationTest(unittest.TestCase):
    model_id = "ll-re/Llama-4-17B-Omni-Instruct"
    # This variable is used to determine which CUDA device are we using for our runners (A10 or T4)
    # Depending on the hardware we get different logits / generations
    cuda_compute_capability_major_version = None

    @classmethod
    def setUpClass(cls):
        if is_torch_available() and torch.cuda.is_available():
            # 8 is for A100 / A10 and 7 for T4
            cls.cuda_compute_capability_major_version = torch.cuda.get_device_capability()[0]
        cls.model = Llama4ForConditionalGeneration.from_pretrained(
            "ll-re/Llama-4-17B-Omni-Instruct", device_map="auto", torch_dtype=torch.float32
        )

    def setUp(self):
        self.processor = Llama4Processor.from_pretrained("ll-re/Llama-4-17B-Omni-Instruct", padding_side="left")

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

    def test_model_17b_16e_fp16(self):
        EXPECTED_TEXT = [
            "The capital of France is Paris, which is located in the north-central part of the country. Paris is known for its iconic landmarks such as the",
            "Roses are red, violets are blue, and this poem is about you. Roses are red, violets are blue, and I love",
        ]

        messages = [
            {"role": "user", "content": "Who are you?"},
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(torch_device)

        output = self.model.generate(**inputs, max_new_tokens=100)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        print(output_text)
        self.assertEqual(output_text, EXPECTED_TEXT)

    def test_model_17b_16e_batch(self):
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

        output = self.model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = [
            'user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nCertainly! \n\nThe image shows a brown cow standing on a sandy beach with clear turquoise water and a blue sky in the background. It looks like',
            "user\nYou are a helpful assistant.\n\n\n\n\n\n\n\n\n\nAre these images identical?\nmodel\nNo, these images are not identical. \n\nHere's a breakdown of the differences:\n\n*   **Image 1:** Shows a cow"
        ]  # fmt: skip
        self.assertEqual(output_text, EXPECTED_TEXTS)
