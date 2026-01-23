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
"""Testing suite for the PyTorch Phi-3 model."""

import tempfile
import unittest

from transformers import (
    AutoProcessor,
    Phi3VConfig,
    Phi3VForConditionalGeneration,
    Phi3VModel,
    is_torch_available,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch


class Phi3VModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_id=0,
        projector_hidden_act="gelu",
        seq_length=25,
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
        text_config={
            "model_type": "llama",
            "seq_length": 7,
            "is_training": True,
            "use_input_mask": True,
            "use_token_type_ids": False,
            "use_labels": True,
            "vocab_size": 99,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 16,
            "max_position_embeddings": 512,
            "num_labels": 3,
            "pad_token_id": 1,
        },
        is_training=True,
        vision_config={
            "use_labels": True,
            "image_size": 4,
            "patch_size": 2,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 4,
            "projection_dim": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 16,
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.image_token_id = image_token_id
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

        self.batch_size = 2
        self.num_channels = 3
        self.num_crops = 2
        self.seq_length = seq_length
        self.num_image_tokens = 5

    def get_config(self):
        return Phi3VConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            ignore_index=self.ignore_index,
            image_token_id=self.image_token_id,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.num_crops + 1,
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
        input_ids[input_ids == config.image_token_id] = self.pad_token_id
        input_ids[:, : self.num_image_tokens] = config.image_token_id
        attention_mask = input_ids.ne(1).to(torch_device)

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
            "image_sizes": torch.tensor(
                [[self.vision_config["image_size"], self.vision_config["image_size"]]] * self.batch_size
            ),
        }
        return config, inputs_dict


@require_torch
class Phi3VForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `Phi3VForConditionalGeneration`.
    """

    all_model_classes = (
        (
            Phi3VModel,
            Phi3VForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (Phi3VForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"image-to-text": Phi3VForConditionalGeneration, "image-text-to-text": Phi3VForConditionalGeneration}
        if is_torch_available()
        else {}
    )
    test_pruning = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = Phi3VModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Phi3VConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip("Not possible now as processor creates a custom attention mask.")
    def test_assisted_decoding_matches_greedy_search_0_random(self):
        pass

    @unittest.skip("Not possible now as processor creates a custom attention mask.")
    def test_assisted_decoding_matches_greedy_search_1_same(self):
        pass

    @unittest.skip("Not possible now as processor creates a custom attention mask.")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @unittest.skip("Not possible now as processor creates a custom attention mask.")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("Not possible now as processor creates a custom attention mask.")
    def test_apply_chat_template_assistant_mask(self):
        pass

    @unittest.skip("Does not work when  num_return_sequences > 1")
    def test_beam_sample_generate(self):
        pass

    @unittest.skip("Does not work when  num_return_sequences > 1")
    def test_beam_sample_generate_dict_output(self):
        pass

    @unittest.skip("Does not work when  num_return_sequences > 1")
    def test_beam_search_generate(self):
        pass

    @unittest.skip("Does not work when  num_return_sequences > 1")
    def test_beam_search_generate_dict_output(self):
        pass

    @unittest.skip("Does not work when  num_return_sequences > 1")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("Does not work when  num_return_sequences > 1")
    def test_generate_from_inputs_embeds_1_beam_search(self):
        pass

    @unittest.skip(
        reason="This architecture seems to not compute gradients because the model uses intermediate hidden states."
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecture seems to not compute gradients because the model uses intermediate hidden states."
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seems to not compute gradients because the model uses intermediate hidden states."
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    def test_sample_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            if self.has_attentions:
                config._attn_implementation = "eager"  # can't output attentions otherwise

            model = model_class(config).to(torch_device).eval()
            output_generate = self._sample_generate(
                model=model,
                inputs_dict=inputs_dict,
                num_return_sequences=1,  # Set it to 1 to avoid test failures.
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=self.has_attentions,
                return_dict_in_generate=True,
                use_cache=False,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[1] == self.max_new_tokens + 1)
                self.assertIsInstance(output_generate, GenerateEncoderDecoderOutput)
            else:
                self.assertTrue(
                    output_generate.sequences.shape[1] == self.max_new_tokens + inputs_dict["input_ids"].shape[1]
                )
                self.assertIsInstance(output_generate, GenerateDecoderOnlyOutput)

            self._check_generate_outputs(output_generate, model.config, num_return_sequences=1)

    def test_sdpa_can_dispatch_composite_models(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # Load the model with SDPA
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                # Load model with eager attention
                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    attn_implementation="eager",
                )
                model_eager = model_eager.eval().to(torch_device)

            vision_attn = language_attn = "sdpa" if model._supports_sdpa else "eager"
            if hasattr(model_sdpa, "vision_model") and hasattr(model_sdpa, "language_model"):
                self.assertTrue(model_sdpa.vision_model.config._attn_implementation == vision_attn)
                self.assertTrue(model_sdpa.language_model.config._attn_implementation == language_attn)
                self.assertTrue(model_eager.vision_model.config._attn_implementation == "eager")
                self.assertTrue(model_eager.language_model.config._attn_implementation == "eager")

            self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
            self.assertTrue(model_eager.config._attn_implementation == "eager")


class Phi3VIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_id = "yaswanthgali/Phi-3.5-vision-instruct"

    @slow
    def test_model_text_generation(self):
        model = Phi3VForConditionalGeneration.from_pretrained(self.model_id, device_map="auto", dtype=torch.bfloat16)
        model.eval()

        processor = AutoProcessor.from_pretrained(self.model_id)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg",
                    },
                    {"type": "text", "text": "Describe what do you see here and tell me about the history behind it?"},
                ],
            },
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        EXPECTED_DECODED_TEXT = "\n \nDescribe what do you see here and tell me about the history behind it? \n The image shows a constellation in the night sky, which is a group of stars that has"
        text = processor.decode(output[0], skip_special_tokens=True)
        self.assertEqual(
            text,
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_model_text_generation_batched(self):
        model = Phi3VForConditionalGeneration.from_pretrained(self.model_id, device_map="auto", dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(self.model_id)

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": "https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg",
                        },
                        {"type": "text", "text": "Describe what do you see here."},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": "https://www.kxan.com/wp-content/uploads/sites/40/2020/10/ORION.jpg"},
                        {"type": "text", "text": "What constellation is this image showing?"},
                    ],
                },
            ],
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        EXPECTED_TEXT_COMPLETION = [
            "\n \nDescribe what do you see here. \n The image shows a starry night sky with a constellation outlined by white lines connecting white",
            "\n \nWhat constellation is this image showing? \n The image shows the constellation of Orion. \n\n\nInstruction ",
        ]
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text = processor.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    def test_model_text_generation_with_multi_image(self):
        model = Phi3VForConditionalGeneration.from_pretrained(self.model_id, device_map="auto", dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(self.model_id)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "https://www.kxan.com/wp-content/uploads/sites/40/2020/10/ORION.jpg"},
                    {
                        "type": "image",
                        "url": "https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg",
                    },
                    {"type": "text", "text": "What do these two images have in common?"},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        EXPECTED_TEXT_COMPLETION = "\n \n \nWhat do these two images have in common? \n Both of"
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text = processor.decode(output[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
