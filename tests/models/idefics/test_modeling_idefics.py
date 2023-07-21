# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch Idefics model. """

import unittest

from transformers import IdeficsConfig, is_torch_available, is_vision_available
from transformers.testing_utils import TestCasePlus, require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        IdeficsForCausalLM,
        IdeficsModel,
    )
    from transformers.models.idefics.modeling_idefics import IDEFICS_PRETRAINED_MODEL_ARCHIVE_LIST

if is_vision_available():
    from PIL import Image

    from transformers import IdeficsProcessor


class IdeficsModelTester:
    def __init__(
        self,
        parent,
        batch_size=1,
        seq_length=7,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        scope=None,
        modality_type_vocab_size=2,
        add_multiple_images=False,
        num_images=-1,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.scope = scope
        self.modality_type_vocab_size = modality_type_vocab_size
        self.add_multiple_images = add_multiple_images
        self.num_images = num_images
        # we set the expected sequence length (which is used in several tests)
        # this is equal to the seq length of the text tokens + number of image patches + 1 for the CLS token
        self.expected_seq_len = self.seq_length + (self.image_size // self.patch_size) ** 2 + 1

    def prepare_config_and_inputs(self):
        self.seq_length = 42

        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        num_images = 2 if self.add_multiple_images else 1
        pixel_values = floats_tensor(
            [self.batch_size, num_images, self.num_channels, self.image_size, self.image_size]
        )
        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        image_attention_mask = random_attention_mask([self.batch_size, self.seq_length, num_images])

        # inputs["input_ids"].shape=torch.Size([1, 41])
        # inputs["attention_mask"].shape=torch.Size([1, 41])
        # inputs["pixel_values"].shape=torch.Size([1, 2, 3, 30, 30])
        # inputs["image_attention_mask"].shape=torch.Size([1, 41, 2])

        # input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        # input_mask = random_attention_mask([self.batch_size, self.seq_length])
        # pixel_values =
        # image_attention_mask =

        config = self.get_config()

        return (config, input_ids, input_mask, pixel_values, image_attention_mask)

    def get_config(self):
        return IdeficsConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            num_labels=self.num_labels,
            modality_type_vocab_size=self.modality_type_vocab_size,
            num_images=self.num_images,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        pixel_values,
        token_labels,
    ):
        model = IdeficsModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, pixel_values=pixel_values)
        result = model(input_ids, token_type_ids=token_type_ids, pixel_values=pixel_values)
        result = model(input_ids, pixel_values=pixel_values)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.expected_seq_len, self.hidden_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            pixel_values,
            image_attention_mask,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "pixel_values": pixel_values,
            "image_attention_mask": image_attention_mask,
        }
        return config, inputs_dict

    def prepare_pixel_values(self):
        return floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])


@require_torch
class IdeficsModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            IdeficsModel,
            IdeficsForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    test_pruning = False
    test_headmasking = False
    test_torchscript = False

    # IdeficsForMaskedLM, IdeficsForQuestionAnswering and IdeficsForImagesAndTextClassification require special treatment
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class.__name__ == "IdeficsForQuestionAnswering":
                inputs_dict["labels"] = torch.zeros(
                    self.model_tester.batch_size, self.model_tester.num_labels, device=torch_device
                )
            elif model_class.__name__ in ["IdeficsForMaskedLM", "IdeficsForTokenClassification"]:
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )
            elif model_class.__name__ == "IdeficsForImagesAndTextClassification":
                inputs_dict["labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )

        return inputs_dict

    def setUp(self):
        self.model_tester = IdeficsModelTester(self)
        self.config_tester = ConfigTester(self, config_class=IdeficsConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_training(self):
        if not self.model_tester.is_training:
            return

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            for k, v in inputs.items():
                print(k, v.shape)
            loss = model(**inputs).loss
            loss.backward()

    def test_training_gradient_checkpointing(self):
        if not self.model_tester.is_training:
            return

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.use_cache = False
            config.return_dict = True

            model = model_class(config)
            model.to(torch_device)
            model.gradient_checkpointing_enable()
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    @unittest.skip(
        reason="""VilT samples image tokens from a multinomial distribution, resulting in not deterministic
                            hidden states"""
    )
    def test_save_load(self):
        pass

    @unittest.skip(
        reason="""VilT samples image tokens from a multinomial distribution, resulting in not deterministic
                            hidden states"""
    )
    def test_determinism(self):
        pass

    @unittest.skip(
        reason="""VilT samples image tokens from a multinomial distribution, resulting in not deterministic
                            hidden states"""
    )
    def test_model_outputs_equivalence(self):
        pass

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "expected_seq_len", None)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            if model_class.__name__ == "IdeficsForImagesAndTextClassification":
                # attentions are a list of length num_images
                # each element contains the attentions of a particular image index
                self.assertEqual(len(attentions), self.model_tester.num_images)
                self.assertEqual(len(attentions[0]), self.model_tester.num_hidden_layers)
            else:
                self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            if model_class.__name__ == "IdeficsForImagesAndTextClassification":
                # attentions are a list of length num_images
                # each element contains the attentions of a particular image index
                self.assertEqual(len(attentions), self.model_tester.num_images)
                self.assertEqual(len(attentions[0]), self.model_tester.num_hidden_layers)
            else:
                self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            if model_class.__name__ == "IdeficsForImagesAndTextClassification":
                self.assertListEqual(
                    list(attentions[0][0].shape[-3:]),
                    [self.model_tester.num_attention_heads, seq_len, seq_len],
                )
            else:
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, seq_len, seq_len],
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

            if model_class.__name__ == "IdeficsForImagesAndTextClassification":
                self.assertEqual(len(self_attentions), self.model_tester.num_images)
                self.assertEqual(len(self_attentions[0]), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(self_attentions[0][0].shape[-3:]),
                    [self.model_tester.num_attention_heads, seq_len, seq_len],
                )
            else:
                self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, seq_len, seq_len],
                )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            if model_class.__name__ == "IdeficsForImagesAndTextClassification":
                # hidden_states are a list of length num_images
                # each element contains the hidden states of a particular image index
                self.assertEqual(len(hidden_states), self.model_tester.num_images)
                self.assertEqual(len(hidden_states[0]), expected_num_layers)
            else:
                self.assertEqual(len(hidden_states), expected_num_layers)

            seq_length = self.model_tester.expected_seq_len

            if model_class.__name__ == "IdeficsForImagesAndTextClassification":
                self.assertListEqual(
                    list(hidden_states[0][0].shape[-2:]),
                    [seq_length, self.model_tester.hidden_size],
                )
            else:
                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            print("Model class:", model_class)
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs[0]

        # Encoder-/Decoder-only models
        hidden_states = outputs.hidden_states[0]
        attentions = outputs.attentions[0]

        if model_class.__name__ == "IdeficsForImagesAndTextClassification":
            # hidden_states are a list of length num_images
            # each element contains the hidden states of a particular image index
            hidden_states[0].retain_grad()
            attentions[0].retain_grad()
        else:
            hidden_states.retain_grad()
            attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        if model_class.__name__ == "IdeficsForImagesAndTextClassification":
            # hidden_states are a list of length num_images
            # each element contains the hidden states of a particular image index
            self.assertIsNotNone(hidden_states[0].grad)
            self.assertIsNotNone(attentions[0].grad)
        else:
            self.assertIsNotNone(hidden_states.grad)
            self.assertIsNotNone(attentions.grad)

    @slow
    def test_model_from_pretrained(self):
        for model_name in IDEFICS_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = IdeficsModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class IdeficsForCausalLMTest(IdeficsModelTest, unittest.TestCase):
    all_model_classes = (IdeficsForCausalLM,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = IdeficsModelTester(
            self, modality_type_vocab_size=3, add_multiple_images=True, num_images=2
        )
        self.config_tester = ConfigTester(self, config_class=IdeficsConfig, hidden_size=37)

    @unittest.skip("We only test the model that takes in multiple images")
    def test_model(self):
        pass

    @unittest.skip("We only test the model that takes in multiple images")
    def test_for_token_classification(self):
        pass


@require_torch
@require_vision
class IdeficsModelIntegrationTest(TestCasePlus):
    @cached_property
    def default_processor(self):
        return IdeficsProcessor.from_pretrained("HuggingFaceM4/idefics-9b") if is_vision_available() else None

    @slow
    def test_inference_natural_language_visual_reasoning(self):
        cat_image_path = self.tests_dir / "fixtures/tests_samples/COCO/000000039769.png"
        cats_image_obj = Image.open(cat_image_path)  # 2 cats
        dogs_image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_nlvr2/raw/main/image1.jpeg"

        prompts = [
            [
                "User:",
                dogs_image_url,
                "Describe this image.\nAssistant: An image of two dogs.\n",
                "User:",
                cats_image_obj,
                "Describe this image.\nAssistant:",
            ],
            [
                "User:",
                cats_image_obj,
                "Describe this image.\nAssistant: An image of two kittens.\n",
                "User:",
                dogs_image_url,
                "Describe this image.\nAssistant:",
            ],
        ]

        model = IdeficsForCausalLM.from_pretrained("HuggingFaceM4/idefics-9b").to(torch_device)
        processor = self.default_processor
        inputs = processor(prompts, eval_mode=True, device=torch_device, return_tensors="pt")
        generated_ids = model.generate(**inputs, max_length=100)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # keep for debugging
        for i, t in enumerate(generated_text):
            t = bytes(t, "utf-8").decode("unicode_escape")
            print(f"{i}:\n{t}\n")

        self.assertIn("image of two cats", generated_text[0])
        self.assertIn("image of two dogs", generated_text[1])
