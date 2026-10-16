# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import copy
import unittest

from transformers import UltraVBertConfig, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch

    from transformers import (
        UltrabertConfig,
        UltraVBertForMaskedLM,
        UltraVBertForSequenceClassification,
        UltraVBertForTokenClassification,
        UltraVBertModel,
    )


class UltraVBertModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_images=2,
        text_config=None,
        is_training=True,
        vision_config=None,
        image_token_id: int = 98,
        pixel_shuffle_factor=2,
        num_labels=3,
        use_labels=True,
        type_sequence_label_size=2,
    ):
        if text_config is None:
            text_config = {
                "vocab_size": 99,
                "pad_token_id": 0,
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 64,
                "prediction_head_size": 16,
                "sliding_window": 2,
                "layer_types": ["sliding_attention", "full_attention"],
                "max_position_embeddings": 512,
                "initializer_range": 0.02,
            }
        if vision_config is None:
            vision_config = {
                "image_size": 16,
                "patch_size": 4,
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 32,
                "dropout": 0.1,
                "attention_dropout": 0.1,
                "initializer_range": 0.02,
            }
        self.parent = parent
        self.batch_size = batch_size
        self.text_config = UltrabertConfig(**text_config)
        self.vision_config = vision_config
        self.num_images = num_images
        self.image_token_id = image_token_id
        self.image_size = vision_config["image_size"]
        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.seq_length = (
            int(((vision_config["image_size"] // vision_config["patch_size"]) ** 2) / (pixel_shuffle_factor**2))
            * self.num_images
        )

        self.vocab_size = text_config["vocab_size"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training
        self.num_labels = num_labels
        self.use_labels = use_labels
        self.type_sequence_label_size = type_sequence_label_size

    def get_config(self):
        config = UltraVBertConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            pixel_shuffle_factor=self.pixel_shuffle_factor,
            vocab_size=self.vocab_size,
            attn_implementation={"text_config": "sdpa"},
        )
        return config

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_images, 3, self.image_size, self.image_size])
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)

        # For simplicity just set the last n tokens to the image token
        n_image_tokens_per_batch = self.seq_length
        input_ids[:, -n_image_tokens_per_batch:] = self.image_token_id
        attention_mask = input_ids.ne(1).to(torch_device)

        sequence_labels = None
        token_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        config = self.get_config()

        # tie text-level args to top-level args for test purposes
        config.pad_token_id = config.text_config.pad_token_id
        config.bos_token_id = config.text_config.bos_token_id
        config.eos_token_id = config.text_config.eos_token_id
        config.tie_word_embeddings = config.text_config.tie_word_embeddings

        return config, input_ids, attention_mask, pixel_values, sequence_labels, token_labels

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values, sequence_labels, token_labels = config_and_inputs

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_model(self, config, input_ids, input_mask, pixel_values, sequence_labels, token_labels):
        model = UltraVBertModel(config=config)
        model.to(torch_device)
        model.eval()

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": input_mask,
        }

        result = model(**inputs_dict)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_masked_lm(
        self, config, input_ids, input_mask, pixel_values, sequence_labels, token_labels
    ):
        model = UltraVBertForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "labels": token_labels,
        }
        result = model(**inputs_dict)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, input_mask, pixel_values, sequence_labels, token_labels
    ):
        config.num_labels = self.num_labels
        model = UltraVBertForSequenceClassification(config)
        model.to(torch_device)
        model.eval()

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "labels": sequence_labels,
        }
        result = model(**inputs_dict)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self, config, input_ids, input_mask, pixel_values, sequence_labels, token_labels
    ):
        config.num_labels = self.num_labels
        model = UltraVBertForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "labels": token_labels,
        }
        result = model(**inputs_dict)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))


@require_torch
class UltraVBertModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            UltraVBertModel,
            UltraVBertForMaskedLM,
            UltraVBertForSequenceClassification,
            UltraVBertForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": UltraVBertModel,
            "fill-mask": UltraVBertForMaskedLM,
            "text-classification": UltraVBertForSequenceClassification,
            "image-classification": UltraVBertForSequenceClassification,
            "token-classification": UltraVBertForTokenClassification,
            "zero-shot": UltraVBertForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    _is_composite = True
    test_mismatched_shapes = False
    skip_test_image_features_output_shape = True  # UltraVBert merges batch_size with num_images in index 0
    model_split_percents = [0.5, 0.8, 0.9]

    def setUp(self):
        self.model_tester = UltraVBertModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=UltraVBertConfig,
            has_text_modality=False,  # Avoid the check for vocab_size, which is now in text_config
            common_properties=None,  # Common properties are now in text_config
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    # We need to override as we need to prepare such that the image token is the last token
    def test_resize_tokens_embeddings(self):
        (original_config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)

            if self.model_tester.is_training is False:
                model.eval()

            model_vocab_size = config.text_config.vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.text_config.vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.text_config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            # Ignore copy
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary - 1 and the image token should be the last token
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 2)
            n_images = self.model_tester.num_images * self.model_tester.seq_length
            model.image_token_id = model_vocab_size - 15 - 1
            inputs_dict["input_ids"][:, -n_images:] = model.image_token_id

            # make sure that decoder_input_ids are resized as well
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True
            for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)

            model_vocab_size = config.text_config.vocab_size
            model.resize_token_embeddings(model_vocab_size + 10, pad_to_multiple_of=1)
            self.assertTrue(model.config.text_config.vocab_size + 10, model_vocab_size)

            model_embed = model.resize_token_embeddings(model_vocab_size, pad_to_multiple_of=64)
            self.assertTrue(model_embed.weight.shape[0] // 64, 0)

            self.assertTrue(model_embed.weight.shape[0], model.config.text_config.vocab_size)
            self.assertTrue(model.config.text_config.vocab_size, model.vocab_size)

            model_embed = model.resize_token_embeddings(model_vocab_size + 13, pad_to_multiple_of=64)
            self.assertTrue(model_embed.weight.shape[0] // 64, 0)

            # Check that resizing a model to a multiple of pad_to_multiple leads to a model of exactly that size
            target_dimension = 128
            model_embed = model.resize_token_embeddings(target_dimension, pad_to_multiple_of=64)
            self.assertTrue(model_embed.weight.shape[0], target_dimension)

            with self.assertRaisesRegex(
                ValueError,
                "Asking to pad the embedding matrix to a multiple of `1.3`, which is not and integer. Please make sure to pass an integer",
            ):
                model.resize_token_embeddings(model_vocab_size, pad_to_multiple_of=1.3)

    # We need to override as we need to prepare such that the image token is the last token
    def test_resize_embeddings_untied(self):
        (original_config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()

        original_config.tie_word_embeddings = False

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config).to(torch_device)
            model.eval()

            # if no output embeddings -> leave test
            if model.get_output_embeddings() is None:
                continue

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_vocab_size = config.text_config.vocab_size
            model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.text_config.vocab_size, model_vocab_size + 10)
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size + 10)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.text_config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size - 15)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size - 15)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary - 1 and the image token should be the last token
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 2)
            n_images = self.model_tester.num_images * self.model_tester.seq_length
            model.image_token_id = model_vocab_size - 15 - 1
            inputs_dict["input_ids"][:, -n_images:] = model.image_token_id

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

    @unittest.skip(reason="Vision head's probe has no gradient.")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Vision head's probe has no gradient.")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Vision head's probe has no gradient.")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass
