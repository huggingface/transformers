# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Blip model."""

import inspect
import tempfile
import unittest

import numpy as np
import requests

from transformers import BlipConfig, BlipTextConfig, BlipVisionConfig
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    require_torch_fp16,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        BlipForConditionalGeneration,
        BlipForImageTextRetrieval,
        BlipForQuestionAnswering,
        BlipModel,
        BlipTextModel,
        BlipVisionModel,
    )


if is_vision_available():
    from PIL import Image

    from transformers import BlipProcessor


class BlipVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=1e-10,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return BlipVisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values):
        model = BlipVisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class BlipVisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Blip does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (BlipVisionModel,) if is_torch_available() else ()

    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = BlipVisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BlipVisionConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Blip does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="This module does not support standalone training")
    def test_training(self):
        pass

    @unittest.skip(reason="This module does not support standalone training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="This module does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="This module does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "Salesforce/blip-vqa-base"
        model = BlipVisionModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


class BlipTextModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        bos_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.scope = scope
        self.bos_token_id = bos_token_id

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        if input_mask is not None:
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return BlipTextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            bos_token_id=self.bos_token_id,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = BlipTextModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids, attention_mask=input_mask)
            result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class BlipTextModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (BlipTextModel,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = BlipTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BlipTextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip
    def test_training(self):
        pass

    @unittest.skip(reason="Blip does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "Salesforce/blip-vqa-base"
        model = BlipTextModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


class BlipModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = BlipTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = BlipVisionModelTester(parent, **vision_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return BlipConfig(
            text_config=self.text_model_tester.get_config().to_dict(),
            vision_config=self.vision_model_tester.get_config().to_dict(),
            projection_dim=64,
        )

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = BlipModel(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_ids, pixel_values, attention_mask)
        self.parent.assertEqual(
            result.logits_per_image.shape, (self.vision_model_tester.batch_size, self.text_model_tester.batch_size)
        )
        self.parent.assertEqual(
            result.logits_per_text.shape, (self.text_model_tester.batch_size, self.vision_model_tester.batch_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "return_loss": True,
        }
        return config, inputs_dict


@require_torch
class BlipModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (BlipModel,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": BlipModel,
            "image-to-text": BlipForConditionalGeneration,
            "image-text-to-text": BlipForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )

    test_resize_embeddings = True
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = BlipModelTester(self)
        common_properties = ["logit_scale_init_value", "image_text_hidden_size", "projection_dim", "label_smoothing"]
        self.config_tester = ConfigTester(
            self, config_class=BlipConfig, has_text_modality=False, common_properties=common_properties
        )

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="BlipModel does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    def test_load_vision_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save BlipConfig and check if we can load BlipVisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = BlipVisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save BlipConfig and check if we can load BlipTextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = BlipTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        model_name = "Salesforce/blip-vqa-base"
        model = BlipModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def test_get_image_features(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        keys_to_pop = ["input_ids", "attention_mask", "return_loss"]

        for key in keys_to_pop:
            inputs_dict.pop(key)

        model = BlipModel(config).to(torch_device)
        model.eval()
        image_features = model.get_image_features(**inputs_dict)
        self.assertEqual(
            image_features.pooler_output.shape,
            (
                self.model_tester.batch_size,
                model.projection_dim,
            ),
        )

    def test_get_text_features(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        keys_to_pop = ["pixel_values", "return_loss"]

        for key in keys_to_pop:
            inputs_dict.pop(key)

        model = BlipModel(config).to(torch_device)
        model.eval()
        text_features = model.get_text_features(**inputs_dict)
        self.assertEqual(
            text_features.pooler_output.shape,
            (
                self.model_tester.batch_size,
                model.projection_dim,
            ),
        )

    def test_get_multimodal_features(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        keys_to_pop = ["return_loss"]

        for key in keys_to_pop:
            inputs_dict.pop(key)

        model = BlipModel(config).to(torch_device)
        model.eval()
        multimodal_features = model.get_multimodal_features(**inputs_dict)
        self.assertEqual(
            multimodal_features.shape,
            (
                self.model_tester.batch_size,
                model.projection_dim,
            ),
        )


class BlipTextRetrievalModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = BlipTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = BlipVisionModelTester(parent, **vision_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return BlipConfig(
            text_config=self.text_model_tester.get_config().to_dict(),
            vision_config=self.vision_model_tester.get_config().to_dict(),
            projection_dim=64,
        )

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = BlipModel(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_ids, pixel_values, attention_mask)
        self.parent.assertEqual(
            result.logits_per_image.shape, (self.vision_model_tester.batch_size, self.text_model_tester.batch_size)
        )
        self.parent.assertEqual(
            result.logits_per_text.shape, (self.text_model_tester.batch_size, self.vision_model_tester.batch_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        return config, inputs_dict


class BlipTextImageModelsModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = BlipTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = BlipVisionModelTester(parent, **vision_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.seq_length = self.text_model_tester.seq_length
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return BlipConfig(
            text_config=self.text_model_tester.get_config().to_dict(),
            vision_config=self.vision_model_tester.get_config().to_dict(),
            projection_dim=64,
        )

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = BlipModel(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_ids, pixel_values, attention_mask)
        self.parent.assertEqual(
            result.logits_per_image.shape, (self.vision_model_tester.batch_size, self.text_model_tester.batch_size)
        )
        self.parent.assertEqual(
            result.logits_per_text.shape, (self.text_model_tester.batch_size, self.vision_model_tester.batch_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        return config, inputs_dict


class BlipVQAModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = BlipTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = BlipVisionModelTester(parent, **vision_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return BlipConfig(
            text_config=self.text_model_tester.get_config().to_dict(),
            vision_config=self.vision_model_tester.get_config().to_dict(),
            projection_dim=64,
        )

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = BlipModel(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_ids, pixel_values, attention_mask)
        self.parent.assertEqual(
            result.logits_per_image.shape, (self.vision_model_tester.batch_size, self.text_model_tester.batch_size)
        )
        self.parent.assertEqual(
            result.logits_per_text.shape, (self.text_model_tester.batch_size, self.vision_model_tester.batch_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "decoder_input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": input_ids,
        }
        return config, inputs_dict


@require_torch
@require_vision
class BlipVQAModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (BlipForQuestionAnswering,) if is_torch_available() else ()
    # Doesn't run generation tests due to custom generation logic -- won't fix
    all_generative_model_classes = ()

    test_resize_embeddings = True
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = BlipVQAModelTester(self)

    def _prepare_inputs_for_vqa(self):
        _, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict["decoder_input_ids"] = inputs_dict["input_ids"]
        inputs_dict.pop("return_loss")
        return inputs_dict

    def test_class_name_consistency(self):
        """
        Tests that all VQA models have a class name that ends with "ForQuestionAnswering"
        """
        for model_class in self.all_model_classes:
            model = model_class(self.model_tester.get_config())
            self.assertTrue(
                model.__class__.__name__.endswith("ForQuestionAnswering"),
                f"Class name should end with 'ForVisualQuestionAnswering' got {model.__class__.__name__}",
            )

    def test_training(self):
        """
        Tests that all VQA models can be trained on a single batch
        """
        for model_class in self.all_model_classes:
            model = model_class(self.model_tester.get_config()).to(torch_device)
            model.train()
            loss = model(**self.model_tester.prepare_config_and_inputs_for_common()[1]).loss
            loss.backward()

            # verify the gradients are not None
            for name, param in model.named_parameters():
                self.assertIsNotNone(param.grad, f"Gradients should not be None - got {param.grad} for {name}")

    def test_forward_signature(self):
        """
        Test if the forward function has the expected arguments.
        """
        for model_class in self.all_model_classes:
            model = model_class(self.model_tester.get_config())
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so args are the first n entries
            args = list(signature.parameters.keys())
            expected_args = [
                "input_ids",
                "attention_mask",
                "labels",
                "decoder_input_ids",
                "decoder_attention_mask",
            ]
            for arg in expected_args:
                self.assertTrue(
                    arg in args,
                    f"Argument {arg} of forward function signature should include {arg}. Found {args}.",
                )

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="BlipModel does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass


@require_torch
class BlipTextRetrievalModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (BlipForImageTextRetrieval,) if is_torch_available() else ()

    test_resize_embeddings = True
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = BlipTextRetrievalModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="BlipModel does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            if model.config.is_encoder_decoder:
                expected_arg_names = [
                    "input_ids",
                    "attention_mask",
                    "decoder_input_ids",
                    "decoder_attention_mask",
                ]
                expected_arg_names.extend(["encoder_outputs"])
                self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)
            else:
                expected_arg_names = ["input_ids"] if model_class != BlipForConditionalGeneration else ["pixel_values"]
                self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_training(self):
        if not self.model_tester.is_training:
            self.skipTest(reason="ModelTester is not setup for training")

        for model_class in self.all_model_classes[:-1]:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)

            # hardcode labels to be the same as input_ids
            inputs["labels"] = inputs["input_ids"]

            loss = model(**inputs).loss
            loss.backward()

    def check_training_gradient_checkpointing(self, gradient_checkpointing_kwargs=None):
        if not self.model_tester.is_training:
            self.skipTest(reason="ModelTester is not setup for training")

        for model_class in self.all_model_classes[:-1]:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.use_cache = False
            config.return_dict = True

            model = model_class(config)
            model.to(torch_device)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)

            # hardcode labels to be the same as input_ids
            inputs["labels"] = inputs["input_ids"]

            loss = model(**inputs).loss
            loss.backward()

    def test_load_vision_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save BlipConfig and check if we can load BlipVisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = BlipVisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save BlipConfig and check if we can load BlipTextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = BlipTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        model_name = "Salesforce/blip-vqa-base"
        model = BlipModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


@require_torch
class BlipTextImageModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (BlipForConditionalGeneration,) if is_torch_available() else ()
    # Doesn't run generation tests due to custom generation logic -- wont fix
    all_generative_model_classes = ()

    test_resize_embeddings = True
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = BlipTextImageModelsModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="BlipModel does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            if model.config.is_encoder_decoder:
                expected_arg_names = [
                    "input_ids",
                    "attention_mask",
                    "decoder_input_ids",
                    "decoder_attention_mask",
                ]
                expected_arg_names.extend(["encoder_outputs"])
                self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)
            else:
                expected_arg_names = ["input_ids"] if model_class != BlipForConditionalGeneration else ["pixel_values"]
                self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_training(self):
        if not self.model_tester.is_training:
            self.skipTest(reason="ModelTester is not setup for training")

        for model_class in self.all_model_classes[:-1]:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)

            # hardcode labels to be the same as input_ids
            inputs["labels"] = inputs["input_ids"]

            loss = model(**inputs).loss
            loss.backward()

    def check_training_gradient_checkpointing(self, gradient_checkpointing_kwargs=None):
        if not self.model_tester.is_training:
            self.skipTest(reason="ModelTester is not setup for training")

        for model_class in self.all_model_classes[:-1]:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.use_cache = False
            config.return_dict = True

            model = model_class(config)
            model.to(torch_device)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)

            # hardcode labels to be the same as input_ids
            inputs["labels"] = inputs["input_ids"]

            loss = model(**inputs).loss
            loss.backward()

    def test_load_vision_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save BlipConfig and check if we can load BlipVisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = BlipVisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save BlipConfig and check if we can load BlipTextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = BlipTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        model_name = "Salesforce/blip-vqa-base"
        model = BlipModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    url = "https://huggingface.co/hf-internal-testing/blip-test-image/resolve/main/demo.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@require_vision
@require_torch
@slow
class BlipModelIntegrationTest(unittest.TestCase):
    def test_inference_image_captioning(self):
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(torch_device)
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        image = prepare_img()

        # image only
        inputs = processor(images=image, return_tensors="pt").to(torch_device)

        predictions = model.generate(**inputs)

        # Test output
        self.assertEqual(predictions[0].tolist(), [30522, 1037, 2450, 3564, 2006, 1996, 3509, 2007, 2014, 3899, 102])

        # image and context
        context = ["a picture of"]
        inputs = processor(images=image, text=context, return_tensors="pt").to(torch_device)

        predictions = model.generate(**inputs)

        # Test output
        self.assertEqual(
            predictions[0].tolist(),
            [30522, 1037, 3861, 1997, 1037, 2450, 1998, 2014, 3899, 2006, 1996, 3509, 102],
        )

    @require_torch_accelerator
    @require_torch_fp16
    def test_inference_image_captioning_fp16(self):
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", dtype=torch.float16
        ).to(torch_device)
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        image = prepare_img()

        # image only
        inputs = processor(images=image, return_tensors="pt").to(torch_device, torch.float16)

        predictions = model.generate(**inputs)

        # Test output
        self.assertEqual(predictions[0].tolist(), [30522, 1037, 2450, 3564, 2006, 1996, 3509, 2007, 2014, 3899, 102])

        # image and context
        context = ["a picture of"]
        inputs = processor(images=image, text=context, return_tensors="pt").to(torch_device, torch.float16)

        predictions = model.generate(**inputs)

        # Test output
        self.assertEqual(
            predictions[0].tolist(),
            [30522, 1037, 3861, 1997, 1037, 2450, 1998, 2014, 3899, 2006, 1996, 3509, 102],
        )

    def test_inference_interpolate_pos_encoding(self):
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(torch_device)
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        processor.image_processor.size = {"height": 500, "width": 500}

        image = prepare_img()
        inputs = processor(images=image, return_tensors="pt").to(torch_device)

        predictions = model.generate(**inputs, interpolate_pos_encoding=True)
        generated_text = processor.batch_decode(predictions, skip_special_tokens=True)[0].strip()

        self.assertEqual(predictions[0].tolist(), [30522, 1037, 2450, 3564, 2006, 1996, 3509, 2007, 1037, 3899, 102])
        self.assertEqual(generated_text, "a woman sitting on the beach with a dog")

    def test_inference_vqa(self):
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(torch_device)
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

        image = prepare_img()
        text = "how many dogs are in the picture?"

        inputs = processor(image, text=text, return_tensors="pt").to(torch_device)
        out = model.generate(**inputs)

        # Test output
        self.assertEqual(out[0].tolist(), [30522, 1015, 102])

    def test_inference_itm(self):
        model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco").to(torch_device)
        processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

        image = prepare_img()
        text = "A woman and her dog sitting in a beach"

        inputs = processor(image, text, return_tensors="pt").to(torch_device)

        out_itm = model(**inputs)
        out = model(**inputs, use_itm_head=False)

        expected_scores = torch.Tensor([[0.0029, 0.9971]])

        torch.testing.assert_close(torch.nn.Softmax()(out_itm[0].cpu()), expected_scores, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(out[0].cpu(), torch.Tensor([[0.5162]]), rtol=1e-3, atol=1e-3)
