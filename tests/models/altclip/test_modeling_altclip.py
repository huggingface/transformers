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
"""Testing suite for the PyTorch AltCLIP model."""

import inspect
import unittest

import numpy as np
import requests

from transformers import AltCLIPConfig, AltCLIPProcessor, AltCLIPTextConfig, AltCLIPVisionConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
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
    import torch.nn as nn

    from transformers import AltCLIPModel, AltCLIPTextModel, AltCLIPVisionModel

if is_vision_available():
    from PIL import Image


class AltCLIPVisionModelTester:
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
        initializer_range=0.02,
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
        return AltCLIPVisionConfig(
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
        model = AltCLIPVisionModel(config=config)
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
class AltCLIPVisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as CLIP does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (AltCLIPVisionModel,) if is_torch_available() else ()

    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = AltCLIPVisionModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=AltCLIPVisionConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="CLIP does not use inputs_embeds")
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

    @unittest.skip(reason="AltCLIPVisionModel use the same cv backbone with CLIP model.")
    def test_model_from_pretrained(self):
        pass


class AltCLIPTextModelTester:
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
        project_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
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
        self.project_dim = project_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.scope = scope

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
        return AltCLIPTextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            project_dim=self.project_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            pad_token_id=1,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = AltCLIPTextModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids, attention_mask=input_mask)
            result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.projection_dim))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class AltCLIPTextModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (AltCLIPTextModel,) if is_torch_available() else ()

    # TODO (@SunMarc): Fix me
    @unittest.skip(reason="It's broken.")
    def test_resize_tokens_embeddings(self):
        super().test_resize_tokens_embeddings()

    def setUp(self):
        self.model_tester = AltCLIPTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=AltCLIPTextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

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

    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="Result of the model is a dict")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="AltCLIP does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "BAAI/AltCLIP"
        model = AltCLIPTextModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


class AltCLIPModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = AltCLIPTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = AltCLIPVisionModelTester(parent, **vision_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()
        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return AltCLIPConfig(
            text_config=self.text_model_tester.get_config().to_dict(),
            vision_config=self.vision_model_tester.get_config().to_dict(),
            projection_dim=64,
        )

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = AltCLIPModel(config=config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            model(input_ids, pixel_values, attention_mask)

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


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@require_torch
class AltCLIPModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (AltCLIPModel,) if is_torch_available() else ()
    pipeline_model_mapping = {"text-embedding": AltCLIPModel} if is_torch_available() else {}

    test_resize_embeddings = False
    test_attention_outputs = False

    # TODO: Fix the failed tests when this model gets more usage
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        if pipeline_test_case_name == "TextEmbeddingPipelineTests":
            return True

        return False

    def setUp(self):
        self.model_tester = AltCLIPModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=AltCLIPConfig,
            has_text_modality=False,
            common_properties=["projection_dim", "logit_scale_init_value"],
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

    @unittest.skip(reason="CLIPModel does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "BAAI/AltCLIP"
        model = AltCLIPModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


@require_vision
@require_torch
class AltCLIPModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model_name = "BAAI/AltCLIP"
        model = AltCLIPModel.from_pretrained(model_name).to(torch_device)
        processor = AltCLIPProcessor.from_pretrained(model_name)

        image = prepare_img()
        inputs = processor(text=["一张猫的照片", "一张狗的照片"], images=image, padding=True, return_tensors="pt").to(torch_device)  # fmt: skip

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        self.assertEqual(
            outputs.logits_per_image.shape,
            torch.Size((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])),
        )
        self.assertEqual(
            outputs.logits_per_text.shape,
            torch.Size((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])),
        )

        probs = outputs.logits_per_image.softmax(dim=1)
        expected_probs = torch.tensor([[9.9942e-01, 5.7805e-04]], device=torch_device)

        torch.testing.assert_close(probs, expected_probs, rtol=5e-3, atol=5e-3)

    @slow
    def test_inference_interpolate_pos_encoding(self):
        # ViT models have an `interpolate_pos_encoding` argument in their forward method,
        # allowing to interpolate the pre-trained position embeddings in order to use
        # the model on higher resolutions. The DINO model by Facebook AI leverages this
        # to visualize self-attention on higher resolution images.
        model_name = "BAAI/AltCLIP"
        model = AltCLIPModel.from_pretrained(model_name).to(torch_device)

        image_processor = AltCLIPProcessor.from_pretrained(
            model_name, size={"shortest_edge": 180}, crop_size={"height": 180, "width": 180}
        )

        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        inputs = image_processor(text="what's in the image", images=image, return_tensors="pt").to(torch_device)

        # interpolate_pos_encodiung false should return value error
        with self.assertRaises(ValueError, msg="doesn't match model"):
            with torch.no_grad():
                model(**inputs, interpolate_pos_encoding=False)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs, interpolate_pos_encoding=True)

        # verify the logits
        expected_shape = torch.Size((1, 145, 1024))
        print("nilesh ")
        print(outputs.vision_model_output.last_hidden_state.shape)
        print(outputs.vision_model_output.last_hidden_state[0, :3, :3])

        self.assertEqual(outputs.vision_model_output.last_hidden_state.shape, expected_shape)

        expected_slice = torch.tensor(
            [
                [-0.3577, -0.5977, 0.3555],
                [0.4544, 0.1660, 0.6583],
                [1.1715, -0.4870, 0.1645],
            ]
        ).to(torch_device)

        torch.testing.assert_close(
            outputs.vision_model_output.last_hidden_state[0, :3, :3], expected_slice, rtol=1e-4, atol=1e-4
        )
