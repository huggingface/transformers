# coding=utf-8
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
""" Testing suite for the PyTorch DPT model. """


import inspect
import unittest

from transformers import DPTConfig
from transformers.file_utils import is_flax_available, is_torch_available, is_vision_available
from transformers.models.auto import get_values
from transformers.testing_utils import is_pt_flax_cross_test, require_torch, require_vision, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch
    from torch import nn

    from transformers import MODEL_MAPPING, DPTForDepthEstimation, DPTForSemanticSegmentation, DPTModel
    from transformers.models.dpt.modeling_dpt import DPT_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import DPTFeatureExtractor

if is_torch_available() and is_flax_available():
    import tempfile

    import numpy as np

    import jax.numpy as jnp
    import transformers
    from transformers.modeling_flax_pytorch_utils import (
        convert_pytorch_state_dict_to_flax,
        load_flax_weights_in_pytorch_model,
    )


class DPTModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_size=32,
        patch_size=16,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=4,
        backbone_out_indices=[0, 1, 2, 3],
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        num_labels=3,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.backbone_out_indices = backbone_out_indices
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.scope = scope
        # sequence length of DPT = num_patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size, self.image_size, self.image_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return DPTConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            backbone_out_indices=self.backbone_out_indices,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = DPTModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_depth_estimation(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = DPTForDepthEstimation(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.predicted_depth.shape, (self.batch_size, self.image_size, self.image_size))

    def create_and_check_for_semantic_segmentation(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = DPTForSemanticSegmentation(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_labels, self.image_size, self.image_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class DPTModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as DPT does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (DPTModel, DPTForDepthEstimation, DPTForSemanticSegmentation) if is_torch_available() else ()

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = DPTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DPTConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="DPT does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_common_attributes(self):
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

    def test_for_depth_estimation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_depth_estimation(*config_and_inputs)

    def test_for_semantic_segmentation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_semantic_segmentation(*config_and_inputs)

    def test_training(self):
        for model_class in self.all_model_classes:
            if model_class.__name__ == "DPTForDepthEstimation":
                continue

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            if model_class in get_values(MODEL_MAPPING):
                continue

            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    def test_training_gradient_checkpointing(self):
        for model_class in self.all_model_classes:
            if model_class.__name__ == "DPTForDepthEstimation":
                continue

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.use_cache = False
            config.return_dict = True

            if model_class in get_values(MODEL_MAPPING) or not model_class.supports_gradient_checkpointing:
                continue
            model = model_class(config)
            model.to(torch_device)
            model.gradient_checkpointing_enable()
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    @is_pt_flax_cross_test
    def test_equivalence_pt_to_flax(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.align_corners = False

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                fx_model_class_name = "Flax" + model_class.__name__

                if not hasattr(transformers, fx_model_class_name):
                    # no flax model exists for this class
                    return

                # Output all for aggressive testing
                config.output_hidden_states = True
                config.output_attentions = self.has_attentions

                fx_model_class = getattr(transformers, fx_model_class_name)

                # load PyTorch class
                pt_model = model_class(config).eval()
                # Flax models don't use the `use_cache` option and cache is not returned as a default.
                # So we disable `use_cache` here for PyTorch model.
                pt_model.config.use_cache = False

                # load Flax class
                fx_model = fx_model_class(config, dtype=jnp.float32)

                # make sure only flax inputs are forward that actually exist in function args
                fx_input_keys = inspect.signature(fx_model.__call__).parameters.keys()

                # prepare inputs
                pt_inputs = self._prepare_for_class(inputs_dict, model_class)

                # remove function args that don't exist in Flax
                pt_inputs = {k: v for k, v in pt_inputs.items() if k in fx_input_keys}

                # send pytorch inputs to the correct device
                pt_inputs = {
                    k: v.to(device=torch_device) if isinstance(v, torch.Tensor) else v for k, v in pt_inputs.items()
                }

                # convert inputs to Flax
                fx_inputs = {k: np.array(v) for k, v in pt_inputs.items() if torch.is_tensor(v)}

                fx_state = convert_pytorch_state_dict_to_flax(pt_model.state_dict(), fx_model)
                fx_model.params = fx_state

                # send pytorch model to the correct device
                pt_model.to(torch_device)

                with torch.no_grad():
                    pt_outputs = pt_model(**pt_inputs)
                fx_outputs = fx_model(**fx_inputs)

                fx_keys = tuple([k for k, v in fx_outputs.items() if v is not None])
                pt_keys = tuple([k for k, v in pt_outputs.items() if v is not None])

                self.assertEqual(fx_keys, pt_keys)
                self.check_pt_flax_outputs(fx_outputs, pt_outputs, model_class)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    pt_model.save_pretrained(tmpdirname)
                    fx_model_loaded = fx_model_class.from_pretrained(tmpdirname, from_pt=True)

                fx_outputs_loaded = fx_model_loaded(**fx_inputs)

                fx_keys = tuple([k for k, v in fx_outputs_loaded.items() if v is not None])
                pt_keys = tuple([k for k, v in pt_outputs.items() if v is not None])

                self.assertEqual(fx_keys, pt_keys)
                self.check_pt_flax_outputs(fx_outputs_loaded, pt_outputs, model_class)

    @is_pt_flax_cross_test
    def test_equivalence_flax_to_pt(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.align_corners = False

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                fx_model_class_name = "Flax" + model_class.__name__

                if not hasattr(transformers, fx_model_class_name):
                    # no flax model exists for this class
                    return

                # Output all for aggressive testing
                config.output_hidden_states = True
                config.output_attentions = self.has_attentions

                fx_model_class = getattr(transformers, fx_model_class_name)

                # load PyTorch class
                pt_model = model_class(config).eval()
                # Flax models don't use the `use_cache` option and cache is not returned as a default.
                # So we disable `use_cache` here for PyTorch model.
                pt_model.config.use_cache = False

                # load Flax class
                fx_model = fx_model_class(config, dtype=jnp.float32)

                # make sure only flax inputs are forward that actually exist in function args
                fx_input_keys = inspect.signature(fx_model.__call__).parameters.keys()

                # prepare inputs
                pt_inputs = self._prepare_for_class(inputs_dict, model_class)

                # remove function args that don't exist in Flax
                pt_inputs = {k: v for k, v in pt_inputs.items() if k in fx_input_keys}

                # send pytorch inputs to the correct device
                pt_inputs = {
                    k: v.to(device=torch_device) if isinstance(v, torch.Tensor) else v for k, v in pt_inputs.items()
                }

                # convert inputs to Flax
                fx_inputs = {k: np.array(v) for k, v in pt_inputs.items() if torch.is_tensor(v)}

                pt_model = load_flax_weights_in_pytorch_model(pt_model, fx_model.params)

                # make sure weights are tied in PyTorch
                pt_model.tie_weights()

                # send pytorch model to the correct device
                pt_model.to(torch_device)

                with torch.no_grad():
                    pt_outputs = pt_model(**pt_inputs)
                fx_outputs = fx_model(**fx_inputs)

                fx_keys = tuple([k for k, v in fx_outputs.items() if v is not None])
                pt_keys = tuple([k for k, v in pt_outputs.items() if v is not None])

                self.assertEqual(fx_keys, pt_keys)
                self.check_pt_flax_outputs(fx_outputs, pt_outputs, model_class)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    fx_model.save_pretrained(tmpdirname)
                    pt_model_loaded = model_class.from_pretrained(tmpdirname, from_flax=True)

                # send pytorch model to the correct device
                pt_model_loaded.to(torch_device)
                pt_model_loaded.eval()

                with torch.no_grad():
                    pt_outputs_loaded = pt_model_loaded(**pt_inputs)

                fx_keys = tuple([k for k, v in fx_outputs.items() if v is not None])
                pt_keys = tuple([k for k, v in pt_outputs_loaded.items() if v is not None])

                self.assertEqual(fx_keys, pt_keys)
                self.check_pt_flax_outputs(fx_outputs, pt_outputs_loaded, model_class)

    @slow
    def test_model_from_pretrained(self):
        for model_name in DPT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = DPTModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
@slow
class DPTModelIntegrationTest(unittest.TestCase):
    def test_inference_depth_estimation(self):
        feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
        model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(torch_device)

        image = prepare_img()
        inputs = feature_extractor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # verify the predicted depth
        expected_shape = torch.Size((1, 384, 384))
        self.assertEqual(predicted_depth.shape, expected_shape)

        expected_slice = torch.tensor(
            [[6.3199, 6.3629, 6.4148], [6.3850, 6.3615, 6.4166], [6.3519, 6.3176, 6.3575]]
        ).to(torch_device)

        self.assertTrue(torch.allclose(outputs.predicted_depth[0, :3, :3], expected_slice, atol=1e-4))

    def test_inference_semantic_segmentation(self):
        feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large-ade")
        model = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade").to(torch_device)

        image = prepare_img()
        inputs = feature_extractor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 150, 480, 480))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[4.0480, 4.2420, 4.4360], [4.3124, 4.5693, 4.8261], [4.5768, 4.8965, 5.2163]]
        ).to(torch_device)

        self.assertTrue(torch.allclose(outputs.logits[0, 0, :3, :3], expected_slice, atol=1e-4))
