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
""" Testing suite for the PyTorch TinyVit model. """

import collections
import inspect
import unittest

from transformers import TinyVitConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import TinyVitBackbone, TinyVitForImageClassification, TinyVitModel
    from transformers.models.tinyvit.modeling_tinyvit import TINYVIT_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import AutoImageProcessor


class TinyVitModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=32,
        num_channels=3,
        hidden_sizes=[48, 96, 192],
        depths=[1, 1, 1],
        num_heads=[2, 2, 2],
        window_sizes=[7, 7, 7],
        mlp_ratio=2.0,
        hidden_dropout_prob=0.0,
        drop_path_rate=0.1,
        hidden_act="gelu",
        initializer_range=0.02,
        is_training=True,
        scope=None,
        use_labels=True,
        num_labels=10,
        out_features=["stage2", "stage3"],
        out_indices=[2, 3],
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.num_heads = num_heads
        self.window_sizes = window_sizes
        self.mlp_ratio = mlp_ratio
        self.hidden_dropout_prob = hidden_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.is_training = is_training
        self.scope = scope
        self.use_labels = use_labels
        self.num_labels = num_labels
        self.out_features = out_features
        self.out_indices = out_indices

        # set expected sequence length of final hidden states
        patches_resolution = self.image_size // 4
        self.seq_length = (patches_resolution // (2 ** (len(depths) - 1))) ** 2

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return TinyVitConfig(
            image_size=self.image_size,
            num_channels=self.num_channels,
            hidden_sizes=self.hidden_sizes,
            depths=self.depths,
            num_heads=self.num_heads,
            window_sizes=self.window_sizes,
            mlp_ratio=self.mlp_ratio,
            hidden_dropout_prob=self.hidden_dropout_prob,
            drop_path_rate=self.drop_path_rate,
            hidden_act=self.hidden_act,
            initializer_range=self.initializer_range,
            out_features=self.out_features,
            out_indices=self.out_indices,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = TinyVitModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_sizes[-1])
        )

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = TinyVitForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

        # test greyscale images
        config.num_channels = 1
        model = TinyVitForImageClassification(config)
        model.to(torch_device)
        model.eval()

        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values,
            labels,
        ) = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class TinyVitModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (TinyVitModel, TinyVitBackbone, TinyVitForImageClassification) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": TinyVitModel, "image-classification": TinyVitForImageClassification}
        if is_torch_available()
        else {}
    )
    fx_compatible = False

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = TinyVitModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TinyVitConfig, embed_dim=37)

    def test_config(self):
        self.create_and_test_config_common_properties()
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def create_and_test_config_common_properties(self):
        return

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    # TODO: check if this works again for PyTorch 2.x.y
    @unittest.skip(reason="Got `CUDA error: misaligned address` with PyTorch 2.0.0.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_training_gradient_checkpointing(self):
        super().test_training_gradient_checkpointing()

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @unittest.skip(reason="TinyVit does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="TinyVit Transformer does not use feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="TinyVit does not support get_input_embeddings")
    def test_model_common_attributes(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

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
            expected_num_attentions = len(self.model_tester.depths) - 1
            self.assertEqual(len(attentions), expected_num_attentions)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            window_size_squared = self.model_tester.window_sizes[0] ** 2
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), expected_num_attentions)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_heads[0], window_size_squared, window_size_squared],
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

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), expected_num_attentions)

            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_heads[0], window_size_squared, window_size_squared],
            )

    def check_hidden_states_output(self, inputs_dict, config, model_class, image_size):
        model = model_class(config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

        hidden_states = outputs.hidden_states

        expected_num_layers = getattr(
            self.model_tester, "expected_num_hidden_layers", len(self.model_tester.depths) + 1
        )
        self.assertEqual(len(hidden_states), expected_num_layers)

        # TinyVit has a different seq_length
        seq_length = (self.model_tester.image_size // 4) ** 2

        self.assertListEqual(
            list(hidden_states[0].shape[-2:]),
            [seq_length, self.model_tester.hidden_sizes[0]],
        )

    def test_hidden_states_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        image_size = (
            self.model_tester.image_size
            if isinstance(self.model_tester.image_size, collections.abc.Iterable)
            else (self.model_tester.image_size, self.model_tester.image_size)
        )

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            self.check_hidden_states_output(inputs_dict, config, model_class, image_size)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            self.check_hidden_states_output(inputs_dict, config, model_class, image_size)

    @slow
    def test_model_from_pretrained(self):
        for model_name in TINYVIT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TinyVitModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if "embeddings" not in name and param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )


@require_vision
@require_torch
class TinyVitModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return AutoImageProcessor.from_pretrained("microsoft/tinyvit-21m-224") if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        model = TinyVitForImageClassification.from_pretrained("microsoft/tinyvit-21m-224").to(torch_device)
        image_processor = self.default_image_processor

        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice = torch.tensor([-0.0948, -0.6454, -0.0921]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4))


@require_torch
class TinyVitBackboneTest(unittest.TestCase, BackboneTesterMixin):
    all_model_classes = (TinyVitBackbone,) if is_torch_available() else ()
    config_class = TinyVitConfig

    def setUp(self):
        self.model_tester = TinyVitModelTester(self)
