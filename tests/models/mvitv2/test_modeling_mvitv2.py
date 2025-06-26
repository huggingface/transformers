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
"""Testing suite for the PyTorch MViTV2 model."""

import copy
import unittest

from huggingface_hub import hf_hub_download

from transformers import MViTV2Config
from transformers.models.auto import get_values
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available
from transformers.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
        MViTV2ForImageClassification,
        MViTV2Model,
    )


if is_vision_available():
    from PIL import Image

    from transformers import ViTImageProcessor


class MViTV2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        depths=[2, 2],
        in_channels=3,
        hidden_size=32,
        num_heads=1,
        image_size=[224, 224],
        patch_kernel_size=[7, 7],
        patch_stride_size=[4, 4],
        patch_padding_size=[3, 3],
        use_cls_token=False,
        mlp_ratio=1,
        num_labels=10,
        use_labels=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.depths = depths
        self.total_layers = sum(depths)
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.hidden_sizes = tuple(self.hidden_size * (2**i) for i in range(len(self.depths)))
        self.num_heads = num_heads
        self.all_num_heads = tuple(self.num_heads * (2**i) for i in range(len(self.depths)))
        self.image_size = image_size
        self.patch_kernel_size = patch_kernel_size
        self.patch_stride_size = patch_stride_size
        self.patch_padding_size = patch_padding_size
        initial_feature_size = self.image_size[0] // self.patch_stride_size[0]
        self.feature_sizes = tuple(initial_feature_size // (2**i) for i in range(len(self.depths)))
        self.num_patches = tuple(i**2 for i in self.feature_sizes)
        self.use_cls_token = use_cls_token
        self.mlp_ratio = mlp_ratio
        self.num_labels = num_labels
        self.use_labels = use_labels
        self.is_training = True

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.in_channels, self.image_size[0], self.image_size[1]])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        config = MViTV2Config(
            depths=self.depths,
            in_channels=self.in_channels,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            image_size=self.image_size,
            patch_kernel_size=self.patch_kernel_size,
            patch_stride_size=self.patch_stride_size,
            patch_padding_size=self.patch_padding_size,
            use_cls_token=self.use_cls_token,
            mlp_ratio=self.mlp_ratio,
        )
        config.num_labels = self.num_labels
        return config

    def create_and_check_model(self, config, pixel_values, labels):
        model = MViTV2Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        # verify the hidden state shape
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.num_patches[-1], self.hidden_sizes[-1])
        )

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        model = MViTV2ForImageClassification(config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values)

        # verify the logits shape
        expected_shape = torch.Size((self.batch_size, self.num_labels))
        self.parent.assertEqual(result.logits.shape, expected_shape)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


def prepare_image():
    file = hf_hub_download(repo_id="hf-internal-testing/fixtures_nlvr2", filename="image2.jpeg", repo_type="dataset")
    image = Image.open(file)
    return image


@require_torch
class MViTV2ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as MViTV2 does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (MViTV2Model, MViTV2ForImageClassification) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": MViTV2Model, "image-classification": MViTV2ForImageClassification}
        if is_torch_available()
        else {}
    )

    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torch_exportable = True

    def setUp(self):
        self.model_tester = MViTV2ModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=MViTV2Config,
            has_text_modality=False,
            common_properties=["hidden_size", "num_heads", "mlp_ratio"],
        )

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)

        if return_labels:
            if model_class in get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING):
                inputs_dict["labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )

        return inputs_dict

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="MViTV2 does not use inputs_embeddings")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "KamilaMila/mvitv2-base"
        model = MViTV2Model.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def test_attention_outputs(self):
        if not self.has_attentions:
            self.skipTest(reason="Model has no attentions")

        else:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            for model_class in self.all_model_classes:
                inputs_dict["output_attentions"] = True
                inputs_dict["output_hidden_states"] = False
                model = model_class(config)
                model.to(torch_device)
                model.eval()
                with torch.no_grad():
                    outputs = model(**self._prepare_for_class(inputs_dict, model_class))
                attentions = outputs.attentions
                self.assertEqual(len(attentions), self.model_tester.total_layers)

                # check that output_attentions also work using config
                del inputs_dict["output_attentions"]
                config.output_attentions = True
                model = model_class(config)
                model.to(torch_device)
                model.eval()
                with torch.no_grad():
                    outputs = model(**self._prepare_for_class(inputs_dict, model_class))
                attentions = outputs.attentions
                self.assertEqual(len(attentions), self.model_tester.total_layers)

                # attentions has shape batch_size x num_heads x num_patches for queries x num_patches for keys
                self.assertListEqual(
                    list(attentions[0].shape),
                    [
                        self.model_tester.batch_size,
                        self.model_tester.all_num_heads[0],
                        self.model_tester.num_patches[0],
                        self.model_tester.num_patches[1] // 4,
                    ],
                )
                self.assertListEqual(
                    list(attentions[3].shape),
                    [
                        self.model_tester.batch_size,
                        self.model_tester.all_num_heads[1],
                        self.model_tester.num_patches[1],
                        self.model_tester.num_patches[1] // 4,
                    ],
                )

                out_len = len(outputs)

                inputs_dict["output_attentions"] = True
                inputs_dict["output_hidden_states"] = True
                model = model_class(config)
                model.to(torch_device)
                model.eval()
                with torch.no_grad():
                    outputs = model(**self._prepare_for_class(inputs_dict, model_class))

                self.assertEqual(out_len + 1, len(outputs))

                self_attentions = outputs.attentions

                self.assertEqual(len(self_attentions), self.model_tester.total_layers)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states
            expected_num_layers = self.model_tester.total_layers + 2
            self.assertEqual(len(hidden_states), expected_num_layers)

            self.assertListEqual(
                list(hidden_states[1].shape),
                [self.model_tester.batch_size, self.model_tester.num_patches[0], self.model_tester.hidden_sizes[0]],
            )

            self.assertListEqual(
                list(hidden_states[4].shape),
                [self.model_tester.batch_size, self.model_tester.num_patches[1], self.model_tester.hidden_sizes[1]],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)


@require_torch
@require_vision
class MViTV2ModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return (
            ViTImageProcessor(image_mean=IMAGENET_DEFAULT_MEAN, image_std=IMAGENET_DEFAULT_STD)
            if is_vision_available()
            else None
        )

    @slow
    def test_inference_for_image_classification(self):
        model = MViTV2ForImageClassification.from_pretrained("KamilaMila/mvitv2-base").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_image()
        inputs = image_processor(image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([-2.4503e-01, 3.0206e-01, 1.0540e-01]).to(torch_device)

        torch.testing.assert_close(outputs.logits[0, :3], expected_slice, rtol=1e-4, atol=1e-4)
