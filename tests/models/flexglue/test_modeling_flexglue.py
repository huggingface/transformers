# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import inspect
import unittest
from functools import cached_property

from transformers import FlexGlueConfig
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import FlexGlueForKeypointMatching, FlexGlueModel

if is_vision_available():
    from transformers import AutoImageProcessor
    from transformers.image_utils import load_image


class FlexGlueModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_height=64,  # needs to be a multiple of the backbone stride (32)
        image_width=96,
        num_keypoints=12,
        backbone_depths=[1, 1, 1, 1],
        backbone_hidden_sizes=[8, 16, 16, 32],
        neck_hidden_size=32,
        feature_dim=16,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        search_dim=8,
        offset_hidden_size=16,
        offset_num_bins=4,
        max_num_samples=40,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.num_keypoints = num_keypoints
        self.backbone_depths = backbone_depths
        self.backbone_hidden_sizes = backbone_hidden_sizes
        self.neck_hidden_size = neck_hidden_size
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.search_dim = search_dim
        self.offset_hidden_size = offset_hidden_size
        self.offset_num_bins = offset_num_bins
        self.max_num_samples = max_num_samples

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, 2, 3, self.image_height, self.image_width])
        # keypoints inside the images, so that the matches of the common tests are deterministic
        keypoints = floats_tensor([self.batch_size, 2, self.num_keypoints, 2], scale=0.8) + 0.1
        config = self.get_config()
        return config, pixel_values, keypoints

    def get_config(self):
        return FlexGlueConfig(
            backbone_config={
                "model_type": "dinov3_convnext",
                "depths": self.backbone_depths,
                "hidden_sizes": self.backbone_hidden_sizes,
                "out_features": ["stage1", "stage2", "stage3", "stage4"],
            },
            neck_hidden_size=self.neck_hidden_size,
            feature_dim=self.feature_dim,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            search_dim=self.search_dim,
            offset_hidden_size=self.offset_hidden_size,
            offset_num_bins=self.offset_num_bins,
            max_num_samples=self.max_num_samples,
            # thresholds off, so that the outputs of the common tests are continuous in the inputs
            filter_threshold=0.0,
            max_distance_px=None,
        )

    def create_and_check_model(self, config, pixel_values, keypoints):
        model = FlexGlueForKeypointMatching(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        grid_height = self.image_height // config.coarse_stride
        grid_width = self.image_width // config.coarse_stride
        num_matches = min(config.max_num_samples, 2 * grid_height * grid_width)
        self.parent.assertEqual(result.keypoints.shape, (self.batch_size, 2, num_matches, 2))
        self.parent.assertEqual(result.matches.shape, (self.batch_size, 2, num_matches))
        self.parent.assertEqual(result.matching_scores.shape, (self.batch_size, 2, num_matches))
        self.parent.assertEqual(result.covisibility_score.shape, (self.batch_size,))
        self.parent.assertEqual(result.dense_correspondences.shape, (self.batch_size, 2, grid_height, grid_width, 2))
        self.parent.assertEqual(result.dense_confidence.shape, (self.batch_size, 2, grid_height, grid_width))
        # sampled points match their k-th counterpart in the other image
        valid = result.matches[:, 0] > -1
        indices = torch.arange(num_matches, device=torch_device).expand(self.batch_size, -1)
        self.parent.assertTrue(torch.equal(result.matches[:, 0][valid], indices[valid]))

    def create_and_check_model_with_keypoints(self, config, pixel_values, keypoints):
        model = FlexGlueForKeypointMatching(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values, keypoints=keypoints)
        # the keypoints of both images, concatenated with their correspondences in the other image
        num_matches = 2 * self.num_keypoints
        self.parent.assertEqual(result.keypoints.shape, (self.batch_size, 2, num_matches, 2))
        self.parent.assertEqual(result.matches.shape, (self.batch_size, 2, num_matches))
        torch.testing.assert_close(result.keypoints[:, 0, : self.num_keypoints], keypoints[:, 0])
        torch.testing.assert_close(result.keypoints[:, 1, self.num_keypoints :], keypoints[:, 1])
        self.parent.assertIsNone(result.dense_correspondences)

        model.config.sparse_matching = True
        with torch.no_grad():
            result = model(pixel_values, keypoints=keypoints)
        # the input keypoints, matched with mutual nearest neighbors
        self.parent.assertEqual(result.keypoints.shape, (self.batch_size, 2, self.num_keypoints, 2))
        self.parent.assertEqual(result.matches.shape, (self.batch_size, 2, self.num_keypoints))
        torch.testing.assert_close(result.keypoints, keypoints)
        matches0, matches1 = result.matches[:, 0], result.matches[:, 1]
        for batch_index in range(self.batch_size):
            for index, match in enumerate(matches0[batch_index].tolist()):
                if match > -1:
                    self.parent.assertEqual(matches1[batch_index, match].item(), index)

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, keypoints = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values, "keypoints": keypoints}
        return config, inputs_dict


@require_torch
class FlexGlueModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (FlexGlueForKeypointMatching, FlexGlueModel) if is_torch_available() else ()

    test_resize_embeddings = False
    has_attentions = True

    def setUp(self):
        self.model_tester = FlexGlueModelTester(self)
        self.config_tester = ConfigTester(self, config_class=FlexGlueConfig, has_text_modality=False)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if model_class is FlexGlueModel:
            inputs_dict.pop("keypoints", None)
        return inputs_dict

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    @unittest.skip(reason="FlexGlue does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="FlexGlue does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="FlexGlue does not use feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="FlexGlue does not support standalone training")
    def test_training(self):
        pass

    @unittest.skip(reason="FlexGlue does not support standalone training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="FlexGlue does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="FlexGlue does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    @unittest.skip(reason="FlexGlue does not output any loss term in the forward pass")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_keypoints(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_keypoints(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]
            self.assertListEqual(arg_names[:1], ["pixel_values"])

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            hidden_states = outputs.hidden_states
            # the input tokens and the output of each layer
            self.assertEqual(len(hidden_states), self.model_tester.num_hidden_layers + 1)
            num_cells = (self.model_tester.image_height // 8) * (self.model_tester.image_width // 8)
            self.assertListEqual(
                list(hidden_states[0].shape), [self.model_tester.batch_size * 2, num_cells, config.hidden_size]
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            check_hidden_states_output(inputs_dict, config, model_class)

    def test_attention_outputs(self):
        def check_attention_output(inputs_dict, config, model_class):
            config._attn_implementation = "eager"
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            # one self attention and one cross attention per layer
            self.assertEqual(len(attentions), 2 * self.model_tester.num_hidden_layers)
            num_cells = (self.model_tester.image_height // 8) * (self.model_tester.image_width // 8)
            for attention in attentions:
                self.assertListEqual(
                    list(attention.shape[-3:]), [self.model_tester.num_attention_heads, num_cells, num_cells]
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            check_attention_output(inputs_dict, config, model_class)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            check_attention_output(inputs_dict, config, model_class)

    def test_forward_labels_should_be_none(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                model_inputs = self._prepare_for_class(inputs_dict, model_class)
                model_inputs["labels"] = torch.rand((128, 128))
                with self.assertRaises(ValueError):
                    model(**model_inputs)

    @slow
    def test_model_from_pretrained(self):
        model = FlexGlueForKeypointMatching.from_pretrained("ETH-CVG/flexglue")
        self.assertIsNotNone(model)


def prepare_imgs():
    image0 = load_image("https://raw.githubusercontent.com/cvg/FlexGlue/main/assets/berlin1.jpg")
    image1 = load_image("https://raw.githubusercontent.com/cvg/FlexGlue/main/assets/berlin2.jpg")
    return [image0, image1]


@require_torch
@require_vision
class FlexGlueModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return AutoImageProcessor.from_pretrained("ETH-CVG/flexglue") if is_vision_available() else None

    @cached_property
    def default_model(self):
        return FlexGlueForKeypointMatching.from_pretrained("ETH-CVG/flexglue").to(torch_device).eval()

    @slow
    def test_inference_dense(self):
        inputs = self.default_image_processor(images=prepare_imgs(), return_tensors="pt").to(torch_device)
        torch.manual_seed(0)
        with torch.no_grad():
            outputs = self.default_model(**inputs)

        self.assertEqual(outputs.matches.shape, (1, 2, 5000))
        self.assertEqual(outputs.dense_correspondences.shape, (1, 2, 64, 64, 2))

        expected_covisibility = Expectations({("cpu", None): 0.9921}).get_expectation()
        expected_dense_correspondences = Expectations(
            {("cpu", None): [[0.3875, 0.5948], [0.4552, 0.5872], [0.5284, 0.5817]]}
        ).get_expectation()
        expected_dense_confidence = Expectations({("cpu", None): [0.6826, 0.4212, 0.1600]}).get_expectation()
        expected_number_of_valid_cells = Expectations({("cpu", None): 1881}).get_expectation()

        torch.testing.assert_close(outputs.covisibility_score[0].item(), expected_covisibility, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(
            outputs.dense_correspondences[0, 0, 32, 47:50],
            torch.tensor(expected_dense_correspondences, device=torch_device),
            rtol=1e-3,
            atol=1e-3,
        )
        torch.testing.assert_close(
            outputs.dense_confidence[0, 0, 32, 47:50],
            torch.tensor(expected_dense_confidence, device=torch_device),
            rtol=1e-3,
            atol=1e-3,
        )
        number_of_valid_cells = (outputs.dense_confidence[0] > 0).sum().item()
        self.assertTrue(abs(number_of_valid_cells - expected_number_of_valid_cells) < 20)

    @slow
    def test_inference_keypoints(self):
        inputs = self.default_image_processor(images=prepare_imgs(), return_tensors="pt").to(torch_device)
        # a grid of keypoints in the first image, their predicted correspondences shifted by 1 pixel in the second
        y, x = torch.meshgrid(torch.linspace(0.1, 0.9, 10), torch.linspace(0.1, 0.9, 10), indexing="ij")
        keypoints0 = torch.stack([x, y], -1).reshape(1, -1, 2).to(torch_device)
        with torch.no_grad():
            dense = self.default_model(**inputs).dense_correspondences
        grid = dense[:, 0].permute(0, 3, 1, 2) * 2 - 1
        keypoints1 = torch.nn.functional.grid_sample(grid, keypoints0[:, None] * 2 - 1, align_corners=False)
        keypoints1 = (keypoints1[:, :, 0].mT + 1) / 2 + 1 / 512
        keypoints = torch.stack([keypoints0, keypoints1], 1)

        with torch.no_grad():
            outputs = self.default_model(**inputs, keypoints=keypoints)

        self.assertEqual(outputs.matches.shape, (1, 2, 200))
        expected_number_of_matches = Expectations({("cpu", None): 50}).get_expectation()
        expected_matching_scores = Expectations(
            {("cpu", None): [0.8587, 0.8015, 0.7063, 0.8438, 0.7524]}
        ).get_expectation()
        number_of_matches = (outputs.matches[0, 0] > -1).sum().item()
        self.assertTrue(abs(number_of_matches - expected_number_of_matches) < 4)
        torch.testing.assert_close(
            outputs.matching_scores[0, 0, [36, 48, 57, 58, 101]],
            torch.tensor(expected_matching_scores, device=torch_device),
            rtol=1e-3,
            atol=1e-3,
        )
