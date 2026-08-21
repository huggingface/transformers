# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from transformers.models.loma.configuration_loma import LoMaConfig
from transformers.testing_utils import get_device_properties, require_torch, require_vision, slow, torch_device
from transformers.utils import is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import LoMaForKeypointMatching
    from transformers.models.loma.modeling_loma import (
        LoMaDescriptorNetwork,
        LoMaMatchAssignmentLayer,
        LoMaPositionalEncoder,
        LoMaTransformerLayer,
    )


class LoMaModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_width=80,
        image_height=60,
        keypoint_detector_config={
            "encoder_hidden_sizes": [32, 32, 64],
            "decoder_hidden_size": 64,
            "keypoint_decoder_dim": 65,
            "descriptor_decoder_dim": 64,
            "keypoint_threshold": 0.005,
            "max_keypoints": 256,
            "nms_radius": 4,
            "border_removal_distance": 4,
        },
        descriptor_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        filter_threshold: float = 0.1,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height

        self.keypoint_detector_config = keypoint_detector_config
        self.descriptor_dim = descriptor_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.filter_threshold = filter_threshold
        self.dinov2_dim = 32

    def prepare_config_and_inputs(self):
        # LoMa expects a grayscale image as input
        pixel_values = floats_tensor([self.batch_size, 2, 3, self.image_height, self.image_width])
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        return LoMaConfig(
            keypoint_detector_config=self.keypoint_detector_config,
            descriptor_dim=self.descriptor_dim,
            descriptor_hidden_blocks=1,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            filter_threshold=self.filter_threshold,
            dinov2_dim=self.dinov2_dim,
            attn_implementation="eager",
        )

    def create_and_check_model(self, config, pixel_values):
        model = LoMaForKeypointMatching(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        maximum_num_matches = result.mask.shape[-1]
        self.parent.assertEqual(
            result.keypoints.shape,
            (self.batch_size, 2, maximum_num_matches, 2),
        )
        self.parent.assertEqual(
            result.matches.shape,
            (self.batch_size, 2, maximum_num_matches),
        )
        self.parent.assertEqual(
            result.matching_scores.shape,
            (self.batch_size, 2, maximum_num_matches),
        )
        self.parent.assertEqual(
            result.prune.shape,
            (self.batch_size, 2, maximum_num_matches),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class LoMaModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (LoMaForKeypointMatching,) if is_torch_available() else ()
    all_generative_model_classes = () if is_torch_available() else ()

    test_resize_embeddings = False
    has_attentions = False
    test_torch_exportable = False  # keypoint matching has data-dependent top-k / non-max suppression

    def setUp(self):
        self.model_tester = LoMaModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LoMaConfig, has_text_modality=False, hidden_size=32)

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def test_config_matches_loma_architecture(self):
        config = LoMaConfig()

        self.assertEqual(config.input_descriptor_dim, 256)
        self.assertEqual(config.descriptor_dim, 256)
        self.assertEqual(config.attention_head_dim, 64)
        self.assertEqual(config.num_attention_heads, 4)
        self.assertEqual(config.num_hidden_layers, 9)
        self.assertEqual(config.filter_threshold, 0.1)

        for attribute in (
            "attention_dropout",
            "depth_confidence",
            "hidden_act",
            "num_key_value_heads",
            "width_confidence",
        ):
            self.assertFalse(hasattr(config, attribute))

    def test_config_derives_attention_heads(self):
        config = LoMaConfig(descriptor_dim=512)
        self.assertEqual(config.num_attention_heads, 8)

        with self.assertRaisesRegex(ValueError, "attention_head_dim"):
            LoMaConfig(descriptor_dim=250)

    def test_descriptor_network(self):
        config = LoMaConfig(input_descriptor_dim=16, descriptor_hidden_blocks=1)
        model = LoMaDescriptorNetwork(config).to(torch_device)
        pixel_values = floats_tensor([2, 3, 32, 48]).to(torch_device)
        keypoints = torch.tensor(
            [[[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]], [[-0.25, 0.25], [0.25, -0.25], [0.0, 0.0]]],
            device=torch_device,
        )

        descriptor_grid = model(pixel_values)
        descriptors = model.describe_keypoints(pixel_values, keypoints)

        self.assertEqual(descriptor_grid.shape, (2, 16, 32, 48))
        self.assertEqual(descriptors.shape, (2, 3, 16))

    def test_descriptor_network_accepts_grayscale_images(self):
        config = LoMaConfig(input_descriptor_dim=16, descriptor_hidden_blocks=1)
        model = LoMaDescriptorNetwork(config).to(torch_device)
        pixel_values = floats_tensor([2, 1, 32, 48]).to(torch_device)
        keypoints = torch.tensor(
            [[[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]], [[-0.25, 0.25], [0.25, -0.25], [0.0, 0.0]]],
            device=torch_device,
        )

        descriptors = model.describe_keypoints(pixel_values, keypoints)

        self.assertEqual(descriptors.shape, (2, 3, 16))

    def test_matching_transformer(self):
        config = LoMaConfig(descriptor_dim=64, num_attention_heads=4, num_hidden_layers=2)
        positional_encoder = LoMaPositionalEncoder(config).to(torch_device)
        transformer_layer = LoMaTransformerLayer(config, layer_idx=0).to(torch_device)
        match_assignment = LoMaMatchAssignmentLayer(config).to(torch_device).eval()
        descriptors_0 = floats_tensor([2, 5, 64]).to(torch_device)
        descriptors_1 = floats_tensor([2, 7, 64]).to(torch_device)
        keypoints_0 = floats_tensor([2, 5, 2]).to(torch_device)
        keypoints_1 = floats_tensor([2, 7, 2]).to(torch_device)

        output_0, output_1 = transformer_layer(
            descriptors_0, descriptors_1, positional_encoder(keypoints_0), positional_encoder(keypoints_1)
        )
        scores = match_assignment(output_0, output_1)

        self.assertEqual(output_0.shape, (2, 5, 64))
        self.assertEqual(output_1.shape, (2, 7, 64))
        self.assertEqual(scores.shape, (2, 5, 7))
        self.assertTrue(torch.all((scores >= 0) & (scores <= 1)))

    def test_batching_equivalence(self, atol=1e-5, rtol=1e-5):
        device_properties = get_device_properties()
        if device_properties[0] == "cuda" and device_properties[1] == 8:
            # TODO: (ydshieh) fix this
            self.skipTest(reason="After switching to A10, this test always fails, but pass on CPU or T4.")
        super().test_batching_equivalence(atol=atol, rtol=rtol)

    @unittest.skip(reason="LoMa includes a VGG-19 descriptor network and is not a small model")
    def test_model_is_small(self):
        pass

    @unittest.skip(reason="LoMaForKeypointMatching does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="LoMaForKeypointMatching does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="LoMaForKeypointMatching does not use feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

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

    @unittest.skip(reason="LoMa does not output any loss term in the forward pass")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states
            maximum_num_matches = outputs.mask.shape[-1]

            self.assertEqual(len(hidden_states), self.model_tester.num_layers)
            for hidden_state in hidden_states:
                self.assertListEqual(
                    list(hidden_state.shape),
                    [self.model_tester.batch_size, 2, maximum_num_matches, self.model_tester.descriptor_dim],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    @unittest.skip(reason="LoMa uses scaled dot-product attention without exposing attention weights")
    def test_attention_outputs(self):
        def check_attention_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            attentions = outputs.attentions
            maximum_num_matches = outputs.mask.shape[-1]

            expected_attention_shape = [self.model_tester.num_heads, maximum_num_matches, maximum_num_matches]

            for i, attention in enumerate(attentions):
                self.assertListEqual(
                    list(attention.shape[-3:]),
                    expected_attention_shape,
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            check_attention_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True

            check_attention_output(inputs_dict, config, model_class)

    # Copied from tests.models.superglue.test_modeling_superglue.SuperGlueModelTest.test_forward_labels_should_be_none
    def test_forward_labels_should_be_none(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                model_inputs = self._prepare_for_class(inputs_dict, model_class)
                # Provide an arbitrary sized Tensor as labels to model inputs
                model_inputs["labels"] = torch.rand((128, 128))

                with self.assertRaises(ValueError) as cm:
                    model(**model_inputs)
                self.assertEqual(ValueError, cm.exception.__class__)


@require_torch
@require_vision
class LoMaModelIntegrationTest(unittest.TestCase):
    checkpoint_id = "Falcon7211/loma-b"

    @slow
    def test_matcher_matches_reference_output(self):
        """Check the converted matcher against outputs from the official LoMa-B checkpoint."""
        model = LoMaForKeypointMatching.from_pretrained(self.checkpoint_id).to(torch_device)
        model.eval()

        keypoints_0 = torch.tensor([[[-0.8, -0.6], [-0.3, 0.1], [0.2, -0.4], [0.7, 0.5]]], device=torch_device)
        keypoints_1 = torch.tensor([[[-0.7, 0.4], [-0.1, -0.2], [0.4, 0.3], [0.8, -0.5]]], device=torch_device)
        descriptors = torch.arange(2048, dtype=torch.float32, device=torch_device).reshape(1, 2, 4, 256) / 1024
        mask = torch.ones(1, 4, dtype=torch.bool, device=torch_device)

        with torch.no_grad():
            descriptors_0 = model.input_projection(descriptors[:, 0])
            descriptors_1 = model.input_projection(descriptors[:, 1])
            position_embeddings_0 = model.positional_encoder(keypoints_0)
            position_embeddings_1 = model.positional_encoder(keypoints_1)
            for layer in model.transformer_layers:
                descriptors_0, descriptors_1 = layer(
                    descriptors_0, descriptors_1, position_embeddings_0, position_embeddings_1
                )
            scores = model.match_assignment(descriptors_0, descriptors_1, mask, mask)

        # Values generated with davnords/LoMa's LoMaB matcher and its official loma_B.pt checkpoint.
        expected_scores = torch.tensor(
            [
                [
                    [0.0205734055, 0.0025816434, 0.0005758349, 0.0000064961],
                    [0.1651729643, 0.0041017337, 0.0011886628, 0.0000239026],
                    [0.5511550903, 0.0038553919, 0.0015511342, 0.0000650399],
                    [0.0011463700, 0.1915852278, 0.2759611607, 0.4865026772],
                ]
            ],
            device=torch_device,
        )
        torch.testing.assert_close(scores, expected_scores, rtol=1e-4, atol=1e-5)

    @slow
    def test_inference(self):
        """Test LoMa inference loads from Hub and produces valid outputs."""
        model = LoMaForKeypointMatching.from_pretrained(self.checkpoint_id).to(torch_device)
        model.eval()

        torch.manual_seed(0)
        pixel_values = torch.rand(1, 2, 3, 120, 160, device=torch_device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        # Verify output structure
        self.assertIsNotNone(outputs.matches)
        self.assertIsNotNone(outputs.matching_scores)

        # Verify shapes: [batch_size, num_pairs, num_keypoints]
        self.assertEqual(outputs.matches.dim(), 3)
        self.assertEqual(outputs.matching_scores.dim(), 3)
        self.assertEqual(outputs.matches.shape[0], 1)  # batch_size
        self.assertEqual(outputs.matches.shape[1], 2)  # num_pairs (one per image)
        self.assertEqual(outputs.matches.shape, outputs.matching_scores.shape)

        # Verify dtypes
        self.assertEqual(outputs.matches.dtype, torch.int64)
        self.assertEqual(outputs.matching_scores.dtype, torch.float32)

        # Verify value ranges
        self.assertTrue((outputs.matching_scores >= 0).all())
        self.assertTrue((outputs.matching_scores <= 1).all())
        self.assertTrue((outputs.matches >= -1).all())

    @slow
    def test_inference_with_keypoints(self):
        """Test LoMa inference with pre-computed keypoints."""
        model = LoMaForKeypointMatching.from_pretrained(self.checkpoint_id).to(torch_device)
        model.eval()

        torch.manual_seed(0)
        num_kp = 32
        pixel_values = torch.rand(1, 2, 3, 120, 160, device=torch_device)
        keypoints0 = torch.rand(1, num_kp, 2, device=torch_device) * 2 - 1
        keypoints1 = torch.rand(1, num_kp, 2, device=torch_device) * 2 - 1
        keypoints = torch.stack([keypoints0, keypoints1], dim=1)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, keypoints=keypoints)

        # With pre-computed keypoints, num_keypoints should match input
        self.assertEqual(outputs.matches.shape[-1], num_kp)
        self.assertEqual(outputs.matching_scores.shape[-1], num_kp)

    @slow
    def test_inference_batched(self):
        """Test LoMa inference with batched image pairs."""
        model = LoMaForKeypointMatching.from_pretrained(self.checkpoint_id).to(torch_device)
        model.eval()

        torch.manual_seed(0)
        batch_size = 2
        pixel_values = torch.rand(batch_size, 2, 3, 120, 160, device=torch_device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        self.assertEqual(outputs.matches.shape[0], batch_size)
        self.assertEqual(outputs.matching_scores.shape[0], batch_size)
