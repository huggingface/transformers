# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from datasets import load_dataset

from transformers.models.lightglue.configuration_lightglue import LightGlueConfig
from transformers.testing_utils import get_device_properties, require_torch, require_vision, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import LightGlueForKeypointMatching

if is_vision_available():
    from transformers import AutoImageProcessor


class LightGlueModelTester:
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
        depth_confidence: float = 1.0,
        width_confidence: float = 1.0,
        filter_threshold: float = 0.1,
        matching_threshold: float = 0.0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height

        self.keypoint_detector_config = keypoint_detector_config
        self.descriptor_dim = descriptor_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.depth_confidence = depth_confidence
        self.width_confidence = width_confidence
        self.filter_threshold = filter_threshold
        self.matching_threshold = matching_threshold

    def prepare_config_and_inputs(self):
        # LightGlue expects a grayscale image as input
        pixel_values = floats_tensor([self.batch_size, 2, 3, self.image_height, self.image_width])
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        return LightGlueConfig(
            keypoint_detector_config=self.keypoint_detector_config,
            descriptor_dim=self.descriptor_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            depth_confidence=self.depth_confidence,
            width_confidence=self.width_confidence,
            filter_threshold=self.filter_threshold,
            matching_threshold=self.matching_threshold,
            attn_implementation="eager",
        )

    def create_and_check_model(self, config, pixel_values):
        model = LightGlueForKeypointMatching(config=config)
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
class LightGlueModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (LightGlueForKeypointMatching,) if is_torch_available() else ()
    all_generative_model_classes = () if is_torch_available() else ()

    test_resize_embeddings = False
    has_attentions = True

    def setUp(self):
        self.model_tester = LightGlueModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LightGlueConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def test_batching_equivalence(self, atol=1e-5, rtol=1e-5):
        device_properties = get_device_properties()
        if device_properties[0] == "cuda" and device_properties[1] == 8:
            # TODO: (ydshieh) fix this
            self.skipTest(reason="After switching to A10, this test always fails, but pass on CPU or T4.")
        super().test_batching_equivalence(atol=atol, rtol=rtol)

    @unittest.skip(reason="LightGlueForKeypointMatching does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="LightGlueForKeypointMatching does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="LightGlueForKeypointMatching does not use feedforward chunking")
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

    @unittest.skip(reason="LightGlue does not output any loss term in the forward pass")
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

            hidden_states_sizes = [
                self.model_tester.descriptor_dim,
                self.model_tester.descriptor_dim,
                self.model_tester.descriptor_dim * 2,
                self.model_tester.descriptor_dim,
                self.model_tester.descriptor_dim,
                self.model_tester.descriptor_dim * 2,
                self.model_tester.descriptor_dim,
            ] * self.model_tester.num_layers

            for i, hidden_states_size in enumerate(hidden_states_sizes):
                self.assertListEqual(
                    list(hidden_states[i].shape[-2:]),
                    [maximum_num_matches, hidden_states_size],
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

    @slow
    def test_model_from_pretrained(self):
        from_pretrained_ids = ["ETH-CVG/lightglue_superpoint"]
        for model_name in from_pretrained_ids:
            model = LightGlueForKeypointMatching.from_pretrained(model_name)
            self.assertIsNotNone(model)

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


def prepare_imgs():
    dataset = load_dataset("hf-internal-testing/image-matching-test-dataset", split="train")
    image0 = dataset[0]["image"]
    image1 = dataset[1]["image"]
    image2 = dataset[2]["image"]
    # [image1, image1] on purpose to test the model early stopping
    return [[image2, image0], [image1, image1]]


@require_torch
@require_vision
class LightGlueModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return AutoImageProcessor.from_pretrained("ETH-CVG/lightglue_superpoint") if is_vision_available() else None

    @slow
    def test_inference(self):
        model = LightGlueForKeypointMatching.from_pretrained(
            "ETH-CVG/lightglue_superpoint", attn_implementation="eager"
        ).to(torch_device)
        preprocessor = self.default_image_processor
        images = prepare_imgs()
        inputs = preprocessor(images=images, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

        predicted_number_of_matches0 = torch.sum(outputs.matches[0][0] != -1).item()
        predicted_matches_values0 = outputs.matches[0, 0, 10:30]
        predicted_matching_scores_values0 = outputs.matching_scores[0, 0, 10:30]

        predicted_number_of_matches1 = torch.sum(outputs.matches[1][0] != -1).item()
        predicted_matches_values1 = outputs.matches[1, 0, 10:30]
        predicted_matching_scores_values1 = outputs.matching_scores[1, 0, 10:30]

        expected_number_of_matches0 = 866
        expected_matches_values0 = torch.tensor(
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            dtype=torch.int64,
            device=torch_device,
        )
        expected_matching_scores_values0 = torch.tensor(
            [
                0.6188,0.7817,0.5686,0.9353,0.9801,0.9193,0.8632,0.9111,0.9821,0.5496,
                0.9906,0.8682,0.9679,0.9914,0.9318,0.1910,0.9669,0.3240,0.9971,0.9923,
            ],
            device=torch_device
        )  # fmt:skip

        expected_number_of_matches1 = 140
        expected_matches_values1 = torch.tensor(
            [14, -1, -1, 15, 17, 13, -1, -1, -1, -1, -1, -1, 5, -1, -1, 19, -1, 10, -1, 11],
            dtype=torch.int64,
            device=torch_device,
        )
        expected_matching_scores_values1 = torch.tensor(
            [0.3796, 0, 0, 0.3772, 0.4439, 0.2411, 0, 0, 0.0032, 0, 0, 0, 0.2997, 0, 0, 0.6762, 0, 0.8826, 0, 0.5583],
            device=torch_device,
        )

        # expected_early_stopping_layer = 2
        # predicted_early_stopping_layer = torch.max(outputs.prune[1]).item()
        # self.assertEqual(predicted_early_stopping_layer, expected_early_stopping_layer)
        # self.assertEqual(predicted_number_of_matches, expected_second_number_of_matches)

        """
        Because of inconsistencies introduced between CUDA versions, the checks here are less strict. SuperGlue relies
        on SuperPoint, which may, depending on CUDA version, return different number of keypoints (866 or 867 in this
        specific test example). The consequence of having different number of keypoints is that the number of matches
        will also be different. In the 20 first matches being checked, having one keypoint less will result in 1 less
        match. The matching scores will also be different, as the keypoints are different. The checks here are less
        strict to account for these inconsistencies.
        Therefore, the test checks that the predicted number of matches, matches and matching scores are close to the
        expected values, individually. Here, the tolerance of the number of values changing is set to 2.

        This was discussed [here](https://github.com/huggingface/transformers/pull/29886#issuecomment-2482752787)
        Such CUDA inconsistencies can be found
        [here](https://github.com/huggingface/transformers/pull/33200/files#r1785980300)
        """
        self.assertTrue(abs(predicted_number_of_matches0 - expected_number_of_matches0) < 4)
        self.assertTrue(abs(predicted_number_of_matches1 - expected_number_of_matches1) < 4)
        self.assertTrue(
            torch.sum(~torch.isclose(predicted_matching_scores_values0, expected_matching_scores_values0, atol=1e-2))
            < 4
        )
        self.assertTrue(
            torch.sum(~torch.isclose(predicted_matching_scores_values1, expected_matching_scores_values1, atol=1e-2))
            < 4
        )
        self.assertTrue(torch.sum(predicted_matches_values0 != expected_matches_values0) < 4)
        self.assertTrue(torch.sum(predicted_matches_values1 != expected_matches_values1) < 4)

    @slow
    def test_inference_without_early_stop(self):
        model = LightGlueForKeypointMatching.from_pretrained(
            "ETH-CVG/lightglue_superpoint", attn_implementation="eager", depth_confidence=1.0
        ).to(torch_device)
        preprocessor = self.default_image_processor
        images = prepare_imgs()
        inputs = preprocessor(images=images, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

        predicted_number_of_matches0 = torch.sum(outputs.matches[0][0] != -1).item()
        predicted_matches_values0 = outputs.matches[0, 0, 10:30]
        predicted_matching_scores_values0 = outputs.matching_scores[0, 0, 10:30]

        predicted_number_of_matches1 = torch.sum(outputs.matches[1][0] != -1).item()
        predicted_matches_values1 = outputs.matches[1, 0, 10:30]
        predicted_matching_scores_values1 = outputs.matching_scores[1, 0, 10:30]

        expected_number_of_matches0 = 134
        expected_matches_values0 = torch.tensor(
            [-1, -1, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 19, -1, 10, -1, 11], dtype=torch.int64
        ).to(torch_device)
        expected_matching_scores_values0 = torch.tensor(
            [0.0083, 0, 0.2022, 0.0621, 0, 0.0828, 0, 0, 0.0003, 0, 0, 0, 0.0960, 0, 0, 0.6940, 0, 0.7167, 0, 0.1512]
        ).to(torch_device)

        expected_number_of_matches1 = 862
        expected_matches_values1 = torch.tensor(
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], dtype=torch.int64
        ).to(torch_device)
        expected_matching_scores_values1 = torch.tensor(
            [
                0.4772,
                0.3781,
                0.0631,
                0.9559,
                0.8746,
                0.9271,
                0.4882,
                0.5406,
                0.9439,
                0.1526,
                0.5028,
                0.4107,
                0.5591,
                0.9130,
                0.7572,
                0.0302,
                0.4532,
                0.0893,
                0.9490,
                0.4880,
            ]
        ).to(torch_device)

        # expected_early_stopping_layer = 2
        # predicted_early_stopping_layer = torch.max(outputs.prune[1]).item()
        # self.assertEqual(predicted_early_stopping_layer, expected_early_stopping_layer)
        # self.assertEqual(predicted_number_of_matches, expected_second_number_of_matches)

        """
        Because of inconsistencies introduced between CUDA versions, the checks here are less strict. SuperGlue relies
        on SuperPoint, which may, depending on CUDA version, return different number of keypoints (866 or 867 in this
        specific test example). The consequence of having different number of keypoints is that the number of matches
        will also be different. In the 20 first matches being checked, having one keypoint less will result in 1 less
        match. The matching scores will also be different, as the keypoints are different. The checks here are less
        strict to account for these inconsistencies.
        Therefore, the test checks that the predicted number of matches, matches and matching scores are close to the
        expected values, individually. Here, the tolerance of the number of values changing is set to 2.

        This was discussed [here](https://github.com/huggingface/transformers/pull/29886#issuecomment-2482752787)
        Such CUDA inconsistencies can be found
        [here](https://github.com/huggingface/transformers/pull/33200/files#r1785980300)
        """

        self.assertTrue(abs(predicted_number_of_matches0 - expected_number_of_matches0) < 4)
        self.assertTrue(abs(predicted_number_of_matches1 - expected_number_of_matches1) < 4)
        self.assertTrue(
            torch.sum(~torch.isclose(predicted_matching_scores_values0, expected_matching_scores_values0, atol=1e-2))
            < 4
        )
        self.assertTrue(
            torch.sum(~torch.isclose(predicted_matching_scores_values1, expected_matching_scores_values1, atol=1e-2))
            < 4
        )
        self.assertTrue(torch.sum(predicted_matches_values0 != expected_matches_values0) < 4)
        self.assertTrue(torch.sum(predicted_matches_values1 != expected_matches_values1) < 4)

    @slow
    def test_inference_without_early_stop_and_keypoint_pruning(self):
        model = LightGlueForKeypointMatching.from_pretrained(
            "ETH-CVG/lightglue_superpoint",
            attn_implementation="eager",
            depth_confidence=1.0,
            width_confidence=1.0,
        ).to(torch_device)
        preprocessor = self.default_image_processor
        images = prepare_imgs()
        inputs = preprocessor(images=images, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

        predicted_number_of_matches0 = torch.sum(outputs.matches[0][0] != -1).item()
        predicted_matches_values0 = outputs.matches[0, 0, 10:30]
        predicted_matching_scores_values0 = outputs.matching_scores[0, 0, 10:30]

        predicted_number_of_matches1 = torch.sum(outputs.matches[1][0] != -1).item()
        predicted_matches_values1 = outputs.matches[1, 0, 10:30]
        predicted_matching_scores_values1 = outputs.matching_scores[1, 0, 10:30]

        expected_number_of_matches0 = 143
        expected_matches_values0 = torch.tensor(
            [-1, -1, -1, -1, 17, 13, -1, -1, -1, -1, -1, -1, 5, -1, -1, 19, -1, 10, -1, 11], dtype=torch.int64
        ).to(torch_device)
        # fmt: off
        expected_matching_scores_values0 = torch.tensor(
            [0.0696, 0.0283, 0.0000, 0.0863, 0.2834, 0.2308, 0.0000, 0.0000, 0.0189, 0.0000, 0.0000, 0.0000, 0.1792, 0.0000, 0.0000, 0.8197, 0.0000, 0.8194, 0.0000, 0.3058]
        ).to(torch_device)
        # fmt: on

        expected_number_of_matches1 = 862
        expected_matches_values1 = torch.tensor(
            [10, 11, -1, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, -1, 26, -1, 28, 29], dtype=torch.int64
        ).to(torch_device)
        # fmt: off
        expected_matching_scores_values1 = torch.tensor(
            [0.4744, 0.3749, 0.0628, 0.9572, 0.8744, 0.9277, 0.4843, 0.5365, 0.9441, 0.1519, 0.5004, 0.4058, 0.5569, 0.9113, 0.7525, 0.0301, 0.4510, 0.0892, 0.9483, 0.4815]
        ).to(torch_device)
        # fmt: on

        # expected_early_stopping_layer = 2
        # predicted_early_stopping_layer = torch.max(outputs.prune[1]).item()
        # self.assertEqual(predicted_early_stopping_layer, expected_early_stopping_layer)
        # self.assertEqual(predicted_number_of_matches, expected_second_number_of_matches)

        """
        Because of inconsistencies introduced between CUDA versions, the checks here are less strict. SuperGlue relies
        on SuperPoint, which may, depending on CUDA version, return different number of keypoints (866 or 867 in this
        specific test example). The consequence of having different number of keypoints is that the number of matches
        will also be different. In the 20 first matches being checked, having one keypoint less will result in 1 less
        match. The matching scores will also be different, as the keypoints are different. The checks here are less
        strict to account for these inconsistencies.
        Therefore, the test checks that the predicted number of matches, matches and matching scores are close to the
        expected values, individually. Here, the tolerance of the number of values changing is set to 2.

        This was discussed [here](https://github.com/huggingface/transformers/pull/29886#issuecomment-2482752787)
        Such CUDA inconsistencies can be found
        [here](https://github.com/huggingface/transformers/pull/33200/files#r1785980300)
        """

        self.assertTrue(abs(predicted_number_of_matches0 - expected_number_of_matches0) < 4)
        self.assertTrue(abs(predicted_number_of_matches1 - expected_number_of_matches1) < 4)
        self.assertTrue(
            torch.sum(~torch.isclose(predicted_matching_scores_values0, expected_matching_scores_values0, atol=1e-2))
            < 4
        )
        self.assertTrue(
            torch.sum(~torch.isclose(predicted_matching_scores_values1, expected_matching_scores_values1, atol=1e-2))
            < 4
        )
        self.assertTrue(torch.sum(predicted_matches_values0 != expected_matches_values0) < 4)
        self.assertTrue(torch.sum(predicted_matches_values1 != expected_matches_values1) < 4)

    @slow
    def test_inference_order_with_early_stop(self):
        model = LightGlueForKeypointMatching.from_pretrained(
            "ETH-CVG/lightglue_superpoint", attn_implementation="eager"
        ).to(torch_device)
        preprocessor = self.default_image_processor
        images = prepare_imgs()
        # [[image2, image0], [image1, image1]] -> [[image2, image0], [image2, image0], [image1, image1]]
        images = [images[0]] + images  # adding a 3rd pair to test batching with early stopping
        inputs = preprocessor(images=images, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

        predicted_number_of_matches_pair0 = torch.sum(outputs.matches[0][0] != -1).item()
        predicted_number_of_matches_pair1 = torch.sum(outputs.matches[1][0] != -1).item()
        predicted_number_of_matches_pair2 = torch.sum(outputs.matches[2][0] != -1).item()

        # pair 0 and 1 are the same, so should have the same number of matches
        # pair 2 is [image1, image1] so should have more matches than first two pairs
        # This ensures that early stopping does not affect the order of the outputs
        # See : https://huggingface.co/ETH-CVG/lightglue_superpoint/discussions/6
        # The bug made the pairs switch order when early stopping was activated
        self.assertTrue(predicted_number_of_matches_pair0 == predicted_number_of_matches_pair1)
        self.assertTrue(predicted_number_of_matches_pair0 < predicted_number_of_matches_pair2)
