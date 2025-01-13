# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import List

from datasets import load_dataset

from transformers.models.superglue.configuration_superglue import SuperGlueConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import SuperGlueForKeypointMatching

if is_vision_available():
    from transformers import AutoImageProcessor


class SuperGlueModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_width=80,
        image_height=60,
        keypoint_detector_config=None,
        hidden_size: int = 64,
        keypoint_encoder_sizes: List[int] = [32, 64],
        gnn_layers_types: List[str] = ["self", "cross"] * 2,
        num_attention_heads: int = 4,
        sinkhorn_iterations: int = 100,
        matching_threshold: float = 0.2,
    ):
        if keypoint_detector_config is None:
            keypoint_detector_config = {
                "encoder_hidden_sizes": [32, 64],
                "decoder_hidden_size": 64,
                "keypoint_decoder_dim": 65,
                "descriptor_decoder_dim": 64,
                "keypoint_threshold": 0.005,
                "max_keypoints": 256,
                "nms_radius": 4,
                "border_removal_distance": 4,
            }
        self.parent = parent
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height

        self.keypoint_detector_config = keypoint_detector_config
        self.hidden_size = hidden_size
        self.keypoint_encoder_sizes = keypoint_encoder_sizes
        self.gnn_layers_types = gnn_layers_types
        self.num_attention_heads = num_attention_heads
        self.sinkhorn_iterations = sinkhorn_iterations
        self.matching_threshold = matching_threshold

    def prepare_config_and_inputs(self):
        # SuperGlue expects a grayscale image as input
        pixel_values = floats_tensor([self.batch_size, 2, 3, self.image_height, self.image_width])
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        return SuperGlueConfig(
            keypoint_detector_config=self.keypoint_detector_config,
            hidden_size=self.hidden_size,
            keypoint_encoder_sizes=self.keypoint_encoder_sizes,
            gnn_layers_types=self.gnn_layers_types,
            num_attention_heads=self.num_attention_heads,
            sinkhorn_iterations=self.sinkhorn_iterations,
            matching_threshold=self.matching_threshold,
        )

    def create_and_check_model(self, config, pixel_values):
        model = SuperGlueForKeypointMatching(config=config)
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

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class SuperGlueModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (SuperGlueForKeypointMatching,) if is_torch_available() else ()
    all_generative_model_classes = () if is_torch_available() else ()

    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = True

    def setUp(self):
        self.model_tester = SuperGlueModelTester(self)
        self.config_tester = ConfigTester(self, config_class=SuperGlueConfig, has_text_modality=False, hidden_size=64)

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    @unittest.skip(reason="SuperGlueForKeypointMatching does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="SuperGlueForKeypointMatching does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="SuperGlueForKeypointMatching does not use feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="SuperGlueForKeypointMatching is not trainable")
    def test_training(self):
        pass

    @unittest.skip(reason="SuperGlueForKeypointMatching is not trainable")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="SuperGlueForKeypointMatching is not trainable")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="SuperGlueForKeypointMatching is not trainable")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="SuperGlue does not output any loss term in the forward pass")
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

            hidden_states_sizes = (
                self.model_tester.keypoint_encoder_sizes
                + [self.model_tester.hidden_size]
                + [self.model_tester.hidden_size, self.model_tester.hidden_size * 2]
                * len(self.model_tester.gnn_layers_types)
                + [self.model_tester.hidden_size] * 2
            )

            for i, hidden_states_size in enumerate(hidden_states_sizes):
                self.assertListEqual(
                    list(hidden_states[i].shape[-2:]),
                    [hidden_states_size, maximum_num_matches],
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

            expected_attention_shape = [
                self.model_tester.num_attention_heads,
                maximum_num_matches,
                maximum_num_matches,
            ]

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
        from_pretrained_ids = ["magic-leap-community/superglue_indoor", "magic-leap-community/superglue_outdoor"]
        for model_name in from_pretrained_ids:
            model = SuperGlueForKeypointMatching.from_pretrained(model_name)
            self.assertIsNotNone(model)

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

    def test_batching_equivalence(self):
        """
        Overwriting ModelTesterMixin.test_batching_equivalence since SuperGlue returns `matching_scores` tensors full of
        zeros which causes the test to fail, because cosine_similarity of two zero tensors is 0.
        Discussed here : https://github.com/huggingface/transformers/pull/29886#issuecomment-2481539481
        """

        def recursive_check(batched_object, single_row_object, model_name, key):
            if isinstance(batched_object, (list, tuple)):
                for batched_object_value, single_row_object_value in zip(batched_object, single_row_object):
                    recursive_check(batched_object_value, single_row_object_value, model_name, key)
            elif isinstance(batched_object, dict):
                for batched_object_value, single_row_object_value in zip(
                    batched_object.values(), single_row_object.values()
                ):
                    recursive_check(batched_object_value, single_row_object_value, model_name, key)
            # do not compare returned loss (0-dim tensor) / codebook ids (int) / caching objects
            elif batched_object is None or not isinstance(batched_object, torch.Tensor):
                return
            elif batched_object.dim() == 0:
                return
            else:
                # indexing the first element does not always work
                # e.g. models that output similarity scores of size (N, M) would need to index [0, 0]
                slice_ids = [slice(0, index) for index in single_row_object.shape]
                batched_row = batched_object[slice_ids]
                self.assertFalse(
                    torch.isnan(batched_row).any(), f"Batched output has `nan` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isinf(batched_row).any(), f"Batched output has `inf` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isnan(single_row_object).any(), f"Single row output has `nan` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isinf(single_row_object).any(), f"Single row output has `inf` in {model_name} for key={key}"
                )
                self.assertTrue(
                    (equivalence(batched_row, single_row_object)) <= 1e-03,
                    msg=(
                        f"Batched and Single row outputs are not equal in {model_name} for key={key}. "
                        f"Difference={equivalence(batched_row, single_row_object)}."
                    ),
                )

        def equivalence(tensor1, tensor2):
            return torch.max(torch.abs(tensor1 - tensor2))

        config, batched_input = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            config.output_hidden_states = True

            model_name = model_class.__name__
            batched_input_prepared = self._prepare_for_class(batched_input, model_class)
            model = model_class(config).to(torch_device).eval()

            batch_size = self.model_tester.batch_size
            single_row_input = {}
            for key, value in batched_input_prepared.items():
                if isinstance(value, torch.Tensor) and value.shape[0] % batch_size == 0:
                    # e.g. musicgen has inputs of size (bs*codebooks). in most cases value.shape[0] == batch_size
                    single_batch_shape = value.shape[0] // batch_size
                    single_row_input[key] = value[:single_batch_shape]
                else:
                    single_row_input[key] = value

            with torch.no_grad():
                model_batched_output = model(**batched_input_prepared)
                model_row_output = model(**single_row_input)

            if isinstance(model_batched_output, torch.Tensor):
                model_batched_output = {"model_output": model_batched_output}
                model_row_output = {"model_output": model_row_output}

            for key in model_batched_output:
                recursive_check(model_batched_output[key], model_row_output[key], model_name, key)


def prepare_imgs():
    dataset = load_dataset("hf-internal-testing/image-matching-test-dataset", split="train")
    image1 = dataset[0]["image"]
    image2 = dataset[1]["image"]
    image3 = dataset[2]["image"]
    return [[image1, image2], [image3, image2]]


@require_torch
@require_vision
class SuperGlueModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return (
            AutoImageProcessor.from_pretrained("magic-leap-community/superglue_outdoor")
            if is_vision_available()
            else None
        )

    @slow
    def test_inference(self):
        model = SuperGlueForKeypointMatching.from_pretrained("magic-leap-community/superglue_outdoor").to(torch_device)
        preprocessor = self.default_image_processor
        images = prepare_imgs()
        inputs = preprocessor(images=images, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

        predicted_number_of_matches = torch.sum(outputs.matches[0][0] != -1).item()
        predicted_matches_values = outputs.matches[0, 0, :30]
        predicted_matching_scores_values = outputs.matching_scores[0, 0, :20]

        expected_number_of_matches = 282
        expected_matches_values = torch.tensor([125,630,137,138,136,143,135,-1,-1,153,
                                                154,156,117,160,-1,149,147,152,168,-1,
                                                165,182,-1,190,187,188,189,112,-1,193],
                                                device=predicted_matches_values.device)  # fmt:skip
        expected_matching_scores_values = torch.tensor([0.9899,0.0033,0.9897,0.9889,0.9879,0.7464,0.7109,0.0,0.0,0.9841,
                                                        0.9889,0.9639,0.0114,0.9559,0.0,0.9735,0.8018,0.5190,0.9157,0.0],
                                                        device=predicted_matches_values.device)  # fmt:skip

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

        self.assertTrue(abs(predicted_number_of_matches - expected_number_of_matches) < 4)
        self.assertTrue(
            torch.sum(~torch.isclose(predicted_matching_scores_values, expected_matching_scores_values, atol=1e-2)) < 4
        )
        self.assertTrue(torch.sum(predicted_matches_values != expected_matches_values) < 4)
