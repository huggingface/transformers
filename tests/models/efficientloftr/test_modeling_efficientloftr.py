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
from functools import reduce

from datasets import load_dataset

from transformers.models.efficientloftr import EfficientLoFTRConfig, EfficientLoFTRModel
from transformers.testing_utils import (
    require_torch,
    require_vision,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    set_model_tester_for_less_flaky_test,
    slow,
    torch_device,
)
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import EfficientLoFTRForKeypointMatching

if is_vision_available():
    from transformers import AutoImageProcessor


class EfficientLoFTRModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_width=80,
        image_height=60,
        stage_num_blocks: list[int] = [1, 1, 1],
        out_features: list[int] = [32, 32, 128],
        stage_stride: list[int] = [2, 1, 2],
        q_aggregation_kernel_size: int = 1,
        kv_aggregation_kernel_size: int = 1,
        q_aggregation_stride: int = 1,
        kv_aggregation_stride: int = 1,
        num_attention_layers: int = 2,
        num_attention_heads: int = 8,
        hidden_size: int = 128,
        coarse_matching_threshold: float = 0.0,
        fine_kernel_size: int = 2,
        coarse_matching_border_removal: int = 0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height

        self.stage_num_blocks = stage_num_blocks
        self.out_features = out_features
        self.stage_stride = stage_stride
        self.q_aggregation_kernel_size = q_aggregation_kernel_size
        self.kv_aggregation_kernel_size = kv_aggregation_kernel_size
        self.q_aggregation_stride = q_aggregation_stride
        self.kv_aggregation_stride = kv_aggregation_stride
        self.num_attention_layers = num_attention_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.coarse_matching_threshold = coarse_matching_threshold
        self.coarse_matching_border_removal = coarse_matching_border_removal
        self.fine_kernel_size = fine_kernel_size

    def prepare_config_and_inputs(self):
        # EfficientLoFTR expects a grayscale image as input
        pixel_values = floats_tensor([self.batch_size, 2, 3, self.image_height, self.image_width])
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        return EfficientLoFTRConfig(
            stage_num_blocks=self.stage_num_blocks,
            out_features=self.out_features,
            stage_stride=self.stage_stride,
            q_aggregation_kernel_size=self.q_aggregation_kernel_size,
            kv_aggregation_kernel_size=self.kv_aggregation_kernel_size,
            q_aggregation_stride=self.q_aggregation_stride,
            kv_aggregation_stride=self.kv_aggregation_stride,
            num_attention_layers=self.num_attention_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_size=self.hidden_size,
            coarse_matching_threshold=self.coarse_matching_threshold,
            coarse_matching_border_removal=self.coarse_matching_border_removal,
            fine_kernel_size=self.fine_kernel_size,
        )

    def create_and_check_model(self, config, pixel_values):
        model = EfficientLoFTRForKeypointMatching(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        maximum_num_matches = result.matches.shape[-1]
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
class EfficientLoFTRModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (EfficientLoFTRForKeypointMatching, EfficientLoFTRModel) if is_torch_available() else ()

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = True

    def setUp(self):
        self.model_tester = EfficientLoFTRModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EfficientLoFTRConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    @unittest.skip(reason="EfficientLoFTRForKeypointMatching does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="EfficientLoFTRForKeypointMatching does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="EfficientLoFTRForKeypointMatching does not use feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="EfficientLoFTRForKeypointMatching is not trainable")
    def test_training(self):
        pass

    @unittest.skip(reason="EfficientLoFTRForKeypointMatching is not trainable")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="EfficientLoFTRForKeypointMatching is not trainable")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="EfficientLoFTRForKeypointMatching is not trainable")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="EfficientLoFTR does not output any loss term in the forward pass")
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

            expected_num_hidden_states = len(self.model_tester.stage_num_blocks) + 1
            self.assertEqual(len(hidden_states), expected_num_hidden_states)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [self.model_tester.image_height, self.model_tester.image_width],
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
            total_stride = reduce(lambda a, b: a * b, config.stage_stride)
            hidden_size = (
                self.model_tester.image_height // total_stride * self.model_tester.image_width // total_stride
            )

            expected_attention_shape = [
                self.model_tester.num_attention_heads,
                hidden_size,
                hidden_size,
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
        from_pretrained_ids = ["zju-community/efficientloftr"]
        for model_name in from_pretrained_ids:
            model = EfficientLoFTRForKeypointMatching.from_pretrained(model_name)
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

    def test_batching_equivalence(self, atol=1e-5, rtol=1e-5):
        """
        This test is overwritten because the model outputs do not contain only regressive values but also keypoint
        locations.
        Similarly to the problem discussed about SuperGlue implementation
        [here](https://github.com/huggingface/transformers/pull/29886#issuecomment-2482752787), the consequence of
        having different scores for matching, makes the maximum indices differ. These indices are being used to compute
        the keypoint coordinates. The keypoint coordinates, in the model outputs, are floating point tensors, so the
        original implementation of this test cover this case. But the resulting tensors may have differences exceeding
        the relative and absolute tolerance.
        Therefore, similarly to SuperGlue integration test, for the key "keypoints" in the model outputs, we check the
        number of differences in keypoint coordinates being less than a TODO given number
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
            # do not compare int or bool outputs as they are mostly computed with max/argmax/topk methods which are
            # very sensitive to the inputs (e.g. tiny differences may give totally different results)
            elif not torch.is_floating_point(batched_object):
                return
            else:
                # indexing the first element does not always work
                # e.g. models that output similarity scores of size (N, M) would need to index [0, 0]
                slice_ids = [slice(0, index) for index in single_row_object.shape]
                batched_row = batched_object[slice_ids]
                if key == "keypoints":
                    batched_row = torch.sum(batched_row, dim=-1)
                    single_row_object = torch.sum(single_row_object, dim=-1)
                    tolerance = 0.02 * single_row_object.shape[-1]
                    self.assertTrue(
                        torch.sum(~torch.isclose(batched_row, single_row_object, rtol=rtol, atol=atol)) < tolerance
                    )
                else:
                    self.assertFalse(
                        torch.isnan(batched_row).any(), f"Batched output has `nan` in {model_name} for key={key}"
                    )
                    self.assertFalse(
                        torch.isinf(batched_row).any(), f"Batched output has `inf` in {model_name} for key={key}"
                    )
                    self.assertFalse(
                        torch.isnan(single_row_object).any(),
                        f"Single row output has `nan` in {model_name} for key={key}",
                    )
                    self.assertFalse(
                        torch.isinf(single_row_object).any(),
                        f"Single row output has `inf` in {model_name} for key={key}",
                    )
                    try:
                        torch.testing.assert_close(batched_row, single_row_object, atol=atol, rtol=rtol)
                    except AssertionError as e:
                        msg = f"Batched and Single row outputs are not equal in {model_name} for key={key}.\n\n"
                        msg += str(e)
                        raise AssertionError(msg)

        set_model_tester_for_less_flaky_test(self)

        config, batched_input = self.model_tester.prepare_config_and_inputs_for_common()
        set_config_for_less_flaky_test(config)

        for model_class in self.all_model_classes:
            config.output_hidden_states = True

            model_name = model_class.__name__
            if hasattr(self.model_tester, "prepare_config_and_inputs_for_model_class"):
                config, batched_input = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
            batched_input_prepared = self._prepare_for_class(batched_input, model_class)
            model = model_class(config).to(torch_device).eval()
            set_model_for_less_flaky_test(model)

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
                # DETR starts from zero-init queries to decoder, leading to cos_similarity = `nan`
                if hasattr(self, "zero_init_hidden_state") and "decoder_hidden_states" in key:
                    model_batched_output[key] = model_batched_output[key][1:]
                    model_row_output[key] = model_row_output[key][1:]
                recursive_check(model_batched_output[key], model_row_output[key], model_name, key)


def prepare_imgs():
    dataset = load_dataset("hf-internal-testing/image-matching-test-dataset", split="train")
    image1 = dataset[0]["image"]
    image2 = dataset[1]["image"]
    image3 = dataset[2]["image"]
    return [[image1, image2], [image3, image2]]


@require_torch
@require_vision
class EfficientLoFTRModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return AutoImageProcessor.from_pretrained("zju-community/efficientloftr") if is_vision_available() else None

    @slow
    def test_inference(self):
        model = EfficientLoFTRForKeypointMatching.from_pretrained(
            "zju-community/efficientloftr", attn_implementation="eager"
        ).to(torch_device)
        preprocessor = self.default_image_processor
        images = prepare_imgs()
        inputs = preprocessor(images=images, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

        predicted_top10 = torch.topk(outputs.matching_scores[0, 0], k=10)
        predicted_top10_matches_indices = predicted_top10.indices
        predicted_top10_matching_scores = predicted_top10.values

        expected_number_of_matches = 4800
        expected_matches_shape = torch.Size((len(images), 2, expected_number_of_matches))
        expected_matching_scores_shape = torch.Size((len(images), 2, expected_number_of_matches))

        expected_top10_matches_indices = torch.tensor(
            [3145, 3065, 3143, 3144, 1397, 1705, 3151, 2422, 3066, 2342], dtype=torch.int64, device=torch_device
        )
        expected_top10_matching_scores = torch.tensor(
            [0.9998, 0.9997, 0.9997, 0.9996, 0.9996, 0.9996, 0.9996, 0.9995, 0.9995, 0.9995], device=torch_device
        )

        self.assertEqual(outputs.matches.shape, expected_matches_shape)
        self.assertEqual(outputs.matching_scores.shape, expected_matching_scores_shape)

        torch.testing.assert_close(
            predicted_top10_matches_indices, expected_top10_matches_indices, rtol=5e-3, atol=5e-3
        )
        torch.testing.assert_close(
            predicted_top10_matching_scores, expected_top10_matching_scores, rtol=5e-3, atol=5e-3
        )
