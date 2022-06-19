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
""" Testing suite for the PyTorch Omnivore model. """

import inspect
import math
import unittest
import warnings

from transformers import MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING, OmnivoreConfig
from transformers.models.auto import get_values
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import OmnivoreForVisionClassification, OmnivoreModel
    from transformers.models.omnivore.modeling_omnivore import OMNIVORE_PRETRAINED_MODEL_ARCHIVE_LIST

if is_vision_available():
    from PIL import Image

    from transformers import AutoFeatureExtractor


class OmnivoreConfigTester(ConfigTester):
    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, "window_size"))
        self.parent.assertTrue(hasattr(config, "num_heads"))
        self.parent.assertTrue(hasattr(config, "patch_size"))
        self.parent.assertTrue(hasattr(config, "depth_mode"))


class OmnivoreModelTester:
    def __init__(
        self,
        parent,
        batch_size=5,
        image_size=32,
        frames=4,
        num_image_labels=2,
        num_video_labels=4,
        num_rgbd_labels=3,
        input_channels=3,
        patch_size=[2, 4, 4],
        embed_dim=96,
        depths=[2, 2, 2, 2],
        num_heads=[3, 3, 4, 4],
        window_size=[8, 7, 7],
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        frozen_stages=-1,
        depth_mode="summed_rgb_d_tokens",
        initializer_range=0.02,
        is_training=True,
        use_labels=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.frames = frames
        self.num_image_labels = num_image_labels
        self.num_video_labels = num_video_labels
        self.num_rgbd_labels = num_rgbd_labels
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.drop_path_rate = drop_path_rate
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.depth_mode = depth_mode
        self.initializer_range = initializer_range
        self.is_training = is_training
        self.use_labels = use_labels

    def prepare_config_and_inputs(self, input_type=None):
        pixel_values_images = floats_tensor(
            [self.batch_size, self.input_channels, 1, self.image_size, self.image_size]
        )
        pixel_values_videos = floats_tensor(
            [self.batch_size, self.input_channels, self.frames, self.image_size, self.image_size]
        )
        pixel_values_rgbds = floats_tensor(
            [self.batch_size, self.input_channels + 1, 1, self.image_size, self.image_size]
        )

        if self.use_labels:
            image_labels = ids_tensor([self.batch_size], self.num_image_labels)
            video_labels = ids_tensor([self.batch_size], self.num_video_labels)
            rgbd_labels = ids_tensor([self.batch_size], self.num_rgbd_labels)
        config = self.get_config()

        if input_type == "image":
            return config, pixel_values_images, image_labels
        elif input_type == "rgbd":
            return config, pixel_values_rgbds, rgbd_labels
        else:
            return config, pixel_values_videos, video_labels

    def get_config(self):
        return OmnivoreConfig(
            num_image_labels=self.num_image_labels,
            num_video_labels=self.num_video_labels,
            num_rgbd_labels=self.num_rgbd_labels,
            input_channels=self.input_channels,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            depths=self.depths,
            num_heads=self.num_heads,
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            drop_path_rate=self.drop_path_rate,
            patch_norm=self.patch_norm,
            frozen_stages=self.frozen_stages,
            depth_mode=self.depth_mode,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = OmnivoreModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        patch_size = self.patch_size

        def get_size(x, i):
            return math.ceil(((x - (patch_size[i] - 1) - 1) / patch_size[i]) + 1)

        ater_patch_embed_dim = (get_size(self.frames, 0), get_size(self.image_size, 1), get_size(self.image_size, 2))
        expected_seq_len = math.ceil(ater_patch_embed_dim[1] // (2 ** (len(config.depths) - 1)))
        expected_dim = int(config.embed_dim * 2 ** (len(config.depths) - 1))
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, expected_dim, ater_patch_embed_dim[0], expected_seq_len, expected_seq_len),
        )

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_image_labels = self.num_image_labels
        model = OmnivoreForVisionClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, "image", labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_image_labels))

    def create_and_check_for_video_classification(self, config, pixel_values, labels):
        config.num_video_labels = self.num_video_labels
        model = OmnivoreForVisionClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, "video", labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_video_labels))

    def create_and_check_for_rgbd_classification(self, config, pixel_values, labels):
        config.num_rgbd_labels = self.num_rgbd_labels
        model = OmnivoreForVisionClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, "rgbd", labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_rgbd_labels))

    def prepare_config_and_inputs_for_common(self, input_type=None):
        config_and_inputs = self.prepare_config_and_inputs(input_type=input_type)
        (
            config,
            pixel_values,
            labels,
        ) = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class OmnivoreModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            OmnivoreModel,
            OmnivoreForVisionClassification,
        )
        if is_torch_available()
        else ()
    )
    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        self.model_tester = OmnivoreModelTester(self)
        self.config_tester = OmnivoreConfigTester(
            self, config_class=OmnivoreConfig, has_text_modality=False, embed_dim=37
        )

    def test_config(self):
        self.config_tester.create_and_test_config_common_properties()
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    @unittest.skip(reason="ViT does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Levit does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="Levit does not output attentions")
    def test_attention_outputs(self):
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

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        return inputs_dict

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = len(self.model_tester.depths)
            self.assertEqual(len(hidden_states), expected_num_layers)

            def get_size(x, i):
                return math.ceil(
                    ((x - (self.model_tester.patch_size[i] - 1) - 1) / self.model_tester.patch_size[i]) + 1
                )

            # verify the first hidden states (first block)
            self.assertListEqual(
                list(hidden_states[0].shape[-4:]),
                [
                    int(self.model_tester.embed_dim * 2),
                    math.ceil(get_size(self.model_tester.frames, 0)),
                    math.ceil(get_size(self.model_tester.image_size, 1) // 2),
                    math.ceil(get_size(self.model_tester.image_size, 2) // 2),
                ],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)
            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            check_hidden_states_output(inputs_dict, config, model_class)

    def test_problem_types(self):
        for input_type in ["rgbd", "image", "video"]:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common(input_type=input_type)
            if input_type == "images":
                problem_types = [
                    {"title": "multi_label_classification", "num_labels": 2, "dtype": torch.float},
                    {"title": "single_label_classification", "num_labels": 1, "dtype": torch.long},
                    {"title": "regression", "num_labels": 1, "dtype": torch.float},
                ]

                for model_class in self.all_model_classes:
                    if model_class not in [
                        *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING),
                    ]:
                        continue

                    for problem_type in problem_types:
                        with self.subTest(msg=f"Testing {model_class} with {problem_type['title']}"):

                            config.problem_type = problem_type["title"]
                            config.num_image_labels = problem_type["num_labels"]

                            model = model_class(config)
                            model.to(torch_device)
                            model.train()

                            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)

                            if problem_type["num_labels"] > 1:
                                inputs["labels"] = inputs["labels"].unsqueeze(1).repeat(1, problem_type["num_labels"])
                            inputs["input_type"] = "image"
                            inputs["labels"] = inputs["labels"].to(problem_type["dtype"])

                            # This tests that we do not trigger the warning form PyTorch "Using a target size that is different
                            # to the input size. This will likely lead to incorrect results due to broadcasting. Please ensure
                            # they have the same size." which is a symptom something in wrong for the regression problem.
                            # See https://github.com/huggingface/transformers/issues/11780
                            with warnings.catch_warnings(record=True) as warning_list:
                                loss = model(**inputs).loss
                            for w in warning_list:
                                if "Using a target size that is different to the input size" in str(w.message):
                                    raise ValueError(
                                        f"Something is going wrong in the regression problem: intercepted {w.message}"
                                    )

                            loss.backward()

            elif input_type == "rgbd":
                problem_types = [
                    {"title": "multi_label_classification", "num_labels": 2, "dtype": torch.float},
                    {"title": "single_label_classification", "num_labels": 1, "dtype": torch.long},
                    {"title": "regression", "num_labels": 1, "dtype": torch.float},
                ]

                for model_class in self.all_model_classes:
                    if model_class not in [
                        *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING),
                    ]:
                        continue

                    for problem_type in problem_types:
                        with self.subTest(msg=f"Testing {model_class} with {problem_type['title']}"):

                            config.problem_type = problem_type["title"]
                            config.num_rgbd_labels = problem_type["num_labels"]

                            model = model_class(config)
                            model.to(torch_device)
                            model.train()

                            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                            inputs["input_type"] = "rgbd"
                            if problem_type["num_labels"] > 1:
                                inputs["labels"] = inputs["labels"].unsqueeze(1).repeat(1, problem_type["num_labels"])

                            inputs["labels"] = inputs["labels"].to(problem_type["dtype"])

                            # This tests that we do not trigger the warning form PyTorch "Using a target size that is different
                            # to the input size. This will likely lead to incorrect results due to broadcasting. Please ensure
                            # they have the same size." which is a symptom something in wrong for the regression problem.
                            # See https://github.com/huggingface/transformers/issues/11780
                            with warnings.catch_warnings(record=True) as warning_list:
                                loss = model(**inputs).loss
                            for w in warning_list:
                                if "Using a target size that is different to the input size" in str(w.message):
                                    raise ValueError(
                                        f"Something is going wrong in the regression problem: intercepted {w.message}"
                                    )

                            loss.backward()

            else:
                problem_types = [
                    {"title": "multi_label_classification", "num_labels": 2, "dtype": torch.float},
                    {"title": "single_label_classification", "num_labels": 1, "dtype": torch.long},
                    {"title": "regression", "num_labels": 1, "dtype": torch.float},
                ]

                for model_class in self.all_model_classes:
                    if model_class not in [
                        *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING),
                    ]:
                        continue

                    for problem_type in problem_types:
                        with self.subTest(msg=f"Testing {model_class} with {problem_type['title']}"):

                            config.problem_type = problem_type["title"]
                            config.num_video_labels = problem_type["num_labels"]

                            model = model_class(config)
                            model.to(torch_device)
                            model.train()
                            inputs["input_type"] = "video"
                            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)

                            if problem_type["num_labels"] > 1:
                                inputs["labels"] = inputs["labels"].unsqueeze(1).repeat(1, problem_type["num_labels"])

                            inputs["labels"] = inputs["labels"].to(problem_type["dtype"])

                            # This tests that we do not trigger the warning form PyTorch "Using a target size that is different
                            # to the input size. This will likely lead to incorrect results due to broadcasting. Please ensure
                            # they have the same size." which is a symptom something in wrong for the regression problem.
                            # See https://github.com/huggingface/transformers/issues/11780
                            with warnings.catch_warnings(record=True) as warning_list:
                                loss = model(**inputs).loss
                            for w in warning_list:
                                if "Using a target size that is different to the input size" in str(w.message):
                                    raise ValueError(
                                        f"Something is going wrong in the regression problem: intercepted {w.message}"
                                    )

                            loss.backward()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs("image")
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    def test_for_video_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs("video")
        self.model_tester.create_and_check_for_video_classification(*config_and_inputs)

    def test_for_rgbd_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs("rgbd")
        self.model_tester.create_and_check_for_rgbd_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in OMNIVORE_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = OmnivoreModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_vision
@require_torch
class OmnivoreModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_feature_extractor(self):
        return AutoFeatureExtractor.from_pretrained("anugunj/omnivore-swinT") if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        model = OmnivoreForVisionClassification.from_pretrained("anugunj/omnivore-swinT").to(torch_device)
        feature_extractor = self.default_feature_extractor

        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        inputs = feature_extractor(images=image, return_tensors="pt").to(torch_device)
        inputs["input_type"] = "image"
        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice = torch.tensor([-0.0948, -0.6454, -0.0921]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4))
