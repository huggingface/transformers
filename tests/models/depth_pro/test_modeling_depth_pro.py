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
"""Testing suite for the PyTorch DepthPro model."""

import unittest

from transformers import DepthProConfig
from transformers.file_utils import is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import DepthProForDepthEstimation, DepthProModel
    from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES


if is_vision_available():
    from PIL import Image

    from transformers import DepthProImageProcessor


class DepthProModelTester:
    def __init__(
        self,
        parent,
        batch_size=8,
        image_size=64,
        patch_size=16,
        num_channels=3,
        is_training=True,
        use_labels=True,
        fusion_hidden_size=16,
        intermediate_hook_ids=[1, 0],
        intermediate_feature_dims=[10, 8],
        scaled_images_ratios=[0.5, 1.0],
        scaled_images_overlap_ratios=[0.0, 0.2],
        scaled_images_feature_dims=[12, 12],
        initializer_range=0.02,
        use_fov_model=False,
        image_model_config={
            "model_type": "dinov2",
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "num_attention_heads": 1,
            "patch_size": 4,
        },
        patch_model_config={
            "model_type": "vit",
            "num_hidden_layers": 2,
            "hidden_size": 24,
            "num_attention_heads": 2,
            "patch_size": 6,
        },
        fov_model_config={
            "model_type": "vit",
            "num_hidden_layers": 2,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "patch_size": 8,
        },
        num_labels=3,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.fusion_hidden_size = fusion_hidden_size
        self.intermediate_hook_ids = intermediate_hook_ids
        self.intermediate_feature_dims = intermediate_feature_dims
        self.scaled_images_ratios = scaled_images_ratios
        self.scaled_images_overlap_ratios = scaled_images_overlap_ratios
        self.scaled_images_feature_dims = scaled_images_feature_dims
        self.initializer_range = initializer_range
        self.use_fov_model = use_fov_model
        self.image_model_config = image_model_config
        self.patch_model_config = patch_model_config
        self.fov_model_config = fov_model_config
        self.num_labels = num_labels

        self.hidden_size = image_model_config["hidden_size"]
        self.num_hidden_layers = image_model_config["num_hidden_layers"]
        self.num_attention_heads = image_model_config["num_attention_heads"]

        # may be different for a backbone other than dinov2
        self.out_size = patch_size // image_model_config["patch_size"]
        self.seq_length = self.out_size**2 + 1  # we add 1 for the [CLS] token

        n_fusion_blocks = len(intermediate_hook_ids) + len(scaled_images_ratios)
        self.expected_depth_size = 2 ** (n_fusion_blocks + 1) * self.out_size

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size, self.image_size, self.image_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return DepthProConfig(
            patch_size=self.patch_size,
            fusion_hidden_size=self.fusion_hidden_size,
            intermediate_hook_ids=self.intermediate_hook_ids,
            intermediate_feature_dims=self.intermediate_feature_dims,
            scaled_images_ratios=self.scaled_images_ratios,
            scaled_images_overlap_ratios=self.scaled_images_overlap_ratios,
            scaled_images_feature_dims=self.scaled_images_feature_dims,
            initializer_range=self.initializer_range,
            image_model_config=self.image_model_config,
            patch_model_config=self.patch_model_config,
            fov_model_config=self.fov_model_config,
            use_fov_model=self.use_fov_model,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = DepthProModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_depth_estimation(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = DepthProForDepthEstimation(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(
            result.predicted_depth.shape, (self.batch_size, self.expected_depth_size, self.expected_depth_size)
        )

    def create_and_check_for_fov(self, config, pixel_values, labels):
        model = DepthProForDepthEstimation(config, use_fov_model=True)
        model.to(torch_device)
        model.eval()

        # check if the fov_model (DinoV2-based encoder) is created
        self.parent.assertIsNotNone(model.fov_model)

        batched_pixel_values = pixel_values
        row_pixel_values = pixel_values[:1]

        with torch.no_grad():
            model_batched_output_fov = model(batched_pixel_values).field_of_view
            model_row_output_fov = model(row_pixel_values).field_of_view

        # check if fov is returned
        self.parent.assertIsNotNone(model_batched_output_fov)
        self.parent.assertIsNotNone(model_row_output_fov)

        # check output shape consistency for fov
        self.parent.assertEqual(model_batched_output_fov.shape, (self.batch_size,))

        # check equivalence between batched and single row outputs for fov
        diff = torch.max(torch.abs(model_row_output_fov - model_batched_output_fov[:1]))
        model_name = model.__class__.__name__
        self.parent.assertTrue(
            diff <= 1e-03,
            msg=(f"Batched and Single row outputs are not equal in {model_name} for fov. Difference={diff}."),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class DepthProModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as DepthPro does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (DepthProModel, DepthProForDepthEstimation) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "depth-estimation": DepthProForDepthEstimation,
            "image-feature-extraction": DepthProModel,
        }
        if is_torch_available()
        else {}
    )

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torch_exportable = True

    def setUp(self):
        self.model_tester = DepthProModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DepthProConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Inductor error: name 'OpaqueUnaryFn_log2' is not defined")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="DepthPro does not use inputs_embeds")
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

    def test_for_depth_estimation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_depth_estimation(*config_and_inputs)

    def test_for_fov(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_fov(*config_and_inputs)

    def test_training(self):
        for model_class in self.all_model_classes:
            if model_class.__name__ == "DepthProForDepthEstimation":
                continue

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            if model_class.__name__ in MODEL_MAPPING_NAMES.values():
                continue

            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    def test_training_gradient_checkpointing(self):
        for model_class in self.all_model_classes:
            if model_class.__name__ == "DepthProForDepthEstimation":
                continue

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.use_cache = False
            config.return_dict = True

            if model_class.__name__ in MODEL_MAPPING_NAMES.values() or not model_class.supports_gradient_checkpointing:
                continue
            model = model_class(config)
            model.to(torch_device)
            model.gradient_checkpointing_enable()
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                non_uniform_init_parms = [
                    # these encoders are vision transformers
                    # any layer outside these encoders is either Conv2d or ConvTranspose2d
                    # which use kaiming initialization
                    "patch_encoder",
                    "image_encoder",
                    "fov_model.encoder",
                ]
                if param.requires_grad:
                    if any(x in name for x in non_uniform_init_parms):
                        # See PR #38607 (to avoid flakiness)
                        data = torch.flatten(param.data)
                        n_elements = torch.numel(data)
                        # skip 2.5% of elements on each side to avoid issues caused by `nn.init.trunc_normal_` described in
                        # https://github.com/huggingface/transformers/pull/27906#issuecomment-1846951332
                        n_elements_to_skip_on_each_side = int(n_elements * 0.025)
                        data_to_check = torch.sort(data).values
                        if n_elements_to_skip_on_each_side > 0:
                            data_to_check = data_to_check[
                                n_elements_to_skip_on_each_side:-n_elements_to_skip_on_each_side
                            ]
                        self.assertIn(
                            ((data_to_check.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    # this started when switched from normal initialization to kaiming_normal initialization
    # maybe because the magnitude of offset values from ViT-encoders increases when followed by many convolution layers
    def test_batching_equivalence(self, atol=1e-4, rtol=1e-4):
        super().test_batching_equivalence(atol=atol, rtol=rtol)

    @slow
    def test_model_from_pretrained(self):
        model_path = "apple/DepthPro-hf"
        model = DepthProModel.from_pretrained(model_path)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
@slow
class DepthProModelIntegrationTest(unittest.TestCase):
    def test_inference_depth_estimation(self):
        model_path = "apple/DepthPro-hf"
        image_processor = DepthProImageProcessor.from_pretrained(model_path)
        model = DepthProForDepthEstimation.from_pretrained(model_path).to(torch_device)
        config = model.config

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the predicted depth
        n_fusion_blocks = len(config.intermediate_hook_ids) + len(config.scaled_images_ratios)
        out_size = config.image_model_config.image_size // config.image_model_config.patch_size
        expected_depth_size = 2 ** (n_fusion_blocks + 1) * out_size

        expected_shape = torch.Size((1, expected_depth_size, expected_depth_size))
        self.assertEqual(outputs.predicted_depth.shape, expected_shape)

        expected_slice = torch.tensor(
            [[1.0582, 1.1225, 1.1335], [1.1154, 1.1398, 1.1486], [1.1434, 1.1500, 1.1643]]
        ).to(torch_device)
        torch.testing.assert_close(outputs.predicted_depth[0, :3, :3], expected_slice, atol=1e-4, rtol=1e-4)

        # verify the predicted fov
        expected_shape = torch.Size((1,))
        self.assertEqual(outputs.field_of_view.shape, expected_shape)

        expected_slice = torch.tensor([47.2459]).to(torch_device)
        torch.testing.assert_close(outputs.field_of_view, expected_slice, atol=1e-4, rtol=1e-4)

    def test_post_processing_depth_estimation(self):
        model_path = "apple/DepthPro-hf"
        image_processor = DepthProImageProcessor.from_pretrained(model_path)
        model = DepthProForDepthEstimation.from_pretrained(model_path)

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt")

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        outputs = image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[[image.height, image.width]],
        )
        predicted_depth = outputs[0]["predicted_depth"]
        expected_shape = torch.Size((image.height, image.width))
        self.assertTrue(predicted_depth.shape == expected_shape)
