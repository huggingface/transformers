# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
# Depth-Anything-V2-Small model is under the Apache-2.0 license.
# Depth-Anything-V2-Base/Large/Giant models are under the CC-BY-NC-4.0 license.

"""Convert Depth Anything V2 checkpoints from the original repository. URL:
https://github.com/MackinationsAi/Upgraded-Depth-Anything-V2"""

"""Testing suite for the PyTorch Depth Anything V2 model."""

import unittest

from transformers import DepthAnythingV2Config, Dinov2Config
from transformers.file_utils import is_torch_available, is_vision_available
from transformers.testing_utils import require_accelerate, require_torch, require_torch_gpu, require_vision, slow, torch_device

from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, compute_module_sizes, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin

if is_torch_available():
    import torch

    from transformers import DepthAnythingV2ForDepthEstimation


if is_vision_available():
    from PIL import Image

    from transformers import DPTImageProcessor


class DepthAnythingV2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_channels=3,
        image_size=32,
        patch_size=16,
        use_labels=True,
        num_labels=3,
        is_training=True,
        hidden_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=8,
        out_features=["stage1", "stage2"],
        apply_layernorm=False,
        reshape_hidden_states=False,
        neck_hidden_sizes=[2, 2],
        fusion_hidden_size=6,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.out_features = out_features
        self.apply_layernorm = apply_layernorm
        self.reshape_hidden_states = reshape_hidden_states
        self.use_labels = use_labels
        self.num_labels = num_labels
        self.is_training = is_training
        self.neck_hidden_sizes = neck_hidden_sizes
        self.fusion_hidden_size = fusion_hidden_size
        self.seq_length = (self.image_size // self.patch_size) ** 2 + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size, self.image_size, self.image_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return DepthAnythingV2Config(
            backbone_config=self.get_backbone_config(),
            reassemble_hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            neck_hidden_sizes=self.neck_hidden_sizes,
            fusion_hidden_size=self.fusion_hidden_size,
        )

    def get_backbone_config(self):
        return Dinov2Config(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            is_training=self.is_training,
            out_features=self.out_features,
            reshape_hidden_states=self.reshape_hidden_states,
        )

    def create_and_check_for_depth_estimation(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = DepthAnythingForDepthEstimation(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.predicted_depth.shape, (self.batch_size, self.image_size, self.image_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class DepthAnythingV2ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Depth Anything V2 does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (DepthAnythingForDepthEstimation,) if is_torch_available() else ()
    pipeline_model_mapping = {"depth-estimation": DepthAnythingForDepthEstimation} if is_torch_available() else {}

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = DepthAnythingV2ModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=DepthAnythingV2Config,
            has_text_modality=False,
            hidden_size=37,
            common_properties=["patch_size"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Depth Anything V2 with AutoBackbone does not have a base model and hence no input_embeddings")
    def test_inputs_embeds(self):
        pass

    def test_for_depth_estimation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_depth_estimation(*config_and_inputs)

    @unittest.skip(reason="Depth Anything V2 does not support training yet")
    def test_training(self):
        pass

    @unittest.skip(reason="Depth Anything V2 does not support training yet")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Depth Anything V2 with AutoBackbone does not have a base model and hence no input_embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Depth Anything V2 with AutoBackbone does not have a base model")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="Depth Anything V2 with AutoBackbone does not have a base model")
    def test_save_load_fast_init_to_base(self):
        pass

    @unittest.skip(
        reason="This architecture seems to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seems to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "MackinationsAi/Depth-Anything-V2_Safetensors/depth_anything_v2_vits.safetensors"
        model = DepthAnythingForDepthEstimation.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def test_backbone_selection(self):
        def _validate_backbone_init():
            for model_class in self.all_model_classes:
                model = model_class(config)
                model.to(torch_device)
                model.eval()
                self.assertEqual(len(model.backbone.out_indices), 2)

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        config.backbone = "resnet18"
        config.use_pretrained_backbone = True
        config.use_timm_backbone = True
        config.backbone_config = None
        config.backbone_kwargs = {"out_indices": (-2, -1)}
        _validate_backbone_init()

        config.backbone = "facebook/dinov2-small"
        config.use_pretrained_backbone = True
        config.use_timm_backbone = False
        config.backbone_config = None
        config.backbone_kwargs = {"out_indices": [-2, -1]}
        _validate_backbone_init()

    @require_accelerate
    @require_torch_gpu
    def test_cpu_offload(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class._no_split_modules is None:
                continue

            inputs_dict_class = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(config).eval()

            # Use accelerate for CPU offloading
            device_map = infer_auto_device_map(model, max_memory={0: "6GiB", "cpu": "48GiB"})
            model = dispatch_model(model, device_map=device_map, offload_buffers=True)

            torch.manual_seed(0)
            base_output = model(**inputs_dict_class)

            model_size = compute_module_sizes(model)[""]
            max_gpu_sizes = [int(p * model_size) for p in self.model_split_percents[1:]]

            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir)
                model = model_class.from_pretrained(tmp_dir).eval()
                model = dispatch_model(model, device_map=device_map, offload_buffers=True)

                for max_gpu_size in max_gpu_sizes:
                    torch.manual_seed(0)
                    offloaded_output = model(**inputs_dict_class)
                    self.assertTrue(torch.allclose(base_output.predicted_depth, offloaded_output.predicted_depth, atol=1e-5))

                # Explicitly delete temp files
                try:
                    os.remove(os.path.join(tmp_dir, 'model.safetensors'))
                except OSError as e:
                    logger.warning(f"Error removing temporary file: {e}")

    @require_accelerate
    @require_torch_gpu
    def test_disk_offload_safetensors(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class._no_split_modules is None:
                continue

            inputs_dict_class = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(config).eval()
            
            # Use accelerate for disk offloading
            device_map = infer_auto_device_map(model, max_memory={0: "6GiB", "cpu": "48GiB"})
            model = dispatch_model(model, device_map=device_map, offload_buffers=True)
            
            torch.manual_seed(0)
            base_output = model(**inputs_dict_class)

            model_size = compute_module_sizes(model)[""]
            max_gpu_sizes = [int(p * model_size) for p in self.model_split_percents[1:]]

            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir)
                model = model_class.from_pretrained(tmp_dir).eval()
                model = dispatch_model(model, device_map=device_map, offload_buffers=True)

                for max_gpu_size in max_gpu_sizes:
                    torch.manual_seed(0)
                    offloaded_output = model(**inputs_dict_class)
                    self.assertTrue(torch.allclose(base_output.predicted_depth, offloaded_output.predicted_depth, atol=1e-5))

                # Explicitly delete temp files
                try:
                    os.remove(os.path.join(tmp_dir, 'model.safetensors'))
                except OSError as e:
                    logger.warning(f"Error removing temporary file: {e}")


def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
@slow
class DepthAnythingV2ModelIntegrationTest(unittest.TestCase):
    def test_inference(self):
        image_processor = DPTImageProcessor.from_pretrained("MackinationsAi/Depth-Anything-V2_Safetensors/depth_anything_v2_vits.safetensors")
        model = DepthAnythingV2ForDepthEstimation.from_pretrained("MackinationsAi/Depth-Anything-V2_Safetensors/depth_anything_v2_vits.safetensors").to(torch_device)

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        expected_shape = torch.Size([1, 518, 686])
        self.assertEqual(predicted_depth.shape, expected_shape)

        expected_slice = torch.tensor(
            [[8.8204, 8.6468, 8.6195], [8.3313, 8.6027, 8.7526], [8.6526, 8.6866, 8.7453]],
        ).to(torch_device)

        self.assertTrue(torch.allclose(outputs.predicted_depth[0, :3, :3], expected_slice, atol=1e-6))
