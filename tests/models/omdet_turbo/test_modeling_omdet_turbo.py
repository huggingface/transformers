# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch OmDet-Turbo model."""

import unittest

import requests

from transformers import OmDetTurboConfig, is_torch_available, is_vision_available
from transformers.file_utils import cached_property
from transformers.testing_utils import (
    require_timm,
    require_torch,
    require_torch_gpu,
    require_vision,
    slow,
    torch_device,
)
from transformers.tokenization_utils_base import BatchEncoding

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    import torch.nn.functional as F

    from transformers import OmDetTurboForObjectDetection


if is_vision_available():
    from PIL import Image

    from transformers import AutoProcessor


class OmDetTurboModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        is_training=False,
        num_channels=3,
        max_text_len=7,
        num_classes=3,
        use_timm_backbone=False,
        backbone=None,
        apply_layernorm=False,
        image_size=224,
        text_projection_in_dim=16,
        text_projection_out_dim=16,
        class_dim=16,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_queries=20,
        encoder_in_channels=(16, 32, 64),
        encoder_dim_feedforward=32,
        num_projection_layers=1,
        decoder_n_points=4,
        num_feature_levels=3,
    ):
        super().__init__()
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.num_channels = num_channels
        self.max_text_len = max_text_len
        self.num_classes = num_classes
        self.use_timm_backbone = use_timm_backbone
        self.backbone = backbone
        self.apply_layernorm = apply_layernorm
        self.image_size = image_size
        self.text_projection_in_dim = text_projection_in_dim
        self.text_projection_out_dim = text_projection_out_dim
        self.class_dim = class_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_queries = num_queries
        self.encoder_in_channels = encoder_in_channels
        self.encoder_dim_feedforward = encoder_dim_feedforward
        self.num_projection_layers = num_projection_layers
        self.decoder_n_points = decoder_n_points
        self.num_feature_levels = num_feature_levels

        self.encoder_seq_length_vision = self.image_size // 32
        self.decoder_seq_length = self.num_queries

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        input_ids_tasks = ids_tensor([self.batch_size, self.max_text_len], self.num_classes)
        input_ids_tasks = input_ids_tasks.to(torch_device)
        input_ids_classes = torch.cat(
            [ids_tensor([self.num_classes, self.max_text_len], self.num_classes) for _ in range(self.batch_size)]
        )
        input_ids_classes = input_ids_classes.to(torch_device)
        attention_mask_tasks = torch.ones_like(input_ids_tasks, device=torch_device)
        attention_mask_classes = torch.ones_like(input_ids_classes, device=torch_device)
        structure_classes = torch.ones(self.batch_size, dtype=torch.long, device=torch_device) * self.num_classes
        classes_encoding = BatchEncoding(
            {"input_ids": input_ids_classes, "attention_mask": attention_mask_classes, "structure": structure_classes}
        )
        task_encoding = BatchEncoding({"input_ids": input_ids_tasks, "attention_mask": attention_mask_tasks})
        config = self.get_config()
        return config, pixel_values, classes_encoding, task_encoding

    def get_config(self):
        text_backbone = {
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 16,
            "max_position_embeddings": 8,
            "model_type": "clip_text_model",
        }
        backbone_config = {
            "embed_dim": self.hidden_size,
            "depths": (1, 1, 1, 1),
            "num_heads": (1, 1, 1, 1),
            "window_size": 7,
            "image_size": self.image_size,
            "out_indices": (2, 3, 4),
            "model_type": "swin",
        }

        return OmDetTurboConfig(
            text_config=text_backbone,
            backbone_config=backbone_config,
            use_timm_backbone=self.use_timm_backbone,
            backbone=self.backbone,
            apply_layernorm=self.apply_layernorm,
            decoder_num_layers=self.num_hidden_layers,
            image_size=self.image_size,
            encoder_in_channels=self.encoder_in_channels,
            num_queries=self.num_queries,
            encoder_layers=self.num_hidden_layers,
            encoder_projection_indices=[2] * self.num_projection_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_num_heads=self.num_attention_heads,
            decoder_num_points=self.decoder_n_points,
            num_feature_levels=self.num_feature_levels,
            encoder_dim_feedforward=self.encoder_dim_feedforward,
            task_encoder_feedforward_dim=self.encoder_dim_feedforward,
            decoder_dim_feedforward=self.encoder_dim_feedforward,
            class_dim=self.class_dim,
            text_projection_in_dim=self.text_projection_in_dim,
            text_projection_out_dim=self.text_projection_out_dim,
            encoder_hidden_dim=self.hidden_size,
            decoder_hidden_dim=self.hidden_size,
            vision_features_channels=[self.hidden_size, self.hidden_size, self.hidden_size],
        )

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, classes_encoding, task_encodings = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values, "classes": classes_encoding, "tasks": task_encodings}
        return config, inputs_dict

    def create_and_check_object_detection_head_model(self, config, pixel_values, classes, tasks):
        model = OmDetTurboForObjectDetection(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values=pixel_values, classes=classes, tasks=tasks)

        self.parent.assertEqual(result.decoder_coord_logits.shape, (self.batch_size, self.num_queries, 4))
        self.parent.assertEqual(
            result.decoder_class_logits.shape, (self.batch_size, self.num_queries, self.num_classes)
        )


@require_torch
class OmDetTurboModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (OmDetTurboForObjectDetection,) if is_torch_available() else ()
    is_encoder_decoder = True
    test_torchscript = False
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False
    pipeline_model_mapping = (
        {"zero-shot-object-detection": OmDetTurboForObjectDetection} if is_torch_available() else {}
    )
    maxDiff = None

    # special case for head models
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        return inputs_dict

    def setUp(self):
        self.model_tester = OmDetTurboModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=OmDetTurboConfig,
            has_text_modality=False,
            common_properties=["d_model", "encoder_attention_heads", "decoder_num_heads"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_object_detection_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_object_detection_head_model(*config_and_inputs)

    @unittest.skip(reason="OmDet-Turbo does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="OmDet-Turbo does not use decoder_outputs")
    def test_save_load_low_cpu_mem_usage(self):
        pass

    @unittest.skip(reason="OmDet-Turbo does not use decoder_outputs")
    def test_save_load_low_cpu_mem_usage_checkpoints(self):
        pass

    @unittest.skip(reason="OmDet-Turbo does not use decoder_outputs")
    def test_save_load_low_cpu_mem_usage_no_safetensors(self):
        pass

    @unittest.skip(reason="OmDet-Turbo does not have a get_input_embeddings method")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="OmDet-Turbo does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

    # Overwrite as `init_reference_points` is not batch dependent and contains `inf` values
    def test_batching_equivalence(self):
        """
        Tests that the model supports batching and that the output is nearly the same for the same input in
        different batch sizes.
        (Why "nearly the same" not "exactly the same"? Batching uses different matmul shapes, which often leads to
        different results: https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535)
        """

        def get_tensor_equivalence_function(batched_input):
            # models operating on continuous spaces have higher abs difference than LMs
            # instead, we can rely on cos distance for image/speech models, similar to `diffusers`
            if "input_ids" not in batched_input:
                return lambda tensor1, tensor2: (
                    1.0 - F.cosine_similarity(tensor1.float().flatten(), tensor2.float().flatten(), dim=0, eps=1e-38)
                )
            return lambda tensor1, tensor2: torch.max(torch.abs(tensor1 - tensor2))

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
            elif key != "init_reference_points":
                # init
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
                    torch.isinf(single_row_object).any(),
                    f"Single row output has `inf` in {model_name} for key={key}",
                )
                self.assertTrue(
                    (equivalence(batched_row, single_row_object)) <= 1e-03,
                    msg=(
                        f"Batched and Single row outputs are not equal in {model_name} for key={key}. "
                        f"Difference={equivalence(batched_row, single_row_object)}."
                    ),
                )

        config, batched_input = self.model_tester.prepare_config_and_inputs_for_common()
        equivalence = get_tensor_equivalence_function(batched_input)

        for model_class in self.all_model_classes:
            config.output_hidden_states = True

            model_name = model_class.__name__
            if hasattr(self.model_tester, "prepare_config_and_inputs_for_model_class"):
                config, batched_input = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
            batched_input_prepared = self._prepare_for_class(batched_input, model_class)
            model = model_class(config).to(torch_device).eval()

            single_row_input = {}
            for key, value in batched_input_prepared.items():
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
                if key == "decoder_class_logits" or key == "encoder_class_logits":
                    # check if all elements are close to 0, if so skip the test as the test strugles with comparing
                    # tensors with all elements close to 0
                    if torch.allclose(
                        model_batched_output[key], torch.zeros_like(model_batched_output[key]), atol=1e-6
                    ) and torch.allclose(model_row_output[key], torch.zeros_like(model_row_output[key]), atol=1e-6):
                        print(f"Skipping {model_name} for key={key} as all elements are close to 0")
                        continue
                    else:
                        # print max abs of both tensors
                        print(
                            f"Max abs batched tensor: {torch.max(torch.abs(model_batched_output[key])).item()} "
                            f"Max abs single row tensor: {torch.max(torch.abs(model_row_output[key])).item()}"
                        )

                recursive_check(model_batched_output[key], model_row_output[key], model_name, key)

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
            attentions = outputs.encoder_attentions[-1]
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions[-1]
            self.assertEqual(
                len(attentions), self.model_tester.num_hidden_layers * self.model_tester.num_projection_layers
            )
            # Rest of the shape seems to depend on backbone output shapes and image size
            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [
                    self.model_tester.num_attention_heads,
                    self.model_tester.encoder_seq_length_vision**2,
                    self.model_tester.encoder_seq_length_vision**2,
                ],
            )
            # decoder attentions
            decoder_attentions = outputs.decoder_attentions[0]
            self.assertIsInstance(decoder_attentions, (list, tuple))
            self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(decoder_attentions[0].shape[-3:]),
                [
                    self.model_tester.num_attention_heads,
                    self.model_tester.num_queries + self.model_tester.max_text_len,
                    self.model_tester.num_queries + self.model_tester.max_text_len,
                ],
            )

            # cross attentions
            cross_attentions = outputs.decoder_attentions[-1]
            self.assertIsInstance(cross_attentions, (list, tuple))
            self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(cross_attentions[0].shape[-3:]),
                [
                    self.model_tester.num_attention_heads,
                    self.model_tester.num_feature_levels,
                    self.model_tester.decoder_n_points,
                ],
            )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            self_attentions = outputs.encoder_attentions[-1]

            self.assertEqual(
                len(self_attentions), self.model_tester.num_hidden_layers * self.model_tester.num_projection_layers
            )
            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [
                    self.model_tester.num_attention_heads,
                    self.model_tester.encoder_seq_length_vision**2,
                    self.model_tester.encoder_seq_length_vision**2,
                ],
            )

    # overwrite since encoder_hidden_states are not typical.
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states
            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_projection_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            seq_len = self.model_tester.encoder_seq_length_vision

            self.assertListEqual(list(hidden_states[0].shape[-3:]), [self.model_tester.hidden_size, seq_len, seq_len])

            hidden_states = outputs.decoder_hidden_states
            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertIsInstance(hidden_states, (list, tuple))
            self.assertEqual(len(hidden_states), expected_num_layers)
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [self.model_tester.decoder_seq_length, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    # removed retain_grad and grad on decoder_hidden_states, as queries don't require grad
    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs[0]

        encoder_hidden_states = outputs.encoder_hidden_states[0]
        encoder_attentions = outputs.encoder_attentions[0][0]
        encoder_hidden_states.retain_grad()
        encoder_attentions.retain_grad()

        cross_attentions = outputs.decoder_attentions[-1][0]
        cross_attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(encoder_hidden_states.grad)
        self.assertIsNotNone(encoder_attentions.grad)
        self.assertIsNotNone(cross_attentions.grad)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if (
                        "embeddings" in name
                        or ".fc" in name
                        or "decoder.channel_projection_layers" in name
                        or "query_position_head" in name
                        or "decoder.encoder_vision_features" in name
                    ):
                        continue
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} seems not properly initialized",
                    )


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image


def prepare_text():
    classes = ["cat", "remote"]
    task = "Detect {}.".format(",".join(classes))
    return classes, task


@require_timm
@require_vision
@slow
class OmDetTurboModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return AutoProcessor.from_pretrained("yonigozlan/omdet-turbo-tiny") if is_vision_available() else None

    def test_inference_object_detection_head(self):
        model = OmDetTurboForObjectDetection.from_pretrained("yonigozlan/omdet-turbo-tiny").to(torch_device)

        processor = self.default_processor
        image = prepare_img()
        classes, task = prepare_text()
        encoding = processor(images=image, text=classes, task=task, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**encoding)

        expected_shape_coord_logits = torch.Size((1, model.config.num_queries, 4))
        expected_shape_class_logits = torch.Size((1, model.config.num_queries, 2))
        self.assertEqual(outputs.decoder_coord_logits.shape, expected_shape_coord_logits)
        self.assertEqual(outputs.decoder_class_logits.shape, expected_shape_class_logits)

        expected_class_logits = torch.tensor([[[0.9427, -2.5958], [0.2105, -3.4569], [-2.6364, -4.1610]]]).to(
            torch_device
        )
        expected_coord_logits = torch.tensor(
            [[[0.2550, 0.5501, 0.4738, 0.8745], [0.7695, 0.4121, 0.4603, 0.7244], [0.7691, 0.4117, 0.4603, 0.7214]]]
        ).to(torch_device)

        self.assertTrue(torch.allclose(outputs.decoder_class_logits[:3, :3], expected_class_logits, atol=1e-1))
        self.assertTrue(torch.allclose(outputs.decoder_coord_logits[:3, :3], expected_coord_logits, atol=1e-3))

        # verify grounded postprocessing
        results = processor.post_process_grounded_object_detection(
            outputs, classes=[classes], target_sizes=[image.size[::-1]]
        )[0]
        expected_scores = torch.tensor([0.7675, 0.7196, 0.5634, 0.5524]).to(torch_device)
        expected_slice_boxes = torch.tensor([39.8870, 70.3522, 176.7424, 118.0354]).to(torch_device)

        self.assertEqual(len(results["scores"]), 4)
        self.assertTrue(torch.allclose(results["scores"], expected_scores, atol=1e-2))
        self.assertTrue(torch.allclose(results["boxes"][0, :], expected_slice_boxes, atol=1e-2))

        expected_classes = ["remote", "cat", "remote", "cat"]
        self.assertListEqual(results["classes"], expected_classes)

    @require_torch_gpu
    def test_inference_object_detection_head_equivalence_cpu_gpu(self):
        processor = self.default_processor
        image = prepare_img()
        classes, task = prepare_text()
        encoding = processor(images=image, text=classes, task=task, return_tensors="pt")
        # 1. run model on CPU
        model = OmDetTurboForObjectDetection.from_pretrained("yonigozlan/omdet-turbo-tiny")

        with torch.no_grad():
            cpu_outputs = model(**encoding)

        # 2. run model on GPU
        model.to("cuda")
        encoding = encoding.to("cuda")
        with torch.no_grad():
            gpu_outputs = model(**encoding)

        # 3. assert equivalence
        expected_class_logits = torch.tensor([[[0.9427, -2.5958], [0.2105, -3.4569], [-2.6364, -4.1610]]])
        expected_coord_logits = torch.tensor(
            [[[0.2550, 0.5501, 0.4738, 0.8745], [0.7695, 0.4121, 0.4603, 0.7244], [0.7691, 0.4117, 0.4603, 0.7214]]]
        )

        self.assertTrue(torch.allclose(cpu_outputs.decoder_class_logits[:3, :3], expected_class_logits, atol=1e-1))
        self.assertTrue(torch.allclose(cpu_outputs.decoder_coord_logits[:3, :3], expected_coord_logits, atol=1e-3))

        # verify grounded postprocessing
        results_cpu = processor.post_process_grounded_object_detection(
            cpu_outputs, classes=[classes], target_sizes=[image.size[::-1]]
        )[0]
        result_gpu = processor.post_process_grounded_object_detection(
            gpu_outputs, classes=[classes], target_sizes=[image.size[::-1]]
        )[0]

        self.assertTrue(torch.allclose(results_cpu["scores"], result_gpu["scores"].cpu(), atol=1e-2))
        self.assertTrue(torch.allclose(results_cpu["boxes"][0, :], result_gpu["boxes"][0, :].cpu(), atol=1e-2))
