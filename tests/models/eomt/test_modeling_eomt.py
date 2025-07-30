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
"""Testing suite for the PyTorch EoMT model."""

import unittest

import requests

from transformers import AutoImageProcessor, EomtConfig, EomtForUniversalSegmentation, pipeline
from transformers.testing_utils import require_torch, require_torch_accelerator, require_torch_fp16, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image


class EomtForUniversalSegmentationTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        is_training=True,
        image_size=40,
        patch_size=2,
        num_queries=5,
        num_register_tokens=19,
        num_labels=4,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.num_queries = num_queries
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_register_tokens = num_register_tokens

        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def get_config(self):
        config = {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_labels": self.num_labels,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "num_register_tokens": self.num_register_tokens,
            "num_queries": self.num_queries,
            "num_blocks": 1,
        }
        return EomtConfig(**config)

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, 3, self.image_size, self.image_size]).to(torch_device)

        mask_labels = (
            torch.rand([self.batch_size, self.num_labels, self.image_size, self.image_size], device=torch_device) > 0.5
        ).float()
        class_labels = (torch.rand((self.batch_size, self.num_labels), device=torch_device) > 0.5).long()

        config = self.get_config()
        return config, pixel_values, mask_labels, class_labels

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, mask_labels, class_labels = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def prepare_config_and_inputs_for_training(self):
        config, pixel_values, mask_labels, class_labels = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values, "mask_labels": mask_labels, "class_labels": class_labels}
        return config, inputs_dict


@require_torch
class EomtForUniversalSegmentationTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (EomtForUniversalSegmentation,) if is_torch_available() else ()
    pipeline_model_mapping = {"image-segmentation": EomtForUniversalSegmentation} if is_torch_available() else {}
    is_encoder_decoder = False
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False
    test_torch_exportable = False

    def setUp(self):
        self.model_tester = EomtForUniversalSegmentationTester(self)
        self.config_tester = ConfigTester(self, config_class=EomtConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_with_labels(self):
        size = (self.model_tester.image_size,) * 2
        inputs = {
            "pixel_values": torch.randn((2, 3, *size), device=torch_device),
            "mask_labels": torch.randn((2, 10, *size), device=torch_device),
            "class_labels": torch.zeros(2, 10, device=torch_device).long(),
        }
        config = self.model_tester.get_config()

        model = EomtForUniversalSegmentation(config).to(torch_device)
        outputs = model(**inputs)
        self.assertTrue(outputs.loss is not None)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # Check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            out_len = len(outputs)
            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.attentions
            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    @unittest.skip(reason="EoMT does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="EoMT does not have a get_input_embeddings method")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="EoMT is not a generative model")
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip(reason="EoMT does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    def test_training(self):
        if not self.model_tester.is_training:
            self.skipTest(reason="ModelTester is not configured to run training tests")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_training()
            config.return_dict = True

            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    def test_initialization(self):
        # Apart from the below params, all other parameters are initialized using kaiming uniform.
        non_uniform_init_parms = [
            "layernorm.bias",
            "layernorm.weight",
            "norm1.bias",
            "norm1.weight",
            "norm2.bias",
            "norm2.weight",
            "layer_scale1.lambda1",
            "layer_scale2.lambda1",
            "register_tokens",
            "cls_token",
        ]

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if any(x in name for x in non_uniform_init_parms):
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )


@require_torch
class EomtForUniversalSegmentationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_id = "tue-mps/coco_panoptic_eomt_large_640"

    @slow
    def test_inference(self):
        model = EomtForUniversalSegmentation.from_pretrained(self.model_id, device_map="auto")
        processor = AutoImageProcessor.from_pretrained(self.model_id)

        image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        inputs = processor(images=image, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            outputs = model(**inputs)

        self.assertTrue(outputs.class_queries_logits.shape == (1, 200, 134))
        self.assertTrue(outputs.masks_queries_logits.shape == (1, 200, 160, 160))

        # fmt: off
        EXPECTED_SLICE = torch.tensor([
            [ 13.2540,   8.9279,   8.6631,  12.3760,  10.1429],
            [ -3.4815, -36.4630, -45.5604, -46.8404, -37.5099],
            [ -6.8689, -44.4206, -62.7591, -59.2928, -47.7035],
            [ -2.9380, -42.0659, -57.4382, -55.1537, -43.5142],
            [ -8.4387, -38.5275, -53.1383, -47.0064, -38.9667],
        ]).to(model.device)
        # fmt: on

        output_slice = outputs.masks_queries_logits[0, 0, :5, :5]
        torch.testing.assert_close(output_slice, EXPECTED_SLICE, rtol=1e-2, atol=1e-2)

        # fmt: off
        EXPECTED_SLICE = torch.tensor([
            [-0.6977, -6.4907, -4.1178, -6.5554, -6.6529],
            [-0.3650, -6.6560, -4.0143, -6.5776, -6.5879],
            [-0.8820, -6.7175, -3.5334, -6.8569, -6.2415],
            [ 0.4502, -5.3911, -3.0232, -5.9411, -6.3243],
            [ 0.3157, -5.6321, -2.6716, -5.5740, -5.5607],
        ]).to(model.device)
        # fmt: on

        output_slice = outputs.class_queries_logits[0, :5, :5]
        torch.testing.assert_close(output_slice, EXPECTED_SLICE, rtol=1e-2, atol=1e-2)

    @require_torch_accelerator
    @require_torch_fp16
    @slow
    def test_inference_fp16(self):
        model = EomtForUniversalSegmentation.from_pretrained(
            self.model_id, dtype=torch.float16, device_map="auto"
        )
        processor = AutoImageProcessor.from_pretrained(self.model_id)

        image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        inputs = processor(images=image, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            outputs = model(**inputs)

        self.assertTrue(outputs.class_queries_logits.shape == (1, 200, 134))
        self.assertTrue(outputs.masks_queries_logits.shape == (1, 200, 160, 160))

    @slow
    def test_semantic_segmentation_inference(self):
        model_id = "tue-mps/ade20k_semantic_eomt_large_512"
        model = EomtForUniversalSegmentation.from_pretrained(model_id, device_map="auto")
        processor = AutoImageProcessor.from_pretrained(model_id)

        image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        inputs = processor(images=image, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            outputs = model(**inputs)

        self.assertTrue(outputs.class_queries_logits.shape == (2, 100, 151))
        self.assertTrue(outputs.masks_queries_logits.shape == (2, 100, 128, 128))

        preds = processor.post_process_semantic_segmentation(outputs, target_sizes=[(image.size[1], image.size[0])])[0]

        self.assertTrue(preds.shape == (image.size[1], image.size[0]))

        # fmt: off
        EXPECTED_SLICE = torch.tensor([
            [39, 39, 39, 39, 39, 39, 39, 39, 39, 39],
            [39, 39, 39, 39, 39, 39, 39, 39, 39, 39],
            [39, 39, 39, 39, 39, 39, 39, 39, 39, 39],
            [39, 39, 39, 39, 39, 39, 39, 39, 39, 39],
            [39, 39, 39, 39, 39, 39, 39, 39, 39, 39],
            [39, 39, 39, 39, 39, 39, 39, 39, 39, 39],
            [39, 39, 39, 39, 39, 39, 39, 39, 39, 39],
            [39, 39, 39, 39, 39, 39, 39, 39, 39, 39],
            [39, 39, 39, 39, 39, 39, 39, 39, 39, 39],
            [39, 39, 39, 39, 39, 39, 39, 39, 39, 39]
        ], device=model.device)
        # fmt: on

        output_slice = preds[:10, :10]
        torch.testing.assert_close(output_slice, EXPECTED_SLICE, rtol=1e-2, atol=1e-2)

    @slow
    def test_panoptic_segmentation_inference(self):
        model = EomtForUniversalSegmentation.from_pretrained(self.model_id, device_map="auto")
        processor = AutoImageProcessor.from_pretrained(self.model_id)

        image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        inputs = processor(images=image, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            outputs = model(**inputs)

        self.assertTrue(outputs.class_queries_logits.shape == (1, 200, 134))
        self.assertTrue(outputs.masks_queries_logits.shape == (1, 200, 160, 160))

        preds = processor.post_process_panoptic_segmentation(outputs, target_sizes=[(image.size[1], image.size[0])])[0]
        segmentation, segments_info = preds["segmentation"], preds["segments_info"]

        # fmt: off
        EXPECTED_SLICE = torch.tensor([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1,  2,  2,  2,  2,  2],
            [-1, -1, -1,  2,  2,  2,  2,  2,  2,  2],
            [ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
            [ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
            [ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
            [ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
            [ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2]
        ], device=model.device)

        EXPECTED_SEGMENTS_INFO = [
            {"id": 0, "label_id": 15, "score": 0.99935},
            {"id": 1, "label_id": 15, "score": 0.998688},
            {"id": 2, "label_id": 57, "score": 0.954325},
            {"id": 3, "label_id": 65, "score": 0.997285},
            {"id": 4, "label_id": 65, "score": 0.99711}
        ]
        # fmt: on

        output_slice = segmentation[:10, :10]
        torch.testing.assert_close(output_slice, EXPECTED_SLICE, rtol=1e-2, atol=1e-2)
        for actual, expected in zip(segments_info, EXPECTED_SEGMENTS_INFO):
            self.assertEqual(actual["id"], expected["id"])
            self.assertEqual(actual["label_id"], expected["label_id"])
            self.assertAlmostEqual(actual["score"], expected["score"], delta=1e-3)

    @slow
    def test_instance_segmentation_inference(self):
        model_id = "tue-mps/coco_instance_eomt_large_640"
        model = EomtForUniversalSegmentation.from_pretrained(model_id, device_map="auto")
        processor = AutoImageProcessor.from_pretrained(model_id)

        image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        inputs = processor(images=image, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            outputs = model(**inputs)

        self.assertTrue(outputs.class_queries_logits.shape == (1, 200, 81))
        self.assertTrue(outputs.masks_queries_logits.shape == (1, 200, 160, 160))

        preds = processor.post_process_instance_segmentation(outputs, target_sizes=[(image.size[1], image.size[0])])[0]
        segmentation, segments_info = preds["segmentation"], preds["segments_info"]

        # fmt: off
        EXPECTED_SLICE = torch.tensor([
            [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
            [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
            [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
            [-1., -1., -1.,  0.,  0.,  1.,  1.,  1.,  1.,  1.],
            [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]
        ], device=model.device)

        EXPECTED_SEGMENTS_INFO = [
            {'id': 0, 'label_id': 57, 'score': 0.871247},
            {'id': 1, 'label_id': 57, 'score': 0.821225},
            {'id': 2, 'label_id': 15, 'score': 0.976252},
            {'id': 3, 'label_id': 65, 'score': 0.972960},
            {'id': 4, 'label_id': 65, 'score': 0.981109},
            {'id': 5, 'label_id': 15, 'score': 0.972689}
        ]
        # fmt: on

        output_slice = segmentation[:10, :10]
        torch.testing.assert_close(output_slice, EXPECTED_SLICE, rtol=1e-2, atol=1e-2)
        for actual, expected in zip(segments_info, EXPECTED_SEGMENTS_INFO):
            self.assertEqual(actual["id"], expected["id"])
            self.assertEqual(actual["label_id"], expected["label_id"])
            self.assertAlmostEqual(actual["score"], expected["score"], delta=1e-3)

    @slow
    def test_segmentation_pipeline(self):
        image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        pipe = pipeline(model=self.model_id, subtask="panoptic", device=torch_device)
        output = pipe(image)

        EXPECTED_OUTPUT_LABELS = ["cat", "cat", "couch", "remote", "remote"]

        output_labels = [segment["label"] for segment in output]
        self.assertEqual(output_labels, EXPECTED_OUTPUT_LABELS)
