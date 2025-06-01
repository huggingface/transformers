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

import tempfile
import unittest

import numpy as np
import requests

from transformers import AutoImageProcessor, EoMTConfig, EoMTForUniversalSegmentation
from transformers.testing_utils import require_torch, require_torch_accelerator, require_torch_fp16, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image


class EoMTForUniversalSegmentationTester:
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
        return EoMTConfig(**config)

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
        inputs_dict = {"pixel_values": pixel_values, "mask_labels": mask_labels, "class_labels": class_labels}
        return config, inputs_dict


@require_torch
class EoMTForUniversalSegmentationTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (EoMTForUniversalSegmentation,) if is_torch_available() else ()
    is_encoder_decoder = False
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False
    test_torch_exportable = False

    def setUp(self):
        self.model_tester = EoMTForUniversalSegmentationTester(self)
        self.config_tester = ConfigTester(self, config_class=EoMTConfig, has_text_modality=False)

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

        model = EoMTForUniversalSegmentation(config).to(torch_device)
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

    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            torch.manual_seed(2)
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            out_2 = outputs[0].cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname)
                model.to(torch_device)

                torch.manual_seed(2)
                with torch.no_grad():
                    after_outputs = model(**self._prepare_for_class(inputs_dict, model_class))

                # Make sure we don't have nans
                out_1 = after_outputs[0].cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    @unittest.skip(reason="Fix Me later")
    def test_determinism(self):
        pass

    @unittest.skip(reason="Fix Me later")
    def test_model_outputs_equivalence(self):
        pass


@require_torch
class EoMTForUniversalSegmentationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_id = "yaswanthgali/coco_panoptic_eomt_large_640-hf"

    @slow
    def test_inference(self):
        model = EoMTForUniversalSegmentation.from_pretrained(self.model_id, device_map="auto")
        processor = AutoImageProcessor.from_pretrained(self.model_id)

        image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        inputs = processor(images=image, segmentation_type="panoptic", return_tensors="pt").to(model.device)

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
        print(output_slice)
        self.assertTrue(torch.allclose(output_slice, EXPECTED_SLICE, atol=1e-3))

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
        self.assertTrue(torch.allclose(output_slice, EXPECTED_SLICE, atol=1e-3))

    @require_torch_accelerator
    @require_torch_fp16
    @slow
    def test_inference_fp16(self):
        model = EoMTForUniversalSegmentation.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map="auto"
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
        model_id = "yaswanthgali/ade20k_semantic_eomt_large_512-hf"
        model = EoMTForUniversalSegmentation.from_pretrained(model_id, device_map="auto")
        processor = AutoImageProcessor.from_pretrained(model_id)

        image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        inputs = processor(images=image, return_tensors="pt").to(model.device)
        patch_offsets = inputs.pop("patch_offsets", None)

        with torch.inference_mode():
            outputs = model(**inputs)

        self.assertTrue(outputs.class_queries_logits.shape == (2, 100, 151))
        self.assertTrue(outputs.masks_queries_logits.shape == (2, 100, 128, 128))

        preds = processor.post_process_semantic_segmentation(
            outputs, original_image_sizes=[image.size], patch_offsets=patch_offsets
        )

        self.assertTrue(preds.shape[1:] == (image.size[0], image.size[1]))

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

        output_slice = preds[0, :10, :10]
        self.assertTrue(torch.allclose(output_slice, EXPECTED_SLICE, atol=1e-4))

    @slow
    def test_panoptic_segmentation_inference(self):
        model = EoMTForUniversalSegmentation.from_pretrained(self.model_id, device_map="auto")
        processor = AutoImageProcessor.from_pretrained(self.model_id)

        image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        inputs = processor(images=image, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            outputs = model(**inputs)

        self.assertTrue(outputs.class_queries_logits.shape == (1, 200, 134))
        self.assertTrue(outputs.masks_queries_logits.shape == (1, 200, 160, 160))

        preds = processor.post_process_panoptic_segmentation(outputs, original_image_sizes=[image.size])[0]
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
        self.assertTrue(torch.allclose(output_slice, EXPECTED_SLICE, atol=1e-4))
        self.assertEqual(segments_info, EXPECTED_SEGMENTS_INFO)
