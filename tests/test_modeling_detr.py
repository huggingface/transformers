# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch DETR model. """


import inspect
import math
import tempfile
import unittest

from PIL import Image

from transformers import is_torch_available, is_vision_available
from transformers.file_utils import cached_property
from transformers.models.auto import get_values
from transformers.testing_utils import require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_generation_utils import GenerationTesterMixin
from .test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import (
        MODEL_MAPPING,
        DetrConfig,
        DetrForObjectDetection,
        DetrModel,
    )


if is_vision_available():
    from PIL import Image

    from transformers import DetrFeatureExtractor


@require_torch
class DetrModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        is_training=True,
        use_labels=True,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_queries=10,
        num_channels=3,
        image_size=200,
        n_targets=8,
        num_labels=91,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_queries = num_queries
        self.num_channels = num_channels
        self.image_size = image_size
        self.n_targets = n_targets
        self.num_labels = num_labels

        # we can also set the expected seq length for both encoder and decoder
        self.encoder_seq_length = math.ceil(self.image_size / 32) ** 2
        self.decoder_seq_length = self.num_queries

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        pixel_mask = torch.ones([self.batch_size, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            # labels is a list of Dict (each Dict being the labels for a given example in the batch)
            labels = []
            for i in range(self.batch_size):
                target = {}
                target["class_labels"] = torch.randint(high=self.num_labels, size=(self.n_targets,))
                target["boxes"] = torch.rand(self.n_targets, 4)
                labels.append(target)

        config = DetrConfig(
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            num_queries=self.num_queries,
            num_labels=self.num_labels,
        )
        return config, pixel_values, pixel_mask, labels

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, pixel_mask, labels = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
        return config, inputs_dict

    def create_and_check_detr_model(self, config, pixel_values, pixel_mask, labels):
        model = DetrModel(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(pixel_values)

        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.decoder_seq_length, self.hidden_size)
        )

    def create_and_check_detr_object_detection_head_model(self, config, pixel_values, pixel_mask, labels):
        model = DetrForObjectDetection(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(pixel_values)

        self.parent.assertEqual(result.pred_logits.shape, (self.batch_size, self.num_queries, self.num_labels + 1))
        self.parent.assertEqual(result.pred_boxes.shape, (self.batch_size, self.num_queries, 4))

        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.pred_logits.shape, (self.batch_size, self.num_queries, self.num_labels + 1))
        self.parent.assertEqual(result.pred_boxes.shape, (self.batch_size, self.num_queries, 4))


@require_torch
class DetrModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            DetrModel,
            DetrForObjectDetection,
        )
        if is_torch_available()
        else ()
    )
    is_encoder_decoder = True
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False

    # special case for DetrForObjectDetection model
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class.__name__ == "DetrForObjectDetection":
                labels = []
                for i in range(self.model_tester.batch_size):
                    target = {}
                    target["class_labels"] = torch.randint(
                        high=self.model_tester.num_labels, size=(self.model_tester.n_targets,)
                    )
                    target["boxes"] = torch.rand(self.model_tester.n_targets, 4)
                    labels.append(target)
                inputs_dict["labels"] = labels

        return inputs_dict

    def setUp(self):
        self.model_tester = DetrModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DetrConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_detr_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_detr_model(*config_and_inputs)

    def test_detr_object_detection_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_detr_object_detection_head_model(*config_and_inputs)

    def test_inputs_embeds(self):
        # DETR does not use inputs_embeds
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (torch.nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, torch.nn.Linear))

    def test_generate_without_input_ids(self):
        # DETR is not a generative model
        pass

    def test_resize_tokens_embeddings(self):
        # DETR does not use token embeddings
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            if model.config.is_encoder_decoder:
                expected_arg_names = [
                    "pixel_values",
                    "pixel_mask",
                ]
                expected_arg_names.extend(
                    ["head_mask", "decoder_head_mask", "encoder_outputs"]
                    if "head_mask" and "decoder_head_mask" in arg_names
                    else []
                )
                self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)
            else:
                expected_arg_names = ["pixel_values", "pixel_mask"]
                self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_save_load_strict(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model2, info = model_class.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertEqual(info["missing_keys"], [])


def assert_tensors_close(a, b, atol=1e-12, prefix=""):
    """If tensors have different shapes, different values or a and b are not both tensors, raise a nice Assertion error."""
    if a is None and b is None:
        return True
    try:
        if torch.allclose(a, b, atol=atol):
            return True
        raise
    except Exception:
        pct_different = (torch.gt((a - b).abs(), atol)).float().mean().item()
        if a.numel() > 100:
            msg = f"tensor values are {pct_different:.1%} percent different."
        else:
            msg = f"{a} != {b}"
        if prefix:
            msg = prefix + ": " + msg
        raise AssertionError(msg)


def _long_tensor(tok_lst):
    return torch.tensor(tok_lst, dtype=torch.long, device=torch_device)


TOLERANCE = 1e-4


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/cats.png")
    return image


@require_torch
@slow
class DetrModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_feature_extractor(self):
        # TODO replace by facebook/detr-resnet-50
        return DetrFeatureExtractor() if is_vision_available() else None

    def test_inference_no_head(self):
        # TODO replace by facebook/detr-resnet-50
        model = DetrModel.from_pretrained("nielsr/detr-resnet-50-new").to(torch_device)
        model.eval()

        feature_extractor = self.default_feature_extractor
        image = prepare_img()
        encoding = feature_extractor(images=image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**encoding)

        expected_shape = torch.Size((1, 100, 256))
        assert outputs.last_hidden_state.shape == expected_shape
        expected_slice = torch.tensor(
            [[0.0616, -0.5146, -0.4032], [-0.7629, -0.4934, -1.7153], [-0.4768, -0.6403, -0.7826]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-4))

    def test_inference_object_detection_head(self):
        model = DetrForObjectDetection.from_pretrained("nielsr/detr-resnet-50-new").to(torch_device)
        model.eval()

        feature_extractor = self.default_feature_extractor
        image = prepare_img()
        encoding = feature_extractor(images=image, return_tensors="pt").to(torch_device)
        pixel_values = encoding["pixel_values"].to(torch_device)
        pixel_mask = encoding["pixel_mask"].to(torch_device)

        with torch.no_grad():
            outputs = model(pixel_values, pixel_mask)

        expected_shape_logits = torch.Size((1, model.config.num_queries, model.config.num_labels + 1))
        self.assertEqual(outputs.pred_logits.shape, expected_shape_logits)
        expected_slice_logits = torch.tensor(
            [[-19.1194, -0.0893, -11.0154], [-17.3640, -1.8035, -14.0219], [-20.0461, -0.5837, -11.1060]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.pred_logits[0, :3, :3], expected_slice_logits, atol=1e-4))

        expected_shape_boxes = torch.Size((1, model.config.num_queries, 4))
        self.assertEqual(outputs.pred_boxes.shape, expected_shape_boxes)
        expected_slice_boxes = torch.tensor(
            [[0.4433, 0.5302, 0.8853], [0.5494, 0.2517, 0.0529], [0.4998, 0.5360, 0.9956]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.pred_boxes[0, :3, :3], expected_slice_boxes, atol=1e-4))
