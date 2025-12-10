# coding = utf-8
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
import os
import tempfile
import unittest
from functools import cached_property

import numpy as np

from transformers import (
    CONFIG_NAME,
    DeformableDetrImageProcessor,
    RfDetrConfig,
    RfDetrDinov2Config,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_vision,
    slow,
    torch_device,
)

from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import RfDetrDinov2Backbone, RfDetrForObjectDetection, RfDetrModel

if is_vision_available():
    from PIL import Image

CHECKPOINT = {
    "base": "stevenbucaille/rf-detr-base",
    "large": "stevenbucaille/rf-detr-large",
}


def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


class RfDetrModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        is_training=True,
        image_size=256,
        num_labels=5,
        n_targets=4,
        use_labels=True,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        batch_norm_eps=1e-5,
        # backbone
        backbone_config=None,
        # projector
        projector_scale_factors=[0.5, 2.0],
        # decoder
        d_model=32,
        decoder_ffn_dim=32,
        decoder_layers=2,
        decoder_self_attention_heads=2,
        decoder_cross_attention_heads=4,
        # model
        num_queries=10,
        group_detr=2,
        dropout=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        attn_implementation="eager",
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.num_channels = 3
        self.image_size = image_size
        self.num_labels = num_labels
        self.n_targets = n_targets
        self.use_labels = use_labels
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.batch_norm_eps = batch_norm_eps
        self.backbone_config = backbone_config
        self.projector_scale_factors = projector_scale_factors
        self.d_model = d_model
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_self_attention_heads = decoder_self_attention_heads
        self.decoder_cross_attention_heads = decoder_cross_attention_heads
        self.num_queries = num_queries
        self.group_detr = group_detr
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.attention_dropout = attention_dropout
        self.attn_implementation = attn_implementation

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        pixel_mask = torch.ones([self.batch_size, self.image_size, self.image_size], device=torch_device)
        labels = None
        if self.use_labels:
            labels = []
            for i in range(self.batch_size):
                target = {}
                target["class_labels"] = torch.randint(
                    high=self.num_labels, size=(self.n_targets,), device=torch_device
                )
                target["boxes"] = torch.rand(self.n_targets, 4, device=torch_device)
                labels.append(target)

        config = self.get_config()
        config.num_labels = self.num_labels
        return config, pixel_values, pixel_mask, labels

    def get_config(self):
        backbone_config = RfDetrDinov2Config(
            attention_probs_dropout_prob=0.0,
            drop_path_rate=0.0,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-06,
            layerscale_value=1.0,
            mlp_ratio=4,
            num_attention_heads=2,
            num_channels=3,
            num_hidden_layers=4,
            qkv_bias=True,
            use_swiglu_ffn=False,
            out_features=["stage2", "stage3"],
            hidden_size=self.d_model,
            patch_size=16,
            num_windows=2,
            image_size=self.image_size,
            attn_implementation=self.attn_implementation,
        )
        return RfDetrConfig(
            backbone_config=backbone_config,
            d_model=self.d_model,
            projector_scale_factors=self.projector_scale_factors,
            decoder_ffn_dim=self.decoder_ffn_dim,
            decoder_layers=self.decoder_layers,
            decoder_self_attention_heads=self.decoder_self_attention_heads,
            decoder_cross_attention_heads=self.decoder_cross_attention_heads,
            num_queries=self.num_queries,
            group_detr=self.group_detr,
            dropout=self.dropout,
            activation_dropout=self.activation_dropout,
            attention_dropout=self.attention_dropout,
            attn_implementation=self.attn_implementation,
            _attn_implementation=self.attn_implementation,
        )

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, pixel_mask, labels = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
        return config, inputs_dict

    def create_and_check_rf_detr_model(self, config, pixel_values, pixel_mask, labels):
        model = RfDetrModel(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(pixel_values)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.num_queries, self.d_model))

    def create_and_check_rf_detr_object_detection_head_model(self, config, pixel_values, pixel_mask, labels):
        model = RfDetrForObjectDetection(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(pixel_values)

        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_queries, self.num_labels))
        self.parent.assertEqual(result.pred_boxes.shape, (self.batch_size, self.num_queries, 4))

        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_queries, self.num_labels))
        self.parent.assertEqual(result.pred_boxes.shape, (self.batch_size, self.num_queries, 4))


@require_torch
class RfDetrModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (RfDetrModel, RfDetrForObjectDetection) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"image-feature-extraction": RfDetrModel, "object-detection": RfDetrForObjectDetection}
        if is_torch_available()
        else {}
    )
    is_encoder_decoder = False
    test_torchscript = False
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False
    test_torch_exportable = True

    # special case for head models
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class.__name__ == "RfDetrForObjectDetection":
                labels = []
                for i in range(self.model_tester.batch_size):
                    target = {}
                    target["class_labels"] = torch.ones(
                        size=(self.model_tester.n_targets,), device=torch_device, dtype=torch.long
                    )
                    target["boxes"] = torch.ones(
                        self.model_tester.n_targets, 4, device=torch_device, dtype=torch.float
                    )
                    labels.append(target)
                inputs_dict["labels"] = labels

        return inputs_dict

    def setUp(self):
        self.model_tester = RfDetrModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=RfDetrConfig,
            has_text_modality=False,
            common_properties=["d_model", "decoder_self_attention_heads"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_rf_detr_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_rf_detr_model(*config_and_inputs)

    def test_rf_detr_object_detection_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_rf_detr_object_detection_head_model(*config_and_inputs)

    @unittest.skip(reason="RTDetr does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="RTDetr does not use test_inputs_embeds_matches_input_ids")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="RTDetr does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="RTDetr does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="RTDetr does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

    def test_attention_outputs(self):
        def check_attention_outputs(inputs_dict, config, model_class):
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.decoder_layers)
            expected_attentions_shape = [
                self.model_tester.batch_size,
                self.model_tester.decoder_self_attention_heads,
                self.model_tester.num_queries,
                self.model_tester.num_queries,
            ]
            for i in range(self.model_tester.decoder_layers):
                self.assertEqual(expected_attentions_shape, list(attentions[i].shape))

            # check cross_attentions outputs
            expected_attentions_shape = [
                self.model_tester.batch_size,
                self.model_tester.num_queries,
                self.model_tester.decoder_cross_attention_heads,
                config.num_feature_levels,
                config.decoder_n_points,
            ]
            cross_attentions = outputs.cross_attentions
            self.assertEqual(len(cross_attentions), self.model_tester.decoder_layers)
            for i in range(self.model_tester.decoder_layers):
                self.assertEqual(expected_attentions_shape, list(cross_attentions[i].shape))

            out_len = len(outputs)

            correct_outlen = 8  # 6 + attentions + cross_attentions

            # Object Detection model returns pred_logits, pred_boxes and auxiliary outputs
            if model_class.__name__ == "RfDetrForObjectDetection":
                correct_outlen += 2
                if "labels" in inputs_dict:
                    correct_outlen += 3  # loss, loss_dict and auxiliary outputs is added to beginning

            self.assertEqual(correct_outlen, out_len)

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        inputs_dict["output_hidden_states"] = False

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            check_attention_outputs(inputs_dict, config, model_class)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            check_attention_outputs(inputs_dict, config, model_class)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_hidden_states = self.model_tester.decoder_layers + 1
            self.assertEqual(len(hidden_states), expected_num_hidden_states)

            for i in range(expected_num_hidden_states):
                self.assertListEqual(
                    list(hidden_states[i].shape),
                    [
                        self.model_tester.batch_size,
                        self.model_tester.num_queries,
                        self.model_tester.d_model,
                    ],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = False
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            check_hidden_states_output(inputs_dict, config, model_class)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            print("Model class:", model_class)
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if (
                        "level_embed" in name
                        or "sampling_offsets.bias" in name
                        or "value_proj" in name
                        or "output_proj" in name
                        or "reference_points" in name
                        or "class_embed" in name
                        or "gamma_1" in name
                        or "gamma_2" in name
                    ):
                        continue
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True

        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        # we take the first output since last_hidden_state is the first item
        output = outputs.last_hidden_state

        hidden_states = outputs.hidden_states[0]
        attentions = outputs.attentions[0]
        hidden_states.retain_grad()
        attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(hidden_states.grad)
        self.assertIsNotNone(attentions.grad)

    def test_save_load(self):
        def check_save_load(out1, out2):
            # make sure we don't have nans
            out_2 = out2.cpu().numpy()
            out_2[np.isnan(out_2)] = 0
            out_2 = out_2[~np.isneginf(out_2)]

            out_1 = out1.cpu().numpy()
            out_1[np.isnan(out_1)] = 0
            out_1 = out_1[~np.isneginf(out_1)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # the config file (and the generation config file, if it can generate) should be saved
                self.assertTrue(os.path.exists(os.path.join(tmpdirname, CONFIG_NAME)))

                model = model_class.from_pretrained(tmpdirname)
                model.config._attn_implementation = "eager"  # TODO Have to force eager for testing, why ?
                model.to(torch_device)
                with torch.no_grad():
                    second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

                # Save and load second time because `from_pretrained` adds a bunch of new config fields
                # so we need to make sure those fields can be loaded back after saving
                # Simply init as `model(config)` doesn't add those fields
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname)

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_save_load(tensor1, tensor2)
            else:
                check_save_load(first, second)

    def test_forward_auxiliary_loss(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.auxiliary_loss = True

        # only test for object detection and segmentation model
        for model_class in self.all_model_classes[1:]:
            model = model_class(config)
            model.to(torch_device)

            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)

            outputs = model(**inputs)

            self.assertIsNotNone(outputs.auxiliary_outputs)
            self.assertEqual(len(outputs.auxiliary_outputs), self.model_tester.decoder_layers - 1)


@require_torch
@require_vision
class RfDetrModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        if is_vision_available():
            return {
                "base": DeformableDetrImageProcessor.from_pretrained(CHECKPOINT["base"]),
                "large": DeformableDetrImageProcessor.from_pretrained(CHECKPOINT["large"]),
            }

    @slow
    def test_inference_object_detection_head_base(self):
        size = "base"
        model = RfDetrForObjectDetection.from_pretrained(CHECKPOINT[size], attn_implementation="eager").to(
            torch_device
        )

        image_processor = self.default_image_processor[size]
        image = prepare_img()
        encoding = image_processor(images=image, return_tensors="pt").to(torch_device)
        pixel_values = encoding["pixel_values"].to(torch_device)
        pixel_mask = encoding["pixel_mask"].to(torch_device)
        with torch.no_grad():
            outputs = model(pixel_values, pixel_mask)

        expected_logits_shape = torch.Size((1, model.config.num_queries, model.config.num_labels))
        self.assertEqual(outputs.logits.shape, expected_logits_shape)

        expectations = Expectations(
            {
                (None, None): [-7.60410, -4.65943, -10.03144, -5.63881, -9.88291],
            }
        )
        expected_logits = torch.tensor(expectations.get_expectation()).to(torch_device)
        torch.testing.assert_close(outputs.logits.flatten()[:5], expected_logits, rtol=2e-4, atol=2e-4)

        expected_boxes_shape = torch.Size((1, model.config.num_queries, 4))
        self.assertEqual(outputs.pred_boxes.shape, expected_boxes_shape)

        expectations = Expectations(
            {
                (None, None): [0.25465, 0.54864, 0.48583, 0.86991, 0.16926],
            }
        )
        expected_boxes = torch.tensor(expectations.get_expectation()).to(torch_device)

        torch.testing.assert_close(outputs.pred_boxes.flatten()[:5], expected_boxes, rtol=2e-4, atol=2e-4)

        results = image_processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=[image.size[::-1]]
        )[0]

        expectations = Expectations(
            {
                (None, None): [0.9596, 0.9335, 0.8987, 0.7271],
            }
        )
        expected_scores = torch.tensor(expectations.get_expectation()).to(torch_device)

        expected_labels = [17, 17, 75, 75]

        expectations = Expectations(
            {
                (None, None): [
                    [7.5101, 54.5667, 318.4421, 472.1259],
                    [343.0202, 23.9165, 639.3325, 372.2062],
                    [40.7178, 72.6109, 175.9414, 117.5903],
                    [333.5370, 76.9845, 370.3848, 187.3158],
                ],
            }
        )
        expected_slice_boxes = torch.tensor(expectations.get_expectation()).to(torch_device)

        torch.testing.assert_close(results["scores"][:4], expected_scores, atol=1e-3, rtol=2e-4)
        self.assertSequenceEqual(results["labels"][:4].tolist(), expected_labels)
        torch.testing.assert_close(results["boxes"][:4], expected_slice_boxes, atol=1e-3, rtol=2e-4)

    @slow
    def test_inference_object_detection_head_large(self):
        size = "large"
        model = RfDetrForObjectDetection.from_pretrained(CHECKPOINT[size], attn_implementation="eager").to(
            torch_device
        )

        image_processor = self.default_image_processor[size]
        image = prepare_img()
        encoding = image_processor(images=image, return_tensors="pt").to(torch_device)
        pixel_values = encoding["pixel_values"].to(torch_device)
        pixel_mask = encoding["pixel_mask"].to(torch_device)
        with torch.no_grad():
            outputs = model(pixel_values, pixel_mask)

        expected_logits_shape = torch.Size((1, model.config.num_queries, model.config.num_labels))
        self.assertEqual(outputs.logits.shape, expected_logits_shape)

        expectations = Expectations(
            {
                (None, None): [-7.60888, -4.36906, -4.98865, -8.06598, -5.52970],
            }
        )
        expected_logits = torch.tensor(expectations.get_expectation()).to(torch_device)
        torch.testing.assert_close(outputs.logits.flatten()[:5], expected_logits, rtol=2e-3, atol=2e-3)

        expected_boxes_shape = torch.Size((1, model.config.num_queries, 4))
        self.assertEqual(outputs.pred_boxes.shape, expected_boxes_shape)

        expectations = Expectations(
            {
                (None, None): [0.25576, 0.55051, 0.47765, 0.87141, 0.76966],
            }
        )
        expected_boxes = torch.tensor(expectations.get_expectation()).to(torch_device)

        torch.testing.assert_close(outputs.pred_boxes.flatten()[:5], expected_boxes, rtol=2e-3, atol=2e-3)

        results = image_processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=[image.size[::-1]]
        )[0]

        expectations = Expectations(
            {
                (None, None): [0.9558, 0.9538, 0.9465, 0.9084],
            }
        )
        expected_scores = torch.tensor(expectations.get_expectation()).to(torch_device)

        expected_labels = [75, 17, 17, 75]

        expectations = Expectations(
            {
                (None, None): [
                    [40.3736, 73.1451, 175.8807, 117.5796],
                    [345.1129, 24.5076, 640.0582, 373.1581],
                    [10.8431, 55.1121, 316.5317, 473.3869],
                    [333.9091, 76.8915, 370.1848, 186.7155],
                ],
            }
        )
        expected_slice_boxes = torch.tensor(expectations.get_expectation()).to(torch_device)

        torch.testing.assert_close(results["scores"][:4], expected_scores, atol=1e-3, rtol=2e-4)
        self.assertSequenceEqual(results["labels"][:4].tolist(), expected_labels)
        torch.testing.assert_close(results["boxes"][:4], expected_slice_boxes, atol=1e-3, rtol=2e-4)


class RfDetrDinov2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=32,
        patch_size=2,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_sequence_label_size=10,
        initializer_range=0.02,
        mask_ratio=0.5,
        num_windows=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range

        # in Dinov2, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1
        self.mask_ratio = mask_ratio
        self.num_masks = int(mask_ratio * self.seq_length)
        self.mask_length = num_patches
        self.num_windows = num_windows

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return RfDetrDinov2Config(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
            num_windows=self.num_windows,
        )

    def create_and_check_backbone(self, config, pixel_values):
        model = RfDetrDinov2Backbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify hidden states
        self.parent.assertEqual(len(result.feature_maps), len(config.out_features))
        expected_size = self.image_size // config.patch_size
        self.parent.assertListEqual(
            list(result.feature_maps[0].shape), [self.batch_size, model.channels[0], expected_size, expected_size]
        )

        # verify channels
        self.parent.assertEqual(len(model.channels), len(config.out_features))

        # verify backbone works with out_features=None
        config.out_features = None
        model = RfDetrDinov2Backbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), 1)
        self.parent.assertListEqual(
            list(result.feature_maps[0].shape), [self.batch_size, model.channels[0], expected_size, expected_size]
        )

        # verify channels
        self.parent.assertEqual(len(model.channels), 1)

        # verify backbone works with apply_layernorm=False and reshape_hidden_states=False
        config.apply_layernorm = False
        config.reshape_hidden_states = False

        model = RfDetrDinov2Backbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), 1)
        self.parent.assertListEqual(
            list(result.feature_maps[0].shape), [self.batch_size, self.seq_length, self.hidden_size]
        )

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class RfDetrDinov2BackboneTest(unittest.TestCase, BackboneTesterMixin):
    all_model_classes = (RfDetrDinov2Backbone,) if is_torch_available() else ()
    config_class = RfDetrDinov2Config

    has_attentions = False

    def setUp(self):
        self.model_tester = RfDetrDinov2ModelTester(self)
