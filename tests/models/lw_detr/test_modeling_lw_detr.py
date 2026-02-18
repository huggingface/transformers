# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import unittest
from functools import cached_property

from transformers import (
    DeformableDetrImageProcessor,
    LwDetrConfig,
    LwDetrViTConfig,
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
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import LwDetrForObjectDetection, LwDetrModel, LwDetrViTBackbone


if is_vision_available():
    from PIL import Image

CHECKPOINT = {
    "tiny": "stevenbucaille/lwdetr_tiny_30e_objects365",
    "xlarge": "stevenbucaille/lwdetr_xlarge_30e_objects365",
}


class LwDetrVitModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        num_labels=3,
        num_channels=3,
        use_labels=True,
        is_training=True,
        image_size=256,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        window_block_indices=[1],
        out_indices=[0],
        num_windows=16,
        dropout_prob=0.0,
        attn_implementation="eager",
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.num_channels = num_channels
        self.use_labels = use_labels
        self.image_size = image_size

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.window_block_indices = window_block_indices
        self.out_indices = out_indices
        self.num_windows = num_windows
        self.dropout_prob = dropout_prob
        self.attn_implementation = attn_implementation

        self.is_training = is_training

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return LwDetrViTConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            window_block_indices=self.window_block_indices,
            out_indices=self.out_indices,
            num_windows=self.num_windows,
            hidden_dropout_prob=self.dropout_prob,
            attention_probs_dropout_prob=self.dropout_prob,
            attn_implementation=self.attn_implementation,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def create_and_check_backbone(self, config, pixel_values, labels):
        model = LwDetrViTBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify hidden states
        self.parent.assertEqual(len(result.feature_maps), len(config.out_features))
        self.parent.assertListEqual(
            list(result.feature_maps[0].shape),
            [
                self.batch_size,
                self.hidden_size,
                self.get_config().num_windows_side ** 2,
                self.get_config().num_windows_side ** 2,
            ],
        )

        # verify channels
        self.parent.assertEqual(len(model.channels), len(config.out_features))
        self.parent.assertListEqual(model.channels, [config.hidden_size])

        # verify backbone works with out_features=None
        config.out_features = None
        model = LwDetrViTBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), 1)
        self.parent.assertListEqual(
            list(result.feature_maps[0].shape),
            [self.batch_size, config.hidden_size, config.patch_size, config.patch_size],
        )

        # verify channels
        self.parent.assertEqual(len(model.channels), 1)
        self.parent.assertListEqual(model.channels, [config.hidden_size])


@require_torch
class LwDetrViTBackboneTest(ModelTesterMixin, BackboneTesterMixin, unittest.TestCase):
    all_model_classes = (LwDetrViTBackbone,) if is_torch_available() else ()
    config_class = LwDetrViTConfig
    test_resize_embeddings = False
    test_torch_exportable = True
    model_split_percents = [0.5, 0.87, 0.9]

    def setUp(self):
        self.model_tester = LwDetrVitModelTester(self)

    def test_backbone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_backbone(*config_and_inputs)

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_attention_outputs(self):
        def check_attention_output(inputs_dict, config, model_class):
            config._attn_implementation = "eager"
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            attentions = outputs.attentions

            windowed_attentions = [attentions[i] for i in self.model_tester.window_block_indices]
            unwindowed_attentions = [attentions[i] for i in self.model_tester.out_indices]

            expected_windowed_attention_shape = [
                self.model_tester.batch_size * self.model_tester.num_windows,
                self.model_tester.num_attention_heads,
                self.model_tester.get_config().num_windows_side ** 2,
                self.model_tester.get_config().num_windows_side ** 2,
            ]

            expected_unwindowed_attention_shape = [
                self.model_tester.batch_size,
                self.model_tester.num_attention_heads,
                self.model_tester.image_size,
                self.model_tester.image_size,
            ]

            for i, attention in enumerate(windowed_attentions):
                self.assertListEqual(
                    list(attention.shape),
                    expected_windowed_attention_shape,
                )

            for i, attention in enumerate(unwindowed_attentions):
                self.assertListEqual(
                    list(attention.shape),
                    expected_unwindowed_attention_shape,
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            check_attention_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True

            check_attention_output(inputs_dict, config, model_class)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_stages = self.model_tester.num_hidden_layers
            self.assertEqual(len(hidden_states), expected_num_stages + 1)

            # VitDet's feature maps are of shape (batch_size, num_channels, height, width)
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [
                    self.model_tester.hidden_size,
                    self.model_tester.hidden_size,
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

    # overwrite since LwDetrVitDet only supports retraining gradients of hidden states
    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = self.has_attentions

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs.feature_maps[0]

        # Encoder-/Decoder-only models
        hidden_states = outputs.hidden_states[0]
        hidden_states.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(hidden_states.grad)


def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


class LwDetrModelTester:
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
                target["boxes"] = torch.rand(self.n_targets, 4, device=torch_device, dtype=pixel_values.dtype)
                labels.append(target)

        config = self.get_config()
        config.num_labels = self.num_labels
        return config, pixel_values, pixel_mask, labels

    def get_config(self):
        backbone_config = LwDetrViTConfig(
            hidden_size=16,
            num_hidden_layers=4,
            num_attention_heads=2,
            window_block_indices=[0, 2],
            out_indices=[1, 3],
            num_windows=16,
            image_size=self.image_size,
            dropout_prob=self.dropout,
            attn_implementation=self.attn_implementation,
        )
        return LwDetrConfig(
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

    def create_and_check_lw_detr_model(self, config, pixel_values, pixel_mask, labels):
        model = LwDetrModel(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(pixel_values)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.num_queries, self.d_model))

    def create_and_check_lw_detr_object_detection_head_model(self, config, pixel_values, pixel_mask, labels):
        model = LwDetrForObjectDetection(config=config)
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
class LwDetrModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (LwDetrModel, LwDetrForObjectDetection) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"image-feature-extraction": LwDetrModel, "object-detection": LwDetrForObjectDetection}
        if is_torch_available()
        else {}
    )
    is_encoder_decoder = False
    test_missing_keys = False
    test_torch_exportable = True
    model_split_percents = [0.5, 0.87, 0.9]

    # special case for head models
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class.__name__ == "LwDetrForObjectDetection":
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
        self.model_tester = LwDetrModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=LwDetrConfig,
            has_text_modality=False,
            common_properties=["d_model", "decoder_self_attention_heads"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_lw_detr_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lw_detr_model(*config_and_inputs)

    def test_lw_detr_object_detection_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lw_detr_object_detection_head_model(*config_and_inputs)

    @unittest.skip(reason="LwDetr does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="LwDetr does not use test_inputs_embeds_matches_input_ids")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="LwDetr does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="LwDetr does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="LwDetr does not use token embeddings")
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
            if model_class.__name__ == "LwDetrForObjectDetection":
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
class LwDetrModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        if is_vision_available():
            return {
                "tiny": DeformableDetrImageProcessor.from_pretrained(CHECKPOINT["tiny"]),
                "xlarge": DeformableDetrImageProcessor.from_pretrained(CHECKPOINT["xlarge"]),
            }

    @slow
    def test_inference_object_detection_head_tiny(self):
        size = "tiny"
        model = LwDetrForObjectDetection.from_pretrained(CHECKPOINT[size], attn_implementation="eager").to(
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
                ("cuda", (8, 0)): [-7.7648, -4.1330, -2.9003, -4.0559, -2.9635],
                ("xpu", (3, 0)): [-7.7693, -4.1270, -2.9018, -4.0605, -2.9575],
            }
        )
        expected_logits = torch.tensor(expectations.get_expectation()).to(torch_device)
        torch.testing.assert_close(outputs.logits.flatten()[:5], expected_logits, rtol=2e-4, atol=2e-4)

        expected_boxes_shape = torch.Size((1, model.config.num_queries, 4))
        self.assertEqual(outputs.pred_boxes.shape, expected_boxes_shape)

        expectations = Expectations(
            {
                ("cuda", (8, 0)): [0.1694, 0.1979, 0.2121, 0.0912, 0.2537],
                ("xpu", (3, 0)): [0.1694, 0.1979, 0.2121, 0.0912, 0.2537],
            }
        )
        expected_boxes = torch.tensor(expectations.get_expectation()).to(torch_device)

        torch.testing.assert_close(outputs.pred_boxes.flatten()[:5], expected_boxes, rtol=2e-4, atol=2e-4)

        results = image_processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=[image.size[::-1]]
        )[0]

        expectations = Expectations(
            {
                ("cuda", (8, 0)): [0.8684, 0.7492, 0.7146, 0.4362],
                ("xpu", (3, 0)): [0.8676, 0.7527, 0.7177, 0.4391],
            }
        )
        expected_scores = torch.tensor(expectations.get_expectation()).to(torch_device)

        expected_labels = [140, 133, 140, 133]

        expectations = Expectations(
            {
                ("cuda", (8, 0)): [
                    [4.9333, 56.6130, 319.7758, 474.7774],
                    [40.5547, 73.0968, 176.2951, 116.8605],
                    [340.3403, 25.1044, 640.2798, 368.7382],
                    [334.2971, 77.0087, 371.2877, 189.8089],
                ],
                ("xpu", (3, 0)): [
                    [4.8948, 56.5549, 319.8077, 474.7937],
                    [40.5620, 73.1059, 176.2996, 116.8567],
                    [340.3327, 25.1026, 640.3193, 368.6754],
                    [334.2945, 76.9876, 371.2914, 189.8221],
                ],
            }
        )
        expected_slice_boxes = torch.tensor(expectations.get_expectation()).to(torch_device)
        torch.testing.assert_close(results["scores"][:4], expected_scores, atol=1e-3, rtol=2e-4)
        self.assertSequenceEqual(results["labels"][:4].tolist(), expected_labels)
        torch.testing.assert_close(results["boxes"][:4], expected_slice_boxes, atol=1e-3, rtol=2e-4)

    @slow
    def test_inference_object_detection_head_xlarge(self):
        size = "xlarge"
        model = LwDetrForObjectDetection.from_pretrained(CHECKPOINT[size], attn_implementation="eager").to(
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
                ("cuda", (8, 0)): [-11.9394, -4.3419, -4.4172, -5.0299, -6.9282],
                ("xpu", (3, 0)): [-11.9292, -4.3307, -4.4075, -5.0207, -6.9211],
            }
        )

        expected_logits = torch.tensor(expectations.get_expectation()).to(torch_device)
        torch.testing.assert_close(outputs.logits.flatten()[:5], expected_logits, rtol=2e-4, atol=2e-4)

        expected_boxes_shape = torch.Size((1, model.config.num_queries, 4))
        self.assertEqual(outputs.pred_boxes.shape, expected_boxes_shape)

        expectations = Expectations(
            {
                ("cuda", (8, 0)): [0.7689, 0.4107, 0.4617, 0.7244, 0.2526],
                ("xpu", (3, 0)): [0.7688, 0.4106, 0.4618, 0.7245, 0.2526],
            }
        )
        expected_boxes = torch.tensor(expectations.get_expectation()).to(torch_device)

        torch.testing.assert_close(outputs.pred_boxes.flatten()[:5], expected_boxes, rtol=2e-4, atol=2e-4)

        results = image_processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=[image.size[::-1]]
        )[0]

        expectations = Expectations(
            {
                ("cuda", (8, 0)): [0.9746, 0.9717, 0.9344, 0.8182],
                ("xpu", (3, 0)): [0.9745, 0.9715, 0.9339, 0.8163],
            }
        )
        expected_scores = torch.tensor(expectations.get_expectation()).to(torch_device)

        expected_labels = [140, 140, 133, 133]

        expectations = Expectations(
            {
                ("cuda", (8, 0)): [
                    [7.4541, 54.2878, 315.8890, 474.8681],
                    [344.3325, 23.2591, 639.7999, 370.9900],
                    [40.4797, 73.3092, 175.6086, 116.9654],
                    [333.9930, 77.1547, 370.4000, 186.1230],
                ],
                ("xpu", (3, 0)): [
                    [7.4487, 54.2931, 315.8945, 474.8726],
                    [344.2597, 23.2305, 639.8082, 370.9894],
                    [40.4780, 73.3095, 175.6083, 116.9673],
                    [333.9890, 77.1453, 370.4069, 186.1300],
                ],
            }
        )
        expected_slice_boxes = torch.tensor(expectations.get_expectation()).to(torch_device)

        torch.testing.assert_close(results["scores"][:4], expected_scores, atol=1e-3, rtol=2e-4)
        self.assertSequenceEqual(results["labels"][:4].tolist(), expected_labels)
        torch.testing.assert_close(results["boxes"][:4], expected_slice_boxes, atol=1e-3, rtol=2e-4)
