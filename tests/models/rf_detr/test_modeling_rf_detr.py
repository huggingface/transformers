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

from transformers import (
    RfDetrConfig,
    RfDetrWindowedDinov2Config,
    is_torch_available,
)
from transformers.testing_utils import require_torch, torch_device

from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        RfDetrForInstanceSegmentation,
        RfDetrForObjectDetection,
        RfDetrModel,
        RfDetrWindowedDinov2Backbone,
    )


class RfDetrWindowedDinov2BackboneModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        is_training=True,
        num_channels=3,
        image_size=256,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        window_block_indexes=[1],
        out_indices=[1],
        num_windows=2,
        dropout_prob=0.0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.num_channels = num_channels
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.window_block_indexes = window_block_indexes
        self.out_indices = out_indices
        self.num_windows = num_windows
        self.dropout_prob = dropout_prob

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        return RfDetrWindowedDinov2Config(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            window_block_indexes=self.window_block_indexes,
            out_indices=self.out_indices,
            num_windows=self.num_windows,
            num_register_tokens=4,
            image_size=self.image_size,
            hidden_dropout_prob=self.dropout_prob,
            attention_probs_dropout_prob=self.dropout_prob,
        )

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def create_and_check_backbone(self, config, pixel_values):
        model = RfDetrWindowedDinov2Backbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        self.parent.assertEqual(len(result.feature_maps), len(config.out_features))
        self.parent.assertListEqual(
            list(result.feature_maps[0].shape),
            [
                self.batch_size,
                self.hidden_size,
                self.image_size // config.patch_size,
                self.image_size // config.patch_size,
            ],
        )

        self.parent.assertEqual(len(model.channels), len(config.out_features))
        self.parent.assertListEqual(model.channels, [config.hidden_size])

        config.out_features = None
        model = RfDetrWindowedDinov2Backbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        self.parent.assertEqual(len(result.feature_maps), 1)
        self.parent.assertListEqual(
            list(result.feature_maps[0].shape),
            [
                self.batch_size,
                config.hidden_size,
                self.image_size // config.patch_size,
                self.image_size // config.patch_size,
            ],
        )

        self.parent.assertEqual(len(model.channels), 1)
        self.parent.assertListEqual(model.channels, [config.hidden_size])


@require_torch
class RfDetrWindowedDinov2BackboneTest(ModelTesterMixin, BackboneTesterMixin, unittest.TestCase):
    all_model_classes = (RfDetrWindowedDinov2Backbone,) if is_torch_available() else ()
    config_class = RfDetrWindowedDinov2Config
    test_resize_embeddings = False
    test_torch_exportable = True
    model_split_percents = [0.5, 0.87, 0.9]

    def setUp(self):
        self.model_tester = RfDetrWindowedDinov2BackboneModelTester(self)

    def test_backbone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_backbone(*config_and_inputs)

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), nn.Module)
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    @unittest.skip(reason="RfDetrWindowedDinov2Backbone does not return attention maps.")
    def test_attention_outputs(self):
        pass

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states
            expected_num_hidden_states = self.model_tester.num_hidden_layers + 1
            self.assertEqual(len(hidden_states), expected_num_hidden_states)
            self.assertEqual(hidden_states[0].shape[-1], self.model_tester.hidden_size)

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            check_hidden_states_output(inputs_dict, config, model_class)

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True

        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)
        outputs = model(**inputs)
        output = outputs.feature_maps[0]

        hidden_states = outputs.hidden_states[0]
        hidden_states.retain_grad()
        output.flatten()[0].backward(retain_graph=True)
        self.assertIsNotNone(hidden_states.grad)


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
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.num_channels = 3
        self.image_size = image_size
        self.num_labels = num_labels
        self.n_targets = n_targets
        self.use_labels = use_labels
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

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        pixel_mask = torch.ones([self.batch_size, self.image_size, self.image_size], device=torch_device)

        labels = None
        if self.use_labels:
            labels = []
            for _ in range(self.batch_size):
                target = {
                    "class_labels": torch.randint(high=self.num_labels, size=(self.n_targets,), device=torch_device),
                    "boxes": torch.rand(self.n_targets, 4, device=torch_device, dtype=pixel_values.dtype),
                }
                labels.append(target)

        config = self.get_config()
        config.num_labels = self.num_labels
        return config, pixel_values, pixel_mask, labels

    def get_config(self):
        backbone_config = RfDetrWindowedDinov2Config(
            hidden_size=16,
            num_hidden_layers=4,
            num_attention_heads=2,
            window_block_indexes=[0, 2],
            out_indices=[1, 3],
            num_windows=2,
            num_register_tokens=4,
            image_size=self.image_size,
            hidden_dropout_prob=self.dropout,
            attention_probs_dropout_prob=self.dropout,
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
            _attn_implementation="eager",
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

    def create_and_check_rf_detr_instance_segmentation_head_model(self, config, pixel_values, pixel_mask, labels):
        model = RfDetrForInstanceSegmentation(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_queries, self.num_labels))
        self.parent.assertEqual(result.pred_boxes.shape, (self.batch_size, self.num_queries, 4))
        self.parent.assertEqual(
            result.pred_masks.shape,
            (
                self.batch_size,
                self.num_queries,
                self.image_size // config.mask_downsample_ratio,
                self.image_size // config.mask_downsample_ratio,
            ),
        )

        with self.parent.assertRaises(NotImplementedError):
            model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)


@require_torch
class RfDetrModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (RfDetrModel, RfDetrForObjectDetection, RfDetrForInstanceSegmentation) if is_torch_available() else ()
    )
    pipeline_model_mapping = (
        {
            "image-feature-extraction": RfDetrModel,
            "object-detection": RfDetrForObjectDetection,
            "image-segmentation": RfDetrForInstanceSegmentation,
        }
        if is_torch_available()
        else {}
    )
    is_encoder_decoder = False
    test_missing_keys = False
    test_torch_exportable = True
    model_split_percents = [0.5, 0.87, 0.9]

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels and model_class.__name__ == "RfDetrForObjectDetection":
            labels = []
            for _ in range(self.model_tester.batch_size):
                labels.append(
                    {
                        "class_labels": torch.ones(
                            size=(self.model_tester.n_targets,), device=torch_device, dtype=torch.long
                        ),
                        "boxes": torch.ones(self.model_tester.n_targets, 4, device=torch_device, dtype=torch.float),
                    }
                )
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

    def test_rf_detr_instance_segmentation_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_rf_detr_instance_segmentation_head_model(*config_and_inputs)

    @unittest.skip(reason="RfDetr does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="RfDetr does not use test_inputs_embeds_matches_input_ids")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="RfDetr does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="RfDetr does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="RfDetr does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

    def test_training(self):
        if not self.model_tester.is_training:
            self.skipTest(reason="ModelTester is not configured to run training tests")

        # Instance segmentation currently has no loss implementation.
        for model_class in [RfDetrForObjectDetection]:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    def test_attention_outputs(self):
        def check_attention_outputs(inputs_dict, config, model_class):
            config._attn_implementation = "eager"
            model = model_class(config)
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

            expected_cross_attentions_shape = [
                self.model_tester.batch_size,
                self.model_tester.num_queries,
                self.model_tester.decoder_cross_attention_heads,
                config.num_feature_levels,
                config.decoder_n_points,
            ]
            cross_attentions = outputs.cross_attentions
            self.assertEqual(len(cross_attentions), self.model_tester.decoder_layers)
            for i in range(self.model_tester.decoder_layers):
                self.assertEqual(expected_cross_attentions_shape, list(cross_attentions[i].shape))

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        inputs_dict["output_hidden_states"] = False

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            check_attention_outputs(inputs_dict, config, model_class)

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

        for model_class in [RfDetrForObjectDetection]:
            model = model_class(config)
            model.to(torch_device)
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            outputs = model(**inputs)

            self.assertIsNotNone(outputs.auxiliary_outputs)
            self.assertEqual(len(outputs.auxiliary_outputs), self.model_tester.decoder_layers - 1)
