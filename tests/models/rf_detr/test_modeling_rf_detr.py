# coding = utf-8
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
    DetrImageProcessor,
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
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import RfDetrDinov2Backbone, RfDetrForInstanceSegmentation, RfDetrForObjectDetection, RfDetrModel

if is_vision_available():
    from PIL import Image


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

        self.num_hidden_layers = decoder_layers
        self.seq_length = num_queries
        self.hidden_size = d_model

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
                target["masks"] = torch.rand(self.n_targets, self.image_size, self.image_size, device=torch_device)
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
    all_model_classes = (
        (RfDetrModel, RfDetrForObjectDetection, RfDetrForInstanceSegmentation) if is_torch_available() else ()
    )
    pipeline_model_mapping = (
        {
            "image-feature-extraction": RfDetrModel,
            "object-detection": RfDetrForObjectDetection,
            "instance-segmentation": RfDetrForInstanceSegmentation,
        }
        if is_torch_available()
        else {}
    )
    is_encoder_decoder = False
    test_missing_keys = False
    test_resize_embeddings = False
    model_split_percents = [0.5, 0.87, 0.9]

    # special case for head models
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class.__name__ in ["RfDetrForObjectDetection", "RfDetrForInstanceSegmentation"]:
                labels = []
                for i in range(self.model_tester.batch_size):
                    target = {}
                    target["class_labels"] = torch.ones(
                        size=(self.model_tester.n_targets,), device=torch_device, dtype=torch.long
                    )
                    target["boxes"] = torch.ones(
                        self.model_tester.n_targets, 4, device=torch_device, dtype=torch.float
                    )
                    target["masks"] = torch.ones(
                        self.model_tester.n_targets,
                        self.model_tester.image_size,
                        self.model_tester.image_size,
                        device=torch_device,
                        dtype=torch.float,
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

    def flash_attn_inference_equivalence(self, **kwargs):
        # RF-DETR's encoder-decoder bridge uses discrete top-k proposal selection. Tiny floating-point
        # differences between flash attention and eager attention in the DINOv2 backbone cause different
        # encoder proposals to be selected, resulting in decoder outputs that exceed the equivalence tolerance.
        self.skipTest(reason="RF-DETR top-k proposal selection is sensitive to flash attention numerics")

    def test_attention_outputs(self):
        # Override test_attention_outputs to support object detection and segmentation heads.
        # Outputs include pred_logits, pred_boxes and auxiliary outputs.
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
                1,
                config.decoder_n_points,
            ]
            cross_attentions = outputs.cross_attentions
            self.assertEqual(len(cross_attentions), self.model_tester.decoder_layers)
            for i in range(self.model_tester.decoder_layers):
                self.assertEqual(expected_attentions_shape, list(cross_attentions[i].shape))

            out_len = len(outputs)

            if model_class.__name__ == "RfDetrModel":
                correct_outlen = 9  # 7 + attentions + cross_attentions
            if model_class.__name__ in "RfDetrForObjectDetection":
                correct_outlen = 11  # 9 + attentions + cross_attentions
                if "labels" in inputs_dict:
                    correct_outlen += 3  # loss, loss_dict and auxiliary outputs is added to beginning
            elif model_class.__name__ == "RfDetrForInstanceSegmentation":
                correct_outlen = 10  # 11 + attentions + cross_attentions
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

    def test_model_outputs_equivalence(self):
        # Override test_model_outputs_equivalence because RfDetr loss has random tensors generated
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                # RfDetr loss has random tensors generated
                torch.manual_seed(0)
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                torch.manual_seed(0)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, (list, tuple)):
                        for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif isinstance(tuple_object, dict):
                        for tuple_iterable_value, dict_iterable_value in zip(
                            tuple_object.values(), dict_object.values()
                        ):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif tuple_object is None:
                        return
                    else:
                        self.assertTrue(
                            torch.allclose(
                                set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5
                            ),
                            msg=(
                                "Tuple and dict output are not equal. Difference:"
                                f" {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                                f" {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has"
                                f" `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}."
                            ),
                        )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(
                model, tuple_inputs, dict_inputs, {"output_hidden_states": True, "output_attentions": True}
            )


@require_torch
@require_vision
@slow
class RfDetrModelIntegrationTest(unittest.TestCase):
    @cached_property
    def annotations(self):
        return {
            "image_id": 0,
            "annotations": [
                {
                    "bbox": [250, 250, 350, 350],
                    "category_id": 0,
                    "iscrowd": 0,
                    "area": 122500,
                    "segments": [[0, 0, 0, 100, 100, 100, 100, 0]],
                }
            ],
        }

    def test_inference_object_detection(self):
        tol = 5e-3
        model = RfDetrForObjectDetection.from_pretrained("Roboflow/rf-detr-base", attn_implementation="eager").to(
            torch_device
        )
        image_processor = DetrImageProcessor.from_pretrained("Roboflow/rf-detr-base")
        image = prepare_img()
        inputs = image_processor(images=image, annotations=self.annotations, return_tensors="pt").to(torch_device)
        inputs["labels"] = [
            {k: v.to(torch_device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in inputs["labels"]
        ]

        torch.manual_seed(0)
        outputs = model(**inputs)

        # Check raw outputs from the model
        # fmt: off
        expectations = Expectations(
            {
                ("cuda", (8, 0)): [-7.58881, -4.64088, -10.02118, -5.65906, -9.8343],
                ("xpu", None): [-7.59672, -4.63663, -10.01015, -5.64823, -9.82786],
            }
        )
        # fmt: on
        expected_logits = torch.tensor(expectations.get_expectation()).to(torch_device)
        expected_logits_shape = torch.Size((1, model.config.num_queries, model.config.num_labels))

        # fmt: off
        expectations = Expectations(
            {
                ("cuda", (8, 0)): [0.25457, 0.54871, 0.48585, 0.86988, 0.16926],
                ("xpu", None): [0.25460, 0.54872, 0.48586, 0.86991, 0.16926],
            }
        )
        # fmt: on
        expected_boxes = torch.tensor(expectations.get_expectation()).to(torch_device)
        expected_boxes_shape = torch.Size((1, model.config.num_queries, 4))

        expectations = Expectations(
            {
                ("cuda", (8, 0)): 21.911297,
                ("xpu", None): 21.834641,
            }
        )
        expected_loss = torch.tensor(expectations.get_expectation()).to(torch_device)

        predicted_logits = outputs.logits.flatten()[:5]
        predicted_boxes = outputs.pred_boxes.flatten()[:5]
        predicted_loss = outputs.loss

        self.assertEqual(outputs.logits.shape, expected_logits_shape)
        self.assertEqual(outputs.pred_boxes.shape, expected_boxes_shape)
        torch.testing.assert_close(predicted_logits, expected_logits, rtol=tol, atol=tol)
        torch.testing.assert_close(predicted_boxes, expected_boxes, rtol=tol, atol=tol)
        torch.testing.assert_close(predicted_loss, expected_loss, rtol=tol, atol=tol)

        # Check post-processed outputs
        post_processed_outputs = image_processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=[image.size[::-1]]
        )[0]
        expectations = Expectations(
            {
                ("cuda", (8, 0)): [17, 75, 17, 75, 63],
                ("xpu", None): [17, 75, 17, 75, 63],
            }
        )
        expected_post_process_labels = torch.tensor(expectations.get_expectation()).to(torch_device)
        # fmt: off
        expectations = Expectations(
            {
                ("cuda", (8, 0)): [0.982765, 0.975941, 0.978163, 0.868452, 0.619554],
                ("xpu", None): [0.98277, 0.976018, 0.977666, 0.868674, 0.615991],
            }
        )
        # fmt: on
        expected_post_process_scores = torch.tensor(expectations.get_expectation()).to(torch_device)

        # fmt: off
        expectations = Expectations(
            {
                ("cuda", (8, 0)): [7.44911, 54.60959, 318.39551, 472.15417],
                ("xpu", None): [7.46919, 54.60617, 318.41934, 472.16153],
            }
        )
        # fmt: on
        expected_post_process_boxes = torch.tensor(expectations.get_expectation()).to(torch_device)

        post_processed_labels = post_processed_outputs["labels"][:5]
        post_processed_scores = post_processed_outputs["scores"][:5]
        post_processed_boxes = post_processed_outputs["boxes"][0]
        torch.testing.assert_close(post_processed_labels, expected_post_process_labels, rtol=tol, atol=tol)
        torch.testing.assert_close(post_processed_scores, expected_post_process_scores, rtol=tol, atol=tol)
        torch.testing.assert_close(post_processed_boxes, expected_post_process_boxes, rtol=1, atol=1)

    def test_inference_segmentation(self):
        tol = 5e-3
        # Loss involves random mask point sampling, so we use a more lenient tolerance
        loss_tol = 1e-2
        model = RfDetrForInstanceSegmentation.from_pretrained(
            "Roboflow/rf-detr-seg-small", attn_implementation="eager"
        ).to(torch_device)

        image_processor = DetrImageProcessor.from_pretrained("Roboflow/rf-detr-seg-small")
        image = prepare_img()
        inputs = image_processor(images=image, annotations=self.annotations, return_tensors="pt").to(torch_device)
        inputs["labels"] = [
            {k: v.to(torch_device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in inputs["labels"]
        ]
        inputs["labels"][0]["masks"] = torch.zeros(
            (1, inputs["pixel_values"].shape[-1], inputs["pixel_values"].shape[-2]), device=torch_device
        )
        torch.manual_seed(0)
        outputs = model(**inputs)

        # Check raw outputs from the model
        # fmt: off
        expectations = Expectations(
            {
                ("cuda", (8, 0)): [-7.3531, -5.14075, -9.63576, -10.81916, -8.3615],
                ("xpu", None): [-7.3542, -5.14673, -9.64132, -10.82835, -8.3629],
            }
        )
        # fmt: on
        expected_logits = torch.tensor(expectations.get_expectation()).to(torch_device)
        expected_logits_shape = torch.Size((1, model.config.num_queries, model.config.num_labels))

        # fmt: off
        expectations = Expectations(
            {
                ("cuda", (8, 0)): [0.25602, 0.54813, 0.48043, 0.87045, 0.77213],
                ("xpu", None): [0.25603, 0.54812, 0.48039, 0.87042, 0.77208],
            }
        )
        # fmt: on
        expected_boxes = torch.tensor(expectations.get_expectation()).to(torch_device)
        expected_boxes_shape = torch.Size((1, model.config.num_queries, 4))

        # fmt: off
        expectations = Expectations(
            {
                ("cuda", (8, 0)): [-13.1366, -13.08283, -13.9058, -13.88317, -13.71717],
                ("xpu", None): [-13.1304, -13.0778, -13.9000, -13.8778, -13.7107],
            }
        )
        # fmt: on
        expected_masks = torch.tensor(expectations.get_expectation()).to(torch_device)
        expected_masks_shape = torch.Size(
            (
                1,
                model.config.num_queries,
                inputs["pixel_values"].shape[-2] // model.config.mask_downsample_ratio,
                inputs["pixel_values"].shape[-1] // model.config.mask_downsample_ratio,
            )
        )

        expectations = Expectations(
            {
                ("cuda", (8, 0)): 88.117493,
                ("xpu", None): 87.474312,
            }
        )
        expected_loss = torch.tensor(expectations.get_expectation()).to(torch_device)

        predicted_logits = outputs.logits.flatten()[:5]
        predicted_boxes = outputs.pred_boxes.flatten()[:5]
        predicted_masks = outputs.pred_masks.flatten()[:5]
        predicted_loss = outputs.loss

        self.assertEqual(outputs.logits.shape, expected_logits_shape)
        self.assertEqual(outputs.pred_boxes.shape, expected_boxes_shape)
        self.assertEqual(outputs.pred_masks.shape, expected_masks_shape)
        torch.testing.assert_close(predicted_logits, expected_logits, rtol=tol, atol=tol)
        torch.testing.assert_close(predicted_boxes, expected_boxes, rtol=tol, atol=tol)
        torch.testing.assert_close(predicted_masks, expected_masks, rtol=tol, atol=tol)
        torch.testing.assert_close(predicted_loss, expected_loss, rtol=loss_tol, atol=loss_tol)

        # Check post-processed outputs
        post_processed_outputs = image_processor.post_process_instance_segmentation(
            outputs, threshold=0.0, target_sizes=[image.size[::-1]]
        )[0]
        expectations = Expectations(
            {
                ("cuda", (8, 0)): [17, 17, 75, 75],
                ("xpu", None): [17, 17, 75, 75],
            }
        )
        expected_post_process_labels = torch.tensor(expectations.get_expectation()).to(torch_device)
        expectations = Expectations(
            {
                ("cuda", (8, 0)): [0.984311, 0.976176, 0.984499, 0.970341],
                ("xpu", None): [0.984291, 0.97617, 0.984517, 0.970308],
            }
        )
        expected_post_process_scores = torch.tensor(expectations.get_expectation()).to(torch_device)

        post_processed_labels = [
            segments_info["label_id"] for segments_info in post_processed_outputs["segments_info"]
        ]
        post_processed_labels = torch.tensor(post_processed_labels).to(torch_device)
        post_processed_scores = [segments_info["score"] for segments_info in post_processed_outputs["segments_info"]]
        post_processed_scores = torch.tensor(post_processed_scores).to(torch_device)
        torch.testing.assert_close(post_processed_labels, expected_post_process_labels, rtol=tol, atol=tol)
        torch.testing.assert_close(post_processed_scores, expected_post_process_scores, rtol=tol, atol=tol)


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

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class RfDetrDinov2BackboneTest(unittest.TestCase, BackboneTesterMixin):
    all_model_classes = (RfDetrDinov2Backbone,) if is_torch_available() else ()
    config_class = RfDetrDinov2Config

    def setUp(self):
        self.model_tester = RfDetrDinov2ModelTester(self)
