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
import os
import tempfile
import unittest
from functools import cached_property

import numpy as np
from parameterized import parameterized

from transformers import (
    CONFIG_NAME,
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
    torch_device,
)

from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import RfDetrDinov2Backbone, RfDetrForInstanceSegmentation, RfDetrForObjectDetection, RfDetrModel

if is_vision_available():
    from PIL import Image

OBJECT_DETECTION_CHECKPOINTS = [
    "stevenbucaille/rf-detr-nano",
    "stevenbucaille/rf-detr-small",
    "stevenbucaille/rf-detr-base",
    "stevenbucaille/rf-detr-base-2",
    "stevenbucaille/rf-detr-medium",
    "stevenbucaille/rf-detr-large",
]

SEGMENTATION_CHECKPOINTS = [
    "stevenbucaille/rf-detr-seg-preview",
    "stevenbucaille/rf-detr-seg-nano",
    "stevenbucaille/rf-detr-seg-small",
    "stevenbucaille/rf-detr-seg-medium",
    "stevenbucaille/rf-detr-seg-large",
    "stevenbucaille/rf-detr-seg-xlarge",
    "stevenbucaille/rf-detr-seg-xxlarge",
]


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
    test_torch_exportable = True

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

    def test_model_outputs_equivalence(self):
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


EXPECTED_OBJECT_DETECTION_OUTPUTS = {
    "stevenbucaille/rf-detr-nano": {
        "logits": [-6.68004, -5.66107, -11.70373, -8.32324, -5.76176],
        "boxes": [0.25828, 0.54991, 0.47220, 0.87432, 0.55099],
        "post_process_labels": [17, 75, 17, 75, 63],
        "post_process_scores": [0.99419, 0.98870, 0.95431, 0.98553, 0.43040],
        "post_process_boxes": [14.19509, 54.12025, 316.40289, 473.79239],
        "loss": 14.893259,
    },
    "stevenbucaille/rf-detr-small": {
        "logits": [-6.83893, -4.55097, -10.53040, -8.20657, -5.55314],
        "boxes": [0.25782, 0.55037, 0.47922, 0.87102, 0.77074],
        "post_process_labels": [17, 17, 75, 75, 65],
        "post_process_scores": [0.99281, 0.98698, 0.99031, 0.98309, 0.51899],
        "post_process_boxes": [11.65812, 55.13201, 318.35583, 473.22156],
        "loss": 19.771887,
    },
    "stevenbucaille/rf-detr-base": {
        "logits": [-7.60410, -4.65943, -10.03144, -5.63881, -9.88291],
        "boxes": [0.25465, 0.54864, 0.48583, 0.86991, 0.16926],
        "post_process_labels": [17, 75, 17, 75, 63],
        "post_process_scores": [0.98291, 0.97628, 0.97799, 0.86630, 0.61596],
        "post_process_boxes": [7.51008, 54.56673, 318.44214, 472.12595],
        "loss": 21.967346,
    },
    "stevenbucaille/rf-detr-base-2": {
        "logits": [-6.81648, -6.80946, -7.72004, -6.06710, -10.37419],
        "boxes": [0.16911, 0.19784, 0.21076, 0.09273, 0.25263],
        "post_process_labels": [75, 17, 17, 75, 63],
        "post_process_scores": [0.98108, 0.97893, 0.96431, 0.82874, 0.43441],
        "post_process_boxes": [40.78329, 72.70991, 175.67207, 117.21776],
        "loss": 21.532478,
    },
    "stevenbucaille/rf-detr-medium": {
        "logits": [-6.58581, -8.07883, -12.52392, -7.78248, -10.47323],
        "boxes": [0.16824, 0.19932, 0.21110, 0.09385, 0.77087],
        "post_process_labels": [75, 17, 17, 75, 63],
        "post_process_scores": [0.98818, 0.98767, 0.98665, 0.98013, 0.42874],
        "post_process_boxes": [40.12718, 73.15034, 175.22226, 118.19680],
        "loss": 26.337656,
    },
    "stevenbucaille/rf-detr-large": {
        "logits": [-6.38973, -8.19355, -12.09174, -7.80438, -10.15835],
        "boxes": [0.16901, 0.19936, 0.21087, 0.09311, 0.77199],
        "post_process_labels": [75, 17, 17, 75, 63],
        "post_process_scores": [0.99024, 0.98684, 0.98713, 0.94908, 0.52595],
        "post_process_boxes": [40.68745, 73.34576, 175.64552, 118.03757],
        "loss": 27.111633,
    },
}

EXPECTED_SEGMENTATION_OUTPUTS = {
    "stevenbucaille/rf-detr-seg-preview": {
        "logits": [-7.05877, -4.23362, -6.54288, -8.13384, -6.36838],
        "boxes": [0.25603, 0.55164, 0.48340, 0.87798, 0.73861],
        "pred_masks": [-16.72129, -16.17153, -17.06426, -17.29409, -17.78559],
        "post_process_labels": [17, 75, 75, 17],
        "post_process_scores": [0.98604, 0.985644, 0.950492, 0.967915],
        "loss": 76.374206,
    },
    "stevenbucaille/rf-detr-seg-nano": {
        "logits": [-7.38427, -5.59449, -9.97889, -11.03668, -8.62285],
        "boxes": [0.25230, 0.54825, 0.48196, 0.86925, 0.77119],
        "pred_masks": [-12.01641, -12.37785, -13.37312, -13.54168, -13.53435],
        "post_process_labels": [17, 17, 75, 75],
        "post_process_scores": [0.987039, 0.984917, 0.978123, 0.966226],
        "loss": 93.131385,
    },
    "stevenbucaille/rf-detr-seg-small": {
        "logits": [-7.35031, -5.09690, -9.58117, -10.80274, -8.35001],
        "boxes": [0.25607, 0.54820, 0.48018, 0.87013, 0.90797],
        "pred_masks": [-13.17243, -13.12057, -13.92742, -13.89896, -13.72802],
        "post_process_labels": [17, 75, 75, 17],
        "post_process_scores": [0.984295, 0.984524, 0.971301, 0.977482],
        "loss": 87.918113,
    },
    "stevenbucaille/rf-detr-seg-medium": {
        "logits": [-7.48751, -5.21394, -9.35906, -9.31897, -8.08021],
        "boxes": [0.76891, 0.41680, 0.46182, 0.72004, 0.16810],
        "pred_masks": [-15.67913, -17.05902, -16.72426, -17.19833, -17.18960],
        "post_process_labels": [17, 75, 17, 75],
        "post_process_scores": [0.982117, 0.982515, 0.986937, 0.94945],
        "loss": 96.400970,
    },
    "stevenbucaille/rf-detr-seg-large": {
        "logits": [-7.37005, -5.04871, -9.19777, -9.37870, -7.96562],
        "boxes": [0.76796, 0.41489, 0.46220, 0.72197, 0.25254],
        "pred_masks": [-15.13846, -16.88754, -16.55486, -17.23686, -17.40160],
        "post_process_labels": [17, 17, 75, 75],
        "post_process_scores": [0.973857, 0.980841, 0.982329, 0.940749],
        "loss": 91.295486,
    },
    "stevenbucaille/rf-detr-seg-xlarge": {
        "logits": [-7.42486, -4.72502, -8.16429, -8.30500, -7.21668],
        "boxes": [0.76863, 0.41618, 0.46055, 0.72461, 0.16735],
        "pred_masks": [-15.15330, -17.61085, -16.79776, -17.33611, -17.39120],
        "post_process_labels": [17, 75, 75, 17],
        "post_process_scores": [0.978207, 0.979888, 0.943645, 0.982371],
        "loss": 105.523971,
    },
    "stevenbucaille/rf-detr-seg-xxlarge": {
        "logits": [-7.33242, -5.11294, -6.31125, -7.06520, -7.07922],
        "boxes": [0.25516, 0.53685, 0.49769, 0.88601, 0.76872],
        "pred_masks": [-7.94849, -8.57010, -8.60056, -8.92854, -8.99288],
        "post_process_labels": [17, 75, 75, 17],
        "post_process_scores": [0.980528, 0.976553, 0.846246, 0.974776],
        "loss": 98.795463,
    },
}


@require_torch
@require_vision
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

    @parameterized.expand([(name, EXPECTED_OBJECT_DETECTION_OUTPUTS[name]) for name in OBJECT_DETECTION_CHECKPOINTS])
    def test_inference_object_detection(self, model_name, expected_outputs):
        tol = 5e-3
        model = RfDetrForObjectDetection.from_pretrained(model_name, attn_implementation="eager").to(torch_device)
        image_processor = DetrImageProcessor.from_pretrained(model_name)
        image = prepare_img()
        inputs = image_processor(images=image, annotations=self.annotations, return_tensors="pt").to(torch_device)

        print(inputs)
        outputs = model(**inputs)

        # Check raw outputs from the model
        expectations = Expectations({("cpu", None): expected_outputs["logits"]})
        expected_logits = torch.tensor(expectations.get_expectation()).to(torch_device)
        expected_logits_shape = torch.Size((1, model.config.num_queries, model.config.num_labels))

        expectations = Expectations({("cpu", None): expected_outputs["boxes"]})
        expected_boxes = torch.tensor(expectations.get_expectation()).to(torch_device)
        expected_boxes_shape = torch.Size((1, model.config.num_queries, 4))

        expectations = Expectations({("cpu", None): expected_outputs["loss"]})
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
        expectations = Expectations({("cpu", None): expected_outputs["post_process_labels"]})
        expected_post_process_labels = torch.tensor(expectations.get_expectation()).to(torch_device)
        expectations = Expectations({("cpu", None): expected_outputs["post_process_scores"]})
        expected_post_process_scores = torch.tensor(expectations.get_expectation()).to(torch_device)
        expectations = Expectations({("cpu", None): expected_outputs["post_process_boxes"]})
        expected_post_process_boxes = torch.tensor(expectations.get_expectation()).to(torch_device)

        post_processed_labels = post_processed_outputs["labels"][:5]
        post_processed_scores = post_processed_outputs["scores"][:5]
        post_processed_boxes = post_processed_outputs["boxes"][0]
        torch.testing.assert_close(post_processed_labels, expected_post_process_labels, rtol=tol, atol=tol)
        torch.testing.assert_close(post_processed_scores, expected_post_process_scores, rtol=tol, atol=tol)
        torch.testing.assert_close(post_processed_boxes, expected_post_process_boxes, rtol=1, atol=1)

    @parameterized.expand([(name, EXPECTED_SEGMENTATION_OUTPUTS[name]) for name in SEGMENTATION_CHECKPOINTS])
    def test_inference_segmentation(self, model_name, expected_outputs):
        tol = 5e-3
        model = RfDetrForInstanceSegmentation.from_pretrained(model_name, attn_implementation="eager").to(torch_device)

        image_processor = DetrImageProcessor.from_pretrained(model_name)
        image = prepare_img()
        inputs = image_processor(images=image, annotations=self.annotations, return_tensors="pt").to(torch_device)
        inputs["labels"][0]["masks"] = torch.zeros(
            (1, inputs["pixel_values"].shape[-1], inputs["pixel_values"].shape[-2])
        )
        outputs = model(**inputs)

        # Check raw outputs from the model
        expectations = Expectations({("cpu", None): expected_outputs["logits"]})
        expected_logits = torch.tensor(expectations.get_expectation()).to(torch_device)
        expected_logits_shape = torch.Size((1, model.config.num_queries, model.config.num_labels))

        expectations = Expectations({("cpu", None): expected_outputs["boxes"]})
        expected_boxes = torch.tensor(expectations.get_expectation()).to(torch_device)
        expected_boxes_shape = torch.Size((1, model.config.num_queries, 4))

        expectations = Expectations({("cpu", None): expected_outputs["pred_masks"]})
        expected_masks = torch.tensor(expectations.get_expectation()).to(torch_device)
        expected_masks_shape = torch.Size(
            (
                1,
                model.config.num_queries,
                inputs["pixel_values"].shape[-2] // model.config.mask_downsample_ratio,
                inputs["pixel_values"].shape[-1] // model.config.mask_downsample_ratio,
            )
        )

        expectations = Expectations({("cpu", None): expected_outputs["loss"]})
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
        torch.testing.assert_close(predicted_loss, expected_loss, rtol=tol, atol=tol)

        # Check post-processed outputs
        post_processed_outputs = image_processor.post_process_instance_segmentation(
            outputs, threshold=0.0, target_sizes=[image.size[::-1]]
        )[0]
        expectations = Expectations({("cpu", None): expected_outputs["post_process_labels"]})
        expected_post_process_labels = torch.tensor(expectations.get_expectation()).to(torch_device)
        expectations = Expectations({("cpu", None): expected_outputs["post_process_scores"]})
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
