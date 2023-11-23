# coding = utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch RT_DETR model. """


import inspect
import unittest

from transformers import RTDetrConfig, RTDetrImageProcessor, is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision, torch_device
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import RTDetrModel
    from transformers.models.rt_detr.modeling_rt_detr import RTDETR_PRETRAINED_MODEL_ARCHIVE_LIST

if is_vision_available():
    from PIL import Image


CHECKPOINT = "rafaelpadilla/porting_rt_detr"  # TODO: replace


def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


class RTDetrConfigTester(ConfigTester):
    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, "initializer_range"))
        self.parent.assertTrue(hasattr(config, "layer_norm_eps"))
        self.parent.assertTrue(hasattr(config, "backbone_config"))
        self.parent.assertTrue(hasattr(config, "feat_strides"))
        self.parent.assertTrue(hasattr(config, "hidden_dim"))
        self.parent.assertTrue(hasattr(config, "num_attention_heads"))
        self.parent.assertTrue(hasattr(config, "dim_feedforward"))
        self.parent.assertTrue(hasattr(config, "dropout"))
        self.parent.assertTrue(hasattr(config, "use_encoder_idx"))
        self.parent.assertTrue(hasattr(config, "num_encoder_layers"))
        self.parent.assertTrue(hasattr(config, "pe_temperature"))
        self.parent.assertTrue(hasattr(config, "act_encoder"))
        self.parent.assertTrue(hasattr(config, "eval_size"))
        self.parent.assertTrue(hasattr(config, "normalize_before"))
        self.parent.assertTrue(hasattr(config, "num_classes"))
        self.parent.assertTrue(hasattr(config, "num_queries"))
        self.parent.assertTrue(hasattr(config, "feat_channels"))
        self.parent.assertTrue(hasattr(config, "num_levels"))
        self.parent.assertTrue(hasattr(config, "num_decoder_points"))
        self.parent.assertTrue(hasattr(config, "num_decoder_layers"))
        self.parent.assertTrue(hasattr(config, "num_denoising"))
        self.parent.assertTrue(hasattr(config, "label_noise_ratio"))
        self.parent.assertTrue(hasattr(config, "box_noise_scale"))
        self.parent.assertTrue(hasattr(config, "learnt_init_query"))
        self.parent.assertTrue(hasattr(config, "eval_spatial_size"))
        self.parent.assertTrue(hasattr(config, "eval_idx"))
        self.parent.assertTrue(hasattr(config, "matcher_alpha"))
        self.parent.assertTrue(hasattr(config, "matcher_gamma"))
        self.parent.assertTrue(hasattr(config, "matcher_class_cost"))
        self.parent.assertTrue(hasattr(config, "matcher_bbox_cost"))
        self.parent.assertTrue(hasattr(config, "matcher_giou_cost"))
        self.parent.assertTrue(hasattr(config, "use_focal_loss"))
        self.parent.assertTrue(hasattr(config, "use_aux_loss"))
        self.parent.assertTrue(hasattr(config, "focal_loss_alpha"))
        self.parent.assertTrue(hasattr(config, "focal_loss_gamma"))
        self.parent.assertTrue(hasattr(config, "weight_loss_vfl"))
        self.parent.assertTrue(hasattr(config, "weight_loss_bbox"))
        self.parent.assertTrue(hasattr(config, "weight_loss_giou"))
        self.parent.assertTrue(hasattr(config, "eos_coefficient"))


class RTDetrModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_size=160,
        num_channels=3,
        is_training=True,
        initializer_range=0.02,
        backbone_config=None,
        feat_strides=[1, 2, 4],
        feat_channels=[4, 4, 4],
        num_levels=3,
        hidden_dim=4,
        num_attention_heads=1,
        dim_feedforward=16,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        act_encoder="silu",
        eval_size=None,
        num_classes=1,
        num_queries=1,
        num_decoder_points=1,
        num_decoder_layers=1,
        num_denoising=1,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        matcher_alpha=0.25,
        matcher_gamma=2.0,
        matcher_class_cost=2.0,
        matcher_bbox_cost=5.0,
        matcher_giou_cost=2.0,
        use_focal_loss=True,
        use_aux_loss=True,
        focal_loss_alpha=0.75,
        focal_loss_gamma=2.0,
        weight_loss_vfl=1.0,
        weight_loss_bbox=5.0,
        weight_loss_giou=2.0,
        eos_coefficient=0.1,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.parent = parent
        self.initializer_range = initializer_range
        self.backbone_config = backbone_config
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.enc_act = enc_act
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.expansion = expansion
        self.act_encoder = act_encoder
        self.eval_size = eval_size
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.feat_channels = feat_channels
        self.num_levels = num_levels
        self.num_decoder_points = num_decoder_points
        self.num_decoder_layers = num_decoder_layers
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.learnt_init_query = learnt_init_query
        self.eval_spatial_size = eval_spatial_size
        self.eval_idx = eval_idx
        self.eps = eps
        self.matcher_alpha = matcher_alpha
        self.matcher_gamma = matcher_gamma
        self.matcher_class_cost = matcher_class_cost
        self.matcher_bbox_cost = matcher_bbox_cost
        self.matcher_giou_cost = matcher_giou_cost
        self.use_focal_loss = use_focal_loss
        self.use_aux_loss = use_aux_loss
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.weight_loss_vfl = weight_loss_vfl
        self.weight_loss_bbox = weight_loss_bbox
        self.weight_loss_giou = weight_loss_giou
        self.eos_coefficient = eos_coefficient

    def prepare_random_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        # labels is a list of Dict (each Dict being the labels for a given example in the batch)
        labels = []
        # simulates number of samples per image
        samples_labels = 3
        for _ in range(self.batch_size):
            target = {}
            target["class_labels"] = ids_tensor([samples_labels], self.num_classes)
            target["boxes"] = torch.rand(samples_labels, 4, device=torch_device)
            labels.append(target)
        return pixel_values, labels

    def prepare_config_and_inputs(self):
        pixel_values, labels = self.prepare_random_inputs()
        config = self.get_config()
        return config, pixel_values, labels

    def get_config(self):
        return RTDetrConfig(
            initializer_range=self.initializer_range,
            backbone_config=self.backbone_config,
            feat_strides=self.feat_strides,
            hidden_dim=self.hidden_dim,
            num_attention_heads=self.num_attention_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            enc_act=self.enc_act,
            use_encoder_idx=self.use_encoder_idx,
            num_encoder_layers=self.num_encoder_layers,
            pe_temperature=self.pe_temperature,
            expansion=self.expansion,
            act_encoder=self.act_encoder,
            eval_size=self.eval_size,
            num_classes=self.num_classes,
            num_queries=self.num_queries,
            feat_channels=self.feat_channels,
            num_levels=self.num_levels,
            num_decoder_points=self.num_decoder_points,
            num_decoder_layers=self.num_decoder_layers,
            num_denoising=self.num_denoising,
            label_noise_ratio=self.label_noise_ratio,
            box_noise_scale=self.box_noise_scale,
            learnt_init_query=self.learnt_init_query,
            eval_spatial_size=self.eval_spatial_size,
            eval_idx=self.eval_idx,
            eps=self.eps,
            matcher_alpha=self.matcher_alpha,
            matcher_gamma=self.matcher_gamma,
            matcher_class_cost=self.matcher_class_cost,
            matcher_bbox_cost=self.matcher_bbox_cost,
            matcher_giou_cost=self.matcher_giou_cost,
            use_focal_loss=self.use_focal_loss,
            use_aux_loss=self.use_aux_loss,
            focal_loss_alpha=self.focal_loss_alpha,
            focal_loss_gamma=self.focal_loss_gamma,
            weight_loss_vfl=self.weight_loss_vfl,
            weight_loss_bbox=self.weight_loss_bbox,
            weight_loss_giou=self.weight_loss_giou,
            eos_coefficient=self.eos_coefficient,
        )

    def create_and_check_model(self, model_class, config, pixel_values, labels):
        """Create and check the shape of the logits and boxes predicted by the model, and their types."""
        config.num_classes = self.num_classes
        model = model_class(config)
        model.to(torch_device)
        model.eval()

        signature = inspect.signature(model_class.forward)
        arg_names = [*signature.parameters.keys()]
        if "labels" in arg_names:
            result = model(pixel_values, labels=labels)
            self.parent.assertIsInstance(result.loss, torch.Tensor)
            self.parent.assertIsInstance(result.loss_dict, dict)
        else:
            result = model(pixel_values)
        # Check model's outputs
        self.parent.assertEqual(result["logits"].shape, (self.batch_size, self.num_queries, self.num_classes))
        self.parent.assertEqual(result["pred_boxes"].shape, (self.batch_size, self.num_queries, 4))

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, _ = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class RTDetrModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (RTDetrModel,) if is_torch_available() else ()

    pipeline_model_mapping = {"object-detection": RTDetrModel} if is_torch_available() else ()

    test_torchscript = False
    test_pruning = False
    test_resize_embeddings = False
    has_attentions = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = RTDetrModelTester(self)
        self.config_tester = RTDetrConfigTester(self, config_class=RTDetrConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="RTDetr does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="RTDetr does not use hidden states")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="RTDetr does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="RTDetr was not designed smaller.")
    def test_model_is_small(self):
        pass

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True

        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)
        encoder_hidden_states = outputs.encoder_hidden_states[0]
        encoder_hidden_states.retain_grad()
        outputs["logits"].flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(encoder_hidden_states.grad)

    def test_forward_signature(self):
        config = self.model_tester.get_config()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        pixel_values, labels = self.model_tester.prepare_random_inputs()
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model(RTDetrModel, config, pixel_values, labels)

    def test_model_from_pretrained(self):
        for model_name in RTDETR_PRETRAINED_MODEL_ARCHIVE_LIST:
            model = RTDetrModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
@require_vision
class RTDetrModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return RTDetrImageProcessor.from_pretrained(CHECKPOINT) if is_vision_available() else None

    def test_inference(self):
        model = RTDetrModel.from_pretrained(CHECKPOINT).to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        batch_size = len(inputs.pixel_values)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 300, 80))
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice = torch.tensor(
            [
                [
                    [-4.64763879776001, -5.001153945922852, -4.978509902954102],
                    [-4.159348487854004, -4.703853607177734, -5.946484565734863],
                    [-4.437461853027344, -4.65836238861084, -6.235235691070557],
                ]
            ]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, :3, :3], expected_slice, atol=1e-4))

        # verify the boxes
        expected_shape = torch.Size((1, 300, 4))
        self.assertEqual(outputs.pred_boxes.shape, expected_shape)
        expected_slice = torch.tensor(
            [
                [
                    [0.1688060760498047, 0.19992263615131378, 0.21225441992282867],
                    [0.768376350402832, 0.41226309537887573, 0.4636859893798828],
                    [0.25953856110572815, 0.5483334064483643, 0.4777486026287079],
                ]
            ]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.pred_boxes[0, :3, :3], expected_slice, atol=1e-4))

        # verify post processor
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=target_sizes, use_focal_loss=model.config.use_focal_loss
        )
        # expecting 1 result per image in the batch
        self.assertEqual(len(results), batch_size)
        # result of the first image
        result = results[0]
        # shapes
        expected_shape = torch.Size([300])
        self.assertEqual(result["scores"].shape, expected_shape)
        self.assertEqual(result["labels"].shape, expected_shape)
        expected_box_shape = torch.Size([300, 4])
        self.assertEqual(result["boxes"].shape, expected_box_shape)
        # labels
        expected_labels = torch.tensor([57, 15, 15, 65]).to(model.device)
        self.assertTrue(torch.equal(result["labels"][:4], expected_labels))
        # scores
        expected_scores = torch.tensor(
            [0.9703017473220825, 0.9599503874778748, 0.9575679302215576, 0.9506784677505493]
        ).to(model.device)
        self.assertTrue(torch.allclose(result["scores"][:4], expected_scores, atol=1e-4))
        # boxes
        expected_boxes = torch.tensor(
            [
                [
                    [0.13774871826171875, 0.37818431854248047, 640.1307373046875, 476.21087646484375],
                    [343.38134765625, 24.276838302612305, 640.140380859375, 371.4957275390625],
                    [13.225126266479492, 54.17934799194336, 318.9842224121094, 472.220703125],
                ]
            ]
        ).to(model.device)
        self.assertTrue(torch.allclose(result["boxes"][:3], expected_boxes, atol=1e-4))
