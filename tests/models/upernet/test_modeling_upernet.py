# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch UperNet framework. """


import inspect
import unittest

from transformers import ConvNextConfig, UperNetConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    from transformers import UperNetForSemanticSegmentation
    from transformers.models.upernet.modeling_upernet import UPERNET_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image


class UperNetModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=32,
        num_channels=3,
        num_stages=4,
        hidden_sizes=[10, 20, 30, 40],
        depths=[2, 2, 3, 2],
        is_training=True,
        use_labels=True,
        intermediate_size=37,
        hidden_act="gelu",
        type_sequence_label_size=10,
        initializer_range=0.02,
        out_features=["stage2", "stage3", "stage4"],
        num_labels=3,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_stages = num_stages
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.is_training = is_training
        self.use_labels = use_labels
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.out_features = out_features
        self.num_labels = num_labels
        self.scope = scope

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_backbone_config(self):
        return ConvNextConfig(
            num_channels=self.num_channels,
            num_stages=self.num_stages,
            hidden_sizes=self.hidden_sizes,
            depths=self.depths,
            is_training=self.is_training,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            out_features=self.out_features,
        )

    def get_config(self):
        return UperNetConfig(
            backbone_config=self.get_backbone_config(),
            hidden_size=512,
            pool_scales=[1, 2, 3, 6],
            use_auxiliary_head=True,
            auxiliary_loss_weight=0.4,
            auxiliary_in_channels=40,
            auxiliary_channels=256,
            auxiliary_num_convs=1,
            auxiliary_concat_input=False,
            loss_ignore_index=255,
            num_labels=self.num_labels,
        )

    def create_and_check_for_semantic_segmentation(self, config, pixel_values, labels):
        model = UperNetForSemanticSegmentation(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels, 4, 4))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values,
            labels,
        ) = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class UperNetModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as UperNet does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (UperNetForSemanticSegmentation,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = UperNetModelTester(self)
        self.config_tester = ConfigTester(self, config_class=UperNetConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="UperNet does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="UperNet does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="UperNet does not have a base model")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="UperNet does not have a base model")
    def test_save_load_fast_init_to_base(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_for_semantic_segmentation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_semantic_segmentation(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in UPERNET_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = UperNetForSemanticSegmentation.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class UperNetModelIntegrationTest(unittest.TestCase):
    # @cached_property
    # def default_feature_extractor(self):
    #     return UperNetFeatureExtractor.from_pretrained("google/vit-base-patch16-224") if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        raise NotImplementedError("To do")
