# coding=utf-8
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
""" Testing suite for the PyTorch Falcon model. """

import unittest

from parameterized import parameterized

from transformers import (
    FastConfig,
    is_torch_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger, require_bitsandbytes, require_torch, slow, tooslow, torch_device
from transformers.utils import logging as transformers_logging

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin

if is_torch_available():
    import torch

    from transformers import (
        FASTForImageCaptioning,
    )


class FastModelTester:
    def __init__(
            self,
            parent,
            backbone_kernel_size=3,
            backbone_stride=2,
            backbone_dilation=1,
            backbone_groups=1,
            backbone_bias=False,
            backbone_has_shuffle=False,
            backbone_in_channels=3,
            backbone_out_channels=64,
            backbone_use_bn=True,
            backbone_act_func="relu",
            backbone_dropout_rate=0,
            backbone_ops_order="weight_bn_act",
            backbone_stage1_in_channels=[64, 64, 64],
            backbone_stage1_out_channels=[64, 64, 64],
            backbone_stage1_kernel_size=[[3, 3], [3, 3], [3, 3]],
            backbone_stage1_stride=[1, 2, 1],
            backbone_stage1_dilation=[1, 1, 1],
            backbone_stage1_groups=[1, 1, 1],
            backbone_stage2_in_channels=[64, 128, 128, 128],
            backbone_stage2_out_channels=[128, 128, 128, 128],
            backbone_stage2_kernel_size=[[3, 3], [1, 3], [3, 3], [3, 1]],
            backbone_stage2_stride=[2, 1, 1, 1],
            backbone_stage2_dilation=[1, 1, 1, 1],
            backbone_stage2_groups=[1, 1, 1, 1],
            backbone_stage3_in_channels=[128, 256, 256, 256],
            backbone_stage3_out_channels=[256, 256, 256, 256],
            backbone_stage3_kernel_size=[[3, 3], [3, 3], [3, 1], [1, 3]],
            backbone_stage3_stride=[2, 1, 1, 1],
            backbone_stage3_dilation=[1, 1, 1, 1],
            backbone_stage3_groups=[1, 1, 1, 1],
            backbone_stage4_in_channels=[256, 512, 512, 512],
            backbone_stage4_out_channels=[512, 512, 512, 512],
            backbone_stage4_kernel_size=[[3, 3], [3, 1], [1, 3], [3, 3]],
            backbone_stage4_stride=[2, 1, 1, 1],
            backbone_stage4_dilation=[1, 1, 1, 1],
            backbone_stage4_groups=[1, 1, 1, 1],
            neck_in_channels=[64, 128, 256, 512],
            neck_out_channels=[128, 128, 128, 128],
            neck_kernel_size=[[3, 3], [3, 3], [3, 3], [3, 3]],
            neck_stride=[1, 1, 1, 1],
            neck_dilation=[1, 1, 1, 1],
            neck_groups=[1, 1, 1, 1],
            head_pooling_size=9,
            head_dropout_ratio=0.1,
            head_conv_in_channels=512,
            head_conv_out_channels=128,
            head_conv_kernel_size=[3, 3],
            head_conv_stride=1,
            head_conv_dilation=1,
            head_conv_groups=1,
            head_final_kernel_size=1,
            head_final_stride=1,
            head_final_dilation=1,
            head_final_groups=1,
            head_final_bias=False,
            head_final_has_shuffle=False,
            head_final_in_channels=128,
            head_final_out_channels=5,
            head_final_use_bn=False,
            head_final_act_func=None,
            head_final_dropout_rate=0,
            head_final_ops_order="weight",
            batch_size=3,
            num_channels=3,
            image_size=500,
            is_training=True,
    ):
        self.parent = parent
        self.backbone_kernel_size = backbone_kernel_size
        self.backbone_stride = backbone_stride
        self.backbone_dilation = backbone_dilation
        self.backbone_groups = backbone_groups
        self.backbone_bias = backbone_bias
        self.backbone_has_shuffle = backbone_has_shuffle
        self.backbone_in_channels = backbone_in_channels
        self.backbone_out_channels = backbone_out_channels
        self.backbone_use_bn = backbone_use_bn
        self.backbone_act_func = backbone_act_func
        self.backbone_dropout_rate = backbone_dropout_rate
        self.backbone_ops_order = backbone_ops_order

        self.backbone_stage1_in_channels = backbone_stage1_in_channels
        self.backbone_stage1_out_channels = backbone_stage1_out_channels
        self.backbone_stage1_kernel_size = backbone_stage1_kernel_size
        self.backbone_stage1_stride = backbone_stage1_stride
        self.backbone_stage1_dilation = backbone_stage1_dilation
        self.backbone_stage1_groups = backbone_stage1_groups

        self.backbone_stage2_in_channels = backbone_stage2_in_channels
        self.backbone_stage2_out_channels = backbone_stage2_out_channels
        self.backbone_stage2_kernel_size = backbone_stage2_kernel_size
        self.backbone_stage2_stride = backbone_stage2_stride
        self.backbone_stage2_dilation = backbone_stage2_dilation
        self.backbone_stage2_groups = backbone_stage2_groups

        self.backbone_stage3_in_channels = backbone_stage3_in_channels
        self.backbone_stage3_out_channels = backbone_stage3_out_channels
        self.backbone_stage3_kernel_size = backbone_stage3_kernel_size
        self.backbone_stage3_stride = backbone_stage3_stride
        self.backbone_stage3_dilation = backbone_stage3_dilation
        self.backbone_stage3_groups = backbone_stage3_groups

        self.backbone_stage4_in_channels = backbone_stage4_in_channels
        self.backbone_stage4_out_channels = backbone_stage4_out_channels
        self.backbone_stage4_kernel_size = backbone_stage4_kernel_size
        self.backbone_stage4_stride = backbone_stage4_stride
        self.backbone_stage4_dilation = backbone_stage4_dilation
        self.backbone_stage4_groups = backbone_stage4_groups

        self.neck_in_channels = neck_in_channels
        self.neck_out_channels = neck_out_channels
        self.neck_kernel_size = neck_kernel_size
        self.neck_stride = neck_stride
        self.neck_dilation = neck_dilation
        self.neck_groups = neck_groups

        self.head_pooling_size = head_pooling_size
        self.head_dropout_ratio = head_dropout_ratio

        self.head_conv_in_channels = head_conv_in_channels
        self.head_conv_out_channels = head_conv_out_channels
        self.head_conv_kernel_size = head_conv_kernel_size
        self.head_conv_stride = head_conv_stride
        self.head_conv_dilation = head_conv_dilation
        self.head_conv_groups = head_conv_groups

        self.head_final_kernel_size = head_final_kernel_size
        self.head_final_stride = head_final_stride
        self.head_final_dilation = head_final_dilation
        self.head_final_groups = head_final_groups
        self.head_final_bias = head_final_bias
        self.head_final_has_shuffle = head_final_has_shuffle
        self.head_final_in_channels = head_final_in_channels
        self.head_final_out_channels = head_final_out_channels
        self.head_final_use_bn = head_final_use_bn
        self.head_final_act_func = head_final_act_func
        self.head_final_dropout_rate = head_final_dropout_rate
        self.head_final_ops_order = head_final_ops_order

        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        pixel_values_meta = {
            "org_img_size": (500, 500),
            "img_size": (500, 500)
        }
        # labels = None
        # if self.use_labels:
        #     labels = ids_tensor([self.batch_size], self.num_labels)
        #
        config = self.get_config()

        return config, {"imgs": pixel_values, "img_metas": pixel_values_meta}

    def get_config(self):
        return FastConfig(
            backbone_kernel_size=self.backbone_kernel_size,
            backbone_stride=self.backbone_stride,
            backbone_dilation=self.backbone_dilation,
            backbone_groups=self.backbone_groups,
            backbone_bias=self.backbone_bias,
            backbone_has_shuffle=self.backbone_has_shuffle,
            backbone_in_channels=self.backbone_in_channels,
            backbone_out_channels=self.backbone_out_channels,
            backbone_use_bn=self.backbone_use_bn,
            backbone_act_func=self.backbone_act_func,
            backbone_dropout_rate=self.backbone_dropout_rate,
            backbone_ops_order=self.backbone_ops_order,
            backbone_stage1_in_channels=self.backbone_stage1_in_channels,
            backbone_stage1_out_channels=self.backbone_stage1_out_channels,
            backbone_stage1_kernel_size=self.backbone_stage1_kernel_size,
            backbone_stage1_stride=self.backbone_stage1_stride,
            backbone_stage1_dilation=self.backbone_stage1_dilation,
            backbone_stage1_groups=self.backbone_stage1_groups,
            backbone_stage2_in_channels=self.backbone_stage2_in_channels,
            backbone_stage2_out_channels=self.backbone_stage2_out_channels,
            backbone_stage2_kernel_size=self.backbone_stage2_kernel_size,
            backbone_stage2_stride=self.backbone_stage2_stride,
            backbone_stage2_dilation=self.backbone_stage2_dilation,
            backbone_stage2_groups=self.backbone_stage2_groups,
            backbone_stage3_in_channels=self.backbone_stage3_in_channels,
            backbone_stage3_out_channels=self.backbone_stage3_out_channels,
            backbone_stage3_kernel_size=self.backbone_stage3_kernel_size,
            backbone_stage3_stride=self.backbone_stage3_stride,
            backbone_stage3_dilation=self.backbone_stage3_dilation,
            backbone_stage3_groups=self.backbone_stage3_groups,
            backbone_stage4_in_channels=self.backbone_stage4_in_channels,
            backbone_stage4_out_channels=self.backbone_stage4_out_channels,
            backbone_stage4_kernel_size=self.backbone_stage4_kernel_size,
            backbone_stage4_stride=self.backbone_stage4_stride,
            backbone_stage4_dilation=self.backbone_stage4_dilation,
            backbone_stage4_groups=self.backbone_stage4_groups,
            neck_in_channels=self.neck_in_channels,
            neck_out_channels=self.neck_out_channels,
            neck_kernel_size=self.neck_kernel_size,
            neck_stride=self.neck_stride,
            neck_dilation=self.neck_dilation,
            neck_groups=self.neck_groups,
            head_pooling_size=self.head_pooling_size,
            head_dropout_ratio=self.head_dropout_ratio,
            head_conv_in_channels=self.head_conv_in_channels,
            head_conv_out_channels=self.head_conv_out_channels,
            head_conv_kernel_size=self.head_conv_kernel_size,
            head_conv_stride=self.head_conv_stride,
            head_conv_dilation=self.head_conv_dilation,
            head_conv_groups=self.head_conv_groups,
            head_final_kernel_size=self.head_final_kernel_size,
            head_final_stride=self.head_final_stride,
            head_final_dilation=self.head_final_dilation,
            head_final_groups=self.head_final_groups,
            head_final_bias=self.head_final_bias,
            head_final_has_shuffle=self.head_final_has_shuffle,
            head_final_in_channels=self.head_final_in_channels,
            head_final_out_channels=self.head_final_out_channels,
            head_final_use_bn=self.head_final_use_bn,
            head_final_act_func=self.head_final_act_func,
            head_final_dropout_rate=self.head_final_dropout_rate,
            head_final_ops_order=self.head_final_ops_order,
        )

    def create_and_check_model(self, config, input):
        model = FASTForImageCaptioning(config=config)
        model.to(torch_device)
        model.eval()
        result = model(imgs=input['imgs'], imgs_mets=input['img_metas'])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, inputs_dict = config_and_inputs
        return config, inputs_dict


@require_torch
class FastModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            FASTForImageCaptioning,
        )
        if is_torch_available()
        else ()
    )

    pipeline_model_mapping = {}
    test_headmasking = False
    test_pruning = False
    test_attention_outputs = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        self.model_tester = FastModelTester(self)
        self.config_tester = ConfigTester(self, config_class=FastConfig, hidden_size=37)

    def test_config(self):
        self.create_and_test_config_common_properties()
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def create_and_test_config_common_properties(self):
        return

    @unittest.skip(reason="Fast does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Fast does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass
