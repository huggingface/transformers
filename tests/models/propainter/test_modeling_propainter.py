# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the S-Lab License, Version 1.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/sczhou/ProPainter/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing suite for the PyTorch ProPainter model."""

import gc
import unittest

import numpy as np
import requests

from datasets import load_dataset
from transformers import ProPainterConfig
from transformers.testing_utils import (
    require_accelerate,
    require_torch,
    require_torch_accelerator,
    require_torch_fp16,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn
    from transformers import ProPainterModel


if is_vision_available():
    from PIL import Image
    from transformers import ProPainterImageProcessor


class ProPainterModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=30,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_frames=8,
        # hidden_act="gelu",
        # hidden_dropout_prob=0.1,
        # attention_probs_dropout_prob=0.1,
        # type_sequence_label_size=10,
        # initializer_range=0.02,
        # GAN_LOSS="hinge",
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_frames = num_frames
        # self.hidden_act = hidden_act
        # self.hidden_dropout_prob = hidden_dropout_prob
        # self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # self.type_sequence_label_size = type_sequence_label_size
        # self.initializer_range = initializer_range

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_frames, 3, self.image_size, self.image_size])

        pixel_values_inp = pixel_values.cpu.numpy()
        pixel_values_inp = [[pixel_values_inp[b, t].transpose(1, 2, 0) for t in range(pixel_values_inp.shape[1])] for b in range(pixel_values_inp.shape[0])]

        masks = ids_tensor([self.batch_size, self.num_frames, 1, self.image_size, self.image_size], vocab_size=2).float()
        flow_masks = masks_dilated = masks
        config = self.get_config()

        return config, pixel_values_inp, pixel_values, flow_masks, masks_dilated, (self.image_size, self.image_size)

    def get_config(self):
        return ProPainterConfig(
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
        )

    def create_and_check_model(self, config, pixel_values_inp, pixel_values, flow_masks, masks_dilated):
        model = ProPainterModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values_inp, pixel_values, flow_masks, masks_dilated)
        self.parent.assertEqual(result.reconstruction.shape, (self.batch_size, self.num_frames, self.image_size, self.image_size, 3))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values_inp,
            pixel_values,
            flow_masks,
            masks_dilated,
        ) = config_and_inputs
        inputs_dict = {"pixel_values_inp": pixel_values_inp, "pixel_values": pixel_values, "flow_masks": flow_masks, "masks_dilated": masks_dilated}
        return config, inputs_dict


@require_torch
class ProPainterModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as ProPainter does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (
            ProPainterModel,
        )
        if is_torch_available()
        else ()
    )
    fx_compatible = False

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = ProPainterModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ProPainterConfig, has_text_modality=False, hidden_size=37)

    @unittest.skip(
        "Since `torch==2.3+cu121`, although this test passes, many subsequent tests have `CUDA error: misaligned address`."
        "If `nvidia-xxx-cu118` are also installed, no failure (even with `torch==2.3+cu121`)."
    )
    def test_multi_gpu_data_parallel_forward(self):
        super().test_multi_gpu_data_parallel_forward()

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="ProPainter does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="ProPainter does not have get_input_embeddings method and get_output_embeddings method")
    def test_model_get_set_embeddings(self):
        pass

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "ruffy369/ProPainter"
        model = ProPainterModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_video():
    ds = load_dataset("ruffy369/propainter-object-removal")
    ds_images = ds['train']["image"]
    num_frames = len(ds_images)//2
    video = [np.array(ds_images[i]) for i in range(num_frames)]
    #stack to convert H,W mask frame to compatible H,W,C frame
    masks = [np.stack([np.array(ds_images[i])] * 3, axis=-1) for i in range(num_frames, 2*num_frames)]
    return video, masks



@require_torch
@require_vision
class ProPainterModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return ProPainterImageProcessor() if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        model = ProPainterModel.from_pretrained("ruffy369/ProPainter").to(torch_device)

        image_processor = self.default_image_processor
        video, masks = prepare_video()
        inputs = image_processor(images=video, masks=masks, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((80, 240, 432, 3))
        self.assertEqual(outputs.reconstruction, expected_shape)

        expected_slice = torch.tensor(
            [[0.5458, 0.5546, 0.5638], [0.5526, 0.5565, 0.5651], [0.5396, 0.5426, 0.5621]]
        ).to(torch_device)

        print("gggggggggggg", outputs.reconstruction[0, 0, :3, :3])
        self.assertTrue(torch.allclose(outputs.reconstruction[0, 0, :3, :3], expected_slice, atol=1e-4))

    @slow
    @require_accelerate
    @require_torch_accelerator
    @require_torch_fp16
    def test_inference_fp16(self):
        r"""
        A small test to make sure that inference work in half precision without any problem.
        """
        model = ProPainterModel.from_pretrained("ruffy369/ProPainter", torch_dtype=torch.float16, device_map="auto")
        image_processor = self.default_image_processor

        video, masks = prepare_video()
        inputs = image_processor(images=video, masks=masks, return_tensors="pt").to(torch_device)

        # forward pass to make sure inference works in fp16
        with torch.no_grad():
            _ = model(**inputs)
