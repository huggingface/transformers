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

import copy
import gc
import unittest

import numpy as np
import requests

from datasets import load_dataset
from transformers import ProPainterConfig
from transformers.models.auto import get_values
from transformers.models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES, MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES
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
        batch_size=8,
        image_size=64,
        is_training=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_frames=8,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_frames = num_frames


    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_frames, 3, self.image_size, self.image_size])

        pixel_values_inp = pixel_values.cpu().numpy().astype(np.uint8)[0]
        pixel_values_inp = [pixel_values_inp[t].transpose(1, 2, 0) for t in range(pixel_values_inp.shape[0])]

        masks = ids_tensor([self.batch_size, self.num_frames, 1, self.image_size, self.image_size], vocab_size=2).float()
        flow_masks = masks_dilated = masks
        config = self.get_config()

        return config, pixel_values_inp, pixel_values, flow_masks, masks_dilated

    def get_config(self):
        return ProPainterConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
        )

    @property
    def encoder_seq_length(self):
        window_size = self.get_config().window_size
        return window_size[0]*window_size[1]

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
    pipeline_model_mapping = {"image-to-image": ProPainterModel} if is_torch_available() else {}
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

    def test_attention_outputs(self):
        if not self.has_attentions:
            self.skipTest(reason="Model does not output attentions")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length", decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
        chunk_length = getattr(self.model_tester, "chunk_length", None)
        if chunk_length is not None and hasattr(self.model_tester, "num_hashes"):
            encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            if chunk_length is not None:
                self.assertListEqual(
                    list(attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertIn(
                    list(attentions[0][1].shape[-4:])[1],
                    [6, 8]  # Allowable values for this dimension
                )
                self.assertListEqual(
                    list(attentions[0][1].shape[-4:]),
                    [self.model_tester.num_attention_heads, list(attentions[0][1].shape[-4:])[1], encoder_seq_length, encoder_key_length],
                )
            out_len = len(outputs)

            if self.is_encoder_decoder:
                correct_outlen = 5

                # loss is at first position
                if "labels" in inputs_dict:
                    correct_outlen += 1  # loss is added to beginning
                # Question Answering model returns start_logits and end_logits
                if model_class.__name__ in [
                    *get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES),
                    *get_values(MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES),
                ]:
                    correct_outlen += 1  # start_logits and end_logits instead of only 1 output
                if "past_key_values" in outputs:
                    correct_outlen += 1  # past_key_values have been returned

                self.assertEqual(out_len, correct_outlen)

                # decoder attentions
                decoder_attentions = outputs.decoder_attentions
                self.assertIsInstance(decoder_attentions, (list, tuple))
                self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(decoder_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
                )

                # cross attentions
                cross_attentions = outputs.cross_attentions
                self.assertIsInstance(cross_attentions, (list, tuple))
                self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(cross_attentions[0].shape[-3:]),
                    [
                        self.model_tester.num_attention_heads,
                        decoder_seq_length,
                        encoder_key_length,
                    ],
                )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            elif self.is_encoder_decoder:
                added_hidden_states = 2
            else:
                added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            if chunk_length is not None:
                self.assertListEqual(
                    list(self_attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertIn(
                    list(attentions[0][1].shape[-4:])[1],
                    [6, 8]  # Allowable values for this dimension
                )
                self.assertListEqual(
                    list(attentions[0][1].shape[-4:]),
                    [self.model_tester.num_attention_heads, list(attentions[0][1].shape[-4:])[1], encoder_seq_length, encoder_key_length],
                )

    def test_feed_forward_chunking(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            torch.manual_seed(0)
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            hidden_states_no_chunk = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            torch.manual_seed(0)
            config.chunk_size_feed_forward = 1
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            hidden_states_with_chunk = model(**self._prepare_for_class(inputs_dict, model_class))[0]
            #As the output at idx 0 is a tuple with three different losses together whihc are generator loss, discriminator loss and flow complete loss
            for hs_no_chunk, hs_with_chunk in zip(hidden_states_no_chunk, hidden_states_with_chunk):
                self.assertTrue(torch.allclose(hs_no_chunk, hs_with_chunk, atol=1e-3))


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
        expected_shape = torch.Size((80, 240,432, 3))
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
