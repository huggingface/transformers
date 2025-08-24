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
"""Testing suite for the PyTorch VideoMAE model."""

import copy
import tempfile
import unittest

import numpy as np
from huggingface_hub import hf_hub_download
from pytest import mark

from transformers import VideoMAEConfig
from transformers.models.auto import get_values
from transformers.testing_utils import (
    is_flaky,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
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

    from transformers import (
        MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING,
        VideoMAEForPreTraining,
        VideoMAEForVideoClassification,
        VideoMAEModel,
    )


if is_vision_available():
    from transformers import VideoMAEImageProcessor


class VideoMAEModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=10,
        num_channels=3,
        patch_size=2,
        tubelet_size=2,
        num_frames=2,
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
        mask_ratio=0.9,
        scope=None,
        attn_implementation="eager",
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames
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
        self.mask_ratio = mask_ratio
        self.scope = scope
        self.attn_implementation = attn_implementation

        # in VideoMAE, the number of tokens equals num_frames/tubelet_size * num_patches per frame
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        self.seq_length = (num_frames // tubelet_size) * self.num_patches_per_frame

        # use this variable to define bool_masked_pos
        self.num_masks = int(mask_ratio * self.seq_length)

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [self.batch_size, self.num_frames, self.num_channels, self.image_size, self.image_size]
        )

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return VideoMAEConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            num_frames=self.num_frames,
            tubelet_size=self.tubelet_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
            decoder_hidden_size=self.hidden_size,
            decoder_intermediate_size=self.intermediate_size,
            decoder_num_attention_heads=self.num_attention_heads,
            decoder_num_hidden_layers=self.num_hidden_layers,
            attn_implementation=self.attn_implementation,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = VideoMAEModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_pretraining(self, config, pixel_values, labels):
        model = VideoMAEForPreTraining(config)
        model.to(torch_device)
        model.eval()
        # important: each video needs to have the same number of masked patches
        # hence we define a single mask, which we then repeat for each example in the batch
        mask = torch.ones((self.num_masks,))
        mask = torch.cat([mask, torch.zeros(self.seq_length - mask.size(0))])
        bool_masked_pos = mask.expand(self.batch_size, -1).bool()

        result = model(pixel_values, bool_masked_pos)
        # model only returns predictions for masked patches
        num_masked_patches = mask.sum().item()
        decoder_num_labels = 3 * self.tubelet_size * self.patch_size**2
        self.parent.assertEqual(result.logits.shape, (self.batch_size, num_masked_patches, decoder_num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class VideoMAEModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as VideoMAE does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (VideoMAEModel, VideoMAEForPreTraining, VideoMAEForVideoClassification) if is_torch_available() else ()
    )
    pipeline_model_mapping = (
        {"feature-extraction": VideoMAEModel, "video-classification": VideoMAEForVideoClassification}
        if is_torch_available()
        else {}
    )

    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torch_exportable = True

    def setUp(self):
        self.model_tester = VideoMAEModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VideoMAEConfig, has_text_modality=False, hidden_size=37)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)

        if model_class == VideoMAEForPreTraining:
            # important: each video needs to have the same number of masked patches
            # hence we define a single mask, which we then repeat for each example in the batch
            mask = torch.ones((self.model_tester.num_masks,))
            mask = torch.cat([mask, torch.zeros(self.model_tester.seq_length - mask.size(0))])
            batch_size = inputs_dict["pixel_values"].shape[0]
            bool_masked_pos = mask.expand(batch_size, -1).bool()
            inputs_dict["bool_masked_pos"] = bool_masked_pos.to(torch_device)

        if return_labels:
            if model_class in [
                *get_values(MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING),
            ]:
                inputs_dict["labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )

        return inputs_dict

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="VideoMAE does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "MCG-NJU/videomae-base"
        model = VideoMAEModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def test_attention_outputs(self):
        if not self.has_attentions:
            self.skipTest(reason="Model does not have attentions")

        else:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            for model_class in self.all_model_classes:
                num_visible_patches = self.model_tester.seq_length - self.model_tester.num_masks
                seq_len = (
                    num_visible_patches if model_class == VideoMAEForPreTraining else self.model_tester.seq_length
                )

                inputs_dict["output_attentions"] = True
                inputs_dict["output_hidden_states"] = False
                config.return_dict = True
                model = model_class(config)
                model.to(torch_device)
                model.eval()
                with torch.no_grad():
                    outputs = model(**self._prepare_for_class(inputs_dict, model_class))
                attentions = outputs.attentions
                self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

                # check that output_attentions also work using config
                del inputs_dict["output_attentions"]
                config.output_attentions = True
                model = model_class(config)
                model.to(torch_device)
                model.eval()
                with torch.no_grad():
                    outputs = model(**self._prepare_for_class(inputs_dict, model_class))
                attentions = outputs.attentions
                self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, seq_len, seq_len],
                )
                out_len = len(outputs)

                # Check attention is always last and order is fine
                inputs_dict["output_attentions"] = True
                inputs_dict["output_hidden_states"] = True
                model = model_class(config)
                model.to(torch_device)
                model.eval()
                with torch.no_grad():
                    outputs = model(**self._prepare_for_class(inputs_dict, model_class))

                self.assertEqual(out_len + 1, len(outputs))

                self_attentions = outputs.attentions

                self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, seq_len, seq_len],
                )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states
            expected_num_layers = self.model_tester.num_hidden_layers + 1
            self.assertEqual(len(hidden_states), expected_num_layers)

            num_visible_patches = self.model_tester.seq_length - self.model_tester.num_masks
            seq_length = num_visible_patches if model_class == VideoMAEForPreTraining else self.model_tester.seq_length

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    @is_flaky()
    def test_flash_attn_2_inference_equivalence(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            inputs_dict["pixel_values"] = inputs_dict["pixel_values"].to(torch.bfloat16)

            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.bfloat16)
                model.to(torch_device)

                outputs = model(**inputs_dict, output_hidden_states=True)
                outputs_fa = model_fa(**inputs_dict, output_hidden_states=True)

                logits = (
                    outputs.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                logits_fa = (
                    outputs_fa.hidden_states[-1]
                    if not model.config.is_encoder_decoder
                    else outputs_fa.decoder_hidden_states[-1]
                )

                assert torch.allclose(logits_fa, logits, atol=4e-2, rtol=4e-2)

                # check with inference + dropout
                model.train()
                _ = model_fa(**inputs_dict)

    @unittest.skip("Not applicable for VideoMAE")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass


# We will verify our results on a video of eating spaghetti
# Frame indices used: [164 168 172 176 181 185 189 193 198 202 206 210 215 219 223 227]
def prepare_video():
    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video", filename="eating_spaghetti.npy", repo_type="dataset"
    )
    video = np.load(file)
    return list(video)


@require_torch
@require_vision
class VideoMAEModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        # logits were tested with a different mean and std, so we use the same here
        return (
            VideoMAEImageProcessor(image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
            if is_vision_available()
            else None
        )

    @slow
    def test_inference_for_video_classification(self):
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics").to(
            torch_device
        )

        image_processor = self.default_image_processor
        video = prepare_video()
        inputs = image_processor(video, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 400))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([0.3669, -0.0688, -0.2421]).to(torch_device)

        torch.testing.assert_close(outputs.logits[0, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_for_pretraining(self):
        model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base-short").to(torch_device)

        image_processor = self.default_image_processor
        video = prepare_video()
        inputs = image_processor(video, return_tensors="pt").to(torch_device)

        # add boolean mask, indicating which patches to mask
        local_path = hf_hub_download(repo_id="hf-internal-testing/bool-masked-pos", filename="bool_masked_pos.pt")
        inputs["bool_masked_pos"] = torch.load(local_path, weights_only=True)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size([1, 1408, 1536])
        expected_slice = torch.tensor(
            [[0.7994, 0.9612, 0.8508], [0.7401, 0.8958, 0.8302], [0.5862, 0.7468, 0.7325]], device=torch_device
        )
        self.assertEqual(outputs.logits.shape, expected_shape)
        torch.testing.assert_close(outputs.logits[0, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

        # verify the loss (`config.norm_pix_loss` = `True`)
        expected_loss = torch.tensor([0.5142], device=torch_device)
        torch.testing.assert_close(outputs.loss, expected_loss, rtol=1e-4, atol=1e-4)

        # verify the loss (`config.norm_pix_loss` = `False`)
        model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base-short", norm_pix_loss=False).to(
            torch_device
        )

        with torch.no_grad():
            outputs = model(**inputs)

        expected_loss = torch.tensor(torch.tensor([0.6469]), device=torch_device)
        torch.testing.assert_close(outputs.loss, expected_loss, rtol=1e-4, atol=1e-4)
