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
""" Testing suite for the PyTorch TVLT model. """

import copy
import inspect
import unittest

import numpy as np

from huggingface_hub import hf_hub_download
from transformers import (
    TvltConfig,
    is_datasets_available,
    is_speech_available,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch
    import torch.nn as nn

    from transformers import TvltForAudioVisualClassification, TvltForPreTraining, TvltForQuestionAnswering, TvltModel
    from transformers.models.tvlt.modeling_tvlt import TVLT_PRETRAINED_MODEL_ARCHIVE_LIST
    from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_10
else:
    is_torch_greater_or_equal_than_1_10 = False


if is_datasets_available():
    from datasets import load_dataset

if is_vision_available():
    from transformers import TvltImageProcessor

if is_speech_available():
    from transformers import TvltFeatureExtractor


class TvltModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        pixel_size=32,
        audio_size=32,
        feature_size=16,
        pixel_patch_size=2,
        audio_patch_size=[2, 2],
        num_pixel_channels=3,
        num_audio_channels=1,
        num_frames=2,
        hidden_size=128,
        num_hidden_layers=12,
        num_attention_heads=4,
        intermediate_size=128,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        qkv_bias=True,
        use_mean_pooling=True,
        decoder_num_attention_heads=4,
        decoder_hidden_size=64,
        decoder_num_hidden_layers=2,
        decoder_intermediate_size=128,
        norm_pix_loss=True,
        pixel_mask_ratio=0.75,
        audio_mask_ratio=0.15,
        task_matching=True,
        task_mae=True,
        num_qa_labels=1,
        num_labels=1,
        is_training=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.pixel_size = pixel_size
        self.audio_size = audio_size
        self.feature_size = feature_size
        self.pixel_patch_size = pixel_patch_size
        self.audio_patch_size = audio_patch_size
        self.num_pixel_channels = num_pixel_channels
        self.num_audio_channels = num_audio_channels
        self.num_frames = num_frames

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.use_mean_pooling = use_mean_pooling

        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.norm_pix_loss = norm_pix_loss
        self.pixel_mask_ratio = pixel_mask_ratio
        self.audio_mask_ratio = audio_mask_ratio

        self.task_matching = task_matching
        self.task_mae = task_mae
        self.num_qa_labels = num_qa_labels
        self.num_labels = num_labels

        self.expected_pixel_seq_len = (self.pixel_size // self.pixel_patch_size) ** 2 * self.num_frames
        self.expected_audio_seq_len = (self.audio_size // self.audio_patch_size[0]) * (
            self.feature_size // self.audio_patch_size[1]
        )
        # we set the expected sequence length (which is used in several tests)
        # this is equal to the seq length of number of image/video patches + number of audio patches
        self.expected_seq_len = self.expected_pixel_seq_len + self.expected_audio_seq_len + 1

        self.pixel_mae_output_dim = pixel_patch_size**2 * num_pixel_channels
        self.audio_mae_output_dim = audio_patch_size[0] * audio_patch_size[1] * num_audio_channels
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [self.batch_size, self.num_frames, self.num_pixel_channels, self.pixel_size, self.pixel_size]
        )
        audio_values = floats_tensor([self.batch_size, self.num_audio_channels, self.audio_size, self.feature_size])

        pixel_masks = floats_tensor([self.batch_size, self.expected_pixel_seq_len])
        audio_masks = floats_tensor([self.batch_size, self.expected_audio_seq_len])

        config = self.get_config()

        return (config, pixel_values, audio_values, pixel_masks, audio_masks)

    def prepare_config_and_inputs_for_pretraining(self):
        pixel_values = floats_tensor(
            [self.batch_size, self.num_frames, self.num_pixel_channels, self.pixel_size, self.pixel_size]
        )
        audio_values = floats_tensor([self.batch_size, self.num_audio_channels, self.audio_size, self.feature_size])

        pixel_masks = floats_tensor([self.batch_size, self.expected_pixel_seq_len])
        audio_masks = floats_tensor([self.batch_size, self.expected_audio_seq_len])

        pixel_mask_position_permutation = torch.argsort(
            floats_tensor(
                [self.batch_size, self.expected_pixel_seq_len],
            ),
            dim=1,
        )
        audio_mask_position_permutation = torch.argsort(
            floats_tensor([self.batch_size, self.expected_audio_seq_len]),
            dim=1,
        )

        pixel_values_mixed = floats_tensor(
            [self.batch_size, self.num_frames, self.num_pixel_channels, self.pixel_size, self.pixel_size]
        )
        pixel_masks_mixed = floats_tensor([self.batch_size, self.expected_pixel_seq_len])
        labels = floats_tensor([self.batch_size])
        config = self.get_config()

        return (
            config,
            pixel_values,
            audio_values,
            pixel_masks,
            audio_masks,
            pixel_mask_position_permutation,
            audio_mask_position_permutation,
            pixel_values_mixed,
            pixel_masks_mixed,
            labels,
        )

    def get_config(self):
        return TvltConfig(
            pixel_size=self.pixel_size,
            audio_size=self.audio_size,
            feature_size=self.feature_size,
            pixel_patch_size=self.pixel_patch_size,
            audio_patch_size=self.audio_patch_size,
            num_pixel_channels=self.num_pixel_channels,
            num_audio_channels=self.num_audio_channels,
            num_frames=self.num_frames,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            qkv_bias=self.qkv_bias,
            use_mean_pooling=self.use_mean_pooling,
            decoder_num_attention_heads=self.decoder_num_attention_heads,
            decoder_hidden_size=self.decoder_hidden_size,
            decoder_num_hidden_layers=self.decoder_num_hidden_layers,
            decoder_intermediate_size=self.decoder_intermediate_size,
            norm_pix_loss=self.norm_pix_loss,
            pixel_mask_ratio=self.pixel_mask_ratio,
            audio_mask_ratio=self.audio_mask_ratio,
            task_matching=self.task_matching,
            task_mae=self.task_mae,
            num_qa_labels=self.num_qa_labels,
            num_labels=self.num_labels,
        )

    def create_and_check_model(self, config, pixel_values, audio_values, pixel_masks, audio_masks):
        model = TvltModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, audio_values, pixel_masks=pixel_masks, audio_masks=audio_masks)
        result = model(pixel_values, audio_values)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.expected_seq_len, self.hidden_size)
        )

    def create_and_check_for_question_answering(self, config, pixel_values, audio_values, pixel_masks, audio_masks):
        model = TvltForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, audio_values, pixel_masks=pixel_masks, audio_masks=audio_masks)
        result = model(pixel_values, audio_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_audiovisual_classification(
        self, config, pixel_values, audio_values, pixel_masks, audio_masks
    ):
        model = TvltForAudioVisualClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, audio_values, pixel_masks=pixel_masks, audio_masks=audio_masks)
        result = model(pixel_values, audio_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_pretraining(
        self,
        config,
        pixel_values,
        audio_values,
        pixel_masks,
        audio_masks,
        pixel_mask_position_permutation,
        audio_mask_position_permutation,
        pixel_values_mixed,
        pixel_masks_mixed,
        labels,
    ):
        model = TvltForPreTraining(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            pixel_values,
            audio_values,
            pixel_masks,
            audio_masks,
            pixel_mask_position_permutation=pixel_mask_position_permutation,
            audio_mask_position_permutation=audio_mask_position_permutation,
            pixel_values_mixed=pixel_values_mixed,
            pixel_masks_mixed=pixel_masks_mixed,
            labels=labels,
        )
        self.parent.assertEqual(
            result.pixel_logits.shape, (self.batch_size, self.expected_pixel_seq_len, self.pixel_mae_output_dim)
        )
        self.parent.assertEqual(
            result.audio_logits.shape, (self.batch_size, self.expected_audio_seq_len, self.audio_mae_output_dim)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, pixel_values, audio_values, pixel_masks, audio_masks) = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "audio_values": audio_values,
            "pixel_masks": pixel_masks,
            "audio_masks": audio_masks,
        }
        return config, inputs_dict

    def prepare_pixel_values(self):
        return floats_tensor(
            [self.batch_size, self.num_frames, self.num_pixel_channels, self.pixel_size, self.pixel_size]
        )

    def prepare_audio_values(self):
        return floats_tensor([self.batch_size, self.num_audio_channels, self.audio_size, self.feature_size])


@require_torch
@unittest.skipIf(not is_torch_greater_or_equal_than_1_10, "TVLT is only available in torch v1.10+")
class TvltModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (TvltModel, TvltForPreTraining, TvltForQuestionAnswering, TvltForAudioVisualClassification)
        if is_torch_available()
        else ()
    )

    fx_compatible = False
    test_pruning = False
    test_headmasking = False
    test_torchscript = False
    test_resize_embeddings = False
    main_input_name = "pixel_values"

    # TvltForQuestionAnswering, TvltForAudioVisualClassification and TvltForPreTraining require special treatment
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=True):
        inputs_dict = copy.deepcopy(inputs_dict)

        if return_labels:
            if model_class.__name__ == "TvltForQuestionAnswering":
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size,), dtype=torch.long, device=torch_device
                )
            elif model_class.__name__ == "TvltForAudioVisualClassification":
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size,), dtype=torch.long, device=torch_device
                )
            elif model_class.__name__ == "TvltForPreTraining":
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size,), dtype=torch.float, device=torch_device
                )
                inputs_dict["pixel_mask_position_permutation"] = torch.argsort(
                    torch.zeros(
                        (self.model_tester.batch_size, self.model_tester.expected_pixel_seq_len),
                        dtype=torch.float,
                        device=torch_device,
                    ),
                    dim=1,
                )
                inputs_dict["audio_mask_position_permutation"] = torch.argsort(
                    torch.zeros(
                        (self.model_tester.batch_size, self.model_tester.expected_audio_seq_len),
                        dtype=torch.float,
                        device=torch_device,
                    ),
                    dim=1,
                )
                inputs_dict["pixel_values_mixed"] = torch.zeros(
                    (
                        self.model_tester.batch_size,
                        self.model_tester.num_frames,
                        self.model_tester.num_pixel_channels,
                        self.model_tester.pixel_size,
                        self.model_tester.pixel_size,
                    ),
                    dtype=torch.float,
                    device=torch_device,
                )
                inputs_dict["pixel_masks_mixed"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.expected_pixel_seq_len),
                    dtype=torch.float,
                    device=torch_device,
                )

        return inputs_dict

    def setUp(self):
        self.model_tester = TvltModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TvltConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="TVLT does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            input_embeddings = model.get_input_embeddings()
            self.assertIsInstance(input_embeddings, (tuple))
            for embedding in input_embeddings:
                self.assertIsInstance(embedding, (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values", "audio_values"]
            self.assertListEqual(arg_names[:2], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_audiovisual_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_audiovisual_classification(*config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_pretraining()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in TVLT_PRETRAINED_MODEL_ARCHIVE_LIST:
            model = TvltModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_training(self):
        if not self.model_tester.is_training:
            return

        for model_class in self.all_model_classes[1:]:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class)
            for k, v in inputs.items():
                print(k, v.shape)
            loss = model(**inputs).loss
            loss.backward()

    def test_training_gradient_checkpointing(self):
        if not self.model_tester.is_training:
            return

        for model_class in self.all_model_classes[1:]:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.use_cache = False
            config.return_dict = True

            model = model_class(config)
            model.to(torch_device)
            model.gradient_checkpointing_enable()
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class)
            loss = model(**inputs).loss
            loss.backward()

    def test_attention_outputs(self):
        if not self.has_attentions:
            pass

        else:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            for model_class in self.all_model_classes[2:]:
                seq_len = self.model_tester.expected_seq_len

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

            seq_length = self.model_tester.expected_seq_len

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes[2:]:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)


# We will verify our results on a video of eating spaghetti
# Frame indices used: [164 168 172 176 181 185 189 193 198 202 206 210 215 219 223 227]
def prepare_video(num_frames=8):
    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video", filename="eating_spaghetti.npy", repo_type="dataset"
    )
    video = np.load(file)[:num_frames]
    return list(video)


def prepare_audio(num_samples=1):
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    # automatic decoding with librispeech
    speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]
    return [x["array"] for x in speech_samples]


@require_torch
@require_vision
class TvltModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_feature_extractor(self):
        # logits were tested with a different mean and std, so we use the same here
        return (TvltImageProcessor() if is_vision_available() else None, TvltFeatureExtractor())

    def test_inference_for_question_answering(self):
        model = TvltForQuestionAnswering.from_pretrained("TVLT/tvlt-base").to(torch_device)

        image_processor, audio_feature_extractor = self.default_feature_extractor
        video = prepare_video()
        audio = prepare_audio()
        video_inputs = image_processor(video, return_tensors="pt").to(torch_device)
        audio_inputs = audio_feature_extractor(audio, return_tensors="pt").to(torch_device)
        inputs = dict()
        inputs.update(video_inputs)
        inputs.update(audio_inputs)
        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1))
        self.assertEqual(outputs.logits.shape, expected_shape)

    def test_inference_for_audiovisual_classification(self):
        model = TvltForAudioVisualClassification.from_pretrained("TVLT/tvlt-base").to(torch_device)

        image_processor, audio_feature_extractor = self.default_feature_extractor
        video = prepare_video()
        audio = prepare_audio()
        video_inputs = image_processor(video, return_tensors="pt").to(torch_device)
        audio_inputs = audio_feature_extractor(audio, return_tensors="pt").to(torch_device)
        inputs = dict()
        inputs.update(video_inputs)
        inputs.update(audio_inputs)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1))
        self.assertEqual(outputs.logits.shape, expected_shape)

    def test_inference_for_pretraining(self):
        model = TvltForPreTraining.from_pretrained("TVLT/tvlt-base").to(torch_device)

        image_processor, audio_feature_extractor = self.default_feature_extractor
        video = prepare_video()
        video_mixed = prepare_video()
        audio = prepare_audio()
        video_inputs = image_processor(video, return_tensors="pt", mask_pixel=True).to(torch_device)
        video_mixed_inputs = image_processor(video_mixed, is_mixed=True, return_tensors="pt").to(torch_device)
        audio_inputs = audio_feature_extractor(audio, return_tensors="pt", mask_audio=True).to(torch_device)
        labels = torch.tensor([[0.0]], device=torch_device)
        inputs = dict()
        inputs.update(video_inputs)
        inputs.update(video_mixed_inputs)
        inputs.update(audio_inputs)
        inputs.update({"labels": labels})

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_pixel_logits_shape = torch.Size([1, 1568, 768])
        expected_audio_logits_shape = torch.Size([1, 96, 256])
        expected_matching_logits_shape = torch.Size([1, 1])

        self.assertEqual(outputs.pixel_logits.shape, expected_pixel_logits_shape)
        self.assertEqual(outputs.audio_logits.shape, expected_audio_logits_shape)
        self.assertTrue(outputs.matching_logits.shape, expected_matching_logits_shape)
