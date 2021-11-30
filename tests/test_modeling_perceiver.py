# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch Perceiver model. """

import copy
import inspect
import os
import random
import tempfile
import unittest
from typing import Dict, List, Tuple

import numpy as np

from transformers import PerceiverConfig
from transformers.file_utils import WEIGHTS_NAME, is_torch_available, is_vision_available
from transformers.models.auto import get_values
from transformers.testing_utils import require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch
    import torch.nn as nn

    from transformers import (
        MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
        MODEL_FOR_MASKED_LM_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        MODEL_MAPPING,
        PerceiverForImageClassification,
        PerceiverForImageClassificationConvProcessing,
        PerceiverForImageClassificationFourier,
        PerceiverForMaskedLM,
        PerceiverForMultimodalAutoencoding,
        PerceiverForOpticalFlow,
        PerceiverModel,
        PerceiverTokenizer,
    )
    from transformers.models.perceiver.modeling_perceiver import PERCEIVER_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import PerceiverFeatureExtractor


class PerceiverModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        num_channels=3,
        image_size=32,
        train_size=[20, 20],
        num_frames=5,
        audio_samples_per_frame=200,
        samples_per_patch=20,
        nchunks=20,
        num_latents=10,
        d_latents=20,
        num_blocks=1,
        num_self_attends_per_block=2,
        num_self_attention_heads=1,
        num_cross_attention_heads=1,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        max_position_embeddings=7,
        num_labels=3,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_channels = num_channels
        self.image_size = image_size
        self.train_size = train_size
        self.num_frames = num_frames
        self.audio_samples_per_frame = audio_samples_per_frame
        self.samples_per_patch = samples_per_patch
        self.nchunks = nchunks
        self.num_latents = num_latents
        self.d_latents = d_latents
        self.num_blocks = num_blocks
        self.num_self_attends_per_block = num_self_attends_per_block
        self.num_self_attention_heads = num_self_attention_heads
        self.num_cross_attention_heads = num_cross_attention_heads
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_act = hidden_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.scope = scope
        # set subsampling for multimodal model (take first chunk)
        image_chunk_size = np.prod((self.num_frames, self.image_size, self.image_size)) // self.nchunks
        audio_chunk_size = self.num_frames * self.audio_samples_per_frame // self.samples_per_patch // self.nchunks
        self.subsampling = {
            "image": torch.arange(0, image_chunk_size),
            "audio": torch.arange(0, audio_chunk_size),
            "label": None,
        }

    def prepare_config_and_inputs(self, model_class=None):
        config = self.get_config()

        input_mask = None
        sequence_labels = None
        token_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.num_labels)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        if model_class is None or model_class.__name__ == "PerceiverModel":
            inputs = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            return config, inputs, input_mask, sequence_labels, token_labels
        elif model_class.__name__ == "PerceiverForMaskedLM":
            inputs = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            # input mask is only relevant for text inputs
            if self.use_input_mask:
                input_mask = random_attention_mask([self.batch_size, self.seq_length])
        elif model_class.__name__ == "PerceiverForImageClassification":
            config.d_model = 512
            inputs = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        elif model_class.__name__ == "PerceiverForImageClassificationFourier":
            config.d_model = 261
            inputs = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        elif model_class.__name__ == "PerceiverForImageClassificationConvProcessing":
            config.d_model = 322
            inputs = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        elif model_class.__name__ == "PerceiverForOpticalFlow":
            config.d_model = 322
            inputs = floats_tensor([self.batch_size, 2, 27, self.train_size[0], self.train_size[1]])
        elif model_class.__name__ == "PerceiverForMultimodalAutoencoding":
            config.d_model = 409
            images = torch.randn(
                (self.batch_size, self.num_frames, self.num_channels, self.image_size, self.image_size)
            )
            audio = torch.randn((self.batch_size, self.num_frames * self.audio_samples_per_frame, 1))
            inputs = dict(image=images, audio=audio, label=torch.zeros((self.batch_size, self.num_labels)))
        else:
            raise ValueError(f"Model class {model_class} not supported")

        return config, inputs, input_mask, sequence_labels, token_labels

    def prepare_config_and_inputs_masked_lm(self):
        inputs = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_labels = None
        if self.use_labels:
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        config = self.get_config()

        return config, inputs, input_mask, token_labels

    def prepare_config_and_inputs_image_classification(self):
        inputs = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        input_mask = None

        image_labels = None
        if self.use_labels:
            image_labels = ids_tensor([self.batch_size], self.num_labels)

        config = self.get_config()

        return config, inputs, input_mask, image_labels

    def get_config(self):
        return PerceiverConfig(
            num_latents=self.num_latents,
            d_latents=self.d_latents,
            num_blocks=self.num_blocks,
            num_self_attends_per_block=self.num_self_attends_per_block,
            num_self_attention_heads=self.num_self_attention_heads,
            num_cross_attention_heads=self.num_cross_attention_heads,
            vocab_size=self.vocab_size,
            hidden_act=self.hidden_act,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            max_position_embeddings=self.max_position_embeddings,
            image_size=self.image_size,
            train_size=self.train_size,
            num_frames=self.num_frames,
            audio_samples_per_frame=self.audio_samples_per_frame,
            samples_per_patch=self.samples_per_patch,
            num_labels=self.num_labels,
        )

    def create_and_check_for_masked_lm(
        self,
        config,
        inputs,
        input_mask,
        token_labels,
    ):
        model = PerceiverForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(inputs, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_image_classification(self, config, inputs, input_mask, image_labels):
        # set d_model and num_labels
        config.d_model = 512
        config.num_labels = self.num_labels
        model = PerceiverForImageClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(inputs, attention_mask=input_mask, labels=image_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, inputs, input_mask, sequence_labels, token_labels = config_and_inputs
        inputs_dict = {"inputs": inputs, "attention_mask": input_mask}
        return config, inputs_dict

    def prepare_config_and_inputs_for_model_class(self, model_class):
        config_and_inputs = self.prepare_config_and_inputs(model_class)
        config, inputs, input_mask, sequence_labels, token_labels = config_and_inputs
        inputs_dict = {"inputs": inputs, "attention_mask": input_mask}

        return config, inputs_dict


@require_torch
class PerceiverModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            PerceiverModel,
            PerceiverForMaskedLM,
            PerceiverForImageClassification,
            PerceiverForImageClassificationConvProcessing,
            PerceiverForImageClassificationFourier,
            PerceiverForOpticalFlow,
            PerceiverForMultimodalAutoencoding,
        )
        if is_torch_available()
        else ()
    )
    test_pruning = False
    test_head_masking = False
    test_torchscript = False

    maxDiff = None

    def setUp(self):
        self.model_tester = PerceiverModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PerceiverConfig, hidden_size=37)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)

        if model_class.__name__ == "PerceiverForMultimodalAutoencoding":
            inputs_dict["subsampled_output_points"] = self.model_tester.subsampling

        if return_labels:
            if model_class in [
                *get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING),
                *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING),
            ]:
                inputs_dict["labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
            elif model_class in [
                *get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING),
                *get_values(MODEL_FOR_MASKED_LM_MAPPING),
            ]:
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )
        return inputs_dict

    def test_config(self):
        # we don't test common_properties and arguments_init as these don't apply for Perceiver
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_masked_lm()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_image_classification()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    def test_model_common_attributes(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
            model = model_class(config)
            # we overwrite this, as the embeddings of Perceiver are an instance of nn.Parameter
            # and Perceiver doesn't support get_output_embeddings
            self.assertIsInstance(model.get_input_embeddings(), (nn.Parameter))

    def test_training(self):
        if not self.model_tester.is_training:
            return

        for model_class in self.all_model_classes:
            if model_class in [*get_values(MODEL_MAPPING), PerceiverForOpticalFlow, PerceiverForMultimodalAutoencoding]:
                continue

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
            config.return_dict = True

            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    def test_forward_signature(self):
        for model_class in self.all_model_classes:
            config, _ = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)

            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["inputs"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_determinism(self):
        for model_class in self.all_model_classes:
            if model_class.__name__ == "PerceiverModel":
                continue

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)

            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                inputs_dict = self._prepare_for_class(inputs_dict, model_class)
                first = model(**inputs_dict)[0]
                second = model(**inputs_dict)[0]

            if model_class.__name__ == "PerceiverForMultimodalAutoencoding":
                # model outputs a dictionary with logits per modality, let's verify each modality
                for modality in first.keys():
                    out_1 = first[modality].cpu().numpy()
                    out_2 = second[modality].cpu().numpy()
                    out_1 = out_1[~np.isnan(out_1)]
                    out_2 = out_2[~np.isnan(out_2)]
                    max_diff = np.amax(np.abs(out_1 - out_2))
                    self.assertLessEqual(max_diff, 1e-5)
            else:
                out_1 = first.cpu().numpy()
                out_2 = second.cpu().numpy()
                out_1 = out_1[~np.isnan(out_1)]
                out_2 = out_2[~np.isnan(out_2)]
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    def test_attention_outputs(self):
        seq_len = getattr(self.model_tester, "num_latents", None)

        for model_class in self.all_model_classes:
            if model_class.__name__ == "PerceiverModel":
                continue

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
            config.return_dict = True

            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self_attentions = outputs.attentions
            cross_attentions = outputs.cross_attentions

            # check expected number of attentions depending on model class
            if model.__class__.__name__ in ["PerceiverForMaskedLM", "PerceiverForImageClassification"]:
                expected_num_self_attentions = (
                    self.model_tester.num_blocks * self.model_tester.num_self_attends_per_block
                )
                # we expect to have 2 cross-attentions, namely one in the PerceiverEncoder, and one in PerceiverBasicDecoder
                expected_num_cross_attentions = 2
            else:
                # todo
                pass
            self.assertEqual(len(self_attentions), expected_num_self_attentions)
            self.assertEqual(len(cross_attentions), expected_num_cross_attentions)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self_attentions = outputs.attentions
            cross_attentions = outputs.cross_attentions
            self.assertEqual(len(self_attentions), expected_num_self_attentions)
            self.assertEqual(len(cross_attentions), expected_num_cross_attentions)

            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_self_attention_heads, seq_len, seq_len],
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

            self.assertEqual(len(self_attentions), expected_num_self_attentions)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_self_attention_heads, seq_len, seq_len],
            )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = self.model_tester.num_blocks * self.model_tester.num_self_attends_per_block + 1
            self.assertEqual(len(hidden_states), expected_num_layers)

            seq_length = self.model_tester.num_latents

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.d_latents],
            )

        for model_class in self.all_model_classes:
            if model_class.__name__ == "PerceiverModel":
                continue

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)

            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_model_outputs_equivalence(self):
        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, (List, Tuple)):
                        for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif isinstance(tuple_object, Dict):
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
                            msg=f"Tuple and dict output are not equal. Difference: {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`: {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}.",
                        )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            if model_class.__name__ == "PerceiverModel":
                continue

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)

            model = model_class(config)
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

            if model_class.__name__ not in ["PerceiverForOpticalFlow", "PerceiverForMultimodalAutoencoding"]:
                # optical flow + multimodal models don't support training for now
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)

            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

            if model_class.__name__ not in ["PerceiverForOpticalFlow", "PerceiverForMultimodalAutoencoding"]:
                # optical flow + multimodal models don't support training for now
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            if model_class.__name__ not in ["PerceiverForOpticalFlow", "PerceiverForMultimodalAutoencoding"]:
                # optical flow + multimodal models don't support training for now
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

            if model_class.__name__ not in ["PerceiverForOpticalFlow", "PerceiverForMultimodalAutoencoding"]:
                # optical flow + multimodal models don't support training for now
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(
                    model, tuple_inputs, dict_inputs, {"output_hidden_states": True, "output_attentions": True}
                )

    def test_retain_grad_hidden_states_attentions(self):
        # no need to test all models as different heads yield the same functionality
        model_class = PerceiverForMaskedLM
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
        config.output_hidden_states = True
        config.output_attentions = True

        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs[0]

        # Encoder-only model
        hidden_states = outputs.hidden_states[0]
        attentions = outputs.attentions[0]

        hidden_states.retain_grad()
        attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(hidden_states.grad)
        self.assertIsNotNone(attentions.grad)

    def test_feed_forward_chunking(self):
        for model_class in self.all_model_classes:
            if model_class.__name__ == "PerceiverModel":
                continue
            original_config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
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
            if model_class.__name__ == "PerceiverForMultimodalAutoencoding":
                # model outputs a dictionary with logits for each modality
                for modality in hidden_states_no_chunk.keys():
                    self.assertTrue(
                        torch.allclose(hidden_states_no_chunk[modality], hidden_states_with_chunk[modality], atol=1e-3)
                    )
            else:
                self.assertTrue(torch.allclose(hidden_states_no_chunk, hidden_states_with_chunk, atol=1e-3))

    def test_save_load(self):
        for model_class in self.all_model_classes:
            if model_class.__name__ == "PerceiverModel":
                continue

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)

            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if model_class.__name__ == "PerceiverForMultimodalAutoencoding":
                for modality in outputs[0].keys():
                    out_2 = outputs[0][modality].cpu().numpy()
                    out_2[np.isnan(out_2)] = 0

                    with tempfile.TemporaryDirectory() as tmpdirname:
                        model.save_pretrained(tmpdirname)
                        model = model_class.from_pretrained(tmpdirname)
                        model.to(torch_device)
                        with torch.no_grad():
                            after_outputs = model(**self._prepare_for_class(inputs_dict, model_class))

                        # Make sure we don't have nans
                        out_1 = after_outputs[0][modality].cpu().numpy()
                        out_1[np.isnan(out_1)] = 0
                        max_diff = np.amax(np.abs(out_1 - out_2))
                        self.assertLessEqual(max_diff, 1e-5)

            else:
                out_2 = outputs[0].cpu().numpy()
                out_2[np.isnan(out_2)] = 0

                with tempfile.TemporaryDirectory() as tmpdirname:
                    model.save_pretrained(tmpdirname)
                    model = model_class.from_pretrained(tmpdirname)
                    model.to(torch_device)
                    with torch.no_grad():
                        after_outputs = model(**self._prepare_for_class(inputs_dict, model_class))

                    # Make sure we don't have nans
                    out_1 = after_outputs[0].cpu().numpy()
                    out_1[np.isnan(out_1)] = 0
                    max_diff = np.amax(np.abs(out_1 - out_2))
                    self.assertLessEqual(max_diff, 1e-5)

    def test_save_load_keys_to_ignore_on_save(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            _keys_to_ignore_on_save = getattr(model, "_keys_to_ignore_on_save", None)
            if _keys_to_ignore_on_save is None:
                continue

            # check the keys are in the original state_dict
            for k in _keys_to_ignore_on_save:
                self.assertIn(k, model.state_dict().keys(), "\n".join(model.state_dict().keys()))

            # check that certain keys didn't get saved with the model
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                output_model_file = os.path.join(tmpdirname, WEIGHTS_NAME)
                state_dict_saved = torch.load(output_model_file)
                for k in _keys_to_ignore_on_save:
                    self.assertNotIn(k, state_dict_saved.keys(), "\n".join(state_dict_saved.keys()))

                # Test we can load the state dict in the model, necessary for the checkpointing API in Trainer.
                load_result = model.load_state_dict(state_dict_saved, strict=False)
                self.assertTrue(
                    len(load_result.missing_keys) == 0
                    or set(load_result.missing_keys) == set(model._keys_to_ignore_on_save)
                )
                self.assertTrue(len(load_result.unexpected_keys) == 0)

    def test_save_load_fast_init_to_base(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        base_class = MODEL_MAPPING[config.__class__]

        if isinstance(base_class, tuple):
            base_class = base_class[0]

        for model_class in self.all_model_classes:

            if model_class == base_class:
                continue

            # make a copy of model class to not break future tests
            # from https://stackoverflow.com/questions/9541025/how-to-copy-a-python-class
            class CopyClass(base_class):
                pass

            base_class_copy = CopyClass

            # make sure that all keys are expected for test
            base_class_copy._keys_to_ignore_on_load_missing = []

            # make init deterministic, but make sure that
            # non-initialized weights throw errors nevertheless
            base_class_copy._init_weights = self._mock_init_weights

            model = model_class(config)
            state_dict = model.state_dict()

            # this will often delete a single weight of a multi-weight module
            # to test an edge case
            random_key_to_del = random.choice(list(state_dict.keys()))
            del state_dict[random_key_to_del]

            # check that certain keys didn't get saved with the model
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.config.save_pretrained(tmpdirname)
                torch.save(state_dict, os.path.join(tmpdirname, "pytorch_model.bin"))

                model_fast_init = base_class_copy.from_pretrained(tmpdirname)
                model_slow_init = base_class_copy.from_pretrained(tmpdirname, _fast_init=False)

                for key in model_fast_init.state_dict().keys():
                    max_diff = (model_slow_init.state_dict()[key] - model_fast_init.state_dict()[key]).sum().item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    def test_correct_missing_keys(self):
        if not self.test_missing_keys:
            return
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            # most Perceiver models don't have a typical head like is the case with BERT
            if model_class in [
                PerceiverForOpticalFlow,
                PerceiverForMultimodalAutoencoding,
                *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING),
            ]:
                continue

            model = model_class(config)
            base_model_prefix = model.base_model_prefix

            if hasattr(model, base_model_prefix):
                with tempfile.TemporaryDirectory() as temp_dir_name:
                    model.base_model.save_pretrained(temp_dir_name)
                    model, loading_info = model_class.from_pretrained(temp_dir_name, output_loading_info=True)
                    with self.subTest(msg=f"Missing keys for {model.__class__.__name__}"):
                        self.assertGreater(len(loading_info["missing_keys"]), 0)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    @unittest.skip(reason="Perceiver doesn't support resize_token_embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Perceiver doesn't support resize_token_embeddings")
    def test_resize_embeddings_untied(self):
        pass

    @unittest.skip(reason="Perceiver doesn't support inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in PERCEIVER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = PerceiverModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
class PerceiverModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):

        # TODO replace by deepmind/language-perceiver
        tokenizer = PerceiverTokenizer.from_pretrained("nielsr/language-perceiver")
        model = PerceiverForMaskedLM.from_pretrained("nielsr/language-perceiver")
        model.to(torch_device)

        # prepare inputs
        text = "This is an incomplete sentence where some words are missing."
        encoding = tokenizer(text, padding="max_length", return_tensors="pt")

        # mask " missing.".
        encoding.input_ids[0, 51:60] = tokenizer.mask_token_id
        inputs, input_mask = encoding.input_ids.to(torch_device), encoding.attention_mask.to(torch_device)

        # forward pass
        outputs = model(inputs=inputs, attention_mask=input_mask)
        logits = outputs.logits

        # verify logits
        expected_shape = torch.Size((1, tokenizer.model_max_length, tokenizer.vocab_size))
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-11.8336, -11.6850, -11.8483], [-12.8149, -12.5863, -12.7904], [-12.8440, -12.6410, -12.8646]]
        )

        self.assertTrue(torch.allclose(logits[0, :3, :3], expected_slice, atol=1e-4))

        expected_greedy_predictions = [38, 115, 111, 121, 121, 111, 116, 109, 52]
        masked_tokens_predictions = logits[0, 51:60].argmax(dim=-1).tolist()
        self.assertListEqual(expected_greedy_predictions, masked_tokens_predictions)

    @slow
    def test_inference_image_classification(self):

        # TODO replace by deepmind/vision-perceiver
        feature_extractor = PerceiverFeatureExtractor()
        model = PerceiverForImageClassification.from_pretrained("nielsr/vision-perceiver")
        model.to(torch_device)

        # prepare inputs
        image = prepare_img()
        inputs = feature_extractor(image, return_tensors="pt").pixel_values.to(torch_device)
        input_mask = None

        # forward pass
        outputs = model(inputs=inputs, attention_mask=input_mask)
        logits = outputs.logits

        # verify logits
        expected_shape = torch.Size((1, model.config.num_labels))
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = torch.tensor([-1.1653, -0.1993, -0.7521])

        self.assertTrue(torch.allclose(logits[0, :3], expected_slice, atol=1e-4))
