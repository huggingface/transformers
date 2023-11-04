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
""" Testing suite for the PyTorch Beit3 model. """

import inspect
import unittest
from typing import Dict, List, Tuple

import numpy as np

from transformers import Beit3Config, Beit3Processor, BeitImageProcessor, XLMRobertaTokenizer
from transformers.models.auto import get_values
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_BACKBONE_MAPPING_NAMES,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        Beit3ForCaptioning,
        Beit3ForImageClassification,
        Beit3ForImageTextRetrieval,
        Beit3ForQuestionAnswering,
        Beit3ForVisualReasoning,
        Beit3Model,
    )

if is_vision_available():
    from PIL import Image


class Beit3ModelTester:
    def __init__(
        self,
        hidden_size=37,
        num_attention_heads=1,
        intermediate_size=2,
        num_hidden_layers=1,
        normalize_before=True,
        activation_fn="gelu",
        dropout=0.0,
        drop_path_rate=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        deepnorm=False,
        subln=True,
        bert_init=False,
        multiway=True,
        max_source_positions=16,
        layernorm_eps=1e-5,
        vocab_size=50,
        image_size=16,
        patch_size=2,
        num_channels=3,
        num_labels=2,
        batch_size=1,
        seq_length=7,
        use_labels=True,
        is_training=True,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.normalize_before = normalize_before
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.deepnorm = deepnorm
        self.subln = subln
        self.bert_init = bert_init
        self.multiway = multiway
        self.max_source_positions = max_source_positions
        self.layernorm_eps = layernorm_eps
        # Text
        self.vocab_size = vocab_size
        # Vision
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        self.num_labels = num_labels
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.use_labels = use_labels
        self.is_training = is_training

        self.encoder_seq_length = ((self.image_size // self.patch_size) ** 2) + self.seq_length + 1
        self.encoder_seq_length_image_only = ((self.image_size // self.patch_size) ** 2) + 1
        self.key_length = ((self.image_size // self.patch_size) ** 2) + self.seq_length + 1

    def get_config(self):
        return Beit3Config(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            normalize_before=self.normalize_before,
            activation_fn=self.activation_fn,
            dropout=self.dropout,
            drop_path_rate=self.drop_path_rate,
            attention_dropout=self.attention_dropout,
            activation_dropout=self.activation_dropout,
            deepnorm=self.deepnorm,
            sub_layernorm=self.subln,
            bert_init=self.bert_init,
            multiway=self.multiway,
            max_source_positions=self.max_source_positions,
            layer_norm_eps=self.layernorm_eps,
            vocab_size=self.vocab_size,
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            num_labels=self.num_labels,
        )

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        attention_mask = torch.zeros_like(input_ids)
        return self.get_config(), {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
        }

    def prepare_config_and_inputs_for_visual_reasoning(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        pixel_values = floats_tensor([self.batch_size, 2, self.num_channels, self.image_size, self.image_size])
        text_padding_mask = torch.zeros((self.batch_size, self.seq_length))
        config = self.get_config()
        model_input = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "text_padding_mask": text_padding_mask,
        }
        labels = ids_tensor([self.batch_size], self.num_labels)
        if self.use_labels:
            model_input["labels"] = labels

        return config, model_input

    def prepare_config_and_inputs_for_image_classification(self):
        pixel_value = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.seq_length
        labels = torch.zeros(self.batch_size, dtype=torch.long, device=torch_device)
        model_input = {"pixel_values": pixel_value}
        if self.use_labels:
            model_input["labels"] = labels
        return config, model_input

    def prepare_config_and_inputs_for_captioning(self):
        language_masked_pos = torch.zeros((self.batch_size, self.seq_length))
        to_fill = list(range(0, self.seq_length, 3))
        language_masked_pos[:, to_fill] = 1
        config = self.get_config()
        label = torch.tensor([20, 5, 2])
        return config, {"language_masked_pos": language_masked_pos, "labels": label}

    def prepare_config_and_inputs_for_basemodel(self):
        language_masked_pos = torch.zeros((self.batch_size, self.seq_length))
        to_fill = list(range(0, self.seq_length, 3))
        language_masked_pos[:, to_fill] = 1
        config = self.get_config()
        torch.tensor([20, 5, 2])
        return config, {}

    def prepare_config_and_inputs_for_visual_question_answering(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        text_padding_mask = torch.zeros((self.batch_size, self.seq_length))
        return self.get_config(), {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": text_padding_mask,
        }

    def prepare_config_and_inputs_for_text_retrieval(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        text_padding_mask = torch.zeros((self.batch_size, self.seq_length))
        return self.get_config(), {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": text_padding_mask,
        }

    def create_and_check_model(self, config, input_dict):
        model = Beit3Model(config=config)
        model.to(torch_device)
        model.eval()
        model(**input_dict)

    def create_and_check_for_visual_reasoning(self, config, input_dict):
        model = Beit3ForVisualReasoning(config=config)
        model.to(torch_device)
        model.eval()
        result = model(**input_dict)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))


@require_torch
class Beit3ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            Beit3Model,
            # Beit3ForVisualReasoning,
            Beit3ForImageTextRetrieval,
            Beit3ForQuestionAnswering,
            Beit3ForImageClassification,
            Beit3ForCaptioning,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": Beit3Model,
            "image-classification": Beit3ForImageClassification,
            "visual-question-answering": Beit3ForQuestionAnswering,
        }
        if is_torch_available()
        else {}
    )
    test_torchscript = False
    test_pruning = False
    test_inputs_embeds = False
    test_head_masking = False

    # special cases for Beit3ForImageClassification, Beit3ForQuestionAnswering, Beit3ForVisualReasoning, Beit3ForCaptioning
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if model_class.__name__ == "Beit3ForImageClassification":
            del inputs_dict["input_ids"]
            del inputs_dict["attention_mask"]

        if return_labels:
            if model_class.__name__ == "Beit3ForQuestionAnswering":
                inputs_dict["labels"] = torch.zeros(
                    self.model_tester.batch_size, self.model_tester.num_labels, device=torch_device
                )
            elif model_class.__name__ == "Beit3ForVisualReasoning":
                inputs_dict["labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
            elif model_class.__name__ == "Beit3ForCaptioning":
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )

        return inputs_dict

    def setUp(self):
        self.model_tester = Beit3ModelTester()
        self.config_tester = ConfigTester(self, config_class=Beit3Config, hidden_size=37)

    def test_training(self):
        if not self.model_tester.is_training:
            return

        for model_class in self.all_model_classes:
            if model_class.__name__ in ["Beit3ForImageTextRetrieval", "Beit3ForCaptioning"]:
                continue

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            if model_class.__name__ in [
                *get_values(MODEL_MAPPING_NAMES),
                *get_values(MODEL_FOR_BACKBONE_MAPPING_NAMES),
            ]:
                continue

            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]

            if model_class.__name__ == "Beit3ForImageClassification":
                # signature.parameters is an OrderedDict => so arg_names order is deterministic

                expected_arg_names = ["pixel_values"]
                self.assertListEqual(arg_names[:1], expected_arg_names)
            else:
                expected_arg_names = ["input_ids"]
                self.assertListEqual(arg_names[:1], expected_arg_names)

    # override as the `logit_scale` parameter initilization is different for Blip
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # check if `logit_scale` is initilized as per the original implementation
                    if name == "logit_scale":
                        self.assertAlmostEqual(
                            param.data.item(),
                            np.log(1 / 0.07),
                            delta=1e-3,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

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
            if model_class.__name__ not in ["Beit3ForImageTextRetrieval", "Beit3ForCaptioning"]:
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
                    if model_class.__name__ == "Beit3ForImageClassification":
                        self.assertListEqual(
                            list(attentions[0].shape[-3:]),
                            [
                                self.model_tester.num_attention_heads,
                                self.model_tester.encoder_seq_length_image_only,
                                self.model_tester.encoder_seq_length_image_only,
                            ],
                        )
                    else:
                        self.assertListEqual(
                            list(attentions[0].shape[-3:]),
                            [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
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
                    if model_class.__name__ == "Beit3ForImageClassification":
                        self.assertListEqual(
                            list(attentions[0].shape[-3:]),
                            [
                                self.model_tester.num_attention_heads,
                                self.model_tester.encoder_seq_length_image_only,
                                self.model_tester.encoder_seq_length_image_only,
                            ],
                        )
                    else:
                        self.assertListEqual(
                            list(self_attentions[0].shape[-3:]),
                            [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                        )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
                if hasattr(self.model_tester, "chunk_length") and self.model_tester.chunk_length > 1:
                    seq_length = seq_length * self.model_tester.chunk_length
            else:
                seq_length = self.model_tester.seq_length

            if model_class.__name__ == "Beit3ForImageClassification":
                seq_length = self.model_tester.encoder_seq_length_image_only

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)
                seq_len = getattr(self.model_tester, "seq_length", None)
                decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)

                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [decoder_seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class.__name__ != "Beit3ForImageTextRetrieval":
                inputs_dict["output_hidden_states"] = True
                check_hidden_states_output(inputs_dict, config, model_class)

                # check that output_hidden_states also work using config
                del inputs_dict["output_hidden_states"]
                config.output_hidden_states = True

                check_hidden_states_output(inputs_dict, config, model_class)

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

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

            if model_class.__name__ != "Beit3ForCaptioning":
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            if model_class.__name__ != "Beit3ForCaptioning":
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            if self.has_attentions:
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

                if model_class.__name__ != "Beit3ForCaptioning":
                    tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                    dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                    check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

                if model_class.__name__ != "Beit3ForCaptioning":
                    tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                    dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                    check_equivalence(
                        model, tuple_inputs, dict_inputs, {"output_hidden_states": True, "output_attentions": True}
                    )


@require_torch
@require_vision
class BeitModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image(self):
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        return image

    @slow
    def test_inference_beit3_image_classification(self):
        model = Beit3ForImageClassification.from_pretrained("Raghavan/beit3_base_patch16_224_in1k").to(torch_device)

        processor = Beit3Processor.from_pretrained("Raghavan/beit3_base_patch16_224_in1k")
        image = self.default_image
        text = "This is a photo of a cat"
        inputs = processor(text=text, images=image, return_tensors="pt")

        # forward pass
        output = model(pixel_values=torch.tensor(inputs["pixel_values"]))
        self.assertEqual(output.logits.shape, torch.Size([1, 1000]))
        expected_slice = torch.tensor([[-0.260473, -0.420061, -0.492118]])
        assert torch.allclose(output.logits.detach()[:, :3], expected_slice)

    @slow
    def test_inference_beit3_vqa(self):
        model = Beit3ForQuestionAnswering.from_pretrained("Raghavan/beit3_base_patch16_480_vqa").to(torch_device)

        processor = Beit3Processor.from_pretrained("Raghavan/beit3_base_patch16_480_vqa")
        image = self.default_image
        text = "This is a photo of a cat"
        inputs = processor(text=text, images=image, return_tensors="pt")

        # forward pass
        output = model(**inputs)
        assert output.logits.shape == torch.Size([1, 3129])
        torch.testing.assert_allclose(
            output.logits.detach()[:, :3], torch.tensor([[-10.862484, -12.388088, -7.6599636]])
        )

    @slow
    def test_inference_beit3_visual_reasoning(self):
        model = Beit3ForVisualReasoning.from_pretrained("Raghavan/beit3_base_patch16_224_nlvr2").to(torch_device)

        processor = Beit3Processor.from_pretrained("Raghavan/beit3_base_patch16_224_nlvr2")
        image = self.default_image
        text = "This is a photo of a cat"
        inputs = processor(text=text, images=image, return_tensors="pt")

        pixel_values = torch.cat(
            (torch.tensor(inputs["pixel_values"]).unsqueeze(1), torch.tensor(inputs["pixel_values"]).unsqueeze(1)),
            dim=1,
        )
        # forward pass
        output = model(
            input_ids=torch.tensor(inputs["input_ids"]),
            pixel_values=pixel_values,
            attention_mask=torch.ones(inputs["input_ids"].shape),
        )
        assert output.logits.shape == torch.Size([1, 2])
        torch.testing.assert_allclose(output.logits.detach(), torch.tensor([[6.593818, -6.582055]]))

    @slow
    def test_inference_beit3_for_image_captioning(self):
        model = Beit3ForCaptioning.from_pretrained("Raghavan/beit3_base_patch16_480_coco_captioning").to(torch_device)

        tokenizer = XLMRobertaTokenizer.from_pretrained("/Users/eaxxkra/Downloads/beit3.spm")
        image_processor = BeitImageProcessor.from_pretrained("Raghavan/beit3_base_patch16_480_coco_captioning")
        processor = Beit3Processor(image_processor, tokenizer)

        # processor = Beit3Processor.from_pretrained("Raghavan/beit3_base_patch16_480_coco_captioning")
        image = self.default_image
        text = "This is photo of a cat"
        inputs = processor(text=text, images=image, return_tensors="pt")

        language_masked_pos = torch.zeros((inputs["input_ids"].shape[0], inputs["input_ids"].shape[1]))
        language_masked_pos[0, 5] = 1
        input_tokens = list(inputs["input_ids"][0])
        input_tokens[5] = 64001
        output = model(
            input_ids=torch.tensor([input_tokens]),
            pixel_values=torch.tensor(inputs["pixel_values"]),
            attention_mask=torch.zeros(language_masked_pos.shape),
            language_masked_pos=language_masked_pos,
        )
        assert output.logits.shape == torch.Size([1, 64010])
        assert torch.allclose(output.logits.detach()[0, :3], torch.tensor([-2.697914, -2.697912, -2.645459]))

    @slow
    def test_inference_beit3_for_image_text_retrieval(self):
        model = Beit3ForImageTextRetrieval.from_pretrained("Raghavan/beit3_base_patch16_384_coco_retrieval").to(
            torch_device
        )

        processor = Beit3Processor.from_pretrained("Raghavan/beit3_base_patch16_384_coco_retrieval")
        image = self.default_image

        inputs = processor(
            text=["This is photo of a cat", "This is photo of a dog"], images=[image, image], return_tensors="pt"
        )
        outputs = model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            return_loss=True,
        )

        self.assertEqual(round(float(outputs.loss.detach().numpy()), 4), 1.8435)
