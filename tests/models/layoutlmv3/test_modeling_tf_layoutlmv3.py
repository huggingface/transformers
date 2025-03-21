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
"""Testing suite for the TensorFlow LayoutLMv3 model."""

from __future__ import annotations

import copy
import inspect
import unittest

import numpy as np

from transformers import is_tf_available, is_vision_available
from transformers.models.auto import get_values
from transformers.testing_utils import require_tf, slow
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers import (
        TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
        TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        LayoutLMv3Config,
        TFLayoutLMv3ForQuestionAnswering,
        TFLayoutLMv3ForSequenceClassification,
        TFLayoutLMv3ForTokenClassification,
        TFLayoutLMv3Model,
    )

if is_vision_available():
    from PIL import Image

    from transformers import LayoutLMv3ImageProcessor


class TFLayoutLMv3ModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_channels=3,
        image_size=4,
        patch_size=2,
        text_seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=36,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        coordinate_size=6,
        shape_size=6,
        num_labels=3,
        num_choices=4,
        scope=None,
        range_bbox=1000,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.range_bbox = range_bbox

        # LayoutLMv3's sequence length equals the number of text tokens + number of patches + 1 (we add 1 for the CLS token)
        self.text_seq_length = text_seq_length
        self.image_seq_length = (image_size // patch_size) ** 2 + 1
        self.seq_length = self.text_seq_length + self.image_seq_length

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.text_seq_length], self.vocab_size)

        bbox = ids_tensor([self.batch_size, self.text_seq_length, 4], self.range_bbox)
        bbox = bbox.numpy()
        # Ensure that bbox is legal
        for i in range(bbox.shape[0]):
            for j in range(bbox.shape[1]):
                if bbox[i, j, 3] < bbox[i, j, 1]:
                    tmp_coordinate = bbox[i, j, 3]
                    bbox[i, j, 3] = bbox[i, j, 1]
                    bbox[i, j, 1] = tmp_coordinate
                if bbox[i, j, 2] < bbox[i, j, 0]:
                    tmp_coordinate = bbox[i, j, 2]
                    bbox[i, j, 2] = bbox[i, j, 0]
                    bbox[i, j, 0] = tmp_coordinate
        bbox = tf.constant(bbox)

        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.text_seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.text_seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.text_seq_length], self.num_labels)

        config = LayoutLMv3Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            coordinate_size=self.coordinate_size,
            shape_size=self.shape_size,
            input_size=self.image_size,
            patch_size=self.patch_size,
        )

        return config, input_ids, bbox, pixel_values, token_type_ids, input_mask, sequence_labels, token_labels

    def create_and_check_model(self, config, input_ids, bbox, pixel_values, token_type_ids, input_mask):
        model = TFLayoutLMv3Model(config=config)

        # text + image
        result = model(input_ids, pixel_values=pixel_values, training=False)
        result = model(
            input_ids,
            bbox=bbox,
            pixel_values=pixel_values,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            training=False,
        )
        result = model(input_ids, bbox=bbox, pixel_values=pixel_values, training=False)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

        # text only
        result = model(input_ids, training=False)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.text_seq_length, self.hidden_size)
        )

        # image only
        result = model({"pixel_values": pixel_values}, training=False)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.image_seq_length, self.hidden_size)
        )

    def create_and_check_for_sequence_classification(
        self, config, input_ids, bbox, pixel_values, token_type_ids, input_mask, sequence_labels
    ):
        config.num_labels = self.num_labels
        model = TFLayoutLMv3ForSequenceClassification(config=config)
        result = model(
            input_ids,
            bbox=bbox,
            pixel_values=pixel_values,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=sequence_labels,
            training=False,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self, config, input_ids, bbox, pixel_values, token_type_ids, input_mask, token_labels
    ):
        config.num_labels = self.num_labels
        model = TFLayoutLMv3ForTokenClassification(config=config)
        result = model(
            input_ids,
            bbox=bbox,
            pixel_values=pixel_values,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=token_labels,
            training=False,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.text_seq_length, self.num_labels))

    def create_and_check_for_question_answering(
        self, config, input_ids, bbox, pixel_values, token_type_ids, input_mask, sequence_labels
    ):
        config.num_labels = 2
        model = TFLayoutLMv3ForQuestionAnswering(config=config)
        result = model(
            input_ids,
            bbox=bbox,
            pixel_values=pixel_values,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
            training=False,
        )
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, bbox, pixel_values, token_type_ids, input_mask, _, _) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "bbox": bbox,
            "pixel_values": pixel_values,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
        }
        return config, inputs_dict


@require_tf
class TFLayoutLMv3ModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            TFLayoutLMv3Model,
            TFLayoutLMv3ForQuestionAnswering,
            TFLayoutLMv3ForSequenceClassification,
            TFLayoutLMv3ForTokenClassification,
        )
        if is_tf_available()
        else ()
    )
    pipeline_model_mapping = (
        {"document-question-answering": TFLayoutLMv3ForQuestionAnswering, "feature-extraction": TFLayoutLMv3Model}
        if is_tf_available()
        else {}
    )

    test_pruning = False
    test_resize_embeddings = False
    test_onnx = False

    # TODO: Fix the failed tests
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return True

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False) -> dict:
        inputs_dict = copy.deepcopy(inputs_dict)

        if model_class in get_values(TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
            inputs_dict = {
                k: tf.tile(tf.expand_dims(v, 1), (1, self.model_tester.num_choices) + (1,) * (v.ndim - 1))
                if isinstance(v, tf.Tensor) and v.ndim > 0
                else v
                for k, v in inputs_dict.items()
            }

        if return_labels:
            if model_class in get_values(TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
                inputs_dict["labels"] = tf.ones(self.model_tester.batch_size, dtype=tf.int32)
            elif model_class in get_values(TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING):
                inputs_dict["start_positions"] = tf.zeros(self.model_tester.batch_size, dtype=tf.int32)
                inputs_dict["end_positions"] = tf.zeros(self.model_tester.batch_size, dtype=tf.int32)
            elif model_class in get_values(TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING):
                inputs_dict["labels"] = tf.zeros(self.model_tester.batch_size, dtype=tf.int32)
            elif model_class in get_values(TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING):
                inputs_dict["labels"] = tf.zeros(
                    (self.model_tester.batch_size, self.model_tester.text_seq_length), dtype=tf.int32
                )

        return inputs_dict

    def setUp(self):
        self.model_tester = TFLayoutLMv3ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LayoutLMv3Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_loss_computation(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            if getattr(model, "hf_compute_loss", None):
                # The number of elements in the loss should be the same as the number of elements in the label
                prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
                added_label = prepared_for_class[
                    sorted(prepared_for_class.keys() - inputs_dict.keys(), reverse=True)[0]
                ]
                expected_loss_size = added_label.shape.as_list()[:1]

                # Test that model correctly compute the loss with kwargs
                prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
                input_ids = prepared_for_class.pop("input_ids")

                loss = model(input_ids, **prepared_for_class)[0]
                self.assertTrue(loss.shape.as_list() == expected_loss_size or loss.shape.as_list() == [1])

                # Test that model correctly compute the loss when we mask some positions
                prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
                input_ids = prepared_for_class.pop("input_ids")
                if "labels" in prepared_for_class:
                    labels = prepared_for_class["labels"].numpy()
                    if len(labels.shape) > 1 and labels.shape[1] != 1:
                        labels[0] = -100
                        prepared_for_class["labels"] = tf.convert_to_tensor(labels)
                        loss = model(input_ids, **prepared_for_class)[0]
                        self.assertTrue(loss.shape.as_list() == expected_loss_size or loss.shape.as_list() == [1])
                        self.assertTrue(not np.any(np.isnan(loss.numpy())))

                # Test that model correctly compute the loss with a dict
                prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
                loss = model(prepared_for_class)[0]
                self.assertTrue(loss.shape.as_list() == expected_loss_size or loss.shape.as_list() == [1])

                # Test that model correctly compute the loss with a tuple
                prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)

                # Get keys that were added with the _prepare_for_class function
                label_keys = prepared_for_class.keys() - inputs_dict.keys()
                signature = inspect.signature(model.call).parameters
                signature_names = list(signature.keys())

                # Create a dictionary holding the location of the tensors in the tuple
                tuple_index_mapping = {0: "input_ids"}
                for label_key in label_keys:
                    label_key_index = signature_names.index(label_key)
                    tuple_index_mapping[label_key_index] = label_key
                sorted_tuple_index_mapping = sorted(tuple_index_mapping.items())
                # Initialize a list with their default values, update the values and convert to a tuple
                list_input = []

                for name in signature_names:
                    if name != "kwargs":
                        list_input.append(signature[name].default)

                for index, value in sorted_tuple_index_mapping:
                    list_input[index] = prepared_for_class[value]

                tuple_input = tuple(list_input)

                # Send to model
                loss = model(tuple_input[:-1])[0]

                self.assertTrue(loss.shape.as_list() == expected_loss_size or loss.shape.as_list() == [1])

    def test_model(self):
        (
            config,
            input_ids,
            bbox,
            pixel_values,
            token_type_ids,
            input_mask,
            _,
            _,
        ) = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, input_ids, bbox, pixel_values, token_type_ids, input_mask)

    def test_model_various_embeddings(self):
        (
            config,
            input_ids,
            bbox,
            pixel_values,
            token_type_ids,
            input_mask,
            _,
            _,
        ) = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config.position_embedding_type = type
            self.model_tester.create_and_check_model(config, input_ids, bbox, pixel_values, token_type_ids, input_mask)

    def test_for_sequence_classification(self):
        (
            config,
            input_ids,
            bbox,
            pixel_values,
            token_type_ids,
            input_mask,
            sequence_labels,
            _,
        ) = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(
            config, input_ids, bbox, pixel_values, token_type_ids, input_mask, sequence_labels
        )

    def test_for_token_classification(self):
        (
            config,
            input_ids,
            bbox,
            pixel_values,
            token_type_ids,
            input_mask,
            _,
            token_labels,
        ) = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(
            config, input_ids, bbox, pixel_values, token_type_ids, input_mask, token_labels
        )

    def test_for_question_answering(self):
        (
            config,
            input_ids,
            bbox,
            pixel_values,
            token_type_ids,
            input_mask,
            sequence_labels,
            _,
        ) = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(
            config, input_ids, bbox, pixel_values, token_type_ids, input_mask, sequence_labels
        )

    @slow
    def test_model_from_pretrained(self):
        model_name = "microsoft/layoutlmv3-base"
        model = TFLayoutLMv3Model.from_pretrained(model_name)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_tf
class TFLayoutLMv3ModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return LayoutLMv3ImageProcessor(apply_ocr=False) if is_vision_available() else None

    @slow
    def test_inference_no_head(self):
        model = TFLayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")

        image_processor = self.default_image_processor
        image = prepare_img()
        pixel_values = image_processor(images=image, return_tensors="tf").pixel_values

        input_ids = tf.constant([[1, 2]])
        bbox = tf.expand_dims(tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]]), axis=0)

        # forward pass
        outputs = model(input_ids=input_ids, bbox=bbox, pixel_values=pixel_values, training=False)

        # verify the logits
        expected_shape = (1, 199, 768)
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)

        expected_slice = tf.constant(
            [[-0.0529, 0.3618, 0.1632], [-0.1587, -0.1667, -0.0400], [-0.1557, -0.1671, -0.0505]]
        )

        self.assertTrue(np.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-4))
