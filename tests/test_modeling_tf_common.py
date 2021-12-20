# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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


import copy
import inspect
import json
import os
import random
import tempfile
import unittest
from importlib import import_module
from typing import List, Tuple

from huggingface_hub import delete_repo, login
from requests.exceptions import HTTPError
from transformers import is_tf_available
from transformers.models.auto import get_values
from transformers.testing_utils import tooslow  # noqa: F401
from transformers.testing_utils import (
    PASS,
    USER,
    CaptureLogger,
    _tf_gpu_memory_limit,
    is_pt_tf_cross_test,
    is_staging_test,
    require_keras2onnx,
    require_tf,
    slow,
)
from transformers.utils import logging


if is_tf_available():
    import numpy as np
    import tensorflow as tf

    from transformers import (
        TF_MODEL_FOR_CAUSAL_LM_MAPPING,
        TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
        TF_MODEL_FOR_MASKED_LM_MAPPING,
        TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
        TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
        TF_MODEL_FOR_PRETRAINING_MAPPING,
        TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        BertConfig,
        TFAutoModel,
        TFAutoModelForSequenceClassification,
        TFBertModel,
        TFSharedEmbeddings,
        tf_top_k_top_p_filtering,
    )
    from transformers.generation_tf_utils import (
        TFBeamSampleDecoderOnlyOutput,
        TFBeamSampleEncoderDecoderOutput,
        TFBeamSearchDecoderOnlyOutput,
        TFBeamSearchEncoderDecoderOutput,
        TFGreedySearchDecoderOnlyOutput,
        TFGreedySearchEncoderDecoderOutput,
        TFSampleDecoderOnlyOutput,
        TFSampleEncoderDecoderOutput,
    )

    if _tf_gpu_memory_limit is not None:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            # Restrict TensorFlow to only allocate x GB of memory on the GPUs
            try:
                tf.config.set_logical_device_configuration(
                    gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=_tf_gpu_memory_limit)]
                )
                logical_gpus = tf.config.list_logical_devices("GPU")
                print("Logical GPUs", logical_gpus)
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)


def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if "_range" in key or "_std" in key:
            setattr(configs_no_init, key, 0.0)
    return configs_no_init


@require_tf
class TFModelTesterMixin:

    model_tester = None
    all_model_classes = ()
    all_generative_model_classes = ()
    test_mismatched_shapes = True
    test_resize_embeddings = True
    test_head_masking = True
    is_encoder_decoder = False

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
            elif model_class in [
                *get_values(TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING),
                *get_values(TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING),
            ]:
                inputs_dict["labels"] = tf.zeros(self.model_tester.batch_size, dtype=tf.int32)
            elif model_class in get_values(TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING):
                inputs_dict["next_sentence_label"] = tf.zeros(self.model_tester.batch_size, dtype=tf.int32)
            elif model_class in [
                *get_values(TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING),
                *get_values(TF_MODEL_FOR_CAUSAL_LM_MAPPING),
                *get_values(TF_MODEL_FOR_MASKED_LM_MAPPING),
                *get_values(TF_MODEL_FOR_PRETRAINING_MAPPING),
                *get_values(TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING),
            ]:
                inputs_dict["labels"] = tf.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=tf.int32
                )
        return inputs_dict

    def test_initialization(self):
        pass

    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname, saved_model=False)
                model = model_class.from_pretrained(tmpdirname)
                after_outputs = model(self._prepare_for_class(inputs_dict, model_class))

                self.assert_outputs_same(after_outputs, outputs)

    def test_save_load_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            model_config = model.get_config()
            # make sure that returned config is jsonifiable, which is required by keras
            json.dumps(model_config)
            new_model = model_class.from_config(model.get_config())
            # make sure it also accepts a normal config
            _ = model_class.from_config(model.config)
            _ = new_model(self._prepare_for_class(inputs_dict, model_class))  # Build model
            new_model.set_weights(model.get_weights())
            after_outputs = new_model(self._prepare_for_class(inputs_dict, model_class))

            self.assert_outputs_same(after_outputs, outputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.call)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            if model.config.is_encoder_decoder:
                expected_arg_names = [
                    "input_ids",
                    "attention_mask",
                    "decoder_input_ids",
                    "decoder_attention_mask",
                ]
                expected_arg_names.extend(
                    ["head_mask", "decoder_head_mask"] if "head_mask" and "decoder_head_mask" in arg_names else []
                )
                # Necessary to handle BART with newly added cross_attn_head_mask
                expected_arg_names.extend(
                    ["cross_attn_head_mask", "encoder_outputs"]
                    if "cross_attn_head_mask" in arg_names
                    else ["encoder_outputs"]
                )
                self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

            else:
                expected_arg_names = ["input_ids"]
                self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_onnx_compliancy(self):
        if not self.test_onnx:
            return

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        INTERNAL_OPS = [
            "Assert",
            "AssignVariableOp",
            "EmptyTensorList",
            "ReadVariableOp",
            "ResourceGather",
            "TruncatedNormal",
            "VarHandleOp",
            "VarIsInitializedOp",
        ]
        onnx_ops = []

        with open(os.path.join(".", "utils", "tf_ops", "onnx.json")) as f:
            onnx_opsets = json.load(f)["opsets"]

        for i in range(1, self.onnx_min_opset + 1):
            onnx_ops.extend(onnx_opsets[str(i)])

        for model_class in self.all_model_classes:
            model_op_names = set()

            with tf.Graph().as_default() as g:
                model = model_class(config)
                model(model.dummy_inputs)

                for op in g.get_operations():
                    model_op_names.add(op.node_def.op)

            model_op_names = sorted(model_op_names)
            incompatible_ops = []

            for op in model_op_names:
                if op not in onnx_ops and op not in INTERNAL_OPS:
                    incompatible_ops.append(op)

            self.assertEqual(len(incompatible_ops), 0, incompatible_ops)

    @require_keras2onnx
    @slow
    def test_onnx_runtime_optimize(self):
        if not self.test_onnx:
            return

        import keras2onnx
        import onnxruntime

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model(model.dummy_inputs)

            onnx_model = keras2onnx.convert_keras(model, model.name, target_opset=self.onnx_min_opset)

            onnxruntime.InferenceSession(onnx_model.SerializeToString())

    def test_keras_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        tf_main_layer_classes = set(
            module_member
            for model_class in self.all_model_classes
            for module in (import_module(model_class.__module__),)
            for module_member_name in dir(module)
            if module_member_name.endswith("MainLayer")
            for module_member in (getattr(module, module_member_name),)
            if isinstance(module_member, type)
            and tf.keras.layers.Layer in module_member.__bases__
            and getattr(module_member, "_keras_serializable", False)
        )
        for main_layer_class in tf_main_layer_classes:
            # T5MainLayer needs an embed_tokens parameter when called without the inputs_embeds parameter
            if "T5" in main_layer_class.__name__:
                # Take the same values than in TFT5ModelTester for this shared layer
                shared = TFSharedEmbeddings(99, 32, name="shared")
                config.use_cache = inputs_dict.pop("use_cache", None)
                main_layer = main_layer_class(config, embed_tokens=shared)
            else:
                main_layer = main_layer_class(config)

            symbolic_inputs = {
                name: tf.keras.Input(tensor.shape[1:], dtype=tensor.dtype) for name, tensor in inputs_dict.items()
            }

            model = tf.keras.Model(symbolic_inputs, outputs=main_layer(symbolic_inputs))
            outputs = model(inputs_dict)

            with tempfile.TemporaryDirectory() as tmpdirname:
                filepath = os.path.join(tmpdirname, "keras_model.h5")
                model.save(filepath)
                if "T5" in main_layer_class.__name__:
                    model = tf.keras.models.load_model(
                        filepath,
                        custom_objects={
                            main_layer_class.__name__: main_layer_class,
                            "TFSharedEmbeddings": TFSharedEmbeddings,
                        },
                    )
                else:
                    model = tf.keras.models.load_model(
                        filepath, custom_objects={main_layer_class.__name__: main_layer_class}
                    )
                assert isinstance(model, tf.keras.Model)
                after_outputs = model(inputs_dict)
                self.assert_outputs_same(after_outputs, outputs)

    def assert_outputs_same(self, after_outputs, outputs):
        # Make sure we don't have nans
        if isinstance(after_outputs, tf.Tensor):
            out_1 = after_outputs.numpy()
        elif isinstance(after_outputs, dict):
            out_1 = after_outputs[list(after_outputs.keys())[0]].numpy()
        else:
            out_1 = after_outputs[0].numpy()
        out_2 = outputs[0].numpy()
        self.assertEqual(out_1.shape, out_2.shape)
        out_1 = out_1[~np.isnan(out_1)]
        out_2 = out_2[~np.isnan(out_2)]
        max_diff = np.amax(np.abs(out_1 - out_2))
        self.assertLessEqual(max_diff, 1e-5)

    @is_pt_tf_cross_test
    def test_pt_tf_model_equivalence(self):
        import torch

        import transformers

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            pt_model_class_name = model_class.__name__[2:]  # Skip the "TF" at the beginning
            pt_model_class = getattr(transformers, pt_model_class_name)

            config.output_hidden_states = True

            tf_model = model_class(config)
            pt_model = pt_model_class(config)

            # Check we can load pt model in tf and vice-versa with model => model functions

            tf_model = transformers.load_pytorch_model_in_tf2_model(
                tf_model, pt_model, tf_inputs=self._prepare_for_class(inputs_dict, model_class)
            )
            pt_model = transformers.load_tf2_model_in_pytorch_model(pt_model, tf_model)

            # Check predictions on first output (logits/hidden-states) are close enought given low-level computational differences
            pt_model.eval()
            pt_inputs_dict = {}
            for name, key in self._prepare_for_class(inputs_dict, model_class).items():
                if type(key) == bool:
                    pt_inputs_dict[name] = key
                elif name == "input_values":
                    pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
                elif name == "pixel_values":
                    pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
                else:
                    pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.long)

            # need to rename encoder-decoder "inputs" for PyTorch
            if "inputs" in pt_inputs_dict and self.is_encoder_decoder:
                pt_inputs_dict["input_ids"] = pt_inputs_dict.pop("inputs")

            with torch.no_grad():
                pto = pt_model(**pt_inputs_dict)
            tfo = tf_model(self._prepare_for_class(inputs_dict, model_class), training=False)

            tf_hidden_states = tfo[0].numpy()
            pt_hidden_states = pto[0].numpy()

            tf_nans = np.copy(np.isnan(tf_hidden_states))
            pt_nans = np.copy(np.isnan(pt_hidden_states))

            pt_hidden_states[tf_nans] = 0
            tf_hidden_states[tf_nans] = 0
            pt_hidden_states[pt_nans] = 0
            tf_hidden_states[pt_nans] = 0

            max_diff = np.amax(np.abs(tf_hidden_states - pt_hidden_states))
            self.assertLessEqual(max_diff, 4e-2)

            # Check we can load pt model in tf and vice-versa with checkpoint => model functions
            with tempfile.TemporaryDirectory() as tmpdirname:
                pt_checkpoint_path = os.path.join(tmpdirname, "pt_model.bin")
                torch.save(pt_model.state_dict(), pt_checkpoint_path)
                tf_model = transformers.load_pytorch_checkpoint_in_tf2_model(tf_model, pt_checkpoint_path)

                tf_checkpoint_path = os.path.join(tmpdirname, "tf_model.h5")
                tf_model.save_weights(tf_checkpoint_path)
                pt_model = transformers.load_tf2_checkpoint_in_pytorch_model(pt_model, tf_checkpoint_path)

            # Check predictions on first output (logits/hidden-states) are close enought given low-level computational differences
            pt_model.eval()
            pt_inputs_dict = {}
            for name, key in self._prepare_for_class(inputs_dict, model_class).items():
                if type(key) == bool:
                    key = np.array(key, dtype=bool)
                    pt_inputs_dict[name] = torch.from_numpy(key).to(torch.long)
                elif name == "input_values":
                    pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
                elif name == "pixel_values":
                    pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
                else:
                    pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.long)
            # need to rename encoder-decoder "inputs" for PyTorch
            if "inputs" in pt_inputs_dict and self.is_encoder_decoder:
                pt_inputs_dict["input_ids"] = pt_inputs_dict.pop("inputs")

            with torch.no_grad():
                pto = pt_model(**pt_inputs_dict)
            tfo = tf_model(self._prepare_for_class(inputs_dict, model_class))
            tfo = tfo[0].numpy()
            pto = pto[0].numpy()
            tf_nans = np.copy(np.isnan(tfo))
            pt_nans = np.copy(np.isnan(pto))

            pto[tf_nans] = 0
            tfo[tf_nans] = 0
            pto[pt_nans] = 0
            tfo[pt_nans] = 0

            max_diff = np.amax(np.abs(tfo - pto))
            self.assertLessEqual(max_diff, 4e-2)

    def test_compile_tf_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        max_input = getattr(self.model_tester, "max_position_embeddings", 512)
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")

        for model_class in self.all_model_classes:
            if self.is_encoder_decoder:
                inputs = {
                    "decoder_input_ids": tf.keras.Input(
                        batch_shape=(2, max_input),
                        name="decoder_input_ids",
                        dtype="int32",
                    ),
                    "input_ids": tf.keras.Input(batch_shape=(2, max_input), name="input_ids", dtype="int32"),
                }
            # TODO: A better way to handle vision models
            elif model_class.__name__ in ["TFViTModel", "TFViTForImageClassification"]:
                inputs = tf.keras.Input(
                    batch_shape=(
                        3,
                        self.model_tester.num_channels,
                        self.model_tester.image_size,
                        self.model_tester.image_size,
                    ),
                    name="pixel_values",
                    dtype="float32",
                )
            elif model_class in get_values(TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
                inputs = tf.keras.Input(batch_shape=(4, 2, max_input), name="input_ids", dtype="int32")
            else:
                inputs = tf.keras.Input(batch_shape=(2, max_input), name="input_ids", dtype="int32")

            # Prepare our model
            model = model_class(config)
            model(self._prepare_for_class(inputs_dict, model_class))  # Model must be called before saving.
            # Let's load it from the disk to be sure we can use pretrained weights
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname, saved_model=False)
                model = model_class.from_pretrained(tmpdirname)

            outputs_dict = model(inputs)
            hidden_states = outputs_dict[0]

            # Add a dense layer on top to test integration with other keras modules
            outputs = tf.keras.layers.Dense(2, activation="softmax", name="outputs")(hidden_states)

            # Compile extended model
            extended_model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
            extended_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    def test_keyword_and_dict_args(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            inputs = self._prepare_for_class(inputs_dict, model_class)

            outputs_dict = model(inputs)

            inputs_keywords = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))
            input_ids = inputs_keywords.pop("input_ids", None)
            if input_ids is None:
                input_ids = inputs_keywords.pop("pixel_values", None)
            outputs_keywords = model(input_ids, **inputs_keywords)
            output_dict = outputs_dict[0].numpy()
            output_keywords = outputs_keywords[0].numpy()

            self.assertLess(np.sum(np.abs(output_dict - output_keywords)), 1e-6)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", self.model_tester.seq_length)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", self.model_tester.seq_length)
        decoder_key_length = getattr(self.model_tester, "key_length", decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        def check_decoder_attentions_output(outputs):
            out_len = len(outputs)
            self.assertEqual(min(out_len % 2, out_len % 5), 0)  # differentiation due to newly added cross_attentions
            decoder_attentions = outputs.decoder_attentions
            self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(decoder_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
            )

        def check_encoder_attentions_output(outputs):
            attentions = [
                t.numpy() for t in (outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions)
            ]
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["use_cache"] = False
            config.output_hidden_states = False
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            out_len = len(outputs)
            self.assertEqual(config.output_hidden_states, False)
            check_encoder_attentions_output(outputs)

            if self.is_encoder_decoder:
                model = model_class(config)
                outputs = model(self._prepare_for_class(inputs_dict, model_class))
                self.assertEqual(config.output_hidden_states, False)
                check_decoder_attentions_output(outputs)

            # Check that output attentions can also be changed via the config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(config.output_hidden_states, False)
            check_encoder_attentions_output(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            config.output_hidden_states = True
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))

            self.assertEqual(out_len + (2 if self.is_encoder_decoder else 1), len(outputs))
            self.assertEqual(model.config.output_hidden_states, True)
            check_encoder_attentions_output(outputs)

    def test_headmasking(self):
        if not self.test_head_masking:
            return

        random.Random().seed(42)
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        random.Random().seed()

        inputs_dict["output_attentions"] = True
        config.output_hidden_states = True
        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)

            # Prepare head_mask
            def prepare_layer_head_mask(i, attention_heads, num_hidden_layers):
                if i == 0:
                    return tf.concat(
                        (tf.zeros(1, dtype=tf.float32), tf.ones(attention_heads - 1, dtype=tf.float32)), 0
                    )
                elif i == num_hidden_layers - 1:
                    return tf.concat(
                        (tf.zeros(attention_heads - 1, dtype=tf.float32), tf.ones(1, dtype=tf.float32)), 0
                    )
                else:
                    return tf.ones(attention_heads, dtype=tf.float32)

            head_mask = tf.stack(
                [
                    prepare_layer_head_mask(i, config.num_attention_heads, config.num_hidden_layers)
                    for i in range(config.num_hidden_layers)
                ],
                0,
            )

            inputs = self._prepare_for_class(inputs_dict, model_class).copy()
            inputs["head_mask"] = head_mask
            if model.config.is_encoder_decoder:
                signature = inspect.signature(model.call)
                arg_names = [*signature.parameters.keys()]
                if "decoder_head_mask" in arg_names:  # necessary diferentiation because of T5 model
                    inputs["decoder_head_mask"] = head_mask
                if "cross_attn_head_mask" in arg_names:
                    inputs["cross_attn_head_mask"] = head_mask

            outputs = model(**inputs, return_dict=True)

            def check_attentions_validity(attentions):
                # Remove Nan
                for t in attentions:
                    self.assertLess(
                        (tf.math.reduce_sum(tf.cast(tf.math.is_nan(t), tf.float32))).numpy(), (tf.size(t) / 4).numpy()
                    )  # Check we don't have more than 25% nans (arbitrary)

                attentions = [
                    tf.where(tf.math.is_nan(t), 0.0, t) for t in attentions
                ]  # remove them (the test is less complete)

                self.assertAlmostEqual(tf.math.reduce_sum(attentions[0][..., 0, :, :]).numpy(), 0.0)
                self.assertNotEqual(tf.math.reduce_sum(attentions[0][..., -1, :, :]).numpy(), 0.0)
                if len(attentions) > 2:  # encoder-decodere models have only 2 layers in each modules
                    self.assertNotEqual(tf.math.reduce_sum(attentions[1][..., 0, :, :]).numpy(), 0.0)
                self.assertAlmostEqual(tf.math.reduce_sum(attentions[-1][..., -2, :, :]).numpy(), 0.0)
                self.assertNotEqual(tf.math.reduce_sum(attentions[-1][..., -1, :, :]).numpy(), 0.0)

            if model.config.is_encoder_decoder:
                check_attentions_validity(outputs.encoder_attentions)
                check_attentions_validity(outputs.decoder_attentions)
                if "cross_attn_head_mask" in arg_names:
                    check_attentions_validity(outputs.cross_attentions)
            else:
                check_attentions_validity(outputs.attentions)

    def test_hidden_states_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_hidden_states_output(config, inputs_dict, model_class):
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )

            if model.config.is_encoder_decoder:
                encoder_hidden_states = outputs.encoder_hidden_states
                decoder_hidden_states = outputs.decoder_hidden_states

                self.assertEqual(config.output_attentions, False)
                self.assertEqual(len(encoder_hidden_states), expected_num_layers)
                self.assertListEqual(
                    list(encoder_hidden_states[0].shape[-2:]),
                    [self.model_tester.seq_length, self.model_tester.hidden_size],
                )
                self.assertEqual(len(decoder_hidden_states), expected_num_layers)
                self.assertListEqual(
                    list(decoder_hidden_states[0].shape[-2:]),
                    [self.model_tester.seq_length, self.model_tester.hidden_size],
                )
            else:
                hidden_states = outputs.hidden_states
                self.assertEqual(config.output_attentions, False)
                self.assertEqual(len(hidden_states), expected_num_layers)
                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [self.model_tester.seq_length, self.model_tester.hidden_size],
                )

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(config, inputs_dict, model_class)

            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            check_hidden_states_output(config, inputs_dict, model_class)

    def test_model_common_attributes(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        list_lm_models = (
            get_values(TF_MODEL_FOR_CAUSAL_LM_MAPPING)
            + get_values(TF_MODEL_FOR_MASKED_LM_MAPPING)
            + get_values(TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING)
        )

        for model_class in self.all_model_classes:
            model = model_class(config)
            assert isinstance(model.get_input_embeddings(), tf.keras.layers.Layer)

            if model_class in list_lm_models:
                x = model.get_output_embeddings()
                assert isinstance(x, tf.keras.layers.Layer)
                name = model.get_bias()
                assert isinstance(name, dict)
                for k, v in name.items():
                    assert isinstance(v, tf.Variable)
            else:
                x = model.get_output_embeddings()
                assert x is None
                name = model.get_bias()
                assert name is None

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            first, second = (
                model(self._prepare_for_class(inputs_dict, model_class), training=False)[0],
                model(self._prepare_for_class(inputs_dict, model_class), training=False)[0],
            )
            out_1 = first.numpy()
            out_2 = second.numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

    def test_model_outputs_equivalence(self):

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            tuple_output = model(tuple_inputs, return_dict=False, **additional_kwargs)
            dict_output = model(dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

            def recursive_check(tuple_object, dict_object):
                if isinstance(tuple_object, (List, Tuple)):
                    for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                        recursive_check(tuple_iterable_value, dict_iterable_value)
                elif tuple_object is None:
                    return
                else:
                    self.assertTrue(
                        all(tf.equal(tuple_object, dict_object)),
                        msg=f"Tuple and dict output are not equal. Difference: {tf.math.reduce_max(tf.abs(tuple_object - dict_object))}",
                    )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(config)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(
                model, tuple_inputs, dict_inputs, {"output_hidden_states": True, "output_attentions": True}
            )

    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)

            inputs = copy.deepcopy(inputs_dict)

            if not self.is_encoder_decoder:
                input_ids = inputs["input_ids"]
                del inputs["input_ids"]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
                del inputs["input_ids"]
                inputs.pop("decoder_input_ids", None)

            if not self.is_encoder_decoder:
                inputs["inputs_embeds"] = model.get_input_embeddings()(input_ids)
            else:
                inputs["inputs_embeds"] = model.get_input_embeddings()(encoder_input_ids)
                inputs["decoder_inputs_embeds"] = model.get_input_embeddings()(decoder_input_ids)

            inputs = self._prepare_for_class(inputs, model_class)

            model(inputs)

    def test_numpy_arrays_inputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def prepare_numpy_arrays(inputs_dict):
            inputs_np_dict = {}
            for k, v in inputs_dict.items():
                if tf.is_tensor(v):
                    inputs_np_dict[k] = v.numpy()
                else:
                    inputs_np_dict[k] = np.array(k)

            return inputs_np_dict

        for model_class in self.all_model_classes:
            model = model_class(config)

            inputs = self._prepare_for_class(inputs_dict, model_class)
            inputs_np = prepare_numpy_arrays(inputs)

            model(inputs_np)

    def test_resize_token_embeddings(self):
        if not self.test_resize_embeddings:
            return
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def _get_word_embedding_weight(model, embedding_layer):
            embeds = getattr(embedding_layer, "weight", None)
            if embeds is not None:
                return embeds

            embeds = getattr(embedding_layer, "decoder", None)
            if embeds is not None:
                return embeds

            model(model.dummy_inputs)

            embeds = getattr(embedding_layer, "weight", None)
            if embeds is not None:
                return embeds

            embeds = getattr(embedding_layer, "decoder", None)
            if embeds is not None:
                return embeds

            return None

        for model_class in self.all_model_classes:
            for size in [config.vocab_size - 10, config.vocab_size + 10, None]:
                # build the embeddings
                model = model_class(config=config)
                old_input_embeddings = _get_word_embedding_weight(model, model.get_input_embeddings())
                old_bias = model.get_bias()
                old_output_embeddings = _get_word_embedding_weight(model, model.get_output_embeddings())
                # reshape the embeddings
                model.resize_token_embeddings(size)
                new_input_embeddings = _get_word_embedding_weight(model, model.get_input_embeddings())
                new_bias = model.get_bias()
                new_output_embeddings = _get_word_embedding_weight(model, model.get_output_embeddings())

                # check that the resized embeddings size matches the desired size.
                assert_size = size if size is not None else config.vocab_size
                self.assertEqual(new_input_embeddings.shape[0], assert_size)

                # check that weights remain the same after resizing
                models_equal = True
                for p1, p2 in zip(old_input_embeddings.value(), new_input_embeddings.value()):
                    if tf.math.reduce_sum(tf.math.abs(p1 - p2)) > 0:
                        models_equal = False
                self.assertTrue(models_equal)

                if old_bias is not None and new_bias is not None:
                    for old_weight, new_weight in zip(old_bias.values(), new_bias.values()):
                        self.assertEqual(new_weight.shape[0], assert_size)

                        models_equal = True
                        for p1, p2 in zip(old_weight.value(), new_weight.value()):
                            if tf.math.reduce_sum(tf.math.abs(p1 - p2)) > 0:
                                models_equal = False
                        self.assertTrue(models_equal)

                if old_output_embeddings is not None and new_output_embeddings is not None:
                    self.assertEqual(new_output_embeddings.shape[0], assert_size)
                    self.assertEqual(new_output_embeddings.shape[1], old_output_embeddings.shape[1])

                    models_equal = True
                    for p1, p2 in zip(old_output_embeddings.value(), new_output_embeddings.value()):
                        if tf.math.reduce_sum(tf.math.abs(p1 - p2)) > 0:
                            models_equal = False
                    self.assertTrue(models_equal)

    def test_lm_head_model_random_no_beam_search_generate(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict.get("input_ids", None)

        # iterate over all generative models
        for model_class in self.all_generative_model_classes:
            model = model_class(config)

            if config.bos_token_id is None:
                # if bos token id is not defined mobel needs input_ids
                with self.assertRaises(AssertionError):
                    model.generate(do_sample=True, max_length=5)
                # num_return_sequences = 1
                self._check_generated_ids(model.generate(input_ids, do_sample=True))
            else:
                # num_return_sequences = 1
                self._check_generated_ids(model.generate(do_sample=True, max_length=5))

            with self.assertRaises(AssertionError):
                # generating multiple sequences when no beam search generation
                # is not allowed as it would always generate the same sequences
                model.generate(input_ids, do_sample=False, num_return_sequences=2)

            # num_return_sequences > 1, sample
            self._check_generated_ids(model.generate(input_ids, do_sample=True, num_return_sequences=2))

            # check bad words tokens language generation
            # create list of 1-seq bad token and list of 2-seq of bad tokens
            bad_words_ids = [self._generate_random_bad_tokens(1, model), self._generate_random_bad_tokens(2, model)]
            output_tokens = model.generate(
                input_ids, do_sample=True, bad_words_ids=bad_words_ids, num_return_sequences=2
            )
            # only count generated tokens
            generated_ids = output_tokens[:, input_ids.shape[-1] :]
            self.assertFalse(self._check_match_tokens(generated_ids.numpy().tolist(), bad_words_ids))

    def test_lm_head_model_no_beam_search_generate_dict_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict.get("input_ids", None)

        # iterate over all generative models
        for model_class in self.all_generative_model_classes:
            model = model_class(config)
            output_greedy = model.generate(
                input_ids,
                do_sample=False,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )
            output_sample = model.generate(
                input_ids,
                do_sample=True,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            if model.config.is_encoder_decoder:
                self.assertIsInstance(output_greedy, TFGreedySearchEncoderDecoderOutput)
                self.assertIsInstance(output_sample, TFSampleEncoderDecoderOutput)
            else:
                self.assertIsInstance(output_greedy, TFGreedySearchDecoderOnlyOutput)
                self.assertIsInstance(output_sample, TFSampleDecoderOnlyOutput)

    def test_lm_head_model_random_beam_search_generate(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict.get("input_ids", None)

        for model_class in self.all_generative_model_classes:
            model = model_class(config)

            if config.bos_token_id is None:
                # if bos token id is not defined mobel needs input_ids, num_return_sequences = 1
                self._check_generated_ids(model.generate(input_ids, do_sample=True, num_beams=2))
            else:
                # num_return_sequences = 1
                self._check_generated_ids(model.generate(do_sample=True, max_length=5, num_beams=2))

            with self.assertRaises(AssertionError):
                # generating more sequences than having beams leads is not possible
                model.generate(input_ids, do_sample=False, num_return_sequences=3, num_beams=2)

            # num_return_sequences > 1, sample
            self._check_generated_ids(
                model.generate(
                    input_ids,
                    do_sample=True,
                    num_beams=2,
                    num_return_sequences=2,
                )
            )
            # num_return_sequences > 1, greedy
            self._check_generated_ids(model.generate(input_ids, do_sample=False, num_beams=2, num_return_sequences=2))

            # check bad words tokens language generation
            # create list of 1-seq bad token and list of 2-seq of bad tokens
            bad_words_ids = [self._generate_random_bad_tokens(1, model), self._generate_random_bad_tokens(2, model)]
            output_tokens = model.generate(
                input_ids, do_sample=False, bad_words_ids=bad_words_ids, num_beams=2, num_return_sequences=2
            )
            # only count generated tokens
            generated_ids = output_tokens[:, input_ids.shape[-1] :]
            self.assertFalse(self._check_match_tokens(generated_ids.numpy().tolist(), bad_words_ids))

    def test_lm_head_model_beam_search_generate_dict_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict.get("input_ids", None)

        # iterate over all generative models
        for model_class in self.all_generative_model_classes:
            model = model_class(config)
            output_beam_search = model.generate(
                input_ids,
                num_beams=2,
                do_sample=False,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )
            output_beam_sample = model.generate(
                input_ids,
                num_beams=2,
                do_sample=True,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            if model.config.is_encoder_decoder:
                self.assertIsInstance(output_beam_search, TFBeamSearchEncoderDecoderOutput)
                self.assertIsInstance(output_beam_sample, TFBeamSampleEncoderDecoderOutput)
            else:
                self.assertIsInstance(output_beam_search, TFBeamSearchDecoderOnlyOutput)
                self.assertIsInstance(output_beam_sample, TFBeamSampleDecoderOnlyOutput)

    def test_loss_computation(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            if getattr(model, "compute_loss", None):
                # The number of elements in the loss should be the same as the number of elements in the label
                prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
                added_label = prepared_for_class[
                    sorted(list(prepared_for_class.keys() - inputs_dict.keys()), reverse=True)[0]
                ]
                loss_size = tf.size(added_label)

                if model.__class__ in get_values(TF_MODEL_FOR_CAUSAL_LM_MAPPING):
                    # if loss is causal lm loss, labels are shift, so that one label per batch
                    # is cut
                    loss_size = loss_size - self.model_tester.batch_size

                # Test that model correctly compute the loss with kwargs
                prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
                input_name = "input_ids" if "input_ids" in prepared_for_class else "pixel_values"
                input_ids = prepared_for_class.pop(input_name)

                loss = model(input_ids, **prepared_for_class)[0]
                self.assertEqual(loss.shape, [loss_size])

                # Test that model correctly compute the loss with a dict
                prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
                loss = model(prepared_for_class)[0]
                self.assertEqual(loss.shape, [loss_size])

                # Test that model correctly compute the loss with a tuple
                prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)

                # Get keys that were added with the _prepare_for_class function
                label_keys = prepared_for_class.keys() - inputs_dict.keys()
                signature = inspect.signature(model.call).parameters
                signature_names = list(signature.keys())

                # Create a dictionary holding the location of the tensors in the tuple
                tuple_index_mapping = {0: input_name}
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

                self.assertEqual(loss.shape, [loss_size])

    def test_generate_with_headmasking(self):
        attention_names = ["encoder_attentions", "decoder_attentions", "cross_attentions"]
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_generative_model_classes:
            model = model_class(config)

            # We want to test only encoder-decoder models
            if not config.is_encoder_decoder:
                continue

            head_masking = {
                "head_mask": tf.zeros((config.encoder_layers, config.encoder_attention_heads)),
                "decoder_head_mask": tf.zeros((config.decoder_layers, config.decoder_attention_heads)),
                "cross_attn_head_mask": tf.zeros((config.decoder_layers, config.decoder_attention_heads)),
            }

            signature = inspect.signature(model.call)
            if set(head_masking.keys()) < set([*signature.parameters.keys()]):
                continue

            for attn_name, (name, mask) in zip(attention_names, head_masking.items()):
                out = model.generate(
                    inputs_dict["input_ids"],
                    num_beams=1,
                    max_length=inputs_dict["input_ids"] + 5,
                    output_attentions=True,
                    return_dict_in_generate=True,
                    **{name: mask},
                )
                # We check the state of decoder_attentions and cross_attentions just from the last step
                attn_weights = out[attn_name] if attn_name == attention_names[0] else out[attn_name][-1]
                self.assertEqual(sum([tf.reduce_sum(w).numpy() for w in attn_weights]), 0.0)

    def test_load_with_mismatched_shapes(self):
        if not self.test_mismatched_shapes:
            return
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class not in get_values(TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING):
                continue

            with self.subTest(msg=f"Testing {model_class}"):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    model = model_class(config)
                    inputs = self._prepare_for_class(inputs_dict, model_class)
                    _ = model(**inputs)
                    model.save_pretrained(tmp_dir)

                    # Fails when we don't set ignore_mismatched_sizes=True
                    with self.assertRaises(ValueError):
                        new_model = TFAutoModelForSequenceClassification.from_pretrained(tmp_dir, num_labels=42)
                    with self.assertRaises(ValueError):
                        new_model_without_prefix = TFAutoModel.from_pretrained(tmp_dir, vocab_size=10)

                    logger = logging.get_logger("transformers.modeling_tf_utils")
                    with CaptureLogger(logger) as cl:
                        new_model = TFAutoModelForSequenceClassification.from_pretrained(
                            tmp_dir, num_labels=42, ignore_mismatched_sizes=True
                        )
                    self.assertIn("the shapes did not match", cl.out)

                    logits = new_model(**inputs).logits
                    self.assertEqual(logits.shape[1], 42)

                    with CaptureLogger(logger) as cl:
                        new_model_without_prefix = TFAutoModel.from_pretrained(
                            tmp_dir, vocab_size=10, ignore_mismatched_sizes=True
                        )
                    self.assertIn("the shapes did not match", cl.out)

                    # Although Tf models always have a prefix pointing to `MainLayer`,
                    # we still add this "without prefix" test to keep a consistency between tf and pt tests.
                    input_ids = ids_tensor((2, 8), 10)
                    if self.is_encoder_decoder:
                        new_model_without_prefix(input_ids, decoder_input_ids=input_ids)
                    else:
                        new_model_without_prefix(input_ids)

    def test_model_main_input_name(self):
        for model_class in self.all_model_classes:
            model_signature = inspect.signature(getattr(model_class, "call"))
            # The main input is the name of the argument after `self`
            observed_main_input_name = list(model_signature.parameters.keys())[1]
            self.assertEqual(model_class.main_input_name, observed_main_input_name)

    def _generate_random_bad_tokens(self, num_bad_tokens, model):
        # special tokens cannot be bad tokens
        special_tokens = []
        if model.config.bos_token_id is not None:
            special_tokens.append(model.config.bos_token_id)
        if model.config.pad_token_id is not None:
            special_tokens.append(model.config.pad_token_id)
        if model.config.eos_token_id is not None:
            special_tokens.append(model.config.eos_token_id)

        # create random bad tokens that are not special tokens
        bad_tokens = []
        while len(bad_tokens) < num_bad_tokens:
            token = tf.squeeze(ids_tensor((1, 1), self.model_tester.vocab_size), 0).numpy()[0]
            if token not in special_tokens:
                bad_tokens.append(token)
        return bad_tokens

    def _check_generated_ids(self, output_ids):
        for token_id in output_ids[0].numpy().tolist():
            self.assertGreaterEqual(token_id, 0)
            self.assertLess(token_id, self.model_tester.vocab_size)

    def _check_match_tokens(self, generated_ids, bad_words_ids):
        # for all bad word tokens
        for bad_word_ids in bad_words_ids:
            # for all slices in batch
            for generated_ids_slice in generated_ids:
                # for all word idx
                for i in range(len(bad_word_ids), len(generated_ids_slice)):
                    # if tokens match
                    if generated_ids_slice[i - len(bad_word_ids) : i] == bad_word_ids:
                        return True
        return False


def ids_tensor(shape, vocab_size, rng=None, name=None, dtype=None):
    """Creates a random int32 tensor of the shape within the vocab size."""
    if rng is None:
        rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    output = tf.constant(values, shape=shape, dtype=dtype if dtype is not None else tf.int32)

    return output


def floats_tensor(shape, scale=1.0, rng=None, name=None, dtype=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return tf.reshape(tf.constant(values, dtype=dtype if dtype is not None else tf.float32), shape=shape)


@require_tf
class UtilsFunctionsTest(unittest.TestCase):

    # tests whether the top_k_top_p_filtering function behaves as expected
    def test_top_k_top_p_filtering(self):
        logits = tf.convert_to_tensor(
            [
                [
                    8.2220991,  # 3rd highest value; idx. 0
                    -0.5620044,
                    5.23229752,
                    4.0386393,
                    -6.8798378,
                    -0.54785802,
                    -3.2012153,
                    2.92777176,
                    1.88171953,
                    7.35341276,  # 5th highest value; idx. 9
                    8.43207833,  # 2nd highest value; idx. 10
                    -9.85711836,
                    -5.96209236,
                    -1.13039161,
                    -7.1115294,
                    -0.8369633,
                    -5.3186408,
                    7.06427407,
                    0.81369344,
                    -0.82023817,
                    -5.9179796,
                    0.58813443,
                    -6.99778438,
                    4.71551189,
                    -0.18771637,
                    7.44020759,  # 4th highest value; idx. 25
                    9.38450987,  # 1st highest value; idx. 26
                    2.12662941,
                    -9.32562038,
                    2.35652522,
                ],  # cummulative prob of 5 highest values <= 0.6
                [
                    0.58425518,
                    4.53139238,
                    -5.57510464,
                    -6.28030699,
                    -7.19529503,
                    -4.02122551,
                    1.39337037,
                    -6.06707057,
                    1.59480517,
                    -9.643119,
                    0.03907799,
                    0.67231762,
                    -8.88206726,
                    6.27115922,  # 4th highest value; idx. 13
                    2.28520723,
                    4.82767506,
                    4.30421368,
                    8.8275313,  # 2nd highest value; idx. 17
                    5.44029958,  # 5th highest value; idx. 18
                    -4.4735794,
                    7.38579536,  # 3rd highest value; idx. 20
                    -2.91051663,
                    2.61946077,
                    -2.5674762,
                    -9.48959302,
                    -4.02922645,
                    -1.35416918,
                    9.67702323,  # 1st highest value; idx. 27
                    -5.89478553,
                    1.85370467,
                ],  # cummulative prob of 5 highest values <= 0.6
            ],
            dtype=tf.float32,
        )

        non_inf_expected_idx = tf.convert_to_tensor(
            [[0, 0], [0, 9], [0, 10], [0, 25], [0, 26], [1, 13], [1, 17], [1, 18], [1, 20], [1, 27]],
            dtype=tf.int32,
        )  # expected non filtered idx as noted above

        non_inf_expected_output = tf.convert_to_tensor(
            [8.222099, 7.3534126, 8.432078, 7.4402075, 9.38451, 6.271159, 8.827531, 5.4402995, 7.3857956, 9.677023],
            dtype=tf.float32,
        )  # expected non filtered values as noted above

        output = tf_top_k_top_p_filtering(logits, top_k=10, top_p=0.6, min_tokens_to_keep=4)

        non_inf_output = output[output != -float("inf")]
        non_inf_idx = tf.cast(
            tf.where(tf.not_equal(output, tf.constant(-float("inf"), dtype=tf.float32))),
            dtype=tf.int32,
        )

        tf.debugging.assert_near(non_inf_output, non_inf_expected_output, rtol=1e-12)
        tf.debugging.assert_equal(non_inf_idx, non_inf_expected_idx)


@require_tf
@is_staging_test
class TFModelPushToHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = login(username=USER, password=PASS)

    @classmethod
    def tearDownClass(cls):
        try:
            delete_repo(token=cls._token, name="test-model-tf")
        except HTTPError:
            pass

        try:
            delete_repo(token=cls._token, name="test-model-tf-org", organization="valid_org")
        except HTTPError:
            pass

    def test_push_to_hub(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        model = TFBertModel(config)
        # Make sure model is properly initialized
        _ = model(model.dummy_inputs)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(os.path.join(tmp_dir, "test-model-tf"), push_to_hub=True, use_auth_token=self._token)

            new_model = TFBertModel.from_pretrained(f"{USER}/test-model-tf")
            models_equal = True
            for p1, p2 in zip(model.weights, new_model.weights):
                if tf.math.reduce_sum(tf.math.abs(p1 - p2)) > 0:
                    models_equal = False
            self.assertTrue(models_equal)

    def test_push_to_hub_with_model_card(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        model = TFBertModel(config)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.push_to_hub(os.path.join(tmp_dir, "test-model-tf"))
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "test-model-card-tf", "README.md")))

    def test_push_to_hub_in_organization(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        model = TFBertModel(config)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(
                os.path.join(tmp_dir, "test-model-tf-org"),
                push_to_hub=True,
                use_auth_token=self._token,
                organization="valid_org",
            )

            new_model = TFBertModel.from_pretrained("valid_org/test-model-tf-org")
            models_equal = True
            for p1, p2 in zip(model.weights, new_model.weights):
                if tf.math.reduce_sum(tf.math.abs(p1 - p2)) > 0:
                    models_equal = False
            self.assertTrue(models_equal)
