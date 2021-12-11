# Copyright 2020 The HuggingFace Team. All rights reserved.
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
import random
import tempfile
import unittest
from typing import List, Tuple

import numpy as np

import transformers
from huggingface_hub import delete_repo, login
from requests.exceptions import HTTPError
from transformers import BertConfig, is_flax_available, is_torch_available
from transformers.models.auto import get_values
from transformers.testing_utils import PASS, USER, CaptureLogger, is_pt_flax_cross_test, is_staging_test, require_flax
from transformers.utils import logging


if is_flax_available():
    import os

    import jax
    import jax.numpy as jnp
    from flax.core.frozen_dict import unfreeze
    from flax.traverse_util import flatten_dict, unflatten_dict
    from transformers import (
        FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        FLAX_MODEL_MAPPING,
        FlaxAutoModel,
        FlaxAutoModelForSequenceClassification,
        FlaxBertModel,
    )
    from transformers.modeling_flax_pytorch_utils import (
        convert_pytorch_state_dict_to_flax,
        load_flax_weights_in_pytorch_model,
    )

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.12"  # assumed parallelism: 8

if is_torch_available():
    import torch


def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if "_range" in key or "_std" in key or "initializer_factor" in key:
            setattr(configs_no_init, key, 1e-10)
    return configs_no_init


def ids_tensor(shape, vocab_size, rng=None):
    """Creates a random int32 tensor of the shape within the vocab size."""
    if rng is None:
        rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    output = np.array(values, dtype=jnp.int32).reshape(shape)

    return output


def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return np.array(values, dtype=jnp.float32).reshape(shape)


def random_attention_mask(shape, rng=None):
    attn_mask = ids_tensor(shape, vocab_size=2, rng=rng)
    # make sure that at least one token is attended to for each batch
    attn_mask[:, -1] = 1
    return attn_mask


@require_flax
class FlaxModelTesterMixin:
    model_tester = None
    all_model_classes = ()
    test_mismatched_shapes = True
    is_encoder_decoder = False

    def _prepare_for_class(self, inputs_dict, model_class):
        inputs_dict = copy.deepcopy(inputs_dict)

        # hack for now until we have AutoModel classes
        if "ForMultipleChoice" in model_class.__name__:
            inputs_dict = {
                k: jnp.broadcast_to(v[:, None], (v.shape[0], self.model_tester.num_choices, v.shape[-1]))
                if isinstance(v, (jnp.ndarray, np.ndarray))
                else v
                for k, v in inputs_dict.items()
            }

        return inputs_dict

    def assert_almost_equals(self, a: np.ndarray, b: np.ndarray, tol: float):
        diff = np.abs((a - b)).max()
        self.assertLessEqual(diff, tol, f"Difference between torch and flax is {diff} (>= {tol}).")

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
            dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

            def recursive_check(tuple_object, dict_object):
                if isinstance(tuple_object, (List, Tuple)):
                    for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                        recursive_check(tuple_iterable_value, dict_iterable_value)
                elif tuple_object is None:
                    return
                else:
                    self.assert_almost_equals(
                        set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), 1e-5
                    )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(config)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

    @is_pt_flax_cross_test
    def test_equivalence_pt_to_flax(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                # prepare inputs
                prepared_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
                pt_inputs = {k: torch.tensor(v.tolist()) for k, v in prepared_inputs_dict.items()}

                # load corresponding PyTorch class
                pt_model_class_name = model_class.__name__[4:]  # Skip the "Flax" at the beginning
                pt_model_class = getattr(transformers, pt_model_class_name)

                pt_model = pt_model_class(config).eval()
                # Flax models don't use the `use_cache` option and cache is not returned as a default.
                # So we disable `use_cache` here for PyTorch model.
                pt_model.config.use_cache = False
                fx_model = model_class(config, dtype=jnp.float32)

                fx_state = convert_pytorch_state_dict_to_flax(pt_model.state_dict(), fx_model)
                fx_model.params = fx_state

                with torch.no_grad():
                    pt_outputs = pt_model(**pt_inputs).to_tuple()

                fx_outputs = fx_model(**prepared_inputs_dict).to_tuple()
                self.assertEqual(len(fx_outputs), len(pt_outputs), "Output lengths differ between Flax and PyTorch")
                for fx_output, pt_output in zip(fx_outputs, pt_outputs):
                    self.assert_almost_equals(fx_output, pt_output.numpy(), 4e-2)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    pt_model.save_pretrained(tmpdirname)
                    fx_model_loaded = model_class.from_pretrained(tmpdirname, from_pt=True)

                fx_outputs_loaded = fx_model_loaded(**prepared_inputs_dict).to_tuple()
                self.assertEqual(
                    len(fx_outputs_loaded), len(pt_outputs), "Output lengths differ between Flax and PyTorch"
                )
                for fx_output_loaded, pt_output in zip(fx_outputs_loaded, pt_outputs):
                    self.assert_almost_equals(fx_output_loaded, pt_output.numpy(), 4e-2)

    @is_pt_flax_cross_test
    def test_equivalence_flax_to_pt(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                # prepare inputs
                prepared_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
                pt_inputs = {k: torch.tensor(v.tolist()) for k, v in prepared_inputs_dict.items()}

                # load corresponding PyTorch class
                pt_model_class_name = model_class.__name__[4:]  # Skip the "Flax" at the beginning
                pt_model_class = getattr(transformers, pt_model_class_name)

                pt_model = pt_model_class(config).eval()
                # Flax models don't use the `use_cache` option and cache is not returned as a default.
                # So we disable `use_cache` here for PyTorch model.
                pt_model.config.use_cache = False
                fx_model = model_class(config, dtype=jnp.float32)

                pt_model = load_flax_weights_in_pytorch_model(pt_model, fx_model.params)

                # make sure weights are tied in PyTorch
                pt_model.tie_weights()

                with torch.no_grad():
                    pt_outputs = pt_model(**pt_inputs).to_tuple()

                fx_outputs = fx_model(**prepared_inputs_dict).to_tuple()
                self.assertEqual(len(fx_outputs), len(pt_outputs), "Output lengths differ between Flax and PyTorch")

                for fx_output, pt_output in zip(fx_outputs, pt_outputs):
                    self.assert_almost_equals(fx_output, pt_output.numpy(), 4e-2)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    fx_model.save_pretrained(tmpdirname)
                    pt_model_loaded = pt_model_class.from_pretrained(tmpdirname, from_flax=True)

                with torch.no_grad():
                    pt_outputs_loaded = pt_model_loaded(**pt_inputs).to_tuple()

                self.assertEqual(
                    len(fx_outputs), len(pt_outputs_loaded), "Output lengths differ between Flax and PyTorch"
                )
                for fx_output, pt_output in zip(fx_outputs, pt_outputs_loaded):
                    self.assert_almost_equals(fx_output, pt_output.numpy(), 4e-2)

    def test_from_pretrained_save_pretrained(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                model = model_class(config)

                prepared_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
                outputs = model(**prepared_inputs_dict).to_tuple()

                # verify that normal save_pretrained works as expected
                with tempfile.TemporaryDirectory() as tmpdirname:
                    model.save_pretrained(tmpdirname)
                    model_loaded = model_class.from_pretrained(tmpdirname)

                outputs_loaded = model_loaded(**prepared_inputs_dict).to_tuple()
                for output_loaded, output in zip(outputs_loaded, outputs):
                    self.assert_almost_equals(output_loaded, output, 1e-3)

                # verify that save_pretrained for distributed training
                # with `params=params` works as expected
                with tempfile.TemporaryDirectory() as tmpdirname:
                    model.save_pretrained(tmpdirname, params=model.params)
                    model_loaded = model_class.from_pretrained(tmpdirname)

                outputs_loaded = model_loaded(**prepared_inputs_dict).to_tuple()
                for output_loaded, output in zip(outputs_loaded, outputs):
                    self.assert_almost_equals(output_loaded, output, 1e-3)

    def test_save_load_from_base(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        base_class = FLAX_MODEL_MAPPING[config.__class__]

        for model_class in self.all_model_classes:
            if model_class == base_class:
                continue

            model = base_class(config)
            base_params = flatten_dict(unfreeze(model.params))

            # check that all base model weights are loaded correctly
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                head_model = model_class.from_pretrained(tmpdirname)

                base_param_from_head = flatten_dict(unfreeze(head_model.params[head_model.base_model_prefix]))

                for key in base_param_from_head.keys():
                    max_diff = (base_params[key] - base_param_from_head[key]).sum().item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    def test_save_load_to_base(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        base_class = FLAX_MODEL_MAPPING[config.__class__]

        for model_class in self.all_model_classes:
            if model_class == base_class:
                continue

            model = model_class(config)
            base_params_from_head = flatten_dict(unfreeze(model.params[model.base_model_prefix]))

            # check that all base model weights are loaded correctly
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                base_model = base_class.from_pretrained(tmpdirname)

                base_params = flatten_dict(unfreeze(base_model.params))

                for key in base_params_from_head.keys():
                    max_diff = (base_params[key] - base_params_from_head[key]).sum().item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    @is_pt_flax_cross_test
    def test_save_load_from_base_pt(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        base_class = FLAX_MODEL_MAPPING[config.__class__]

        for model_class in self.all_model_classes:
            if model_class == base_class:
                continue

            model = base_class(config)
            base_params = flatten_dict(unfreeze(model.params))

            # convert Flax model to PyTorch model
            pt_model_class = getattr(transformers, base_class.__name__[4:])  # Skip the "Flax" at the beginning
            pt_model = pt_model_class(config).eval()
            pt_model = load_flax_weights_in_pytorch_model(pt_model, model.params)

            # check that all base model weights are loaded correctly
            with tempfile.TemporaryDirectory() as tmpdirname:
                # save pt model
                pt_model.save_pretrained(tmpdirname)
                head_model = model_class.from_pretrained(tmpdirname, from_pt=True)

                base_param_from_head = flatten_dict(unfreeze(head_model.params[head_model.base_model_prefix]))

                for key in base_param_from_head.keys():
                    max_diff = (base_params[key] - base_param_from_head[key]).sum().item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    @is_pt_flax_cross_test
    def test_save_load_to_base_pt(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        base_class = FLAX_MODEL_MAPPING[config.__class__]

        for model_class in self.all_model_classes:
            if model_class == base_class:
                continue

            model = model_class(config)
            base_params_from_head = flatten_dict(unfreeze(model.params[model.base_model_prefix]))

            # convert Flax model to PyTorch model
            pt_model_class = getattr(transformers, model_class.__name__[4:])  # Skip the "Flax" at the beginning
            pt_model = pt_model_class(config).eval()
            pt_model = load_flax_weights_in_pytorch_model(pt_model, model.params)

            # check that all base model weights are loaded correctly
            with tempfile.TemporaryDirectory() as tmpdirname:
                pt_model.save_pretrained(tmpdirname)
                base_model = base_class.from_pretrained(tmpdirname, from_pt=True)

                base_params = flatten_dict(unfreeze(base_model.params))

                for key in base_params_from_head.keys():
                    max_diff = (base_params[key] - base_params_from_head[key]).sum().item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    @is_pt_flax_cross_test
    def test_save_load_bf16_to_base_pt(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        base_class = FLAX_MODEL_MAPPING[config.__class__]

        for model_class in self.all_model_classes:
            if model_class == base_class:
                continue

            model = model_class(config)
            model.params = model.to_bf16(model.params)
            base_params_from_head = flatten_dict(unfreeze(model.params[model.base_model_prefix]))

            # convert Flax model to PyTorch model
            pt_model_class = getattr(transformers, model_class.__name__[4:])  # Skip the "Flax" at the beginning
            pt_model = pt_model_class(config).eval()
            pt_model = load_flax_weights_in_pytorch_model(pt_model, model.params)

            # check that all base model weights are loaded correctly
            with tempfile.TemporaryDirectory() as tmpdirname:
                pt_model.save_pretrained(tmpdirname)
                base_model = base_class.from_pretrained(tmpdirname, from_pt=True)

                base_params = flatten_dict(unfreeze(base_model.params))

                for key in base_params_from_head.keys():
                    max_diff = (base_params[key] - base_params_from_head[key]).sum().item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    def test_jit_compilation(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                prepared_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
                model = model_class(config)

                @jax.jit
                def model_jitted(input_ids, attention_mask=None, **kwargs):
                    return model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

                with self.subTest("JIT Enabled"):
                    jitted_outputs = model_jitted(**prepared_inputs_dict).to_tuple()

                with self.subTest("JIT Disabled"):
                    with jax.disable_jit():
                        outputs = model_jitted(**prepared_inputs_dict).to_tuple()

                self.assertEqual(len(outputs), len(jitted_outputs))
                for jitted_output, output in zip(jitted_outputs, outputs):

                    self.assertEqual(jitted_output.shape, output.shape)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.__call__)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            if model.config.is_encoder_decoder:
                expected_arg_names = [
                    "input_ids",
                    "attention_mask",
                    "decoder_input_ids",
                    "decoder_attention_mask",
                ]
                self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)
            else:
                expected_arg_names = ["input_ids", "attention_mask"]
                self.assertListEqual(arg_names[:2], expected_arg_names)

    def test_naming_convention(self):
        for model_class in self.all_model_classes:
            model_class_name = model_class.__name__
            module_class_name = (
                model_class_name[:-5] + "Module" if model_class_name[-5:] == "Model" else model_class_name + "Module"
            )
            bert_modeling_flax_module = __import__(model_class.__module__, fromlist=[module_class_name])
            module_cls = getattr(bert_modeling_flax_module, module_class_name)

            self.assertIsNotNone(module_cls)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
            else:
                seq_length = self.model_tester.seq_length

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
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_length = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_length)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_length)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length", decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )
            out_len = len(outputs)

            if self.is_encoder_decoder:
                correct_outlen = 5

                # Question Answering model returns start_logits and end_logits
                if model_class in get_values(FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING):
                    correct_outlen += 1  # start_logits and end_logits instead of only 1 output

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

            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )

    def test_load_with_mismatched_shapes(self):
        if not self.test_mismatched_shapes:
            return
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class not in get_values(FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING):
                continue

            with self.subTest(msg=f"Testing {model_class}"):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    model = model_class(config)
                    model.save_pretrained(tmp_dir)

                    # Fails when we don't set ignore_mismatched_sizes=True
                    with self.assertRaises(ValueError):
                        new_model = FlaxAutoModelForSequenceClassification.from_pretrained(tmp_dir, num_labels=42)
                    with self.assertRaises(ValueError):
                        new_model_without_prefix = FlaxAutoModel.from_pretrained(tmp_dir, vocab_size=10)

                    logger = logging.get_logger("transformers.modeling_flax_utils")
                    with CaptureLogger(logger) as cl:
                        new_model = FlaxAutoModelForSequenceClassification.from_pretrained(
                            tmp_dir, num_labels=42, ignore_mismatched_sizes=True
                        )
                    self.assertIn("the shapes did not match", cl.out)

                    logits = new_model(**inputs_dict)["logits"]
                    self.assertEqual(logits.shape[1], 42)

                    with CaptureLogger(logger) as cl:
                        new_model_without_prefix = FlaxAutoModel.from_pretrained(
                            tmp_dir, vocab_size=10, ignore_mismatched_sizes=True
                        )
                    self.assertIn("the shapes did not match", cl.out)
                    input_ids = ids_tensor((2, 8), 10)
                    if self.is_encoder_decoder:
                        new_model_without_prefix(input_ids, decoder_input_ids=input_ids)
                    else:
                        new_model_without_prefix(input_ids)

    def test_default_params_dtype(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            # check if all params are still in float32 when dtype of computation is half-precision
            model = model_class(config, dtype=jnp.float16)
            types = jax.tree_map(lambda x: x.dtype, model.params)
            types = flatten_dict(types)

            for name, type_ in types.items():
                self.assertEquals(type_, jnp.float32, msg=f"param {name} is not initialized in fp32.")

    def test_to_bf16(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)

            # cast all params to bf16
            params = model.to_bf16(model.params)
            types = flatten_dict(jax.tree_map(lambda x: x.dtype, params))
            # test if all params are in bf16
            for name, type_ in types.items():
                self.assertEqual(type_, jnp.bfloat16, msg=f"param {name} is not in bf16.")

            # test masking
            flat_params = flatten_dict(params)
            key = random.choice(list(flat_params.keys()))  # choose a random param
            mask = {path: path != key for path in flat_params}  # don't cast the key
            mask = unflatten_dict(mask)

            params = model.to_bf16(model.params, mask)
            types = flatten_dict(jax.tree_map(lambda x: x.dtype, params))
            # test if all params are in bf16 except key
            for name, type_ in types.items():
                if name == key:
                    self.assertEqual(type_, jnp.float32, msg=f"param {name} should be in fp32.")
                else:
                    self.assertEqual(type_, jnp.bfloat16, msg=f"param {name} is not in bf16.")

    def test_to_fp16(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)

            # cast all params to fp16
            params = model.to_fp16(model.params)
            types = flatten_dict(jax.tree_map(lambda x: x.dtype, params))
            # test if all params are in fp16
            for name, type_ in types.items():
                self.assertEqual(type_, jnp.float16, msg=f"param {name} is not in fp16.")

            # test masking
            flat_params = flatten_dict(params)
            key = random.choice(list(flat_params.keys()))  # choose a random param
            mask = {path: path != key for path in flat_params}  # don't cast the key
            mask = unflatten_dict(mask)

            params = model.to_fp16(model.params, mask)
            types = flatten_dict(jax.tree_map(lambda x: x.dtype, params))
            # test if all params are in fp16 except key
            for name, type_ in types.items():
                if name == key:
                    self.assertEqual(type_, jnp.float32, msg=f"param {name} should be in fp32.")
                else:
                    self.assertEqual(type_, jnp.float16, msg=f"param {name} is not in fp16.")

    def test_to_fp32(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)

            # cast all params to fp16 and back to fp32
            params = model.to_fp16(model.params)
            params = model.to_fp32(params)

            # test if all params are in fp32
            types = flatten_dict(jax.tree_map(lambda x: x.dtype, params))
            for name, type_ in types.items():
                self.assertEqual(type_, jnp.float32, msg=f"param {name} is not in fp32.")

            # test masking
            flat_params = flatten_dict(params)
            key = random.choice(list(flat_params.keys()))  # choose a random param
            mask = {path: path != key for path in flat_params}  # don't cast the key
            mask = unflatten_dict(mask)

            # cast to fp16 and back to fp32 with mask
            params = model.to_fp16(model.params)
            params = model.to_fp32(params, mask)

            # test if all params are in fp32 except key
            types = flatten_dict(jax.tree_map(lambda x: x.dtype, params))
            for name, type_ in types.items():
                if name == key:
                    self.assertEqual(type_, jnp.float16, msg=f"param {name} should be in fp16.")
                else:
                    self.assertEqual(type_, jnp.float32, msg=f"param {name} is not in fp32.")

    def test_save_load_in_fp16(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)

        # convert weights to fp16 and save
        params = model.to_fp16(model.params)
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname, params=params)

            # load the weights again and check if they are still in fp16
            model = model_class.from_pretrained(tmpdirname)
            types = flatten_dict(jax.tree_map(lambda x: x.dtype, model.params))
            for name, type_ in types.items():
                self.assertEqual(type_, jnp.float16, msg=f"param {name} is not in fp16.")

    def test_save_load_in_bf16(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)

        # convert weights to bf16 and save
        params = model.to_bf16(model.params)
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname, params=params)

            # load the weights again and check if they are still in fp16
            model = model_class.from_pretrained(tmpdirname)
            types = flatten_dict(jax.tree_map(lambda x: x.dtype, model.params))
            for name, type_ in types.items():
                self.assertEqual(type_, jnp.bfloat16, msg=f"param {name} is not in bf16.")


@require_flax
@is_staging_test
class FlaxModelPushToHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = login(username=USER, password=PASS)

    @classmethod
    def tearDownClass(cls):
        try:
            delete_repo(token=cls._token, name="test-model-flax")
        except HTTPError:
            pass

        try:
            delete_repo(token=cls._token, name="test-model-flax-org", organization="valid_org")
        except HTTPError:
            pass

    def test_push_to_hub(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        model = FlaxBertModel(config)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(
                os.path.join(tmp_dir, "test-model-flax"), push_to_hub=True, use_auth_token=self._token
            )

            new_model = FlaxBertModel.from_pretrained(f"{USER}/test-model-flax")

            base_params = flatten_dict(unfreeze(model.params))
            new_params = flatten_dict(unfreeze(new_model.params))

            for key in base_params.keys():
                max_diff = (base_params[key] - new_params[key]).sum().item()
                self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

    def test_push_to_hub_in_organization(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        model = FlaxBertModel(config)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(
                os.path.join(tmp_dir, "test-model-flax-org"),
                push_to_hub=True,
                use_auth_token=self._token,
                organization="valid_org",
            )

            new_model = FlaxBertModel.from_pretrained("valid_org/test-model-flax-org")

            base_params = flatten_dict(unfreeze(model.params))
            new_params = flatten_dict(unfreeze(new_model.params))

            for key in base_params.keys():
                max_diff = (base_params[key] - new_params[key]).sum().item()
                self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")
