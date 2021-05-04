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
from typing import List, Tuple

import numpy as np

import transformers
from transformers import is_flax_available, is_torch_available
from transformers.testing_utils import is_pt_flax_cross_test, require_flax


if is_flax_available():
    import os

    import jax
    import jax.numpy as jnp
    import jaxlib.xla_extension as jax_xla
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


def random_attention_mask(shape, rng=None):
    attn_mask = ids_tensor(shape, vocab_size=2, rng=rng)
    # make sure that at least one token is attended to for each batch
    attn_mask[:, -1] = 1
    return attn_mask


@require_flax
class FlaxModelTesterMixin:
    model_tester = None
    all_model_classes = ()
    test_head_masking = False

    def _prepare_for_class(self, inputs_dict, model_class):
        inputs_dict = copy.deepcopy(inputs_dict)

        # hack for now until we have AutoModel classes
        if "ForMultipleChoice" in model_class.__name__:
            inputs_dict = {
                k: jnp.broadcast_to(v[:, None], (v.shape[0], self.model_tester.num_choices, v.shape[-1]))
                for k, v in inputs_dict.items()
                if isinstance(v, (jax_xla.DeviceArray, np.ndarray))
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

        for model_class in self.all_model_classes[
            :-2
        ]:  # TODO(Patrick, Daniel) - ForSeqClassification and QA doesn't work yet -> need to investigate
            with self.subTest(model_class.__name__):
                # prepare inputs
                prepared_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
                pt_inputs = {k: torch.tensor(v.tolist()) for k, v in prepared_inputs_dict.items()}

                # load corresponding PyTorch class
                pt_model_class_name = model_class.__name__[4:]  # Skip the "Flax" at the beginning
                pt_model_class = getattr(transformers, pt_model_class_name)

                pt_model = pt_model_class(config).eval()
                fx_model = model_class(config, dtype=jnp.float32)

                fx_state = convert_pytorch_state_dict_to_flax(pt_model.state_dict(), fx_model)
                fx_model.params = fx_state

                with torch.no_grad():
                    pt_outputs = pt_model(**pt_inputs).to_tuple()

                fx_outputs = fx_model(**prepared_inputs_dict).to_tuple()
                self.assertEqual(len(fx_outputs), len(pt_outputs), "Output lengths differ between Flax and PyTorch")
                for fx_output, pt_output in zip(fx_outputs, pt_outputs):
                    if not isinstance(fx_output, tuple):  # TODO(Patrick, Daniel) - let's discard use_cache for now
                        self.assert_almost_equals(fx_output, pt_output.numpy(), 1e-3)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    pt_model.save_pretrained(tmpdirname)
                    fx_model_loaded = model_class.from_pretrained(tmpdirname, from_pt=True)

                fx_outputs_loaded = fx_model_loaded(**prepared_inputs_dict).to_tuple()
                self.assertEqual(
                    len(fx_outputs_loaded), len(pt_outputs), "Output lengths differ between Flax and PyTorch"
                )
                for fx_output_loaded, pt_output in zip(fx_outputs_loaded, pt_outputs):
                    if not isinstance(
                        fx_output_loaded, tuple
                    ):  # TODO(Patrick, Daniel) - let's discard use_cache for now
                        self.assert_almost_equals(fx_output_loaded, pt_output.numpy(), 1e-3)

    @is_pt_flax_cross_test
    def test_equivalence_flax_to_pt(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes[
            :-2
        ]:  # TODO(Patrick, Daniel) - ForSeqClassification and QA doesn't work yet -> need to investigate
            with self.subTest(model_class.__name__):
                # prepare inputs
                prepared_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
                pt_inputs = {k: torch.tensor(v.tolist()) for k, v in prepared_inputs_dict.items()}

                # load corresponding PyTorch class
                pt_model_class_name = model_class.__name__[4:]  # Skip the "Flax" at the beginning
                pt_model_class = getattr(transformers, pt_model_class_name)

                pt_model = pt_model_class(config).eval()
                fx_model = model_class(config, dtype=jnp.float32)

                pt_model = load_flax_weights_in_pytorch_model(pt_model, fx_model.params)

                # make sure weights are tied in PyTorch
                pt_model.tie_weights()

                with torch.no_grad():
                    pt_outputs = pt_model(**pt_inputs).to_tuple()

                fx_outputs = fx_model(**prepared_inputs_dict).to_tuple()
                self.assertEqual(len(fx_outputs), len(pt_outputs), "Output lengths differ between Flax and PyTorch")

                for fx_output, pt_output in zip(fx_outputs, pt_outputs):
                    if not isinstance(fx_output, tuple):  # TODO(Patrick, Daniel) - let's discard use_cache for now
                        self.assert_almost_equals(fx_output, pt_output.numpy(), 1e-3)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    fx_model.save_pretrained(tmpdirname)
                    pt_model_loaded = pt_model_class.from_pretrained(tmpdirname, from_flax=True)

                with torch.no_grad():
                    pt_outputs_loaded = pt_model_loaded(**pt_inputs).to_tuple()

                self.assertEqual(
                    len(fx_outputs), len(pt_outputs_loaded), "Output lengths differ between Flax and PyTorch"
                )
                for fx_output, pt_output in zip(fx_outputs, pt_outputs_loaded):
                    if not isinstance(fx_output, tuple):  # TODO(Patrick, Daniel) - let's discard use_cache for now
                        self.assert_almost_equals(fx_output, pt_output.numpy(), 5e-3)

    def test_from_pretrained_save_pretrained(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class.__name__ != "FlaxBertModel":
                continue

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

    def test_jit_compilation(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                prepared_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
                model = model_class(config)

                if config.is_encoder_decoder:

                    @jax.jit
                    def model_jitted(
                        input_ids,
                        decoder_input_ids=None,
                        attention_mask=None,
                        decoder_attention_mask=None,
                        head_mask=None,
                        decoder_head_mask=None,
                        cross_attn_head_mask=None,
                    ):
                        return model(
                            input_ids=input_ids,
                            decoder_input_ids=decoder_input_ids,
                            attention_mask=attention_mask,
                            decoder_attention_mask=decoder_attention_mask,
                            head_mask=head_mask,
                            decoder_head_mask=decoder_head_mask,
                            cross_attn_head_mask=cross_attn_head_mask,
                        ).to_tuple()

                else:

                    @jax.jit
                    def model_jitted(input_ids, attention_mask=None, token_type_ids=None):
                        return model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                        ).to_tuple()

                # Necessary implementation due to encoder-decoder past_key_values which are nested tuples
                def assertEqual_nested(jitted_output, output):
                    if isinstance(jitted_output, tuple) and isinstance(output, tuple):
                        for jitted_out, out in zip(jitted_output, output):
                            assertEqual_nested(jitted_out, out)
                    else:
                        self.assertEqual(jitted_output.shape, output.shape)

                with self.subTest("JIT Enabled"):
                    jitted_outputs = model_jitted(**prepared_inputs_dict)

                with self.subTest("JIT Disabled"):
                    with jax.disable_jit():
                        outputs = model_jitted(**prepared_inputs_dict)

                self.assertEqual(len(outputs), len(jitted_outputs))
                for jitted_output, output in zip(jitted_outputs, outputs):
                    assertEqual_nested(jitted_output, output)

                if config.is_encoder_decoder:

                    @jax.jit
                    def model_jitted_return_dict(
                        input_ids,
                        decoder_input_ids=None,
                        attention_mask=None,
                        decoder_attention_mask=None,
                        head_mask=None,
                        decoder_head_mask=None,
                        cross_attn_head_mask=None,
                    ):
                        return model(
                            input_ids=input_ids,
                            decoder_input_ids=decoder_input_ids,
                            attention_mask=attention_mask,
                            decoder_attention_mask=decoder_attention_mask,
                            head_mask=head_mask,
                            decoder_head_mask=decoder_head_mask,
                            cross_attn_head_mask=cross_attn_head_mask,
                        )

                else:

                    @jax.jit
                    def model_jitted_return_dict(input_ids, attention_mask=None, token_type_ids=None):
                        return model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                        )

                # jitted function cannot return OrderedDict
                with self.assertRaises(TypeError):
                    model_jitted_return_dict(**prepared_inputs_dict)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.__call__)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

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

    def test_headmasking(self):
        if not self.test_head_masking:
            return

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        inputs_dict["output_attentions"] = True
        inputs_dict["output_hidden_states"] = True
        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)

            # Prepare head_mask
            # Prepare head_mask
            def prepare_layer_head_mask(i, attention_heads, num_hidden_layers):
                if i == 0:
                    return jnp.concatenate(
                        (jnp.zeros(1, dtype=jnp.int32), jnp.ones(attention_heads - 1, dtype=jnp.int32)), 0
                    ).reshape(1, -1)
                elif i == num_hidden_layers - 1:
                    return jnp.concatenate(
                        (jnp.zeros(attention_heads - 1, dtype=jnp.int32), jnp.ones(1, dtype=jnp.int32)), 0
                    ).reshape(1, -1)
                else:
                    return jnp.ones(attention_heads, dtype=jnp.int32).reshape(1, -1)

            head_mask = jnp.concatenate(
                [
                    prepare_layer_head_mask(i, config.num_attention_heads, config.num_hidden_layers)
                    for i in range(config.num_hidden_layers)
                ],
                0,
            )

            inputs = self._prepare_for_class(inputs_dict, model_class).copy()
            inputs["head_mask"] = head_mask

            if model.config.is_encoder_decoder:
                signature = inspect.signature(model.__call__)
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
                        jnp.sum(jnp.isnan(t)), t.size / 4
                    )  # Check we don't have more than 25% nans (arbitrary)
                attentions = [
                    jnp.where(jnp.isnan(t), 0.0, t) for t in attentions
                ]  # remove them (the test is less complete)

                self.assertAlmostEqual(attentions[0][..., 0, :, :].flatten().sum().item(), 0.0)
                self.assertNotEqual(attentions[0][..., -1, :, :].flatten().sum().item(), 0.0)
                if len(attentions) > 2:  # encoder-decoder models have only 2 layers in each module
                    self.assertNotEqual(attentions[1][..., 0, :, :].flatten().sum().item(), 0.0)
                self.assertAlmostEqual(attentions[-1][..., -2, :, :].flatten().sum().item(), 0.0)
                self.assertNotEqual(attentions[-1][..., -1, :, :].flatten().sum().item(), 0.0)

            if model.config.is_encoder_decoder:
                check_attentions_validity(outputs.encoder_attentions)
                check_attentions_validity(outputs.decoder_attentions)
                check_attentions_validity(outputs.cross_attentions)
            else:
                check_attentions_validity(outputs.attentions)
