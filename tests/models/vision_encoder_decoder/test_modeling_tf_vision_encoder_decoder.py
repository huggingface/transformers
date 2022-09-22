# coding=utf-8
# Copyright 2022 HuggingFace Inc. team.
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
""" Testing suite for the TensorFlow VisionEncoderDecoder model. """


import copy
import os
import tempfile
import unittest

import numpy as np

from transformers import is_tf_available, is_torch_available, is_vision_available
from transformers.testing_utils import (
    is_pt_tf_cross_test,
    require_tf,
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils.generic import ModelOutput

from ...test_modeling_tf_common import floats_tensor, ids_tensor
from ..gpt2.test_modeling_tf_gpt2 import TFGPT2ModelTester
from ..vit.test_modeling_tf_vit import TFViTModelTester


if is_tf_available():
    import tensorflow as tf

    from transformers import (
        AutoConfig,
        AutoFeatureExtractor,
        AutoTokenizer,
        TFAutoModel,
        TFAutoModelForCausalLM,
        TFGPT2LMHeadModel,
        TFVisionEncoderDecoderModel,
        TFViTModel,
        VisionEncoderDecoderConfig,
    )
    from transformers.modeling_tf_outputs import TFBaseModelOutput

if is_torch_available():
    import torch

    from transformers import GPT2LMHeadModel, VisionEncoderDecoderModel, ViTModel

if is_vision_available():
    from PIL import Image

    from transformers import ViTFeatureExtractor


@require_tf
class TFVisionEncoderDecoderMixin:
    def get_encoder_decoder_model(self, config, decoder_config):
        raise NotImplementedError

    def prepare_config_and_inputs(self):
        raise NotImplementedError

    def get_pretrained_model(self):
        raise NotImplementedError

    def check_encoder_decoder_model_from_pretrained_configs(
        self,
        config,
        pixel_values,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs
    ):
        encoder_decoder_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config, decoder_config)
        self.assertTrue(encoder_decoder_config.decoder.is_decoder)

        enc_dec_model = TFVisionEncoderDecoderModel(encoder_decoder_config)

        self.assertTrue(enc_dec_model.config.is_encoder_decoder)

        outputs_encoder_decoder = enc_dec_model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            kwargs=kwargs,
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )
        self.assertEqual(outputs_encoder_decoder["encoder_last_hidden_state"].shape[0], pixel_values.shape[0])
        self.assertEqual(outputs_encoder_decoder["encoder_last_hidden_state"].shape[-1], config.hidden_size)

    def check_encoder_decoder_model(
        self,
        config,
        pixel_values,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = TFVisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        self.assertTrue(enc_dec_model.config.decoder.is_decoder)
        self.assertTrue(enc_dec_model.config.decoder.add_cross_attention)
        self.assertTrue(enc_dec_model.config.is_encoder_decoder)

        outputs_encoder_decoder = enc_dec_model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            kwargs=kwargs,
        )
        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )
        self.assertEqual(outputs_encoder_decoder["encoder_last_hidden_state"].shape[0], pixel_values.shape[0])
        self.assertEqual(outputs_encoder_decoder["encoder_last_hidden_state"].shape[-1], config.hidden_size)

        encoder_outputs = TFBaseModelOutput(last_hidden_state=encoder_hidden_states)
        outputs_encoder_decoder = enc_dec_model(
            pixel_values=None,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            kwargs=kwargs,
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )
        self.assertEqual(outputs_encoder_decoder["encoder_last_hidden_state"].shape[0], pixel_values.shape[0])
        self.assertEqual(outputs_encoder_decoder["encoder_last_hidden_state"].shape[-1], config.hidden_size)

    def check_encoder_decoder_model_from_pretrained(
        self,
        config,
        pixel_values,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        return_dict,
        **kwargs
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        kwargs = {"encoder_model": encoder_model, "decoder_model": decoder_model, "return_dict": return_dict}
        enc_dec_model = TFVisionEncoderDecoderModel.from_encoder_decoder_pretrained(**kwargs)
        outputs_encoder_decoder = enc_dec_model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
            kwargs=kwargs,
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )
        self.assertEqual(outputs_encoder_decoder["encoder_last_hidden_state"].shape[0], pixel_values.shape[0])
        self.assertEqual(outputs_encoder_decoder["encoder_last_hidden_state"].shape[-1], config.hidden_size)

    def check_save_and_load(
        self,
        config,
        pixel_values,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = TFVisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)

        outputs = enc_dec_model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            kwargs=kwargs,
        )
        out_2 = np.array(outputs[0])
        out_2[np.isnan(out_2)] = 0

        with tempfile.TemporaryDirectory() as tmpdirname:
            enc_dec_model.save_pretrained(tmpdirname)
            enc_dec_model = TFVisionEncoderDecoderModel.from_pretrained(tmpdirname)

            after_outputs = enc_dec_model(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                kwargs=kwargs,
            )
            out_1 = np.array(after_outputs[0])
            out_1[np.isnan(out_1)] = 0
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

    def check_encoder_decoder_model_labels(
        self,
        config,
        pixel_values,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        labels,
        **kwargs
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = TFVisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)

        outputs_encoder_decoder = enc_dec_model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            kwargs=kwargs,
        )

        # Make sure `loss` exist
        self.assertIn("loss", outputs_encoder_decoder)

        batch_size, seq_len = decoder_input_ids.shape
        expected_shape = (batch_size, seq_len, decoder_config.vocab_size)
        self.assertEqual(outputs_encoder_decoder["logits"].shape, expected_shape)
        self.assertEqual(outputs_encoder_decoder["encoder_last_hidden_state"].shape[0], pixel_values.shape[0])
        self.assertEqual(outputs_encoder_decoder["encoder_last_hidden_state"].shape[-1], config.hidden_size)

    def check_encoder_decoder_model_output_attentions(
        self,
        config,
        pixel_values,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs
    ):
        # make the decoder inputs a different shape from the encoder inputs to harden the test
        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_attention_mask = decoder_attention_mask[:, :-1]
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = TFVisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        outputs_encoder_decoder = enc_dec_model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=True,
            kwargs=kwargs,
        )

        encoder_attentions = outputs_encoder_decoder["encoder_attentions"]
        self.assertEqual(len(encoder_attentions), config.num_hidden_layers)

        self.assertEqual(encoder_attentions[0].shape[-3:-2], (config.num_attention_heads,))

        decoder_attentions = outputs_encoder_decoder["decoder_attentions"]
        num_decoder_layers = (
            decoder_config.num_decoder_layers
            if hasattr(decoder_config, "num_decoder_layers")
            else decoder_config.num_hidden_layers
        )
        self.assertEqual(len(decoder_attentions), num_decoder_layers)

        self.assertEqual(
            decoder_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, decoder_input_ids.shape[-1], decoder_input_ids.shape[-1]),
        )

        cross_attentions = outputs_encoder_decoder["cross_attentions"]
        self.assertEqual(len(cross_attentions), num_decoder_layers)

        cross_attention_input_seq_len = decoder_input_ids.shape[-1] * (
            1 + (decoder_config.ngram if hasattr(decoder_config, "ngram") else 0)
        )
        self.assertEqual(
            cross_attentions[0].shape[-3:-1],
            (decoder_config.num_attention_heads, cross_attention_input_seq_len),
        )

    def check_encoder_decoder_model_generate(self, pixel_values, config, decoder_config, **kwargs):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = TFVisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)

        # Generate until max length
        if hasattr(enc_dec_model.config, "eos_token_id"):
            enc_dec_model.config.eos_token_id = None
        if hasattr(enc_dec_model.config, "decoder") and hasattr(enc_dec_model.config.decoder, "eos_token_id"):
            enc_dec_model.config.decoder.eos_token_id = None

        # Bert does not have a bos token id, so use pad_token_id instead
        generated_output = enc_dec_model.generate(
            pixel_values, decoder_start_token_id=enc_dec_model.config.decoder.pad_token_id
        )
        self.assertEqual(
            tuple(generated_output.shape.as_list()), (pixel_values.shape[0],) + (decoder_config.max_length,)
        )

    def check_pt_tf_outputs(self, tf_outputs, pt_outputs, model_class, tol=1e-5, name="outputs", attributes=None):
        """Check the outputs from PyTorch and TensorFlow models are close enough. Checks are done in a recursive way.

        Args:
            model_class: The class of the model that is currently testing. For example, `TFBertModel`,
                TFBertForMaskedLM`, `TFBertForSequenceClassification`, etc. Mainly used for providing more informative
                error messages.
            name (`str`): The name of the output. For example, `output.hidden_states`, `output.attentions`, etc.
            attributes (`Tuple[str]`): The names of the output's element if the output is a tuple/list with each element
                being a named field in the output.
        """

        self.assertEqual(type(name), str)
        if attributes is not None:
            self.assertEqual(type(attributes), tuple, f"{name}: The argument `attributes` should be a `tuple`")

        # Allow `ModelOutput` (e.g. `CLIPOutput` has `text_model_output` and `vision_model_output`).
        if isinstance(tf_outputs, ModelOutput):
            self.assertTrue(
                isinstance(pt_outputs, ModelOutput),
                f"{name}: `pt_outputs` should an instance of `ModelOutput` when `tf_outputs` is",
            )

            tf_keys = [k for k, v in tf_outputs.items() if v is not None]
            pt_keys = [k for k, v in pt_outputs.items() if v is not None]

            self.assertEqual(tf_keys, pt_keys, f"{name}: Output keys differ between TF and PyTorch")

            # convert to the case of `tuple`
            # appending each key to the current (string) `names`
            attributes = tuple([f"{name}.{k}" for k in tf_keys])
            self.check_pt_tf_outputs(
                tf_outputs.to_tuple(), pt_outputs.to_tuple(), model_class, tol=tol, name=name, attributes=attributes
            )

        # Allow `list` (e.g. `TransfoXLModelOutput.mems` is a list of tensors.)
        elif type(tf_outputs) in [tuple, list]:
            self.assertEqual(type(tf_outputs), type(pt_outputs), f"{name}: Output types differ between TF and PyTorch")
            self.assertEqual(len(tf_outputs), len(pt_outputs), f"{name}: Output lengths differ between TF and PyTorch")

            if attributes is not None:
                # case 1: each output has assigned name (e.g. a tuple form of a `ModelOutput`)
                self.assertEqual(
                    len(attributes),
                    len(tf_outputs),
                    f"{name}: The tuple `names` should have the same length as `tf_outputs`",
                )
            else:
                # case 2: each output has no assigned name (e.g. hidden states of each layer) -> add an index to `names`
                attributes = tuple([f"{name}_{idx}" for idx in range(len(tf_outputs))])

            for tf_output, pt_output, attr in zip(tf_outputs, pt_outputs, attributes):
                self.check_pt_tf_outputs(tf_output, pt_output, model_class, tol=tol, name=attr)

        elif isinstance(tf_outputs, tf.Tensor):
            self.assertTrue(
                isinstance(pt_outputs, torch.Tensor), f"{name}: `pt_outputs` should a tensor when `tf_outputs` is"
            )

            tf_outputs = tf_outputs.numpy()
            pt_outputs = pt_outputs.detach().to("cpu").numpy()

            self.assertEqual(
                tf_outputs.shape, pt_outputs.shape, f"{name}: Output shapes differ between TF and PyTorch"
            )

            # deal with NumPy's scalars to make replacing nan values by 0 work.
            if np.isscalar(tf_outputs):
                tf_outputs = np.array([tf_outputs])
                pt_outputs = np.array([pt_outputs])

            tf_nans = np.isnan(tf_outputs)
            pt_nans = np.isnan(pt_outputs)

            pt_outputs[tf_nans] = 0
            tf_outputs[tf_nans] = 0
            pt_outputs[pt_nans] = 0
            tf_outputs[pt_nans] = 0

            max_diff = np.amax(np.abs(tf_outputs - pt_outputs))
            self.assertLessEqual(max_diff, tol, f"{name}: Difference between torch and tf is {max_diff} (>= {tol}).")
        else:
            raise ValueError(
                "`tf_outputs` should be an instance of `tf.Tensor`, a `tuple`, or an instance of `tf.Tensor`. Got"
                f" {type(tf_outputs)} instead."
            )

    def prepare_pt_inputs_from_tf_inputs(self, tf_inputs_dict):

        pt_inputs_dict = {}
        for name, key in tf_inputs_dict.items():
            if type(key) == bool:
                pt_inputs_dict[name] = key
            elif name == "input_values":
                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
            elif name == "pixel_values":
                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
            elif name == "input_features":
                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
            # other general float inputs
            elif tf_inputs_dict[name].dtype.is_floating:
                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
            else:
                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.long)

        return pt_inputs_dict

    def check_pt_tf_models(self, tf_model, pt_model, tf_inputs_dict):

        pt_inputs_dict = self.prepare_pt_inputs_from_tf_inputs(tf_inputs_dict)

        # send pytorch inputs to the correct device
        pt_inputs_dict = {
            k: v.to(device=torch_device) if isinstance(v, torch.Tensor) else v for k, v in pt_inputs_dict.items()
        }

        # send pytorch model to the correct device
        pt_model.to(torch_device)

        # Check predictions on first output (logits/hidden-states) are close enough given low-level computational differences
        pt_model.eval()

        with torch.no_grad():
            pt_outputs = pt_model(**pt_inputs_dict)
        tf_outputs = tf_model(tf_inputs_dict)

        # tf models returned loss is usually a tensor rather than a scalar.
        # (see `hf_compute_loss`: it uses `tf.keras.losses.Reduction.NONE`)
        # Change it here to a scalar to match PyTorch models' loss
        tf_loss = getattr(tf_outputs, "loss", None)
        if tf_loss is not None:
            tf_outputs.loss = tf.math.reduce_mean(tf_loss)

        self.check_pt_tf_outputs(tf_outputs, pt_outputs, type(tf_model))

    def check_pt_tf_equivalence(self, tf_model, pt_model, tf_inputs_dict):
        """Wrap `check_pt_tf_models` to further check PT -> TF again"""

        self.check_pt_tf_models(tf_model, pt_model, tf_inputs_dict)

        # PT -> TF
        with tempfile.TemporaryDirectory() as encoder_tmp_dirname, tempfile.TemporaryDirectory() as decoder_tmp_dirname:

            pt_model.encoder.save_pretrained(encoder_tmp_dirname)
            pt_model.decoder.save_pretrained(decoder_tmp_dirname)
            tf_model_loaded = TFVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                encoder_tmp_dirname, decoder_tmp_dirname, encoder_from_pt=True, decoder_from_pt=True
            )
            # This is only for copying some specific attributes of this particular model.
            tf_model_loaded.config = pt_model.config

        self.check_pt_tf_models(tf_model, pt_model, tf_inputs_dict)

    def check_pt_to_tf_equivalence(self, config, decoder_config, tf_inputs_dict):
        """EncoderDecoderModel requires special way to cross load (PT -> TF)"""

        encoder_decoder_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config, decoder_config)
        # Output all for aggressive testing
        encoder_decoder_config.output_hidden_states = True
        # All models tested in this file have attentions
        encoder_decoder_config.output_attentions = True

        pt_model = VisionEncoderDecoderModel(encoder_decoder_config)

        with tempfile.TemporaryDirectory() as encoder_tmp_dirname, tempfile.TemporaryDirectory() as decoder_tmp_dirname:

            pt_model.encoder.save_pretrained(encoder_tmp_dirname)
            pt_model.decoder.save_pretrained(decoder_tmp_dirname)
            tf_model = TFVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                encoder_tmp_dirname, decoder_tmp_dirname, encoder_from_pt=True, decoder_from_pt=True
            )
            # This is only for copying some specific attributes of this particular model.
            tf_model.config = pt_model.config

        self.check_pt_tf_equivalence(tf_model, pt_model, tf_inputs_dict)

    def check_tf_to_pt_equivalence(self, config, decoder_config, tf_inputs_dict):
        """EncoderDecoderModel requires special way to cross load (TF -> PT)"""

        encoder_decoder_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config, decoder_config)
        # Output all for aggressive testing
        encoder_decoder_config.output_hidden_states = True
        # TODO: A generalizable way to determine this attribute
        encoder_decoder_config.output_attentions = True

        # Using `_tf_model`, the test will fail, because the weights of `_tf_model` get extended before saving
        # the encoder/decoder models.
        # There was a (very) ugly potential fix, which wasn't integrated to `transformers`: see
        #   https://github.com/huggingface/transformers/pull/13222/commits/dbb3c9de76eee235791d2064094654637c99f36d#r697304245
        #   (the change in `src/transformers/modeling_tf_utils.py`)
        _tf_model = TFVisionEncoderDecoderModel(encoder_decoder_config)
        # Make sure model is built
        _tf_model(**tf_inputs_dict)

        # Using `tf_model` to pass the test.
        encoder = _tf_model.encoder.__class__(encoder_decoder_config.encoder)
        decoder = _tf_model.decoder.__class__(encoder_decoder_config.decoder)
        # Make sure models are built
        encoder(encoder.dummy_inputs)
        decoder(decoder.dummy_inputs)
        tf_model = TFVisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
        tf_model.config = encoder_decoder_config

        with tempfile.TemporaryDirectory() as encoder_tmp_dirname, tempfile.TemporaryDirectory() as decoder_tmp_dirname:

            tf_model.encoder.save_pretrained(encoder_tmp_dirname)
            tf_model.decoder.save_pretrained(decoder_tmp_dirname)
            pt_model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                encoder_tmp_dirname, decoder_tmp_dirname, encoder_from_tf=True, decoder_from_tf=True
            )
            # This is only for copying some specific attributes of this particular model.
            pt_model.config = tf_model.config

        self.check_pt_tf_equivalence(tf_model, pt_model, tf_inputs_dict)

    def test_encoder_decoder_model(self):
        config_inputs_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model(**config_inputs_dict)

    def test_encoder_decoder_model_from_pretrained_configs(self):
        config_inputs_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_from_pretrained_configs(**config_inputs_dict)

    def test_encoder_decoder_model_from_pretrained(self):
        config_inputs_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_from_pretrained(**config_inputs_dict, return_dict=False)

    def test_encoder_decoder_model_from_pretrained_return_dict(self):
        config_inputs_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_from_pretrained(**config_inputs_dict, return_dict=True)

    def test_save_and_load_from_pretrained(self):
        config_inputs_dict = self.prepare_config_and_inputs()
        self.check_save_and_load(**config_inputs_dict)

    def test_encoder_decoder_model_labels(self):
        config_inputs_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_labels(**config_inputs_dict)

    def test_encoder_decoder_model_output_attentions(self):
        config_inputs_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_output_attentions(**config_inputs_dict)

    def test_encoder_decoder_model_generate(self):
        config_inputs_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_generate(**config_inputs_dict)

    def assert_almost_equals(self, a: np.ndarray, b: np.ndarray, tol: float):
        diff = np.abs((a - b)).max()
        self.assertLessEqual(diff, tol, f"Difference between torch and tf is {diff} (>= {tol}).")

    @is_pt_tf_cross_test
    def test_pt_tf_model_equivalence(self):

        config_inputs_dict = self.prepare_config_and_inputs()
        labels = config_inputs_dict.pop("decoder_token_labels")

        # Keep only common arguments
        arg_names = [
            "config",
            "pixel_values",
            "decoder_config",
            "decoder_input_ids",
            "decoder_attention_mask",
            "encoder_hidden_states",
        ]
        config_inputs_dict = {k: v for k, v in config_inputs_dict.items() if k in arg_names}

        config = config_inputs_dict.pop("config")
        decoder_config = config_inputs_dict.pop("decoder_config")

        # Output all for aggressive testing
        config.output_hidden_states = True
        decoder_config.output_hidden_states = True
        # All models tested in this file have attentions
        config.output_attentions = True
        decoder_config.output_attentions = True

        tf_inputs_dict = config_inputs_dict
        # `encoder_hidden_states` is not used in model call/forward
        del tf_inputs_dict["encoder_hidden_states"]

        # Make sure no sequence has all zeros as attention mask, otherwise some tests fail due to the inconsistency
        # of the usage `1e-4`, `1e-9`, `1e-30`, `-inf`.
        for k in ["decoder_attention_mask"]:
            attention_mask = tf_inputs_dict[k]

            # Make sure no all 0s attention masks - to avoid failure at this moment.
            # Put `1` at the beginning of sequences to make it still work when combining causal attention masks.
            # TODO: remove this line once a fix regarding large negative values for attention mask is done.
            attention_mask = tf.concat(
                [tf.ones_like(attention_mask[:, :1], dtype=attention_mask.dtype), attention_mask[:, 1:]], axis=-1
            )
            tf_inputs_dict[k] = attention_mask

        tf_inputs_dict_with_labels = copy.copy(tf_inputs_dict)
        tf_inputs_dict_with_labels["labels"] = labels

        self.assertTrue(decoder_config.cross_attention_hidden_size is None)

        # Original test: check without `labels` and  without `enc_to_dec_proj` projection
        self.assertTrue(config.hidden_size == decoder_config.hidden_size)
        self.check_pt_to_tf_equivalence(config, decoder_config, tf_inputs_dict)
        self.check_tf_to_pt_equivalence(config, decoder_config, tf_inputs_dict)

        # check with `labels`
        self.check_pt_to_tf_equivalence(config, decoder_config, tf_inputs_dict_with_labels)
        self.check_tf_to_pt_equivalence(config, decoder_config, tf_inputs_dict_with_labels)

        # This is not working, because pt/tf equivalence test for encoder-decoder use `from_encoder_decoder_pretrained`,
        # which randomly initialize `enc_to_dec_proj`.
        # check `enc_to_dec_proj` work as expected
        # decoder_config.hidden_size = decoder_config.hidden_size * 2
        # self.assertTrue(config.hidden_size != decoder_config.hidden_size)
        # self.check_pt_to_tf_equivalence(config, decoder_config, tf_inputs_dict)
        # self.check_tf_to_pt_equivalence(config, decoder_config, tf_inputs_dict)

        # Let's just check `enc_to_dec_proj` can run for now
        decoder_config.hidden_size = decoder_config.hidden_size * 2
        self.assertTrue(config.hidden_size != decoder_config.hidden_size)
        encoder_decoder_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config, decoder_config)
        model = TFVisionEncoderDecoderModel(encoder_decoder_config)
        model(tf_inputs_dict)

    @slow
    def test_real_model_save_load_from_pretrained(self):
        model_2 = self.get_pretrained_model()
        pixel_values = floats_tensor(
            [
                13,
                model_2.config.encoder.num_channels,
                model_2.config.encoder.image_size,
                model_2.config.encoder.image_size,
            ]
        )
        decoder_input_ids = ids_tensor([13, 1], model_2.config.decoder.vocab_size)

        outputs = model_2(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
        )
        out_2 = np.array(outputs[0])
        out_2[np.isnan(out_2)] = 0

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model_2.save_pretrained(tmp_dirname)
            model_1 = TFVisionEncoderDecoderModel.from_pretrained(tmp_dirname)

            after_outputs = model_1(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
            out_1 = np.array(after_outputs[0])
            out_1[np.isnan(out_1)] = 0
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)


@require_tf
class TFViT2GPT2EncoderDecoderModelTest(TFVisionEncoderDecoderMixin, unittest.TestCase):
    def get_pretrained_model(self):
        return TFVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "google/vit-base-patch16-224-in21k", "../gpt2"
        )

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = TFViTModel(config, name="encoder")
        decoder_model = TFGPT2LMHeadModel(decoder_config, name="decoder")
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester_encoder = TFViTModelTester(self, batch_size=13)
        model_tester_decoder = TFGPT2ModelTester(self)
        encoder_config_and_inputs = model_tester_encoder.prepare_config_and_inputs()
        decoder_config_and_inputs = model_tester_decoder.prepare_config_and_inputs_for_decoder()
        (config, pixel_values, labels) = encoder_config_and_inputs
        (
            decoder_config,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_head_mask,
            decoder_token_type_ids,
            decoder_sequence_labels,
            decoder_token_labels,
            decoder_choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        ) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        # disable cache for now
        decoder_config.use_cache = False
        return {
            "config": config,
            "pixel_values": pixel_values,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_token_labels": decoder_token_labels,
            "encoder_hidden_states": encoder_hidden_states,  # This is not used in the tests.
            "labels": decoder_token_labels,
        }


@require_tf
class TFVisionEncoderDecoderModelTest(unittest.TestCase):
    def get_from_encoderdecoder_pretrained_model(self):
        return TFVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "google/vit-base-patch16-224-in21k", "../gpt2"
        )

    def get_decoder_config(self):
        config = AutoConfig.from_pretrained("../gpt2")
        config.is_decoder = True
        config.add_cross_attention = True
        return config

    def get_encoderdecoder_model(self):
        return TFVisionEncoderDecoderModel.from_pretrained("ydshieh/vit-gpt2-coco-en")

    def get_encoder_decoder_models(self):
        encoder_model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k", name="encoder")
        decoder_model = TFGPT2LMHeadModel.from_pretrained("../gpt2", config=self.get_decoder_config(), name="decoder")
        return {"encoder": encoder_model, "decoder": decoder_model}

    def _check_configuration_tie(self, model):
        assert id(model.decoder.config) == id(model.config.decoder)
        assert id(model.encoder.config) == id(model.config.encoder)

    @slow
    def test_configuration_tie(self):
        model = self.get_from_encoderdecoder_pretrained_model()
        self._check_configuration_tie(model)

        model = TFVisionEncoderDecoderModel(**self.get_encoder_decoder_models())
        self._check_configuration_tie(model)

        model = self.get_encoderdecoder_model()
        self._check_configuration_tie(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_tf
class TFVisionEncoderDecoderModelSaveLoadTests(unittest.TestCase):
    def get_encoder_decoder_config(self):
        encoder_config = AutoConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        decoder_config = AutoConfig.from_pretrained("../gpt2", is_decoder=True, add_cross_attention=True)
        return VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)

    def get_encoder_decoder_config_small(self):
        encoder_config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-vit")
        decoder_config = AutoConfig.from_pretrained(
            "hf-internal-testing/tiny-random-gpt2", is_decoder=True, add_cross_attention=True
        )
        return VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)

    def test_encoder_decoder_save_load_from_encoder_decoder(self):
        config = self.get_encoder_decoder_config_small()

        # create two random ViT/GPT2 models for vit-gpt2 & initialize weights (+cross_attention weights)
        encoder = TFViTModel(config.encoder)
        encoder(encoder.dummy_inputs)
        decoder = TFGPT2LMHeadModel(config.decoder)
        decoder(decoder.dummy_inputs)

        encoder_decoder_orig = TFVisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

        pixel_values = floats_tensor(
            [
                13,
                encoder.config.num_channels,
                encoder.config.image_size,
                encoder.config.image_size,
            ]
        )
        decoder_input_ids = ids_tensor([13, 1], decoder.config.vocab_size)

        logits_orig = encoder_decoder_orig(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids).logits

        with tempfile.TemporaryDirectory() as tmp_dirname:
            encoder_path = os.path.join(tmp_dirname, "encoder")
            decoder_path = os.path.join(tmp_dirname, "decoder")

            encoder.save_pretrained(encoder_path)
            decoder.save_pretrained(decoder_path)

            encoder_decoder = TFVisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_path, decoder_path)

        logits_1 = encoder_decoder(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids).logits

        self.assertTrue(logits_orig.numpy().sum() - logits_1.numpy().sum() < 1e-3)

        max_diff = np.max(np.abs(logits_1.numpy() - logits_orig.numpy()))
        self.assertAlmostEqual(max_diff, 0.0, places=4)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            encoder_decoder.save_pretrained(tmp_dirname)
            encoder_decoder = TFVisionEncoderDecoderModel.from_pretrained(tmp_dirname)

        logits_2 = encoder_decoder(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids).logits

        max_diff = np.max(np.abs(logits_2.numpy() - logits_orig.numpy()))
        self.assertAlmostEqual(max_diff, 0.0, places=4)

    @require_torch
    @is_pt_tf_cross_test
    def test_encoder_decoder_save_load_from_encoder_decoder_from_pt(self):
        config = self.get_encoder_decoder_config_small()

        # create two random ViT/GPT2 models for vit-gpt2 & initialize weights (+cross_attention weights)
        encoder_pt = ViTModel(config.encoder).to(torch_device).eval()
        decoder_pt = GPT2LMHeadModel(config.decoder).to(torch_device).eval()

        encoder_decoder_pt = VisionEncoderDecoderModel(encoder=encoder_pt, decoder=decoder_pt).to(torch_device).eval()

        pixel_values = floats_tensor(
            [
                13,
                encoder_pt.config.num_channels,
                encoder_pt.config.image_size,
                encoder_pt.config.image_size,
            ]
        )
        decoder_input_ids = ids_tensor([13, 1], decoder_pt.config.vocab_size)

        pt_pixel_values = torch.tensor(pixel_values.numpy(), device=torch_device, dtype=torch.float)
        pt_decoder_input_ids = torch.tensor(decoder_input_ids.numpy(), device=torch_device, dtype=torch.long)

        logits_pt = encoder_decoder_pt(pixel_values=pt_pixel_values, decoder_input_ids=pt_decoder_input_ids).logits

        # PyTorch => TensorFlow
        with tempfile.TemporaryDirectory() as tmp_dirname_1, tempfile.TemporaryDirectory() as tmp_dirname_2:
            encoder_decoder_pt.encoder.save_pretrained(tmp_dirname_1)
            encoder_decoder_pt.decoder.save_pretrained(tmp_dirname_2)
            encoder_decoder_tf = TFVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                tmp_dirname_1, tmp_dirname_2, encoder_from_pt=True, decoder_from_pt=True
            )

        logits_tf = encoder_decoder_tf(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids).logits

        max_diff = np.max(np.abs(logits_pt.detach().cpu().numpy() - logits_tf.numpy()))
        self.assertAlmostEqual(max_diff, 0.0, places=3)

        # Make sure `from_pretrained` following `save_pretrained` work and give the same result
        # (See https://github.com/huggingface/transformers/pull/14016)
        with tempfile.TemporaryDirectory() as tmp_dirname:
            encoder_decoder_tf.save_pretrained(tmp_dirname)
            encoder_decoder_tf = TFVisionEncoderDecoderModel.from_pretrained(tmp_dirname)

            logits_tf_2 = encoder_decoder_tf(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids).logits

            max_diff = np.max(np.abs(logits_tf_2.numpy() - logits_tf.numpy()))
            self.assertAlmostEqual(max_diff, 0.0, places=3)

    @require_vision
    @slow
    def test_encoder_decoder_from_pretrained(self):
        load_weight_prefix = TFVisionEncoderDecoderModel.load_weight_prefix

        config = self.get_encoder_decoder_config()
        feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        decoder_tokenizer = AutoTokenizer.from_pretrained("../gpt2")

        img = prepare_img()
        pixel_values = feature_extractor(images=img, return_tensors="tf").pixel_values
        decoder_input_ids = decoder_tokenizer("Linda Davis", return_tensors="tf").input_ids

        with tempfile.TemporaryDirectory() as tmp_dirname:

            # Since most of HF's models don't have pretrained cross-attention layers, they are randomly
            # initialized even if we create models using `from_pretrained` method.
            # For the tests, the decoder need to be a model with pretrained cross-attention layers.
            # So we create pretrained models (without `load_weight_prefix`), save them, and later,
            # we load them using `from_pretrained`.
            # (we don't need to do this for encoder, but let's make the code more similar between encoder/decoder)
            encoder = TFAutoModel.from_pretrained("google/vit-base-patch16-224-in21k", name="encoder")
            # It's necessary to specify `add_cross_attention=True` here.
            decoder = TFAutoModelForCausalLM.from_pretrained(
                "../gpt2", is_decoder=True, add_cross_attention=True, name="decoder"
            )
            pretrained_encoder_dir = os.path.join(tmp_dirname, "pretrained_encoder")
            pretrained_decoder_dir = os.path.join(tmp_dirname, "pretrained_decoder")
            encoder.save_pretrained(pretrained_encoder_dir)
            decoder.save_pretrained(pretrained_decoder_dir)
            del encoder
            del decoder

            enc_dec_model = TFVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                pretrained_encoder_dir,
                pretrained_decoder_dir,
            )
            # check that the from pretrained methods work
            enc_dec_model.save_pretrained(tmp_dirname)
            enc_dec_model = TFVisionEncoderDecoderModel.from_pretrained(tmp_dirname)

            output = enc_dec_model(pixel_values, decoder_input_ids=decoder_input_ids, labels=decoder_input_ids)

            loss_pretrained = output.loss
            del enc_dec_model

            # Create the model using `__init__` with loaded ``pretrained`` encoder / decoder
            encoder = TFAutoModel.from_pretrained(
                pretrained_encoder_dir, load_weight_prefix=load_weight_prefix, name="encoder"
            )
            decoder = TFAutoModelForCausalLM.from_pretrained(
                pretrained_decoder_dir, load_weight_prefix=load_weight_prefix, name="decoder"
            )
            enc_dec_model = TFVisionEncoderDecoderModel(config=config, encoder=encoder, decoder=decoder)

        output = enc_dec_model(pixel_values, decoder_input_ids=decoder_input_ids, labels=decoder_input_ids)

        loss_init = output.loss

        max_diff = np.max(np.abs(loss_pretrained - loss_init))
        expected_diff = 0.0

        self.assertAlmostEqual(max_diff, expected_diff, places=4)


@require_vision
@require_tf
class TFViT2GPT2ModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_coco_en(self):

        loc = "ydshieh/vit-gpt2-coco-en"

        feature_extractor = ViTFeatureExtractor.from_pretrained(loc)
        tokenizer = AutoTokenizer.from_pretrained(loc)
        model = TFVisionEncoderDecoderModel.from_pretrained(loc)

        # We will verify our results on an image of cute cats
        img = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        pixel_values = feature_extractor(images=img, return_tensors="tf").pixel_values

        decoder_input_ids = tf.constant([[model.config.decoder_start_token_id]])

        logits = model(pixel_values, decoder_input_ids)[0].numpy()

        # verify the logits
        expected_shape = (1, 1, model.config.decoder.vocab_size)
        self.assertEqual(logits.shape, expected_shape)

        EXPECTED_LOGIT_SLICE = np.array(
            [
                -38.705807,
                -30.639929,
                -31.41903,
                -39.012012,
                -38.38696,
                -34.887207,
                -33.290855,
                -35.68447,
                -38.508484,
                -36.124645,
            ]
        )
        max_diff = np.amax(np.abs(logits[0, 0, :10] - EXPECTED_LOGIT_SLICE))
        self.assertLessEqual(max_diff, 1e-4)

        def generate_step(pixel_values):
            outputs = model.generate(
                pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True, output_scores=True
            )
            output_ids = outputs.sequences
            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            preds = [pred.strip() for pred in preds]

            return preds, outputs.scores.numpy()

        preds, scores = generate_step(pixel_values)

        # should produce
        # ["a cat laying on top of a couch next to another cat"]
        self.assertEqual(preds, ["a cat laying on top of a couch next to another cat"])
