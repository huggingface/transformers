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
"""Testing suite for the TensorFlow VisionEncoderDecoder model."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from transformers import is_tf_available, is_torch_available, is_vision_available
from transformers.testing_utils import (
    require_tf,
    require_vision,
    slow,
)

from ...test_modeling_tf_common import floats_tensor, ids_tensor
from ..gpt2.test_modeling_tf_gpt2 import TFGPT2ModelTester
from ..vit.test_modeling_tf_vit import TFViTModelTester


if is_tf_available():
    import tensorflow as tf

    from transformers import (
        AutoConfig,
        AutoImageProcessor,
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
    pass

if is_vision_available():
    from PIL import Image

    from transformers import ViTImageProcessor


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
        **kwargs,
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
        **kwargs,
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
        **kwargs,
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
        **kwargs,
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
        **kwargs,
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
        **kwargs,
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
        if hasattr(enc_dec_model.generation_config, "eos_token_id"):
            enc_dec_model.generation_config.eos_token_id = None

        # Bert does not have a bos token id, so use pad_token_id instead
        generated_output = enc_dec_model.generate(
            pixel_values, decoder_start_token_id=enc_dec_model.config.decoder.pad_token_id
        )
        self.assertEqual(
            tuple(generated_output.shape.as_list()), (pixel_values.shape[0],) + (decoder_config.max_length,)
        )

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
            "google/vit-base-patch16-224-in21k", "openai-community/gpt2"
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
            "google/vit-base-patch16-224-in21k", "openai-community/gpt2"
        )

    def get_decoder_config(self):
        config = AutoConfig.from_pretrained("openai-community/gpt2")
        config.is_decoder = True
        config.add_cross_attention = True
        return config

    def get_encoderdecoder_model(self):
        return TFVisionEncoderDecoderModel.from_pretrained("ydshieh/vit-gpt2-coco-en")

    def get_encoder_decoder_models(self):
        encoder_model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k", name="encoder")
        decoder_model = TFGPT2LMHeadModel.from_pretrained(
            "openai-community/gpt2", config=self.get_decoder_config(), name="decoder"
        )
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
        decoder_config = AutoConfig.from_pretrained("openai-community/gpt2", is_decoder=True, add_cross_attention=True)
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
        encoder.build_in_name_scope()
        decoder = TFGPT2LMHeadModel(config.decoder)
        decoder.build_in_name_scope()

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

    @require_vision
    @slow
    def test_encoder_decoder_from_pretrained(self):
        load_weight_prefix = TFVisionEncoderDecoderModel.load_weight_prefix

        config = self.get_encoder_decoder_config()
        image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        decoder_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

        img = prepare_img()
        pixel_values = image_processor(images=img, return_tensors="tf").pixel_values
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
                "openai-community/gpt2", is_decoder=True, add_cross_attention=True, name="decoder"
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
            enc_dec_model.build_in_name_scope()
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

        image_processor = ViTImageProcessor.from_pretrained(loc)
        tokenizer = AutoTokenizer.from_pretrained(loc)
        model = TFVisionEncoderDecoderModel.from_pretrained(loc)

        # We will verify our results on an image of cute cats
        img = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        pixel_values = image_processor(images=img, return_tensors="tf").pixel_values

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
            outputs = model.generate(pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True)
            output_ids = outputs.sequences
            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            preds = [pred.strip() for pred in preds]

            return preds

        preds = generate_step(pixel_values)

        # should produce
        # ["a cat laying on top of a couch next to another cat"]
        self.assertEqual(preds, ["a cat laying on top of a couch next to another cat"])
