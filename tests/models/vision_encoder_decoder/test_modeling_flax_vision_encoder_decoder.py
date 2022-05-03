# coding=utf-8
# Copyright 2021 HuggingFace Inc. team.
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


import tempfile
import unittest

import numpy as np

from transformers import is_flax_available, is_torch_available, is_vision_available
from transformers.testing_utils import is_pt_flax_cross_test, require_flax, require_vision, slow, torch_device

from ...test_modeling_flax_common import floats_tensor, ids_tensor
from ..gpt2.test_modeling_flax_gpt2 import FlaxGPT2ModelTester
from ..vit.test_modeling_flax_vit import FlaxViTModelTester


if is_flax_available():
    from transformers import (
        AutoTokenizer,
        FlaxGPT2LMHeadModel,
        FlaxVisionEncoderDecoderModel,
        FlaxViTModel,
        VisionEncoderDecoderConfig,
    )
    from transformers.modeling_flax_pytorch_utils import (
        convert_pytorch_state_dict_to_flax,
        load_flax_weights_in_pytorch_model,
    )

if is_torch_available():
    import torch

    from transformers import VisionEncoderDecoderModel

if is_vision_available():
    from PIL import Image

    from transformers import ViTFeatureExtractor


@require_flax
class FlaxEncoderDecoderMixin:
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

        enc_dec_model = FlaxVisionEncoderDecoderModel(encoder_decoder_config)

        self.assertTrue(enc_dec_model.config.is_encoder_decoder)

        outputs_encoder_decoder = enc_dec_model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
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
        enc_dec_model = FlaxVisionEncoderDecoderModel.from_encoder_decoder_pretrained(**kwargs)
        outputs_encoder_decoder = enc_dec_model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
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
        kwargs = {"encoder_model": encoder_model, "decoder_model": decoder_model}
        enc_dec_model = FlaxVisionEncoderDecoderModel.from_encoder_decoder_pretrained(**kwargs)

        outputs = enc_dec_model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        out_2 = np.array(outputs[0])
        out_2[np.isnan(out_2)] = 0

        with tempfile.TemporaryDirectory() as tmpdirname:
            enc_dec_model.save_pretrained(tmpdirname)
            FlaxVisionEncoderDecoderModel.from_pretrained(tmpdirname)

            after_outputs = enc_dec_model(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            out_1 = np.array(after_outputs[0])
            out_1[np.isnan(out_1)] = 0
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

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
        kwargs = {"encoder_model": encoder_model, "decoder_model": decoder_model}
        enc_dec_model = FlaxVisionEncoderDecoderModel.from_encoder_decoder_pretrained(**kwargs)
        outputs_encoder_decoder = enc_dec_model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=True,
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
        kwargs = {"encoder_model": encoder_model, "decoder_model": decoder_model}
        enc_dec_model = FlaxVisionEncoderDecoderModel.from_encoder_decoder_pretrained(**kwargs)

        pad_token_id = enc_dec_model.config.decoder.pad_token_id
        eos_token_id = enc_dec_model.config.decoder.eos_token_id
        decoder_start_token_id = enc_dec_model.config.decoder.decoder_start_token_id

        # Copied from generation_utils (GPT2 doesn't have `pad_token_id`)
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = enc_dec_model.config.decoder.bos_token_id

        # Bert does not have a bos token id, so use pad_token_id instead
        # Copied from `test_modeling_encoder_decoder.py`
        if decoder_start_token_id is None:
            decoder_start_token_id = pad_token_id

        generated_output = enc_dec_model.generate(
            pixel_values,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
        )
        generated_sequences = generated_output.sequences
        self.assertEqual(generated_sequences.shape, (pixel_values.shape[0],) + (decoder_config.max_length,))

    def check_pt_flax_equivalence(self, pt_model, fx_model, inputs_dict):

        pt_model.to(torch_device)
        pt_model.eval()

        # prepare inputs
        flax_inputs = inputs_dict
        pt_inputs = {k: torch.tensor(v.tolist()) for k, v in flax_inputs.items()}

        with torch.no_grad():
            pt_outputs = pt_model(**pt_inputs).to_tuple()

        fx_outputs = fx_model(**inputs_dict).to_tuple()
        self.assertEqual(len(fx_outputs), len(pt_outputs), "Output lengths differ between Flax and PyTorch")
        for fx_output, pt_output in zip(fx_outputs, pt_outputs):
            self.assert_almost_equals(fx_output, pt_output.numpy(), 1e-5)

        # PT -> Flax
        with tempfile.TemporaryDirectory() as tmpdirname:
            pt_model.save_pretrained(tmpdirname)
            fx_model_loaded = FlaxVisionEncoderDecoderModel.from_pretrained(tmpdirname, from_pt=True)

        fx_outputs_loaded = fx_model_loaded(**inputs_dict).to_tuple()
        self.assertEqual(len(fx_outputs_loaded), len(pt_outputs), "Output lengths differ between Flax and PyTorch")
        for fx_output_loaded, pt_output in zip(fx_outputs_loaded, pt_outputs):
            self.assert_almost_equals(fx_output_loaded, pt_output.numpy(), 1e-5)

        # Flax -> PT
        with tempfile.TemporaryDirectory() as tmpdirname:
            fx_model.save_pretrained(tmpdirname)
            pt_model_loaded = VisionEncoderDecoderModel.from_pretrained(tmpdirname, from_flax=True)

        pt_model_loaded.to(torch_device)
        pt_model_loaded.eval()

        with torch.no_grad():
            pt_outputs_loaded = pt_model_loaded(**pt_inputs).to_tuple()

        self.assertEqual(len(fx_outputs), len(pt_outputs_loaded), "Output lengths differ between Flax and PyTorch")
        for fx_output, pt_output_loaded in zip(fx_outputs, pt_outputs_loaded):
            self.assert_almost_equals(fx_output, pt_output_loaded.numpy(), 1e-5)

    def check_equivalence_pt_to_flax(self, config, decoder_config, inputs_dict):

        encoder_decoder_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config, decoder_config)

        pt_model = VisionEncoderDecoderModel(encoder_decoder_config)
        fx_model = FlaxVisionEncoderDecoderModel(encoder_decoder_config)

        fx_state = convert_pytorch_state_dict_to_flax(pt_model.state_dict(), fx_model)
        fx_model.params = fx_state

        self.check_pt_flax_equivalence(pt_model, fx_model, inputs_dict)

    def check_equivalence_flax_to_pt(self, config, decoder_config, inputs_dict):

        encoder_decoder_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config, decoder_config)

        pt_model = VisionEncoderDecoderModel(encoder_decoder_config)
        fx_model = FlaxVisionEncoderDecoderModel(encoder_decoder_config)

        pt_model = load_flax_weights_in_pytorch_model(pt_model, fx_model.params)

        self.check_pt_flax_equivalence(pt_model, fx_model, inputs_dict)

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

    def test_encoder_decoder_model_output_attentions(self):
        config_inputs_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_output_attentions(**config_inputs_dict)

    def test_encoder_decoder_model_generate(self):
        config_inputs_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_generate(**config_inputs_dict)

    def assert_almost_equals(self, a: np.ndarray, b: np.ndarray, tol: float):
        diff = np.abs((a - b)).max()
        self.assertLessEqual(diff, tol, f"Difference between torch and flax is {diff} (>= {tol}).")

    @is_pt_flax_cross_test
    def test_pt_flax_equivalence(self):

        config_inputs_dict = self.prepare_config_and_inputs()
        config = config_inputs_dict.pop("config")
        decoder_config = config_inputs_dict.pop("decoder_config")

        inputs_dict = config_inputs_dict
        # `encoder_hidden_states` is not used in model call/forward
        del inputs_dict["encoder_hidden_states"]

        # Avoid the case where a sequence has no place to attend (after combined with the causal attention mask)
        batch_size = inputs_dict["decoder_attention_mask"].shape[0]
        inputs_dict["decoder_attention_mask"] = np.concatenate(
            [np.ones(shape=(batch_size, 1)), inputs_dict["decoder_attention_mask"][:, 1:]], axis=1
        )

        # Flax models don't use the `use_cache` option and cache is not returned as a default.
        # So we disable `use_cache` here for PyTorch model.
        decoder_config.use_cache = False

        self.assertTrue(decoder_config.cross_attention_hidden_size is None)

        # check without `enc_to_dec_proj` projection
        self.assertTrue(config.hidden_size == decoder_config.hidden_size)
        self.check_equivalence_pt_to_flax(config, decoder_config, inputs_dict)
        self.check_equivalence_flax_to_pt(config, decoder_config, inputs_dict)

        # check `enc_to_dec_proj` work as expected
        decoder_config.hidden_size = decoder_config.hidden_size * 2
        self.assertTrue(config.hidden_size != decoder_config.hidden_size)
        self.check_equivalence_pt_to_flax(config, decoder_config, inputs_dict)
        self.check_equivalence_flax_to_pt(config, decoder_config, inputs_dict)

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
            model_1 = FlaxVisionEncoderDecoderModel.from_pretrained(tmp_dirname)

            after_outputs = model_1(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
            )
            out_1 = np.array(after_outputs[0])
            out_1[np.isnan(out_1)] = 0
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)


@require_flax
class FlaxViT2GPT2EncoderDecoderModelTest(FlaxEncoderDecoderMixin, unittest.TestCase):
    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = FlaxViTModel(config)
        decoder_model = FlaxGPT2LMHeadModel(decoder_config)
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester_encoder = FlaxViTModelTester(self, batch_size=13)
        model_tester_decoder = FlaxGPT2ModelTester(self, batch_size=13)
        encoder_config_and_inputs = model_tester_encoder.prepare_config_and_inputs()
        decoder_config_and_inputs = model_tester_decoder.prepare_config_and_inputs_for_decoder()
        (config, pixel_values) = encoder_config_and_inputs
        (
            decoder_config,
            decoder_input_ids,
            decoder_attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
        ) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        return {
            "config": config,
            "pixel_values": pixel_values,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "encoder_hidden_states": encoder_hidden_states,  # This is not used in the tests.
        }

    def get_pretrained_model(self):
        return FlaxVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "google/vit-base-patch16-224-in21k", "gpt2"
        )


@require_flax
class FlaxVisionEncoderDecoderModelTest(unittest.TestCase):
    def get_from_encoderdecoder_pretrained_model(self):
        return FlaxVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "google/vit-base-patch16-224-in21k", "gpt2"
        )

    def _check_configuration_tie(self, model):

        module = model.module.bind(model.params)

        assert id(module.decoder.config) == id(model.config.decoder)
        assert id(module.encoder.config) == id(model.config.encoder)

    @slow
    def test_configuration_tie(self):
        model = self.get_from_encoderdecoder_pretrained_model()
        self._check_configuration_tie(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_vision
@require_flax
class FlaxViT2GPT2ModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_coco_en(self):

        loc = "ydshieh/vit-gpt2-coco-en"

        feature_extractor = ViTFeatureExtractor.from_pretrained(loc)
        tokenizer = AutoTokenizer.from_pretrained(loc)
        model = FlaxVisionEncoderDecoderModel.from_pretrained(loc)

        img = prepare_img()
        pixel_values = feature_extractor(images=img, return_tensors="np").pixel_values

        decoder_input_ids = np.array([[model.config.decoder_start_token_id]])
        logits = model(pixel_values, decoder_input_ids)[0]
        logits = np.array(logits)

        # verify the logits
        expected_shape = (1, 1, model.config.decoder.vocab_size)
        self.assertEqual(logits.shape, expected_shape)

        EXPECTED_LOGIT_SLICE = np.array(
            [
                -38.705837,
                -30.639936,
                -31.41905,
                -39.01204,
                -38.38698,
                -34.887215,
                -33.29087,
                -35.684475,
                -38.50852,
                -36.124676,
            ]
        )
        max_diff = np.amax(np.abs(logits[0, 0, :10] - EXPECTED_LOGIT_SLICE))
        self.assertLessEqual(max_diff, 1e-4)

        def generate_step(pixel_values):

            outputs = model.generate(pixel_values, max_length=16, num_beams=4)
            output_ids = outputs.sequences
            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            preds = [pred.strip() for pred in preds]

            return preds, outputs.scores

        preds, scores = generate_step(pixel_values)

        EXPECTED_SCORES = np.array([-0.59563464])
        scores = np.array(scores)
        max_diff = np.amax(np.abs(scores - EXPECTED_SCORES))
        self.assertLessEqual(max_diff, 1e-4)

        # should produce
        # ["a cat laying on top of a couch next to another cat"]
        self.assertEqual(preds, ["a cat laying on top of a couch next to another cat"])
