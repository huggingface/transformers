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

from datasets import load_dataset

from transformers.file_utils import cached_property, is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision, slow, torch_device

from .test_modeling_bert import BertModelTester
from .test_modeling_common import floats_tensor, ids_tensor, random_attention_mask
from .test_modeling_deit import DeiTModelTester
from .test_modeling_trocr import TrOCRStandaloneDecoderModelTester
from .test_modeling_vit import ViTModelTester


if is_torch_available():
    import numpy as np
    import torch

    from transformers import (
        BertLMHeadModel,
        DeiTModel,
        TrOCRForCausalLM,
        VisionEncoderDecoderConfig,
        VisionEncoderDecoderModel,
        ViTModel,
    )
    from transformers.modeling_outputs import BaseModelOutput
    from transformers.models.vit.modeling_vit import to_2tuple


if is_vision_available():
    from PIL import Image

    from transformers import TrOCRProcessor


@require_torch
class EncoderDecoderMixin:
    def get_encoder_decoder_model(self, config, decoder_config):
        pass

    def prepare_config_and_inputs(self):
        pass

    def get_pretrained_model_and_inputs(self):
        pass

    def check_encoder_decoder_model_from_pretrained_configs(
        self,
        config,
        attention_mask,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        pixel_values=None,
        **kwargs
    ):
        encoder_decoder_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config, decoder_config)
        self.assertTrue(encoder_decoder_config.decoder.is_decoder)

        enc_dec_model = VisionEncoderDecoderModel(encoder_decoder_config)
        enc_dec_model.to(torch_device)
        enc_dec_model.eval()

        self.assertTrue(enc_dec_model.config.is_encoder_decoder)

        outputs_encoder_decoder = enc_dec_model(
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )

    def check_encoder_decoder_model(
        self,
        config,
        attention_mask,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        pixel_values=None,
        **kwargs
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = VisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        self.assertTrue(enc_dec_model.config.decoder.is_decoder)
        self.assertTrue(enc_dec_model.config.decoder.add_cross_attention)
        self.assertTrue(enc_dec_model.config.is_encoder_decoder)
        enc_dec_model.to(torch_device)
        outputs_encoder_decoder = enc_dec_model(
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
        )
        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )
        encoder_outputs = BaseModelOutput(last_hidden_state=outputs_encoder_decoder.encoder_hidden_states[-1])
        outputs_encoder_decoder = enc_dec_model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )

    def check_encoder_decoder_model_from_pretrained(
        self,
        config,
        attention_mask,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        return_dict,
        pixel_values=None,
        **kwargs
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        kwargs = {"encoder_model": encoder_model, "decoder_model": decoder_model, "return_dict": return_dict}
        enc_dec_model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(**kwargs)
        enc_dec_model.to(torch_device)
        outputs_encoder_decoder = enc_dec_model(
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )

    def check_save_and_load(
        self,
        config,
        attention_mask,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        pixel_values=None,
        **kwargs
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = VisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)
        enc_dec_model.eval()
        with torch.no_grad():
            outputs = enc_dec_model(
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            out_2 = outputs[0].cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as tmpdirname:
                enc_dec_model.save_pretrained(tmpdirname)
                enc_dec_model = VisionEncoderDecoderModel.from_pretrained(tmpdirname)
                enc_dec_model.to(torch_device)

                after_outputs = enc_dec_model(
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                )
                out_1 = after_outputs[0].cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    def check_save_and_load_encoder_decoder_model(
        self,
        config,
        attention_mask,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        pixel_values=None,
        **kwargs
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = VisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)
        enc_dec_model.eval()
        with torch.no_grad():
            outputs = enc_dec_model(
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            out_2 = outputs[0].cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as encoder_tmp_dirname, tempfile.TemporaryDirectory() as decoder_tmp_dirname:
                enc_dec_model.encoder.save_pretrained(encoder_tmp_dirname)
                enc_dec_model.decoder.save_pretrained(decoder_tmp_dirname)
                VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                    encoder_pretrained_model_name_or_path=encoder_tmp_dirname,
                    decoder_pretrained_model_name_or_path=decoder_tmp_dirname,
                )

                after_outputs = enc_dec_model(
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                )
                out_1 = after_outputs[0].cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    def check_encoder_decoder_model_output_attentions(
        self,
        config,
        attention_mask,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        labels=None,
        pixel_values=None,
        **kwargs
    ):
        # make the decoder inputs a different shape from the encoder inputs to harden the test
        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_attention_mask = decoder_attention_mask[:, :-1]
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = VisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)
        outputs_encoder_decoder = enc_dec_model(
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=True,
        )

        encoder_attentions = outputs_encoder_decoder["encoder_attentions"]
        self.assertEqual(len(encoder_attentions), config.num_hidden_layers)

        # in ViT, the seq_len equals the number of patches + 1 (we add 1 for the [CLS] token)
        image_size = to_2tuple(encoder_model.config.image_size)
        patch_size = to_2tuple(encoder_model.config.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        seq_len = num_patches + 1
        self.assertEqual(encoder_attentions[0].shape[-3:], (config.num_attention_heads, seq_len, seq_len))

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

        cross_attention_input_seq_len = decoder_input_ids.shape[-1]
        self.assertEqual(
            cross_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, cross_attention_input_seq_len, seq_len),
        )

    def check_encoder_decoder_model_generate(self, config, decoder_config, pixel_values=None, **kwargs):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = VisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)

        inputs = pixel_values

        # Bert does not have a bos token id, so use pad_token_id instead
        generated_output = enc_dec_model.generate(
            inputs, decoder_start_token_id=enc_dec_model.config.decoder.pad_token_id
        )
        self.assertEqual(generated_output.shape, (inputs.shape[0],) + (decoder_config.max_length,))

    def test_encoder_decoder_model(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model(**input_ids_dict)

    def test_encoder_decoder_model_from_pretrained_configs(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_from_pretrained_configs(**input_ids_dict)

    def test_encoder_decoder_model_from_pretrained(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_from_pretrained(**input_ids_dict, return_dict=False)

    def test_encoder_decoder_model_from_pretrained_return_dict(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_from_pretrained(**input_ids_dict, return_dict=True)

    def test_save_and_load_from_pretrained(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_save_and_load(**input_ids_dict)

    def test_save_and_load_from_encoder_decoder_pretrained(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_save_and_load_encoder_decoder_model(**input_ids_dict)

    def test_encoder_decoder_model_output_attentions(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_output_attentions(**input_ids_dict)

    def test_encoder_decoder_model_generate(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_generate(**input_ids_dict)

    @slow
    def test_real_model_save_load_from_pretrained(self):
        model_2, inputs = self.get_pretrained_model_and_inputs()
        model_2.to(torch_device)

        with torch.no_grad():
            outputs = model_2(**inputs)
            out_2 = outputs[0].cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as tmp_dirname:
                model_2.save_pretrained(tmp_dirname)
                model_1 = VisionEncoderDecoderModel.from_pretrained(tmp_dirname)
                model_1.to(torch_device)

                after_outputs = model_1(**inputs)
                out_1 = after_outputs[0].cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)


@require_torch
class DeiT2RobertaModelTest(EncoderDecoderMixin, unittest.TestCase):
    def get_pretrained_model_and_inputs(self):
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "hf-internal-testing/tiny-random-deit", "hf-internal-testing/tiny-random-roberta"
        )
        batch_size = 13
        pixel_values = floats_tensor(
            [
                batch_size,
                model.encoder.config.num_channels,
                model.encoder.config.image_size,
                model.encoder.config.image_size,
            ]
        )
        # for DEiT, the sequence length is equal to the number of patches + 2 (for the [CLS] and distillation tokens)
        seq_len = (model.encoder.config.image_size // model.encoder.config.patch_size) ** 2 + 2
        attention_mask = random_attention_mask([batch_size, seq_len])
        decoder_input_ids = ids_tensor([batch_size, 4], model.decoder.config.vocab_size)
        decoder_attention_mask = random_attention_mask([batch_size, 4])
        inputs = {
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }

        return model, inputs

    def check_encoder_decoder_model_output_attentions(
        self,
        config,
        attention_mask,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        labels=None,
        pixel_values=None,
        **kwargs
    ):
        # make the decoder inputs a different shape from the encoder inputs to harden the test
        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_attention_mask = decoder_attention_mask[:, :-1]
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = VisionEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)
        outputs_encoder_decoder = enc_dec_model(
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=True,
        )

        encoder_attentions = outputs_encoder_decoder["encoder_attentions"]
        self.assertEqual(len(encoder_attentions), config.num_hidden_layers)

        # in DEiT, the seq_len equals the number of patches + 2 (we add 2 for the [CLS] and distillation tokens)
        image_size = to_2tuple(encoder_model.config.image_size)
        patch_size = to_2tuple(encoder_model.config.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        seq_len = num_patches + 2
        self.assertEqual(encoder_attentions[0].shape[-3:], (config.num_attention_heads, seq_len, seq_len))

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

        cross_attention_input_seq_len = decoder_input_ids.shape[-1]
        self.assertEqual(
            cross_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, cross_attention_input_seq_len, seq_len),
        )

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = DeiTModel(config).eval()
        decoder_model = BertLMHeadModel(decoder_config).eval()
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        bert_model_tester = BertModelTester(self)
        deit_model_tester = DeiTModelTester(self)
        encoder_config_and_inputs = deit_model_tester.prepare_config_and_inputs()
        decoder_config_and_inputs = bert_model_tester.prepare_config_and_inputs_for_decoder()
        config, pixel_values, _ = encoder_config_and_inputs
        input_mask = None  # TODO add once attention_mask is supported for vision models
        (
            decoder_config,
            decoder_input_ids,
            decoder_token_type_ids,
            decoder_input_mask,
            decoder_sequence_labels,
            decoder_token_labels,
            decoder_choice_labels,
            encoder_attention_mask,
            _,
        ) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        return {
            "config": config,
            "pixel_values": pixel_values,
            "attention_mask": input_mask,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_token_type_ids": decoder_token_type_ids,
            "decoder_attention_mask": decoder_input_mask,
            "decoder_sequence_labels": decoder_sequence_labels,
            "decoder_token_labels": decoder_token_labels,
            "decoder_choice_labels": decoder_choice_labels,
            "labels": decoder_token_labels,
        }


@require_torch
class ViT2BertModelTest(EncoderDecoderMixin, unittest.TestCase):
    def get_pretrained_model_and_inputs(self):
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "hf-internal-testing/tiny-random-vit", "hf-internal-testing/tiny-bert"
        )
        batch_size = 13
        pixel_values = floats_tensor(
            [
                batch_size,
                model.encoder.config.num_channels,
                model.encoder.config.image_size,
                model.encoder.config.image_size,
            ]
        )
        # for ViT, the sequence length is equal to the number of patches + 1 (for the [CLS] token)
        seq_len = (model.encoder.config.image_size // model.encoder.config.patch_size) ** 2 + 1
        attention_mask = random_attention_mask([batch_size, seq_len])
        decoder_input_ids = ids_tensor([batch_size, 4], model.decoder.config.vocab_size)
        decoder_attention_mask = random_attention_mask([batch_size, 4])
        inputs = {
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }

        return model, inputs

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = ViTModel(config).eval()
        decoder_model = BertLMHeadModel(decoder_config).eval()
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        vit_model_tester = ViTModelTester(self)
        bert_model_tester = BertModelTester(self)
        encoder_config_and_inputs = vit_model_tester.prepare_config_and_inputs()
        decoder_config_and_inputs = bert_model_tester.prepare_config_and_inputs_for_decoder()

        config, pixel_values, _ = encoder_config_and_inputs
        input_mask = None  # TODO add once attention_mask is supported for vision models

        (
            decoder_config,
            decoder_input_ids,
            decoder_token_type_ids,
            decoder_input_mask,
            decoder_sequence_labels,
            decoder_token_labels,
            decoder_choice_labels,
            encoder_attention_mask,
            _,
        ) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        return {
            "config": config,
            "pixel_values": pixel_values,
            "attention_mask": input_mask,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_token_type_ids": decoder_token_type_ids,
            "decoder_attention_mask": decoder_input_mask,
            "decoder_sequence_labels": decoder_sequence_labels,
            "decoder_token_labels": decoder_token_labels,
            "decoder_choice_labels": decoder_choice_labels,
            "labels": decoder_token_labels,
        }


@require_torch
class ViT2TrOCR(EncoderDecoderMixin, unittest.TestCase):
    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = ViTModel(config).eval()
        decoder_model = TrOCRForCausalLM(decoder_config).eval()
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester_encoder = ViTModelTester(self, batch_size=13)
        model_tester_decoder = TrOCRStandaloneDecoderModelTester(
            self, batch_size=13, d_model=32, max_position_embeddings=512
        )
        encoder_config_and_inputs = model_tester_encoder.prepare_config_and_inputs()
        decoder_config_and_inputs = model_tester_decoder.prepare_config_and_inputs()
        config, pixel_values, _ = encoder_config_and_inputs
        input_mask = None  # TODO add once attention_mask is supported for vision models
        (decoder_config, decoder_input_ids, decoder_attention_mask, _) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        #  disable cache for now
        decoder_config.use_cache = False
        return {
            "config": config,
            "pixel_values": pixel_values,
            "attention_mask": input_mask,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }

    # there are no published pretrained TrOCR checkpoints for now
    def test_real_model_save_load_from_pretrained(self):
        pass


@require_vision
@require_torch
class TrOCRModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten") if is_vision_available() else None

    @slow
    def test_inference_handwritten(self):
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(torch_device)

        ds = load_dataset("hf-internal-testing/fixtures_ocr", split="test")
        image = Image.open(ds[0]["file"]).convert("RGB")

        processor = self.default_processor
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(torch_device)

        # forward pass
        decoder_input_ids = torch.tensor([[model.config.decoder.decoder_start_token_id]]).to(torch_device)
        outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits

        # verify the logits
        expected_shape = torch.Size((1, 1, model.decoder.config.vocab_size))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [-1.4502, -4.6683, -0.5347, -2.9291, 9.1435, -3.0571, 8.9764, 1.7560, 8.7358, -1.5311]
        ).to(torch_device)

        self.assertTrue(torch.allclose(logits[0, 0, :10], expected_slice, atol=1e-4))

    @slow
    def test_inference_printed(self):
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(torch_device)

        ds = load_dataset("hf-internal-testing/fixtures_ocr", split="test")
        image = Image.open(ds[1]["file"]).convert("RGB")

        processor = self.default_processor
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(torch_device)

        # forward pass
        decoder_input_ids = torch.tensor([[model.config.decoder.decoder_start_token_id]]).to(torch_device)
        outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits

        # verify the logits
        expected_shape = torch.Size((1, 1, model.decoder.config.vocab_size))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [-5.6816, -5.8388, 1.1398, -6.9034, 6.8505, -2.4393, 1.2284, -1.0232, -1.9661, -3.9210]
        ).to(torch_device)

        self.assertTrue(torch.allclose(logits[0, 0, :10], expected_slice, atol=1e-4))
