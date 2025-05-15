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

import tempfile
import unittest

import numpy as np

from transformers import is_flax_available, is_torch_available
from transformers.testing_utils import require_flax, slow

from ...test_modeling_flax_common import floats_tensor, ids_tensor, random_attention_mask
from ..bart.test_modeling_flax_bart import FlaxBartStandaloneDecoderModelTester
from ..bert.test_modeling_flax_bert import FlaxBertModelTester
from ..gpt2.test_modeling_flax_gpt2 import FlaxGPT2ModelTester
from ..wav2vec2.test_modeling_flax_wav2vec2 import FlaxWav2Vec2ModelTester


if is_flax_available():
    import jax
    import jax.numpy as jnp
    from flax.training.common_utils import onehot
    from flax.traverse_util import flatten_dict

    from transformers import (
        FlaxBartForCausalLM,
        FlaxBertForCausalLM,
        FlaxGPT2LMHeadModel,
        FlaxSpeechEncoderDecoderModel,
        FlaxWav2Vec2Model,
        SpeechEncoderDecoderConfig,
    )
    from transformers.modeling_flax_outputs import FlaxBaseModelOutput

if is_torch_available():
    from transformers import SpeechEncoderDecoderModel


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
        inputs,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs,
    ):
        encoder_decoder_config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(config, decoder_config)
        self.assertTrue(encoder_decoder_config.decoder.is_decoder)

        enc_dec_model = FlaxSpeechEncoderDecoderModel(encoder_decoder_config)

        self.assertTrue(enc_dec_model.config.is_encoder_decoder)
        self.assertFalse(enc_dec_model.config.tie_word_embeddings)

        outputs_encoder_decoder = enc_dec_model(
            inputs=inputs,
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
        inputs,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs,
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = SpeechEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        self.assertTrue(enc_dec_model.config.decoder.is_decoder)
        self.assertTrue(enc_dec_model.config.decoder.add_cross_attention)
        self.assertTrue(enc_dec_model.config.is_encoder_decoder)

        outputs_encoder_decoder = enc_dec_model(
            inputs=inputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )

        encoder_outputs = FlaxBaseModelOutput(last_hidden_state=outputs_encoder_decoder.encoder_hidden_states[-1])

        outputs_encoder_decoder = enc_dec_model(
            attention_mask, decoder_input_ids, decoder_attention_mask, encoder_outputs=encoder_outputs
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )

    def check_encoder_decoder_model_from_pretrained(
        self,
        config,
        inputs,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        return_dict,
        **kwargs,
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        kwargs = {"encoder_model": encoder_model, "decoder_model": decoder_model, "return_dict": return_dict}
        enc_dec_model = FlaxSpeechEncoderDecoderModel.from_encoder_decoder_pretrained(**kwargs)
        outputs_encoder_decoder = enc_dec_model(
            inputs=inputs,
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
        inputs,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs,
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        kwargs = {"encoder_model": encoder_model, "decoder_model": decoder_model}
        enc_dec_model = FlaxSpeechEncoderDecoderModel.from_encoder_decoder_pretrained(**kwargs)

        outputs = enc_dec_model(
            inputs=inputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        out_2 = np.array(outputs[0])
        out_2[np.isnan(out_2)] = 0

        with tempfile.TemporaryDirectory() as tmpdirname:
            enc_dec_model.save_pretrained(tmpdirname)
            FlaxSpeechEncoderDecoderModel.from_pretrained(tmpdirname)

            after_outputs = enc_dec_model(
                inputs=inputs,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            out_1 = np.array(after_outputs[0])
            out_1[np.isnan(out_1)] = 0
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 4e-2)

    def check_encoder_decoder_model_from_encoder_decoder_pretrained(
        self,
        config,
        inputs,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs,
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        # assert that loading encoder and decoder models from configs has been correctly executed
        self.assertEqual(config.add_adapter, encoder_model.config.add_adapter)
        self.assertEqual(decoder_config.use_cache, decoder_model.config.use_cache)

        with tempfile.TemporaryDirectory() as enc_tmpdir:
            with tempfile.TemporaryDirectory() as dec_tmpdir:
                encoder_model.save_pretrained(enc_tmpdir)
                decoder_model.save_pretrained(dec_tmpdir)
                # load a model from pretrained encoder and decoder checkpoints, setting one encoder and one decoder kwarg opposite to that specified in their respective configs
                enc_dec_model = FlaxSpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
                    encoder_pretrained_model_name_or_path=enc_tmpdir,
                    decoder_pretrained_model_name_or_path=dec_tmpdir,
                    encoder_add_adapter=not config.add_adapter,
                    decoder_use_cache=not decoder_config.use_cache,
                )

        # assert that setting encoder and decoder kwargs opposite to those in the configs has correctly been applied
        self.assertNotEqual(config.add_adapter, enc_dec_model.config.encoder.add_adapter)
        self.assertNotEqual(decoder_config.use_cache, enc_dec_model.config.decoder.use_cache)

        outputs_encoder_decoder = enc_dec_model(
            inputs=inputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )

    def check_encoder_decoder_model_output_attentions(
        self,
        config,
        inputs,
        attention_mask,
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
        kwargs = {"encoder_model": encoder_model, "decoder_model": decoder_model}
        enc_dec_model = FlaxSpeechEncoderDecoderModel.from_encoder_decoder_pretrained(**kwargs)
        outputs_encoder_decoder = enc_dec_model(
            inputs=inputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=True,
        )

        encoder_attentions = outputs_encoder_decoder["encoder_attentions"]
        self.assertEqual(len(encoder_attentions), config.num_hidden_layers)

        seq_len = enc_dec_model._get_feat_extract_output_lengths(inputs.shape[1])
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

    def check_encoder_decoder_model_generate(self, inputs, config, decoder_config, **kwargs):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        kwargs = {"encoder_model": encoder_model, "decoder_model": decoder_model}
        enc_dec_model = FlaxSpeechEncoderDecoderModel.from_encoder_decoder_pretrained(**kwargs)

        pad_token_id = enc_dec_model.config.decoder.pad_token_id
        eos_token_id = enc_dec_model.config.decoder.eos_token_id
        decoder_start_token_id = enc_dec_model.config.decoder.decoder_start_token_id

        # Copied from generation.utils (GPT2 doesn't have `pad_token_id`)
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = enc_dec_model.config.decoder.bos_token_id

        # Bert does not have a bos token id, so use pad_token_id instead
        # Copied from `test_modeling_encoder_decoder.py`
        if decoder_start_token_id is None:
            decoder_start_token_id = pad_token_id

        generated_output = enc_dec_model.generate(
            inputs,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
        )
        generated_sequences = generated_output.sequences
        self.assertEqual(generated_sequences.shape, (inputs.shape[0],) + (decoder_config.max_length,))

    def check_freeze_feature_encoder(
        self,
        config,
        inputs,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs,
    ):
        encoder_decoder_config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(config, decoder_config)
        enc_dec_model = FlaxSpeechEncoderDecoderModel(encoder_decoder_config)
        params = enc_dec_model.params

        def cross_entropy(logits, labels):
            return -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)

        # define a dummy loss function for computing the loss over a forward pass
        def compute_loss(
            params,
            inputs,
            attention_mask,
            decoder_input_ids,
            freeze_feature_encoder: bool = False,
        ):
            outputs_enc_dec = enc_dec_model(
                inputs=inputs,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                freeze_feature_encoder=freeze_feature_encoder,
                params=params,
            )
            logits = outputs_enc_dec.logits
            vocab_size = logits.shape[-1]
            loss = cross_entropy(logits, onehot(labels=decoder_input_ids, num_classes=vocab_size)).sum()
            return (loss, logits)

        # transform the loss function to get the gradients
        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)

        # compute the loss, logits, and gradients for the unfrozen model
        (loss, logits), grads = grad_fn(
            params, inputs, attention_mask, decoder_input_ids, freeze_feature_encoder=False
        )

        # compare to the loss, logits and gradients for the frozen model
        (loss_frozen, logits_frozen), grads_frozen = grad_fn(
            params, inputs, attention_mask, decoder_input_ids, freeze_feature_encoder=True
        )

        # ensure that the logits and losses remain precisely equal
        self.assertTrue((logits == logits_frozen).all())
        self.assertEqual(loss, loss_frozen)

        grads = flatten_dict(grads)
        grads_frozen = flatten_dict(grads_frozen)

        # ensure that the dicts of gradients contain the same keys
        self.assertEqual(grads.keys(), grads_frozen.keys())

        # ensure that the gradients of the feature extractor layers are precisely zero when frozen and contain non-zero entries when unfrozen
        feature_extractor_grads = tuple(grads[k] for k in grads if "feature_extractor" in k)
        feature_extractor_grads_frozen = tuple(grads_frozen[k] for k in grads_frozen if "feature_extractor" in k)

        for feature_extractor_grad, feature_extractor_grad_frozen in zip(
            feature_extractor_grads, feature_extractor_grads_frozen
        ):
            self.assertTrue((feature_extractor_grad_frozen == 0.0).all())
            self.assertTrue((feature_extractor_grad > 0.0).any())

        # ensure that the gradients of all unfrozen layers remain precisely equal, i.e. all layers excluding the frozen 'feature_extractor'
        grads = tuple(grads[k] for k in grads if "feature_extractor" not in k)
        grads_frozen = tuple(grads_frozen[k] for k in grads_frozen if "feature_extractor" not in k)

        for grad, grad_frozen in zip(grads, grads_frozen):
            self.assertTrue((grad == grad_frozen).all())

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

    def test_encoder_decoder_model_from_encoder_decoder_pretrained(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_from_encoder_decoder_pretrained(**input_ids_dict)

    def test_encoder_decoder_model_output_attentions(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_output_attentions(**input_ids_dict)

    def test_freeze_feature_encoder(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_freeze_feature_encoder(**input_ids_dict)

    def test_encoder_decoder_model_generate(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_generate(**input_ids_dict)

    def assert_almost_equals(self, a: np.ndarray, b: np.ndarray, tol: float):
        diff = np.abs(a - b).max()
        self.assertLessEqual(diff, tol, f"Difference between torch and flax is {diff} (>= {tol}).")

    @slow
    def test_real_model_save_load_from_pretrained(self):
        model_2 = self.get_pretrained_model()
        inputs = ids_tensor([13, 5], model_2.config.encoder.vocab_size)
        decoder_input_ids = ids_tensor([13, 1], model_2.config.decoder.vocab_size)
        attention_mask = ids_tensor([13, 5], vocab_size=2)

        outputs = model_2(
            inputs=inputs,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
        )
        out_2 = np.array(outputs[0])
        out_2[np.isnan(out_2)] = 0

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model_2.save_pretrained(tmp_dirname)
            model_1 = FlaxSpeechEncoderDecoderModel.from_pretrained(tmp_dirname)

            after_outputs = model_1(
                inputs=inputs,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
            )
            out_1 = np.array(after_outputs[0])
            out_1[np.isnan(out_1)] = 0
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 4e-2)


@require_flax
class FlaxWav2Vec2GPT2ModelTest(FlaxEncoderDecoderMixin, unittest.TestCase):
    def get_pretrained_model_and_inputs(self):
        model = FlaxSpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
            "facebook/wav2vec2-large-lv60", "openai-community/gpt2-medium"
        )
        batch_size = 13
        input_values = floats_tensor([batch_size, 512], scale=1.0)
        attention_mask = random_attention_mask([batch_size, 512])
        decoder_input_ids = ids_tensor([batch_size, 4], model.config.decoder.vocab_size)
        decoder_attention_mask = random_attention_mask([batch_size, 4])
        inputs = {
            "inputs": input_values,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }

        return model, inputs

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = FlaxWav2Vec2Model(config)
        decoder_model = FlaxGPT2LMHeadModel(decoder_config)
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester_encoder = FlaxWav2Vec2ModelTester(self, batch_size=13)
        model_tester_decoder = FlaxGPT2ModelTester(self, batch_size=13)
        encoder_config_and_inputs = model_tester_encoder.prepare_config_and_inputs()
        decoder_config_and_inputs = model_tester_decoder.prepare_config_and_inputs_for_decoder()
        (config, inputs, attention_mask) = encoder_config_and_inputs
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
            "inputs": inputs,
            "attention_mask": attention_mask,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
        }


@require_flax
class FlaxWav2Vec2BartModelTest(FlaxEncoderDecoderMixin, unittest.TestCase):
    def get_pretrained_model_and_inputs(self):
        model = FlaxSpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
            "facebook/wav2vec2-large-lv60", "bart-large"
        )
        batch_size = 13
        input_values = floats_tensor([batch_size, 512], scale=1.0)
        attention_mask = random_attention_mask([batch_size, 512])
        decoder_input_ids = ids_tensor([batch_size, 4], model.config.decoder.vocab_size)
        decoder_attention_mask = random_attention_mask([batch_size, 4])
        inputs = {
            "inputs": input_values,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }

        return model, inputs

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = FlaxWav2Vec2Model(config)
        decoder_model = FlaxBartForCausalLM(decoder_config)
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester_encoder = FlaxWav2Vec2ModelTester(self, batch_size=13)
        model_tester_decoder = FlaxBartStandaloneDecoderModelTester(self, batch_size=13)
        encoder_config_and_inputs = model_tester_encoder.prepare_config_and_inputs()
        decoder_config_and_inputs = model_tester_decoder.prepare_config_and_inputs_for_decoder()
        (config, inputs, attention_mask) = encoder_config_and_inputs
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
            "inputs": inputs,
            "attention_mask": attention_mask,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
        }


@require_flax
class FlaxWav2Vec2BertModelTest(FlaxEncoderDecoderMixin, unittest.TestCase):
    def get_pretrained_model_and_inputs(self):
        model = FlaxSpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
            "facebook/wav2vec2-large-lv60", "google-bert/bert-large-uncased"
        )
        batch_size = 13
        input_values = floats_tensor([batch_size, 512], model.config.encoder.vocab_size)
        attention_mask = random_attention_mask([batch_size, 512])
        decoder_input_ids = ids_tensor([batch_size, 4], model.config.decoder.vocab_size)
        decoder_attention_mask = random_attention_mask([batch_size, 4])
        inputs = {
            "inputs": input_values,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }

        return model, inputs

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = FlaxWav2Vec2Model(config)
        decoder_model = FlaxBertForCausalLM(decoder_config)
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester_encoder = FlaxWav2Vec2ModelTester(self, batch_size=13)
        model_tester_decoder = FlaxBertModelTester(self, batch_size=13)
        encoder_config_and_inputs = model_tester_encoder.prepare_config_and_inputs()
        decoder_config_and_inputs = model_tester_decoder.prepare_config_and_inputs_for_decoder()
        (config, inputs, attention_mask) = encoder_config_and_inputs
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
            "inputs": inputs,
            "attention_mask": attention_mask,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
        }
