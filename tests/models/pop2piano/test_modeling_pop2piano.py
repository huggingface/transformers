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
""" Testing suite for the PyTorch Pop2Piano model. """

import copy
import inspect
import os
import tempfile
import unittest

import numpy as np

import transformers
from transformers import Pop2PianoConfig
from transformers.testing_utils import is_pt_flax_cross_test, require_torch, require_torchaudio, slow, torch_device
from transformers.utils import cached_property, is_flax_available, is_torch_available
from transformers.feature_extraction_utils import BatchFeature

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        Pop2PianoFeatureExtractor,
        Pop2PianoModel,
        set_seed,
    )


def prepare_pop2piano_inputs_dict(
    input_features,
    composer="composer1",
    max_batch_size = None,
    n_bars:int = 2,
    return_dict:bool = False,
):
    return {"inputs":BatchFeature({"input_features": input_features}),
            "composer":composer,
            "max_batch_size":max_batch_size,
            "n_bars":n_bars,
            "return_dict":return_dict}

@require_torch
@require_torchaudio
class Pop2PianoModelTester:
    def __init__(
        self,
        parent,
        is_training=False,
        vocab_size=2400,
        d_model=512,
        d_kv=64,
        d_ff=2048,
        num_layers=6,
        num_decoder_layers=None,
        num_heads=8,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="gated-gelu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        dense_act_fn="relu",
        dataset_target_length=256,
        dataset_n_bars=2,
        dataset_sample_rate=22050,
        dataset_mel_is_conditioned=True,
        n_fft=4096,
        hop_length=1024,
        f_min=10.0,
        n_mels=512
    ):
        self.parent = parent
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.is_encoder_decoder = is_encoder_decoder
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.dense_act_fn = dense_act_fn
        self.dataset_target_length = dataset_target_length
        self.dataset_n_bars = dataset_n_bars
        self.dataset_sample_rate = dataset_sample_rate
        self.dataset_mel_is_conditioned = dataset_mel_is_conditioned
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.n_mels = n_mels



    def prepare_config_and_inputs(self):
        input_features = [torch.ones([120, 65024], device=torch_device)] # (120, 65024) is randomly taken from examples
        composer = "composer1"
        max_batch_size = 64
        n_bars = 2
        return_dict = False

        config = self.get_config()
        inputs_dict = prepare_pop2piano_inputs_dict(
            input_features=input_features,
            composer=composer,
            max_batch_size=max_batch_size,
            n_bars=n_bars,
            return_dict=return_dict,
        )
        return config, inputs_dict

    def get_config(self):
        return Pop2PianoConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            d_kv=self.d_kv,
            d_ff=self.d_ff,
            num_layers=self.num_layers,
            num_decoder_layers=self.num_decoder_layers,
            num_heads=self.num_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            relative_attention_max_distance=self.relative_attention_max_distance,
            dropout_rate=self.dropout_rate,
            layer_norm_epsilon=self.layer_norm_epsilon,
            initializer_factor=self.initializer_factor,
            feed_forward_proj=self.feed_forward_proj,
            is_encoder_decoder=self.is_encoder_decoder,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            dense_act_fn=self.dense_act_fn,
            dataset_target_length=self.dataset_target_length,
            dataset_n_bars=self.dataset_n_bars,
            dataset_sample_rate=self.dataset_sample_rate,
            dataset_mel_is_conditioned=self.dataset_sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            f_min=self.f_min,
            n_mels=self.n_mels
        )

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_model_forward(self, config, inputs_dict, freeze_encoder=False):
        model = Pop2PianoModel(config=config).to(torch_device).eval()

        if freeze_encoder:
            model.freeze_encoder()

        # first forward pass
        last_hidden_state = model(inputs_dict, return_dict=False)
        self.parent.assertTrue(type(last_hidden_state), list)

    def create_and_check_model_log_mel_spectogram_and_mel_conditioner(self, config):
        composer = "composer1"
        config["mel_is_conditioned"] = True
        model = Pop2PianoModel(config=config).to(torch_device).eval()
        input_features = torch.ones([120, 65024], device=torch_device)

        # test spectogram
        spectogram_output = model.spectrogram(input_features).transpose(-1, -2)
        self.parent.assertTrue(spectogram_output.size(-1), 512)

        # test mel_conditioner
        composer_to_feature_token = self.config.composer_to_feature_token
        composer_value = composer_to_feature_token[composer]
        composer_value = torch.tensor(composer_value, device=self.device)
        composer_value = composer_value.repeat(spectogram_output.size(0))
        mel_conditioner_output = model.mel_conditioner(spectogram_output, composer_value)
        self.parent.assertTrue(mel_conditioner_output.size(-1), 512)

@require_torch
@require_torchaudio
class Pop2PianoModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (Pop2PianoModel, ) if is_torch_available() else ()
    all_generative_model_classes = ()
    is_encoder_decoder = True
    fx_compatible = False
    test_pruning = False
    test_missing_keys = False
    test_torchscript = False

    input_name = "input_features"

    def setUp(self):
        self.model_tester = Pop2PianoModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Pop2PianoConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_save_load_strict(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model2, info = model_class.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertEqual(info["missing_keys"], [])

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_model_forward_with_frozen_encoder(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs, freeze_encoder=True)

    def test_requires_grad_with_frozen_encoder(self):
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.freeze_encoder()

            try:
                encoder_grads = [param.requires_grad for param in model.encoder.parameters()]
                decoder_grads = [param.requires_grad for param in model.decoder.parameters()]
            except AttributeError:
                encoder_grads = [param.requires_grad for param in model.model.encoder.parameters()]
                decoder_grads = [param.requires_grad for param in model.model.decoder.parameters()]

            self.assertFalse(all(encoder_grads))
            self.assertTrue(all(decoder_grads))

    # not implemented currently
    def test_inputs_embeds(self):
        pass

    # training is not supported yet
    def test_training(self):
        pass

    def test_training_gradient_checkpointing(self):
        pass

    def test_generate_with_head_masking(self):
        pass

    def test_generate_fp16(self): #
        config, input_dict = self.model_tester.prepare_config_and_inputs()
        config.max_target_positions = 400
        input_features = input_dict["input_features"]
        model = Pop2PianoForConditionalGeneration(config).eval().to(torch_device)
        if torch_device == "cuda":
            input_features = input_features.half()
            model.half()
        model.generate(input_features)
        model.generate(input_features, num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)

    def test_resize_tokens_embeddings(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            return

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)

            if self.model_tester.is_training is False:
                model.eval()

            model_vocab_size = config.vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            # make sure that decoder_input_ids are resized
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True
            for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    def test_resize_embeddings_untied(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            return

        original_config.tie_word_embeddings = False

        # if model cannot untied embeddings -> leave test
        if original_config.tie_word_embeddings:
            return

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config).to(torch_device)

            # if no output embeddings -> leave test
            if model.get_output_embeddings() is None:
                continue

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_vocab_size = config.vocab_size
            model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size + 10)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size - 15)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size - 15)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            return

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()
            inputs = self._prepare_for_class(inputs_dict, model_class)

            try:
                model.config.use_cache = False  # FSTM still requires this hack -> FSTM should probably be refactored similar to BART afterward
                input_features = inputs["input_features"]
                decoder_input_ids = inputs["decoder_input_ids"]
                decoder_attention_mask = inputs["decoder_attention_mask"]
                traced_model = torch.jit.trace(model, (input_features, decoder_input_ids, decoder_attention_mask))
            except RuntimeError:
                self.fail("Couldn't trace module.")

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

                try:
                    torch.jit.save(traced_model, pt_file_name)
                except Exception:
                    self.fail("Couldn't save module.")

                try:
                    loaded_model = torch.jit.load(pt_file_name)
                except Exception:
                    self.fail("Couldn't load module.")

            model.to(torch_device)
            model.eval()

            loaded_model.to(torch_device)
            loaded_model.eval()

            model_state_dict = model.state_dict()
            loaded_model_state_dict = loaded_model.state_dict()

            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

            models_equal = True
            for layer_name, p1 in model_state_dict.items():
                p2 = loaded_model_state_dict[layer_name]
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

@require_torch
@require_torchaudio
class Pop2PianoModelIntegrationTests(unittest.TestCase):
    def test_log_mel_spectrogram_integration(self):
        model = Pop2PianoModel.from_pretrained("susnato/pop2piano_dev")
        inputs = torch.ones([10, 100000])
        output = model.spectrogram(inputs)

        # check shape
        self.assertEqual(output.size(), torch.Size([10, 512, 98]))

        # check values
        self.assertEqual(output[0, :3, :3].cpu().numpy().tolist(),
                         [[-13.815510749816895, -13.815510749816895, -13.815510749816895],
                          [-13.815510749816895, -13.815510749816895, -13.815510749816895],
                          [-13.815510749816895, -13.815510749816895, -13.815510749816895]]
                         )

    @slow
    def test_mel_conditioner_integration(self):
        composer = "composer1"
        model = Pop2PianoModel.from_pretrained("susnato/pop2piano_dev")
        input_embeds = torch.ones([10, 100, 512])

        composer_value = model.config.composer_to_feature_token[composer]
        composer_value = torch.tensor(composer_value)
        composer_value = composer_value.repeat(input_embeds.size(0))
        outputs = model.mel_conditioner(input_embeds, composer_value)

        # check shape
        self.assertEqual(outputs.size(), torch.Size([10, 101, 512]))

        # check values
        self.assertEqual(outputs[0, :3, :3].detach().cpu().numpy().tolist(),
                         [[1.0475305318832397, 0.29052114486694336, -0.47778210043907166],
                          [1.0, 1.0, 1.0],
                          [1.0, 1.0, 1.0]]
                         )
    @slow
    def test_transformer_integration(self):
        model = Pop2PianoModel.from_pretrained("susnato/pop2piano_dev")
        inputs_embeds = torch.arange(51200).resize(10, 10, 512)
        outputs = model.transformer.generate(input_id=None, inputs_embeds=inputs_embeds)

        # check shape
        self.assertEqual(outputs.size(0), 10)

        # check values
        self.assertEqual(outputs[0].detach().cpu().numpy().tolist(),
                         [0, 134, 133, 55, 61, 64, 135, 135, 88, 135, 135, 132, 135, 55, 135, 61, 64, 135, 88, 135]
                         )

    @slow
    def full_model_integration(self):
        model = Pop2PianoModel.from_pretrained("susnato/pop2piano_dev")
        model.eval()
        inputs_embeds = BatchFeature({'input_features': [torch.ones([100, 100000])]})
        outputs = model(inputs=inputs_embeds, return_dict=False)

        # check for shapes
        self.assertEqual(outputs[0].size(0), 100)

        # check for values
        self.assertEqual(outputs[0][0, :3].detach().cpu().numpy().tolist(),
                         [0, 134, 133]
                         )

    @slow
    def full_model_integration_batched(self):
        model = Pop2PianoModel.from_pretrained("susnato/pop2piano_dev")
        model.eval()
        inputs_embeds = BatchFeature({'input_features': [torch.ones([100, 100000]),
                                                         torch.arange(10000000).type(torch.float32).resize(100, 100000)]})
        outputs = model(inputs=inputs_embeds, return_dict=False)

        # check for shapes
        self.assertEqual(outputs[0].size(0), 100)
        self.assertEqual(outputs[1].size(0), 100)

        # check for values
        self.assertEqual(outputs[0].detach().cpu().numpy().tolist()[0][:3],
                         [0, 134, 133]
                         )
        self.assertEqual(outputs[1].detach().cpu().numpy().tolist()[0][:3],
                         [0, 1, 0]
                         )
