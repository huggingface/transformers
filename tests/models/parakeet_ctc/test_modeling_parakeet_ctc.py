# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch ParakeetCTC model."""

import unittest

from transformers import is_datasets_available, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, random_attention_mask


if is_datasets_available():
    from datasets import load_dataset

if is_torch_available():
    import torch

    from transformers import AutoConfig, AutoFeatureExtractor, AutoModel, AutoTokenizer
    from transformers.models.fastconformer import FastConformerConfig, FastConformerFeatureExtractor
    from transformers.models.parakeet_ctc import ParakeetCTC, ParakeetCTCConfig
    from transformers.models.parakeet_ctc.modeling_parakeet_ctc import ParakeetCTCDecoder


class ParakeetCTCModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=1024,  # mel-spectrogram frames
        is_training=False,
        vocab_size=128,  # CTC vocab size
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        encoder_layerdrop=0.1,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        initializer_range=0.02,
        conv_kernel_size=9,
        subsampling_factor=8,
        subsampling_conv_channels=32,
        use_bias=False,
        num_mel_bins=128,
        xscaling=False,
        dropout_emb=0.0,
        blank_token_id=0,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.encoder_layerdrop = encoder_layerdrop
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.initializer_range = initializer_range
        self.conv_kernel_size = conv_kernel_size
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.use_bias = use_bias
        self.num_mel_bins = num_mel_bins
        self.xscaling = xscaling
        self.dropout_emb = dropout_emb
        self.blank_token_id = blank_token_id
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity
        self.scope = scope

        # Calculate output sequence length after subsampling
        self.output_seq_length = self.seq_length // self.subsampling_factor
        self.encoder_seq_length = self.output_seq_length
        self.key_length = self.output_seq_length

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.seq_length, self.num_mel_bins])
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        # Calculate input_lengths based on attention_mask
        input_lengths = attention_mask.sum(-1)

        config = self.get_config()

        return config, input_features, attention_mask, input_lengths

    def get_config(self):
        # Create FastConformer encoder config
        encoder_config = FastConformerConfig(
            vocab_size=1024,  # Not used by encoder
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            encoder_layerdrop=self.encoder_layerdrop,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_dropout,
            activation_dropout=self.activation_dropout,
            initializer_range=self.initializer_range,
            conv_kernel_size=self.conv_kernel_size,
            subsampling_factor=self.subsampling_factor,
            subsampling_conv_channels=self.subsampling_conv_channels,
            use_bias=self.use_bias,
            num_mel_bins=self.num_mel_bins,
            xscaling=self.xscaling,
            dropout_emb=self.dropout_emb,
        )

        # Create ParakeetCTC config
        return ParakeetCTCConfig(
            vocab_size=self.vocab_size,
            blank_token_id=self.blank_token_id,
            ctc_loss_reduction=self.ctc_loss_reduction,
            ctc_zero_infinity=self.ctc_zero_infinity,
            encoder_config=encoder_config,
        )

    def create_and_check_model(self, config, input_features, attention_mask, input_lengths):
        model = ParakeetCTC(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_features, attention_mask=attention_mask, input_lengths=input_lengths)

        # Check output shape - should be reduced due to subsampling
        expected_seq_length = input_features.shape[1] // config.encoder_config.subsampling_factor
        self.parent.assertEqual(result.logits.shape, (self.batch_size, expected_seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask, input_lengths = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_features": input_features,
            "attention_mask": attention_mask,
            "input_lengths": input_lengths,
        }
        return config, inputs_dict


@require_torch
class ParakeetCTCDecoderTest(unittest.TestCase):
    """Test the ParakeetCTCDecoder component."""

    def setUp(self):
        self.config = ParakeetCTCConfig(
            vocab_size=128,
            blank_token_id=127,
            ctc_loss_reduction="mean",
            ctc_zero_infinity=True,
            encoder_config=FastConformerConfig(hidden_size=64, num_hidden_layers=2),
        )

    @require_torch
    def test_decoder_initialization(self):
        """Test decoder initialization."""
        decoder = ParakeetCTCDecoder(self.config)

        # Check CTC head
        self.assertIsInstance(decoder.ctc_head, torch.nn.Linear)
        self.assertEqual(decoder.ctc_head.in_features, 64)  # encoder hidden_size
        self.assertEqual(decoder.ctc_head.out_features, 128)  # vocab_size

        # Check CTC parameters
        self.assertEqual(decoder.blank_token_id, 127)
        self.assertEqual(decoder.ctc_loss_reduction, "mean")
        self.assertEqual(decoder.ctc_zero_infinity, True)

    @require_torch
    def test_decoder_forward(self):
        """Test decoder forward pass."""
        decoder = ParakeetCTCDecoder(self.config)
        decoder.to(torch_device)
        decoder.eval()

        # Create test input
        batch_size, seq_len, hidden_size = 2, 50, 64
        hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(torch_device)

        # Forward pass
        with torch.no_grad():
            logits = decoder(hidden_states)

        # Check output shape
        self.assertEqual(logits.shape, (batch_size, seq_len, 128))

    @require_torch
    def test_decoder_ctc_loss(self):
        """Test CTC loss computation."""
        decoder = ParakeetCTCDecoder(self.config)
        decoder.to(torch_device)

        # Create test data
        batch_size, seq_len, vocab_size = 2, 20, 128
        logits = torch.randn(batch_size, seq_len, vocab_size).to(torch_device)
        labels = torch.tensor([[1, 2, 3, -100], [4, 5, -100, -100]]).to(torch_device)
        input_lengths = torch.tensor([seq_len, seq_len // 2]).to(torch_device)
        label_lengths = torch.tensor([3, 2]).to(torch_device)

        # Compute loss
        loss = decoder.compute_ctc_loss(logits, labels, input_lengths, label_lengths)

        # Check loss is finite and positive
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0)


@require_torch
class ParakeetCTCModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ParakeetCTC,) if is_torch_available() else ()

    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = ParakeetCTCModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ParakeetCTCConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_encoder_decoder_architecture(self):
        """Test the encoder-decoder architecture."""
        config = self.model_tester.get_config()
        model = ParakeetCTC(config).to(torch_device)

        # Test encoder access
        encoder = model.get_encoder()
        self.assertIsNotNone(encoder)
        self.assertEqual(type(encoder).__name__, "FastConformerEncoder")

        # Test decoder access
        decoder = model.get_decoder()
        self.assertIsNotNone(decoder)
        self.assertIsInstance(decoder, ParakeetCTCDecoder)

        # Test decoder properties
        self.assertEqual(decoder.blank_token_id, config.blank_token_id)
        self.assertEqual(decoder.ctc_loss_reduction, config.ctc_loss_reduction)
        self.assertEqual(decoder.ctc_zero_infinity, config.ctc_zero_infinity)

        # Test encoder/decoder setter methods
        original_encoder = model.get_encoder()
        original_decoder = model.get_decoder()

        model.set_encoder(original_encoder)
        model.set_decoder(original_decoder)

        # Should still work the same
        self.assertEqual(model.get_encoder(), original_encoder)
        self.assertEqual(model.get_decoder(), original_decoder)

    @unittest.skip(reason="ParakeetCTC has no input_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="ParakeetCTC has no tokens embeds")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="ParakeetCTC has no input_embeds")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="ParakeetCTC has issues with global device setting")
    def test_can_load_with_global_device_set(self):
        pass

    @unittest.skip(reason="ParakeetCTC attention gradients are not properly retained")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    def test_model_name_list(self):
        pass

    @slow
    def test_ctc_model_integration_generate(self):
        ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
        audio = ds[0]["audio"]["array"]

        model_name = "nvidia/parakeet-ctc-1.1b"
        model = ParakeetCTC.from_pretrained(model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model.eval()
        model.to(torch_device)

        audio = torch.tensor(audio).to(torch_device)
        features = feature_extractor(audio, sampling_rate=16000)

        decoded_tokens = model.generate(features.input_features, features.attention_mask, features.input_lengths)
        print(decoded_tokens)
        text = tokenizer.decode(decoded_tokens[0], ctc_decode=True)

        EXPECTED_TOKENS = [[130, 103, 38, 994, 62]]
        EXPECTED_TEXT = "what are you working on"
        self.assertEqual(decoded_tokens, EXPECTED_TOKENS)
        self.assertEqual(text, EXPECTED_TEXT)

    @slow
    def test_model_from_pretrained(self):
        """Test loading the actual nvidia/parakeet-ctc-1.1b model from the Hub."""
        model_name = "nvidia/parakeet-ctc-1.1b"

        # Test AutoModel loading
        auto_model = AutoModel.from_pretrained(model_name).to(torch_device)
        self.assertIsInstance(auto_model, ParakeetCTC)
        self.assertEqual(auto_model.config.model_type, "parakeet_ctc")
        self.assertEqual(auto_model.config.vocab_size, 1025)

        # Test direct ParakeetCTC loading
        ctc_model = ParakeetCTC.from_pretrained(model_name).to(torch_device)
        self.assertIsInstance(ctc_model, ParakeetCTC)
        self.assertEqual(ctc_model.config.model_type, "parakeet_ctc")
        self.assertEqual(ctc_model.config.vocab_size, 1025)

        # Test config loading
        config = AutoConfig.from_pretrained(model_name)
        self.assertIsInstance(config, ParakeetCTCConfig)
        self.assertEqual(config.model_type, "parakeet_ctc")
        self.assertEqual(config.vocab_size, 1025)

        # Test feature extractor loading
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.assertIsInstance(feature_extractor, FastConformerFeatureExtractor)
        self.assertEqual(feature_extractor.feature_size, 80)
        self.assertEqual(feature_extractor.sampling_rate, 16000)

        # Test forward pass with real model
        auto_model.eval()
        ctc_model.eval()

        # Create test input
        batch_size, seq_len, mel_bins = 1, 100, 80
        input_features = torch.randn(batch_size, seq_len, mel_bins).to(torch_device)
        input_lengths = torch.tensor([seq_len], dtype=torch.long).to(torch_device)

        with torch.no_grad():
            # Test AutoModel forward pass
            auto_outputs = auto_model(
                input_features=input_features,
                input_lengths=input_lengths,
            )
            self.assertIsNotNone(auto_outputs.logits)
            self.assertEqual(auto_outputs.logits.shape[0], batch_size)
            self.assertEqual(auto_outputs.logits.shape[2], 1025)  # vocab_size

            # Test ParakeetCTC forward pass
            ctc_outputs = ctc_model(
                input_features=input_features,
                input_lengths=input_lengths,
            )
            self.assertIsNotNone(ctc_outputs.logits)
            self.assertEqual(ctc_outputs.logits.shape[0], batch_size)
            self.assertEqual(ctc_outputs.logits.shape[2], 1025)  # vocab_size

            # Test that both models produce the same output (they should be identical)
            torch.testing.assert_close(auto_outputs.logits, ctc_outputs.logits, rtol=1e-5, atol=1e-5)

            # Test CTC generation
            decoded_sequences = ctc_model.generate(
                input_features=input_features,
                input_lengths=input_lengths,
            )
            self.assertEqual(len(decoded_sequences), batch_size)
            self.assertIsInstance(decoded_sequences[0], list)

    def test_ctc_model_with_ctc_config(self):
        """Test that ParakeetCTC works with ParakeetCTCConfig."""
        # Create ParakeetCTCConfig
        fastconformer_config = FastConformerConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_mel_bins=80,
        )

        ctc_config = ParakeetCTCConfig(
            vocab_size=128,
            blank_token_id=1,
            ctc_loss_reduction="sum",
            ctc_zero_infinity=False,
            encoder_config=fastconformer_config,
        )

        model = ParakeetCTC(ctc_config)
        model.to(torch_device)
        model.eval()

        # Create test input
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        _, input_features, attention_mask, input_lengths = config_and_inputs

        # Adjust input features for smaller configs
        input_features = input_features[:, :200, :80]  # Smaller for testing
        attention_mask = attention_mask[:, :200]
        input_lengths = torch.clamp(input_lengths, max=200)

        with torch.no_grad():
            outputs = model(
                input_features.to(torch_device),
                attention_mask=attention_mask.to(torch_device),
                input_lengths=input_lengths.to(torch_device),
            )

        # Should produce outputs
        self.assertIsNotNone(outputs.logits)
        self.assertEqual(outputs.logits.shape[-1], 128)

        # Check that CTC parameters are set correctly in the decoder
        self.assertEqual(model.decoder.blank_token_id, 1)  # Custom
        self.assertEqual(model.decoder.ctc_loss_reduction, "sum")  # Custom
        self.assertEqual(model.decoder.ctc_zero_infinity, False)  # Custom

    def test_ctc_model_loss_computation(self):
        """Test CTC loss computation."""
        fastconformer_config = FastConformerConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_mel_bins=40,
        )

        config = ParakeetCTCConfig(
            vocab_size=10,
            blank_token_id=0,
            ctc_loss_reduction="mean",
            encoder_config=fastconformer_config,
        )

        model = ParakeetCTC(config)
        model.to(torch_device)
        model.eval()

        # Create test input and labels
        batch_size, seq_len, mel_bins = 2, 100, 40
        input_features = torch.randn(batch_size, seq_len, mel_bins).to(torch_device)
        input_lengths = torch.tensor([seq_len, seq_len // 2], dtype=torch.long).to(torch_device)

        # Create dummy labels (non-blank tokens)
        labels = torch.tensor(
            [
                [1, 2, 3, -100, -100],  # First sequence
                [4, 5, -100, -100, -100],  # Second sequence (shorter)
            ],
            dtype=torch.long,
        ).to(torch_device)

        # Forward pass with labels should compute loss
        outputs = model(
            input_features=input_features,
            input_lengths=input_lengths,
            labels=labels,
        )

        # Check that loss is computed and finite
        self.assertIsNotNone(outputs.loss)
        self.assertTrue(torch.isfinite(outputs.loss))
        # Check that logits have the right dimensions (output sequence length varies based on exact subsampling)
        self.assertEqual(outputs.logits.shape[0], batch_size)  # Batch size
        self.assertEqual(outputs.logits.shape[2], 10)  # Vocab size
        self.assertGreater(outputs.logits.shape[1], 0)  # Some positive sequence length

    def test_encoder_decoder_loss_consistency(self):
        """Test that CTC loss computed via decoder matches model loss."""
        fastconformer_config = FastConformerConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_mel_bins=40,
        )

        config = ParakeetCTCConfig(
            vocab_size=10,
            blank_token_id=0,
            ctc_loss_reduction="mean",
            encoder_config=fastconformer_config,
        )

        model = ParakeetCTC(config)
        model.to(torch_device)
        model.eval()

        # Create test input and labels
        batch_size, seq_len, mel_bins = 2, 100, 40
        input_features = torch.randn(batch_size, seq_len, mel_bins).to(torch_device)
        input_lengths = torch.tensor([seq_len, seq_len // 2], dtype=torch.long).to(torch_device)
        labels = torch.tensor([[1, 2, 3, -100], [4, 5, -100, -100]], dtype=torch.long).to(torch_device)

        # Forward pass through full model
        model_outputs = model(
            input_features=input_features,
            input_lengths=input_lengths,
            labels=labels,
        )

        # Manual encoder-decoder forward pass
        encoder_outputs = model.encoder(
            input_features=input_features,
            input_lengths=input_lengths,
            return_dict=True,
        )
        decoder_logits = model.decoder(encoder_outputs.last_hidden_state)

        # Check that logits match
        torch.testing.assert_close(model_outputs.logits, decoder_logits)

        # Check that loss is computed correctly
        self.assertIsNotNone(model_outputs.loss)
        self.assertTrue(torch.isfinite(model_outputs.loss))


if __name__ == "__main__":
    unittest.main()
