# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch GraniteSpeechNar model."""

import tempfile
import unittest

from transformers import is_datasets_available, is_torch_available
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_datasets_available():
    from datasets import load_dataset

if is_torch_available():
    import torch

    from transformers import (
        AutoModel,
        AutoProcessor,
        GraniteConfig,
        GraniteSpeechNarConfig,
    )
    from transformers.models.granite_speech_nar.configuration_granite_speech_nar import (
        GraniteSpeechNarEncoderConfig,
        GraniteSpeechNarProjectorConfig,
    )
    from transformers.models.granite_speech_nar.modeling_granite_speech_nar import (
        GraniteSpeechNarCTCEncoder,
        GraniteSpeechNarForCTC,
    )


class GraniteSpeechNarEncoderModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=100,
        is_training=True,
        num_layers=2,
        hidden_dim=64,
        num_heads=4,
        dim_head=16,
        input_dim=160,
        output_dim=10,
        context_size=50,
        self_conditioning_layer=1,
        bpe_output_dim=51,
        bpe_pooling_window=4,
        dropout=0.0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training

        self.num_layers = num_layers
        self.num_hidden_layers = num_layers
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_size = context_size
        self.self_conditioning_layer = self_conditioning_layer
        self.bpe_output_dim = bpe_output_dim
        self.bpe_pooling_window = bpe_pooling_window
        self.dropout = dropout

    def get_config(self):
        return GraniteSpeechNarEncoderConfig(
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dim_head=self.dim_head,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            context_size=self.context_size,
            self_conditioning_layer=self.self_conditioning_layer,
            bpe_output_dim=self.bpe_output_dim,
            bpe_pooling_window=self.bpe_pooling_window,
            dropout=self.dropout,
            blank_token_id=0,
        )

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.seq_length, self.input_dim])
        attention_mask = torch.ones(self.batch_size, self.seq_length, dtype=torch.bool)
        attention_mask[1, 80:] = False
        config = self.get_config()
        return config, input_features, attention_mask

    def create_and_check_model(self, config, input_features, attention_mask):
        model = GraniteSpeechNarCTCEncoder(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_features, attention_mask=attention_mask, output_hidden_states=True)

        self.parent.assertIsNotNone(result.logits)
        self.parent.assertEqual(result.logits.shape[-1], self.bpe_output_dim)
        self.parent.assertIsNotNone(result.hidden_states)
        self.parent.assertEqual(len(result.hidden_states), self.num_layers + 1)

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_features": input_features,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


class GraniteSpeechNarForCTCModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=100,
        is_training=True,
        vocab_size=51,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = vocab_size

        self.encoder_model_tester = GraniteSpeechNarEncoderModelTester(
            parent,
            batch_size=batch_size,
            seq_length=seq_length,
            bpe_output_dim=vocab_size,
        )

    def get_config(self):
        encoder_config = self.encoder_model_tester.get_config()
        projector_config = GraniteSpeechNarProjectorConfig(
            encoder_dim=self.encoder_model_tester.hidden_dim,
            llm_dim=128,
            downsample_rate=5,
            num_encoder_layers=2,
            hidden_size=128,
            num_heads=4,
            num_layers=1,
            block_size=15,
        )
        text_config = GraniteConfig(
            vocab_size=self.vocab_size,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=512,
            tie_word_embeddings=True,
            embedding_multiplier=1.0,
            attention_multiplier=1.0,
            residual_multiplier=1.0,
            logits_scaling=1.0,
        )
        return GraniteSpeechNarConfig(
            encoder_config=encoder_config,
            projector_config=projector_config,
            text_config=text_config.to_dict(),
            encoder_layer_indices=[1, -1],
            scale_projected_embeddings=False,
        )

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.seq_length, self.encoder_model_tester.input_dim])
        attention_mask = torch.ones(self.batch_size, self.seq_length, dtype=torch.bool)
        attention_mask[1, 80:] = False
        config = self.get_config()
        return config, input_features, attention_mask

    def create_and_check_model(self, config, input_features, attention_mask):
        model = GraniteSpeechNarForCTC(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_features=input_features, attention_mask=attention_mask)

        self.parent.assertIsNotNone(result.logits)
        self.parent.assertIsInstance(result.logits, list)
        self.parent.assertEqual(len(result.logits), self.batch_size)
        for logits in result.logits:
            self.parent.assertEqual(logits.ndim, 2)
            self.parent.assertEqual(logits.shape[1], self.vocab_size)

    def create_and_check_generate(self, config, input_features, attention_mask):
        model = GraniteSpeechNarForCTC(config=config)
        model.to(torch_device)
        model.eval()
        output = model.generate(input_features=input_features, attention_mask=attention_mask)

        self.parent.assertIsNotNone(output.preds)
        self.parent.assertEqual(len(output.preds), self.batch_size)
        for pred in output.preds:
            self.parent.assertIsInstance(pred, torch.Tensor)
            self.parent.assertEqual(pred.ndim, 1)

    def create_and_check_generate_multi_step(self, config, input_features, attention_mask):
        model = GraniteSpeechNarForCTC(config=config)
        model.to(torch_device)
        model.eval()
        output = model.generate(input_features=input_features, attention_mask=attention_mask, num_editing_steps=3)

        self.parent.assertIsNotNone(output.preds)
        self.parent.assertEqual(len(output.preds), self.batch_size)
        for pred in output.preds:
            self.parent.assertIsInstance(pred, torch.Tensor)
            self.parent.assertEqual(pred.ndim, 1)

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_features": input_features,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class GraniteSpeechNarEncoderModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (GraniteSpeechNarCTCEncoder,) if is_torch_available() else ()

    test_resize_embeddings = False
    test_attention_outputs = False
    has_attentions = False

    @unittest.skip(reason="GraniteSpeechNarCTCEncoder does not use inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="GraniteSpeechNarCTCEncoder does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="GraniteSpeechNarCTCEncoder does not use inputs_embeds")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="Conformer encoder does not expose attention outputs")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Self-conditioning injection between layers causes hidden_states mismatch in tuple vs dict")
    def test_model_outputs_equivalence(self):
        pass

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if return_labels:
            batch_size = self.model_tester.batch_size
            inputs_dict["labels"] = torch.randint(0, self.model_tester.bpe_output_dim, (batch_size, 5))
            inputs_dict["label_lengths"] = torch.tensor([5] * batch_size)
        return inputs_dict

    def setUp(self):
        self.model_tester = GraniteSpeechNarEncoderModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GraniteSpeechNarEncoderConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)


@require_torch
class GraniteSpeechNarForCTCModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (GraniteSpeechNarForCTC,) if is_torch_available() else ()

    test_resize_embeddings = False
    test_attention_outputs = False
    has_attentions = False
    _is_composite = True

    @unittest.skip(reason="GraniteSpeechNarForCTC takes audio input_features, not input_ids/inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="GraniteSpeechNarForCTC takes audio input_features, not input_ids/inputs_embeds")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="GraniteSpeechNarForCTC has a custom generate method, not standard GenerationMixin")
    def test_generation_tester_mixin_inheritance(self):
        pass

    @unittest.skip(reason="Non-standard output format (logits is a list of tensors)")
    def test_determinism(self):
        pass

    @unittest.skip(reason="Non-standard output format (logits is a list of tensors)")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="Non-standard output format (logits is a list of tensors)")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Non-standard output format (logits is a list of tensors)")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Encoder does not have standard embedding layer for gradient checkpointing")
    def test_enable_input_require_grads_with_gradient_checkpointing(self):
        pass

    def test_can_init_all_missing_weights(self):
        super().test_can_init_all_missing_weights()

    def setUp(self):
        self.model_tester = GraniteSpeechNarForCTCModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GraniteSpeechNarConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_generate(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_generate(*config_and_inputs)

    def test_generate_multi_step(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_generate_multi_step(*config_and_inputs)

    def test_loss(self):
        config, input_features, attention_mask = self.model_tester.prepare_config_and_inputs()
        model = GraniteSpeechNarForCTC(config).to(torch_device).train()

        labels = torch.randint(0, self.model_tester.vocab_size, (self.model_tester.batch_size, 5))
        label_lengths = torch.tensor([5, 3])

        output = model(
            input_features=input_features,
            attention_mask=attention_mask,
            labels=labels,
            label_lengths=label_lengths,
        )

        self.assertIsNotNone(output.loss)
        self.assertEqual(output.loss.ndim, 0)
        self.assertTrue(output.loss.requires_grad)
        output.loss.backward()

    def test_loss_with_ce(self):
        config, input_features, attention_mask = self.model_tester.prepare_config_and_inputs()
        config.ce_loss_lambda = 0.5
        model = GraniteSpeechNarForCTC(config).to(torch_device).train()

        labels = torch.randint(0, self.model_tester.vocab_size, (self.model_tester.batch_size, 4))
        label_lengths = torch.tensor([4, 3])

        output = model(
            input_features=input_features, attention_mask=attention_mask, labels=labels, label_lengths=label_lengths
        )
        self.assertIsNotNone(output.loss)
        self.assertTrue(output.loss.requires_grad)
        output.loss.backward()

    def test_loss_with_encoder_ctc(self):
        config, input_features, attention_mask = self.model_tester.prepare_config_and_inputs()
        config.encoder_ctc_loss_lambda = 0.3
        model = GraniteSpeechNarForCTC(config).to(torch_device).train()

        labels = torch.randint(0, self.model_tester.vocab_size, (self.model_tester.batch_size, 4))
        label_lengths = torch.tensor([4, 3])

        output = model(
            input_features=input_features, attention_mask=attention_mask, labels=labels, label_lengths=label_lengths
        )
        self.assertIsNotNone(output.loss)
        self.assertTrue(output.loss.requires_grad)
        output.loss.backward()

    def test_no_loss_without_labels(self):
        config, input_features, attention_mask = self.model_tester.prepare_config_and_inputs()
        model = GraniteSpeechNarForCTC(config).to(torch_device).eval()

        with torch.no_grad():
            output = model(input_features=input_features, attention_mask=attention_mask)

        self.assertIsNone(output.loss)

    def test_bidirectional_attention(self):
        config = self.model_tester.get_config()
        model = GraniteSpeechNarForCTC(config).to(torch_device).eval()
        granite_model = model.model.language_model

        embeds_a = torch.randn(1, 10, 128, device=torch_device)
        embeds_b = embeds_a.clone()
        embeds_b[0, -1, :] = torch.randn(128, device=torch_device)

        with torch.no_grad():
            out_a = granite_model(inputs_embeds=embeds_a).last_hidden_state
            out_b = granite_model(inputs_embeds=embeds_b).last_hidden_state

        diff_first = (out_a[0, 0] - out_b[0, 0]).abs().max().item()
        self.assertGreater(diff_first, 1e-5, "First token unchanged — attention appears causal.")

    def test_is_causal_false_on_layers(self):
        config = self.model_tester.get_config()
        model = GraniteSpeechNarForCTC(config)
        for i, layer in enumerate(model.model.language_model.layers):
            self.assertFalse(layer.self_attn.is_causal, f"Layer {i} is_causal is not False")

    def test_sdpa_can_dispatch_composite_models(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(model_eager.config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        raise ValueError("The eager model should not have SDPA attention layers")


@require_torch
class GraniteSpeechNarIntegrationTest(unittest.TestCase):
    checkpoint_name = "ibm-granite/granite-speech-4.1-2b-nar"
    _dataset = None

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @classmethod
    def _load_dataset(cls):
        if cls._dataset is None:
            cls._dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    def _load_datasamples(self, num_samples):
        self._load_dataset()
        samples = self._dataset.sort("id")[:num_samples]["audio"]
        return [torch.tensor(x["array"], dtype=torch.float32) for x in samples]

    @slow
    def test_single_sample_transcription(self):
        model = AutoModel.from_pretrained(
            self.checkpoint_name,
            attn_implementation="flash_attention_2",
            device_map=torch_device,
            dtype=torch.bfloat16,
        ).eval()
        processor = AutoProcessor.from_pretrained(self.checkpoint_name)

        waveforms = self._load_datasamples(1)
        inputs = processor(waveforms, device=torch_device)
        output = model.generate(**inputs)
        transcriptions = processor.batch_decode(output.preds)

        expected = "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel"
        self.assertEqual(transcriptions[0], expected)

    @slow
    def test_batch_transcription(self):
        model = AutoModel.from_pretrained(
            self.checkpoint_name,
            attn_implementation="flash_attention_2",
            device_map=torch_device,
            dtype=torch.bfloat16,
        ).eval()
        processor = AutoProcessor.from_pretrained(self.checkpoint_name)

        waveforms = self._load_datasamples(2)
        inputs = processor(waveforms, device=torch_device)
        output = model.generate(**inputs)
        transcriptions = processor.batch_decode(output.preds)

        expected = [
            "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel",
            "nor is mister quilter's manner less interesting than his matter",
        ]
        self.assertEqual(len(transcriptions), 2)
        self.assertEqual(transcriptions, expected)
