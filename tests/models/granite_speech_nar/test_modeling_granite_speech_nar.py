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

import json
import tempfile
import unittest
from pathlib import Path

from transformers import is_datasets_available, is_torch_available
from transformers.testing_utils import cleanup, require_torch, require_torchaudio, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_datasets_available():
    from datasets import Audio, load_dataset


FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures/granite_speech_nar"

if is_torch_available():
    import torch

    from transformers import (
        AutoProcessor,
        GraniteSpeechNarConfig,
    )
    from transformers.models.granite_speech_nar.configuration_granite_speech_nar import (
        GraniteSpeechNarEncoderConfig,
        GraniteSpeechNarProjectorConfig,
        GraniteSpeechNarTextConfig,
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
            cat_hidden_layers=[1],
        )

    def prepare_config_and_inputs(self):
        # The conformer encoder consumes every frame (it has no attention mask), so no mask is needed.
        input_features = floats_tensor([self.batch_size, self.seq_length, self.input_dim])
        config = self.get_config()
        return config, input_features

    def create_and_check_model(self, config, input_features):
        model = GraniteSpeechNarCTCEncoder(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_features, output_hidden_states=True)

        self.parent.assertIsNotNone(result.last_hidden_state)
        self.parent.assertEqual(result.last_hidden_state.shape[0], self.batch_size)
        self.parent.assertIsNotNone(result.pooled_hidden_states)
        self.parent.assertIsNotNone(result.hidden_states)
        self.parent.assertEqual(len(result.hidden_states), self.num_layers + 1)

    def prepare_config_and_inputs_for_common(self):
        config, input_features = self.prepare_config_and_inputs()
        inputs_dict = {"input_features": input_features}
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
            downsample_rate=5,
            num_encoder_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            num_layers=1,
            intermediate_size=256,
            block_size=15,
        )
        text_config = GraniteSpeechNarTextConfig(
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
            blank_token_id=text_config.eos_token_id,
        )

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.seq_length, self.encoder_model_tester.input_dim])
        input_features_mask = torch.ones(self.batch_size, self.seq_length, dtype=torch.bool)
        input_features_mask[1, 80:] = False
        config = self.get_config()
        return config, input_features, input_features_mask

    def create_and_check_model(self, config, input_features, input_features_mask):
        model = GraniteSpeechNarForCTC(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_features=input_features, input_features_mask=input_features_mask)

        # Logits are packed as `[1, sum(seq_lengths), vocab_size]` (batch folded into a flat sequence).
        self.parent.assertIsNotNone(result.logits)
        self.parent.assertIsInstance(result.logits, torch.Tensor)
        self.parent.assertEqual(result.logits.ndim, 3)
        self.parent.assertEqual(result.logits.shape[0], 1)
        self.parent.assertEqual(result.logits.shape[-1], self.vocab_size)
        self.parent.assertIsNotNone(result.seq_lengths)
        self.parent.assertEqual(len(result.seq_lengths), self.batch_size)
        self.parent.assertEqual(sum(result.seq_lengths), result.logits.shape[1])

    def create_and_check_generate(self, config, input_features, input_features_mask):
        model = GraniteSpeechNarForCTC(config=config)
        model.to(torch_device)
        model.eval()
        sequences = model.generate(input_features=input_features, input_features_mask=input_features_mask)

        self.parent.assertEqual(len(sequences), self.batch_size)
        for sequence in sequences:
            self.parent.assertIsInstance(sequence, torch.Tensor)
            self.parent.assertEqual(sequence.ndim, 1)

    def prepare_config_and_inputs_for_common(self):
        config, input_features, input_features_mask = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_features": input_features,
            "input_features_mask": input_features_mask,
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

    @unittest.skip(reason="GraniteSpeechNarCTCEncoder is a backbone component with no standalone training loss")
    def test_training(self):
        pass

    @unittest.skip(reason="GraniteSpeechNarCTCEncoder is a backbone component with no standalone training loss")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="GraniteSpeechNarCTCEncoder is a backbone component with no standalone training loss")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="GraniteSpeechNarCTCEncoder is a backbone component with no standalone training loss")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    def setUp(self):
        self.model_tester = GraniteSpeechNarEncoderModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=GraniteSpeechNarEncoderConfig,
            has_text_modality=False,
            common_properties=["hidden_dim", "num_layers", "num_heads"],
        )

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

    @unittest.skip(
        reason="Output packs logits as `[1, sum(seq_lengths), vocab]` and carries a `seq_lengths` list, "
        "which the mixin's recursive tuple/dict equivalence comparison cannot handle."
    )
    def test_model_outputs_equivalence(self):
        pass

    def test_hidden_states_output(self):
        # The mixin assumes a `[batch, seq_length, hidden]` layout, but this model runs the LM on a
        # flat packed sequence, so `hidden_states` are `[1, packed_len, hidden_size]`. Check that they
        # are exposed (one per LM layer + the initial embedding) with the packed shape.
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**inputs_dict, output_hidden_states=True)

            hidden_states = outputs.hidden_states
            self.assertEqual(len(hidden_states), config.text_config.num_hidden_layers + 1)
            for hidden_state in hidden_states:
                self.assertEqual(hidden_state.shape[0], 1)
                self.assertEqual(hidden_state.shape[-1], config.text_config.hidden_size)
            # every layer shares the same packed sequence length
            self.assertEqual(len({hidden_state.shape[1] for hidden_state in hidden_states}), 1)

    @unittest.skip(
        reason="forward/generate pack the batch into one flat, variable-length sequence, so "
        "nn.DataParallel cannot gather the differently-sized per-replica outputs."
    )
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip(
        reason="The packed forward manually concatenates projector audio embeddings with LLM text "
        "embeddings; naive device_map layer-splitting places them on different devices."
    )
    def test_model_parallelism(self):
        pass

    def test_can_init_all_missing_weights(self):
        super().test_can_init_all_missing_weights()

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if return_labels:
            # GraniteSpeechNarForCTC's CTC loss needs explicit target lengths alongside `labels`
            # (unlike CTC models that derive them from a padding id), which the generic harness omits.
            batch_size = self.model_tester.batch_size
            inputs_dict["labels"] = torch.randint(1, self.model_tester.vocab_size, (batch_size, 4))
            inputs_dict["label_lengths"] = torch.tensor([3] * batch_size)
        return inputs_dict

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

    def test_loss(self):
        config, input_features, attention_mask = self.model_tester.prepare_config_and_inputs()
        model = GraniteSpeechNarForCTC(config).to(torch_device).train()

        labels = torch.randint(0, self.model_tester.vocab_size, (self.model_tester.batch_size, 5))
        label_lengths = torch.tensor([5, 3])

        output = model(
            input_features=input_features,
            input_features_mask=attention_mask,
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
            input_features=input_features,
            input_features_mask=attention_mask,
            labels=labels,
            label_lengths=label_lengths,
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
            input_features=input_features,
            input_features_mask=attention_mask,
            labels=labels,
            label_lengths=label_lengths,
        )
        self.assertIsNotNone(output.loss)
        self.assertTrue(output.loss.requires_grad)
        output.loss.backward()

    def test_no_loss_without_labels(self):
        config, input_features, attention_mask = self.model_tester.prepare_config_and_inputs()
        model = GraniteSpeechNarForCTC(config).to(torch_device).eval()

        with torch.no_grad():
            output = model(input_features=input_features, input_features_mask=attention_mask)

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
@require_torchaudio
class GraniteSpeechNarForCTCIntegrationTest(unittest.TestCase):
    _dataset = None

    @classmethod
    def setUp(cls):
        cls.checkpoint_name = "ibm-granite/granite-speech-4.1-2b-nar"
        cls.revision = "refs/pr/6"
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint_name, revision=cls.revision)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @classmethod
    def _load_dataset(cls):
        if cls._dataset is None:
            cls._dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
            cls._dataset = cls._dataset.cast_column(
                "audio", Audio(sampling_rate=cls.processor.feature_extractor.sampling_rate)
            )

    def _load_datasamples(self, num_samples):
        self._load_dataset()
        ds = self._dataset
        speech_samples = ds.sort("id")[:num_samples]["audio"]
        return [x["array"] for x in speech_samples]

    @slow
    def test_model_integration(self):
        """
        reproducer: https://gist.github.com/eustlb/77deb553184fbc64d3565e8767448c65#file-reproducer_single-py
        """
        RESULTS_PATH = FIXTURES_DIR / "expected_results_single.json"
        with open(RESULTS_PATH, "r") as f:
            raw_data = json.load(f)
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(len(EXPECTED_TRANSCRIPTIONS))
        model = GraniteSpeechNarForCTC.from_pretrained(self.checkpoint_name, revision=self.revision, device_map="auto")

        inputs = self.processor(samples, sampling_rate=self.processor.feature_extractor.sampling_rate)
        inputs.to(model.device, dtype=model.dtype)
        output = model.generate(**inputs, return_dict_in_generate=True)
        predicted_transcripts = self.processor.batch_decode(output.sequences, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)

    @slow
    def test_model_integration_batched(self):
        """
        reproducer: https://gist.github.com/eustlb/77deb553184fbc64d3565e8767448c65#file-reproducer_batch-py
        """
        RESULTS_PATH = FIXTURES_DIR / "expected_results_batch.json"
        with open(RESULTS_PATH, "r") as f:
            raw_data = json.load(f)
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(len(EXPECTED_TRANSCRIPTIONS))
        model = GraniteSpeechNarForCTC.from_pretrained(self.checkpoint_name, revision=self.revision, device_map="auto")

        inputs = self.processor(samples, sampling_rate=self.processor.feature_extractor.sampling_rate)
        inputs.to(model.device, dtype=model.dtype)
        output = model.generate(**inputs, return_dict_in_generate=True)
        predicted_transcripts = self.processor.batch_decode(output.sequences, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)
