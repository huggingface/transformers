# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Nemotron3_5Asr model."""

import json
import tempfile
import unittest
from pathlib import Path
from threading import Thread

from transformers import is_datasets_available, is_torch_available
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_datasets_available():
    from datasets import Audio, load_dataset

if is_torch_available():
    import torch

    from transformers import (
        AutoProcessor,
        Nemotron3_5AsrConfig,
        Nemotron3_5AsrEncoderConfig,
        Nemotron3_5AsrForRNNT,
        TextIteratorStreamer,
    )
    from transformers.audio_utils import load_audio


class Nemotron3_5AsrForRNNTModelTester:
    def __init__(
        self,
        parent,
        batch_size=4,
        seq_length=256,
        num_mel_bins=80,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        subsampling_factor=8,
        subsampling_conv_channels=32,
        vocab_size=128,
        decoder_hidden_size=32,
        joint_hidden_size=32,
        num_decoder_layers=1,
        hidden_act="relu",
        max_symbols_per_step=5,
        pad_token_id=2,
        num_prompts=8,
        prompt_intermediate_size=16,
        is_training=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_mel_bins = num_mel_bins
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.vocab_size = vocab_size
        self.decoder_hidden_size = decoder_hidden_size
        self.joint_hidden_size = joint_hidden_size
        self.num_decoder_layers = num_decoder_layers
        self.hidden_act = hidden_act
        self.max_symbols_per_step = max_symbols_per_step
        self.pad_token_id = pad_token_id
        self.num_prompts = num_prompts
        self.prompt_intermediate_size = prompt_intermediate_size
        self.is_training = is_training

        self.blank_token_id = vocab_size - 1
        self.output_seq_length = seq_length // subsampling_factor
        self.seq_length = seq_length
        self.encoder_seq_length = self.output_seq_length

    def get_config(self):
        encoder_config = Nemotron3_5AsrEncoderConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=0.0,
            dropout_positions=0.0,
            layerdrop=0.0,
            activation_dropout=0.0,
            attention_dropout=0.0,
            subsampling_factor=self.subsampling_factor,
            subsampling_conv_channels=self.subsampling_conv_channels,
            num_mel_bins=self.num_mel_bins,
            scale_input=True,
        )
        return Nemotron3_5AsrConfig(
            vocab_size=self.vocab_size,
            decoder_hidden_size=self.decoder_hidden_size,
            joint_hidden_size=self.joint_hidden_size,
            num_decoder_layers=self.num_decoder_layers,
            hidden_act=self.hidden_act,
            max_symbols_per_step=self.max_symbols_per_step,
            encoder_config=encoder_config.to_dict(),
            pad_token_id=self.pad_token_id,
            blank_token_id=self.blank_token_id,
            num_prompts=self.num_prompts,
            prompt_intermediate_size=self.prompt_intermediate_size,
        )

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.seq_length, self.num_mel_bins])
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        config = self.get_config()
        return config, input_features, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_features": input_features,
            "attention_mask": attention_mask,
            "decoder_input_ids": ids_tensor([self.batch_size, 1], self.vocab_size),
            "prompt_ids": ids_tensor([self.batch_size], self.num_prompts),
        }
        return config, inputs_dict

    def create_and_check_model(self, config, inputs_dict):
        model = Nemotron3_5AsrForRNNT(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(**inputs_dict)
        # The causal subsampling length is computed by the model; just check batch and hidden dims.
        self.parent.assertEqual(result.last_hidden_state.shape[0], self.batch_size)
        self.parent.assertEqual(result.last_hidden_state.shape[-1], self.hidden_size)


@require_torch
class Nemotron3_5AsrForRNNTModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Nemotron3_5AsrForRNNT,) if is_torch_available() else ()
    pipeline_model_mapping = {"automatic-speech-recognition": Nemotron3_5AsrForRNNT} if is_torch_available() else {}

    test_attention_outputs = False
    test_resize_embeddings = False
    test_torch_exportable = False
    _is_composite = True
    # The FastConformer encoder is reused as-is from NemotronAsr and built via AutoModel; this RNN-T model
    # does not re-expose the encoder's attentions/hidden_states through its own output recorder, so the
    # attention-dependent parity tests (which would otherwise re-test the reused encoder) do not apply.
    # Eager/sdpa equivalence of the encoder is covered by NemotronAsr's own test suite.
    has_attentions = False

    @unittest.skip(reason="No available flash-SDPA kernels for Nemotron3_5Asr test shapes on this setup")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def setUp(self):
        self.model_tester = Nemotron3_5AsrForRNNTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Nemotron3_5AsrConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_prompt_conditioning_changes_logits(self):
        """The language-ID prompt must actually steer the model: different `prompt_ids` -> different logits."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Nemotron3_5AsrForRNNT(config=config).to(torch_device).eval()
        inputs_dict = {k: v.to(torch_device) for k, v in inputs_dict.items()}
        base = {k: v for k, v in inputs_dict.items() if k != "prompt_ids"}
        batch_size = base["input_features"].shape[0]
        with torch.no_grad():
            logits_a = model(**base, prompt_ids=torch.zeros(batch_size, dtype=torch.long, device=torch_device)).logits
            logits_b = model(
                **base,
                prompt_ids=torch.full((batch_size,), config.num_prompts - 1, dtype=torch.long, device=torch_device),
            ).logits
        self.assertFalse(torch.allclose(logits_a, logits_b))

    def test_missing_prompt_ids_defaults(self):
        """Without `prompt_ids` the model defaults to prompt index 0 (with a warning) and still runs."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Nemotron3_5AsrForRNNT(config=config).to(torch_device).eval()
        no_prompt = {k: v.to(torch_device) for k, v in inputs_dict.items() if k != "prompt_ids"}
        zeros = torch.zeros(no_prompt["input_features"].shape[0], dtype=torch.long, device=torch_device)
        with torch.no_grad():
            default_logits = model(**no_prompt).logits
            explicit_logits = model(**no_prompt, prompt_ids=zeros).logits
        torch.testing.assert_close(default_logits, explicit_logits)

    @unittest.skip(reason="Nemotron3_5AsrForRNNT does not use inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(
        reason="Nemotron3_5AsrForRNNT is a transducer, not a standard encoder-decoder: no separate text config to set"
    )
    def test_attn_implementation_composite_models(self):
        pass

    @unittest.skip(
        reason="Nemotron3_5AsrForRNNT is a transducer with an LSTM prediction network; "
        "it does not expose encoder_hidden_states in the standard encoder-decoder sense"
    )
    def test_hidden_states_output(self):
        pass

    @unittest.skip(
        reason="Nemotron3_5AsrForRNNT is a transducer with an LSTM prediction network; "
        "it does not expose encoder_hidden_states in the standard encoder-decoder sense"
    )
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(
        reason="Nemotron3_5AsrForRNNT has a custom generate() that is not fully compatible with GenerationTesterMixin"
    )
    def test_generation_tester_mixin_inheritance(self):
        pass

    @unittest.skip(reason="Nemotron3_5AsrForRNNT is a flat composite model without a separate base_model sub-module")
    def test_model_base_model_prefix(self):
        pass

    @unittest.skip(reason="Nemotron3_5AsrForRNNT decoder is an LSTM prediction network without attention")
    def test_flex_attention_with_grads(self):
        pass

    # Original function assumes vision+text model, so overwrite since Nemotron3_5Asr is audio+text
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
                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(model_eager.config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        raise ValueError("The eager model should not have SDPA attention layers")


# Local conversion of `nvidia/nemotron-3.5-asr-streaming-0.6b`. Replace with the public Hub id once the
# converted weights are published.
NEMOTRON_3_5_ASR_CHECKPOINT = "/raid/eustache/nemotron3_5_asr_hf"
# Long, single-speaker sample (~16 min of Obama's farewell address), used to exercise chunked streaming.
OBAMA_AUDIO_URL = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"
FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures/nemotron3_5_asr"


@slow
@require_torch
class Nemotron3_5AsrForRNNTIntegrationTest(unittest.TestCase):
    """Integration tests for the multilingual, prompt-conditioned Nemotron3_5Asr RNN-T model.

    Expected transcriptions are the NeMo reference outputs of `nvidia/nemotron-3.5-asr-streaming-0.6b` on
    the same audio, loaded from the `fixtures/nemotron3_5_asr/expected_results_*.json` fixtures. The
    reproducers that regenerate these reference values live in `reproducer/` (run `run_reproducers.sh`).
    Inference runs in float32 to track the NeMo reference as closely as possible.

    - Offline single/batched `generate` uses the model's default attention context `[56, 3]` (the first
      entry of `att_context_size`, the 320 ms setting), conditioned on `language="en-US"`.
    - Streaming feeds mel-frame chunks through a cache-aware `generate` at `num_lookahead_tokens=6`
      (att_context `[56, 6]`, the 560 ms setting), the transcript consumed via a `TextIteratorStreamer`.
    """

    _dataset = None

    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(NEMOTRON_3_5_ASR_CHECKPOINT)

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
        speech_samples = self._dataset.sort("id")[:num_samples]["audio"]
        return [x["array"] for x in speech_samples]

    @staticmethod
    def _load_expected(name):
        with open(FIXTURES_DIR / name) as f:
            return json.load(f)

    def test_rnnt_model_integration(self):
        # NeMo `nvidia/nemotron-3.5-asr-streaming-0.6b` reference; HF matches it.
        expected = self._load_expected("expected_results_single.json")
        samples = self._load_datasamples(len(expected["transcriptions"]))
        model = Nemotron3_5AsrForRNNT.from_pretrained(
            NEMOTRON_3_5_ASR_CHECKPOINT, dtype=torch.float32, device_map="auto"
        ).eval()

        inputs = self.processor(samples, sampling_rate=16000, language=expected["language"])
        num_lookahead_tokens = inputs.pop("num_lookahead_tokens")
        inputs.to(model.device, dtype=model.dtype)
        output = model.generate(**inputs, num_lookahead_tokens=num_lookahead_tokens, return_dict_in_generate=True)
        predicted = self.processor.batch_decode(output.sequences, skip_special_tokens=True)
        self.assertListEqual(predicted, expected["transcriptions"])

    def test_rnnt_model_integration_batched(self):
        # NeMo reference; all five HF transcripts match it.
        expected = self._load_expected("expected_results_batch.json")
        samples = self._load_datasamples(len(expected["transcriptions"]))
        model = Nemotron3_5AsrForRNNT.from_pretrained(
            NEMOTRON_3_5_ASR_CHECKPOINT, dtype=torch.float32, device_map="auto"
        ).eval()

        inputs = self.processor(samples, sampling_rate=16000, language=expected["language"])
        num_lookahead_tokens = inputs.pop("num_lookahead_tokens")
        inputs.to(model.device, dtype=model.dtype)
        output = model.generate(**inputs, num_lookahead_tokens=num_lookahead_tokens, return_dict_in_generate=True)
        predicted = self.processor.batch_decode(output.sequences, skip_special_tokens=True)
        self.assertListEqual(predicted, expected["transcriptions"])

    def test_rnnt_model_integration_streaming(self):
        """Cache-aware streaming generation from a generator of mel-frame chunks.

        The full mel spectrogram is sliced into contiguous chunks (`1 + subsampling_factor * right` frames
        for the first chunk, then `subsampling_factor * (right + 1)`) and fed to `generate` as a generator
        together with `num_lookahead_tokens` and the language `prompt_ids`. The decoder/encoder caches are
        threaded across chunks internally and the transcript is consumed incrementally via a
        `TextIteratorStreamer`.
        """
        expected = self._load_expected("expected_results_streaming.json")
        right = expected["att_context_size"][1]
        model = Nemotron3_5AsrForRNNT.from_pretrained(
            NEMOTRON_3_5_ASR_CHECKPOINT, dtype=torch.float32, device_map="auto"
        ).eval()

        audio = load_audio(OBAMA_AUDIO_URL, sampling_rate=self.processor.feature_extractor.sampling_rate)
        inputs = self.processor(audio, sampling_rate=16000, language=expected["language"])
        inputs.to(model.device, dtype=model.dtype)
        prompt_ids = inputs["prompt_ids"]

        def input_features_generator():
            start_idx = 0
            subsampling_factor = model.config.encoder_config.subsampling_factor
            first_chunk_size = 1 + subsampling_factor * right
            chunk_size = subsampling_factor * (right + 1)
            chunk = first_chunk_size
            input_length = inputs.input_features.shape[1]
            while start_idx < input_length:
                end_idx = min(start_idx + chunk, input_length)
                yield inputs.input_features[:, start_idx:end_idx, :]
                start_idx = end_idx
                chunk = chunk_size

        streamer = TextIteratorStreamer(
            self.processor.tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        generate_kwargs = {
            "input_features": input_features_generator(),
            "num_lookahead_tokens": right,
            "prompt_ids": prompt_ids,
            "streamer": streamer,
        }
        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()
        streamed_text = "".join(streamer)
        thread.join()

        self.assertEqual(streamed_text, expected["transcription"])

    def test_transcription_auto_language_tag(self):
        # In `auto` mode the model emits the detected language tag (a special token): `skip_special_tokens`
        # controls whether it is kept (language labeling) or stripped (clean transcript).
        model = Nemotron3_5AsrForRNNT.from_pretrained(
            NEMOTRON_3_5_ASR_CHECKPOINT, dtype=torch.float32, device_map="auto"
        ).eval()
        inputs = self.processor(self._load_datasamples(1), sampling_rate=16000, language="auto")
        num_lookahead_tokens = inputs.pop("num_lookahead_tokens")
        inputs.to(model.device, dtype=model.dtype)
        output = model.generate(**inputs, num_lookahead_tokens=num_lookahead_tokens)
        kept = self.processor.batch_decode(output.sequences, skip_special_tokens=False)[0]
        stripped = self.processor.batch_decode(output.sequences, skip_special_tokens=True)[0]
        self.assertIn("<en-US>", kept)
        self.assertNotIn("<en-US>", stripped)
