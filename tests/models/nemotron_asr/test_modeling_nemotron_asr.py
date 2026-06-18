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
"""Testing suite for the PyTorch NemotronAsr model."""

import json
import unittest
from pathlib import Path
from threading import Thread

from transformers import is_datasets_available, is_torch_available
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import floats_tensor
from ..parakeet.test_modeling_parakeet import (
    ParakeetEncoderModelTest,
    ParakeetEncoderModelTester,
    ParakeetForRNNTModelTest,
    ParakeetForRNNTModelTester,
)


if is_datasets_available():
    from datasets import Audio, load_dataset

if is_torch_available():
    import torch

    from transformers import (
        AutoProcessor,
        NemotronAsrRNNTConfig,
        NemotronAsrEncoder,
        NemotronAsrEncoderConfig,
        NemotronAsrForRNNT,
        TextIteratorStreamer,
    )
    from transformers.audio_utils import load_audio


# NemotronAsr's modeling/config inherit from Parakeet (see `modular_nemotron_asr.py`), so the tests inherit
# from the Parakeet testers/tests too and override only the NemotronAsr-specific classes and behaviours.
class NemotronAsrEncoderModelTester(ParakeetEncoderModelTester):
    def get_config(self):
        return NemotronAsrEncoderConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            dropout=self.dropout,
            dropout_positions=self.dropout,
            layerdrop=self.dropout,
            activation_dropout=self.dropout,
            attention_dropout=self.dropout,
            conv_kernel_size=self.conv_kernel_size,
            subsampling_factor=self.subsampling_factor,
            subsampling_conv_channels=self.subsampling_conv_channels,
            attention_bias=self.attention_bias,
            num_mel_bins=self.num_mel_bins,
            scale_input=self.scale_input,
        )

    def create_and_check_model(self, config, input_features, attention_mask):
        model = NemotronAsrEncoder(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_features, attention_mask=attention_mask)

        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, config.hidden_size)
        )


@require_torch
class NemotronAsrEncoderModelTest(ParakeetEncoderModelTest):
    all_model_classes = (NemotronAsrEncoder,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = NemotronAsrEncoderModelTester(self)
        self.config_tester = ConfigTester(self, config_class=NemotronAsrEncoderConfig, has_text_modality=False)


class NemotronAsrForRNNTModelTester(ParakeetForRNNTModelTester):
    def __init__(self, parent, encoder_kwargs=None, vocab_size=128, joint_hidden_size=32, **kwargs):
        super().__init__(parent, encoder_kwargs=encoder_kwargs, vocab_size=vocab_size, **kwargs)
        # NemotronAsr uses its own encoder tester (Parakeet's `__init__` wired the Parakeet one). The two share
        # identical settings, so the encoder-derived attributes set by `super().__init__` stay valid.
        self.encoder_model_tester = NemotronAsrEncoderModelTester(parent, **(encoder_kwargs or {}))
        self.joint_hidden_size = joint_hidden_size

    def get_config(self):
        return NemotronAsrRNNTConfig(
            vocab_size=self.vocab_size,
            decoder_hidden_size=self.decoder_hidden_size,
            joint_hidden_size=self.joint_hidden_size,
            num_decoder_layers=self.num_decoder_layers,
            hidden_act=self.hidden_act,
            max_symbols_per_step=self.max_symbols_per_step,
            encoder_config=self.encoder_model_tester.get_config().to_dict(),
            pad_token_id=self.pad_token_id,
            blank_token_id=self.blank_token_id,
        )

    def create_and_check_model(self, config, inputs_dict):
        model = NemotronAsrForRNNT(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(**inputs_dict)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.output_seq_length, self.encoder_model_tester.hidden_size),
        )

    def create_and_check_streaming_state(self, config, input_features, attention_mask):
        """Build a cache-aware variant of the test config and check the streaming state shape.

        The default tester uses an offline encoder (att_context_size=None); force a small
        cache-aware context here so the code path is actually exercised rather than skipped.
        """
        config.encoder_config.att_context_size = [3, 1]
        model = NemotronAsrForRNNT(config=config)
        model.to(torch_device).eval()
        state = model.get_initial_streaming_state(batch_size=self.batch_size, device=torch_device, dtype=torch.float32)
        self.parent.assertIn("cache_last_channel", state)
        self.parent.assertIn("last_token", state)
        self.parent.assertEqual(state["last_token"].shape, (self.batch_size, 1))
        self.parent.assertEqual(state["last_token"].dtype, torch.long)
        # cache_last_channel: (num_layers, batch, left_ctx, hidden_size)
        self.parent.assertEqual(
            state["cache_last_channel"].shape,
            (self.num_hidden_layers, self.batch_size, 3, self.hidden_size),
        )


@require_torch
class NemotronAsrForRNNTModelTest(ParakeetForRNNTModelTest):
    all_model_classes = (NemotronAsrForRNNT,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": NemotronAsrEncoder,
            "automatic-speech-recognition": NemotronAsrForRNNT,
        }
        if is_torch_available()
        else {}
    )

    def setUp(self):
        self.model_tester = NemotronAsrForRNNTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=NemotronAsrRNNTConfig)

    def test_streaming_state_init(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_streaming_state(*config_and_inputs)

    def test_streaming_generate_requires_num_lookahead_tokens(self):
        """Streaming `generate` (input_features passed as a generator) must be given `num_lookahead_tokens`
        explicitly. It sets both the attention right context used in every forward and the exact mel-chunk
        sizes the encoder consumes, so silently falling back to the model default could mismatch the chunks
        the processor produced and corrupt the transcript. The guard raises before the generator is consumed.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = NemotronAsrForRNNT(config=config).to(torch_device).eval()

        consumed = False

        def input_features_generator():
            nonlocal consumed
            consumed = True
            yield floats_tensor([self.model_tester.batch_size, 9, config.encoder_config.num_mel_bins])

        with self.assertRaisesRegex(ValueError, "must be passed explicitly"):
            model.generate(input_features=input_features_generator())
        # The guard must fire before the stream is touched, otherwise a chunk-size mismatch (not the missing
        # argument) is what gets reported.
        self.assertFalse(consumed, "streaming `generate` consumed the stream before validating num_lookahead_tokens")


# HF-format conversion of the original NeMo `nvidia/nemotron-speech-streaming-en-0.6b` checkpoint, published as a
# PR on the Hub until it is merged into the main revision.
NEMOTRON_ASR_CHECKPOINT = "nvidia/nemotron-speech-streaming-en-0.6b"
NEMOTRON_ASR_REVISION = "refs/pr/17"
# Long, single-speaker sample (~16 min of Obama's farewell address), used to exercise chunked streaming.
OBAMA_AUDIO_URL = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"


@require_torch
class NemotronAsrForRNNTIntegrationTest(unittest.TestCase):
    """Integration tests for NemotronAsrForRNNT.

    Expected transcriptions are the outputs of the original NeMo cache-aware streaming model
    `nvidia/nemotron-speech-streaming-en-0.6b` on the same audio, loaded from the
    `fixtures/nemotron_asr/expected_results_*.json` fixtures. The reproducers that regenerate these reference
    values live at https://gist.github.com/eustlb/a395a94b508dd9f20d405c63b45ab8eb (run `run_reproducers.sh`).
    Inference runs in float32 to track the NeMo reference as closely as possible.

    - Offline (non-streaming) `generate` uses the model's default attention context `[70, 13]` (the widest /
      best-WER setting, the first entry of `att_context_size`), matching the NeMo offline reference. The HF
      offline transcripts match NeMo exactly.
    - The streaming test feeds mel-frame chunks through a cache-aware `generate` at `[70, 6]`, mirroring the
      `test_stream_generate.py` recipe. Because the HF FastConformer encoder is a re-implementation and chunks
      differently from NeMo's `CacheAwareStreamingAudioBuffer`, a single sub-word can drift (~1e-3 numerical
      noise over the conformer stack flipping a borderline greedy emission); the one difference is annotated
      inline below.
    """

    _dataset = None

    @classmethod
    def setUp(cls):
        cls.checkpoint_name = NEMOTRON_ASR_CHECKPOINT
        cls.revision = NEMOTRON_ASR_REVISION
        cls.dtype = torch.float32
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
    def test_processor_set_num_lookahead_tokens(self):
        """`set_num_lookahead_tokens` is the only way to select the right attention context: it re-derives
        every streaming chunk-size property and the `num_lookahead_tokens` emitted by `__call__`. The processor
        is the single source of truth for the supported set, so it rejects unsupported values, and
        `streaming_latency_ms` is no longer accepted by `__call__`."""
        import inspect

        processor = self.processor
        subsampling = processor._subsampling_factor

        # The latency knob is gone — `set_num_lookahead_tokens` is the only entry point.
        self.assertNotIn("streaming_latency_ms", inspect.signature(processor.__call__).parameters)

        sample = self._load_datasamples(1)[0]
        for right in processor.supported_num_lookahead_tokens:
            processor.set_num_lookahead_tokens(right)
            self.assertEqual(processor.default_num_lookahead_tokens, right)
            self.assertEqual(processor.num_mel_frames_first_audio_chunk, 1 + subsampling * right)
            self.assertEqual(processor.num_mel_frames_per_audio_chunk, subsampling * (right + 1))
            inputs = processor(sample, sampling_rate=processor.feature_extractor.sampling_rate)
            self.assertEqual(inputs["num_lookahead_tokens"], right)

        # The processor owns the supported set and rejects unsupported values.
        with self.assertRaises(ValueError):
            processor.set_num_lookahead_tokens(max(processor.supported_num_lookahead_tokens) + 1)

    @slow
    def test_processor_streaming_latencies(self):
        """`streaming_latency_ms` reports the latency of the currently-selected right context, and
        `supported_streaming_latencies_ms` maps every supported right context to its latency. The streaming
        delay of a right context `r` is `(r + 1)` encoder frames, each `subsampling_factor * hop_length /
        sampling_rate` seconds long."""
        processor = self.processor
        fe = processor.feature_extractor
        frame_ms = processor._subsampling_factor * fe.hop_length / fe.sampling_rate * 1000

        for right in processor.supported_num_lookahead_tokens:
            processor.set_num_lookahead_tokens(right)
            self.assertEqual(processor.streaming_latency_ms, round((right + 1) * frame_ms))

        self.assertEqual(
            processor.supported_streaming_latencies_ms,
            {right: round((right + 1) * frame_ms) for right in processor.supported_num_lookahead_tokens},
        )

    @slow
    def test_rnnt_model_integration(self):
        # NeMo `nvidia/nemotron-speech-streaming-en-0.6b` reference; HF matches it exactly.
        # reproducer: https://gist.github.com/eustlb/a395a94b508dd9f20d405c63b45ab8eb#file-reproducer_single_rnnt-py
        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/nemotron_asr/expected_results_single.json"
        with open(RESULTS_PATH) as f:
            EXPECTED_TRANSCRIPTIONS = json.load(f)["transcriptions"]

        samples = self._load_datasamples(len(EXPECTED_TRANSCRIPTIONS))
        model = NemotronAsrForRNNT.from_pretrained(
            self.checkpoint_name, revision=self.revision, dtype=self.dtype, device_map="auto"
        )

        inputs = self.processor(samples, sampling_rate=self.processor.feature_extractor.sampling_rate)
        inputs.to(model.device, dtype=model.dtype)
        output = model.generate(**inputs, return_dict_in_generate=True)
        predicted_transcripts = self.processor.batch_decode(output.sequences, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)

    @slow
    def test_rnnt_model_integration_batched(self):
        # NeMo reference; all five HF transcripts match it exactly.
        # reproducer: https://gist.github.com/eustlb/a395a94b508dd9f20d405c63b45ab8eb#file-reproducer_batch_rnnt-py
        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/nemotron_asr/expected_results_batch.json"
        with open(RESULTS_PATH) as f:
            EXPECTED_TRANSCRIPTIONS = json.load(f)["transcriptions"]

        samples = self._load_datasamples(len(EXPECTED_TRANSCRIPTIONS))
        model = NemotronAsrForRNNT.from_pretrained(
            self.checkpoint_name, revision=self.revision, dtype=self.dtype, device_map="auto"
        )

        inputs = self.processor(samples, sampling_rate=self.processor.feature_extractor.sampling_rate)
        inputs.to(model.device, dtype=model.dtype)
        output = model.generate(**inputs, return_dict_in_generate=True)
        predicted_transcripts = self.processor.batch_decode(output.sequences, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)

    @slow
    def test_rnnt_model_integration_streaming(self):
        """Cache-aware streaming generation from a generator of mel-frame chunks.

        Mirrors `test_stream_generate.py`: the full mel spectrogram is sliced into contiguous chunks (49 frames
        for the first chunk, then 56) and fed to `generate` as a generator together with the streaming
        attention context `[70, 6]`. The decoder/encoder caches are threaded across chunks internally and the
        transcript is consumed incrementally via a `TextIteratorStreamer`.

        reproducer: https://gist.github.com/eustlb/a395a94b508dd9f20d405c63b45ab8eb#file-reproducer_streaming_rnnt-py
        """
        # The fixture holds the NeMo cache-aware streaming reference. The HF re-implementation matches it
        # except a single sub-word: NeMo emits "...rescued our economy from the worst crisis of our
        # lifetimes...", the Transformers re-implementation emits "cris" instead — a borderline greedy
        # emission flipped by ~1e-3 numerical drift in the re-implemented FastConformer encoder (WER-neutral).
        # We assert against the NeMo reference, so this test is expected to fail on that one sub-word until the
        # encoder drift is closed.
        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/nemotron_asr/expected_results_streaming.json"
        with open(RESULTS_PATH) as f:
            EXPECTED_TRANSCRIPTION = json.load(f)["transcription"]

        audio = load_audio(OBAMA_AUDIO_URL, sampling_rate=self.processor.feature_extractor.sampling_rate)
        model = NemotronAsrForRNNT.from_pretrained(
            self.checkpoint_name, revision=self.revision, dtype=self.dtype, device_map="auto"
        )

        # Select the streaming right attention context (lookahead, in subsampled encoder frames). This sizes
        # the mel chunks below (49 then 56 frames) and must be passed to `generate` so the forward matches.
        self.processor.set_num_lookahead_tokens(6)

        inputs = self.processor(audio, sampling_rate=self.processor.feature_extractor.sampling_rate)
        inputs.to(model.device, dtype=model.dtype)

        def input_features_generator():
            start_idx = 0
            chunk = self.processor.num_mel_frames_first_audio_chunk
            input_length = inputs.input_features.shape[1]
            while start_idx < input_length:
                end_idx = min(start_idx + chunk, input_length)
                yield inputs.input_features[:, start_idx:end_idx, :]
                start_idx = end_idx
                chunk = self.processor.num_mel_frames_per_audio_chunk

        streamer = TextIteratorStreamer(
            self.processor.tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        generate_kwargs = {
            "input_features": input_features_generator(),
            "num_lookahead_tokens": self.processor.default_num_lookahead_tokens,
            "streamer": streamer,
        }
        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()
        streamed_text = "".join(streamer)
        thread.join()

        self.assertEqual(streamed_text, EXPECTED_TRANSCRIPTION)
