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
"""Testing suite for the PyTorch Nemotron3Diarization model."""

import os
import tempfile
import unittest

from huggingface_hub import download_bucket_files
from safetensors.torch import load_file

from transformers import is_torch_available
from transformers.audio_utils import load_audio
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        AutoModel,
        AutoProcessor,
        Nemotron3DiarizationConfig,
        Nemotron3DiarizationEncoderConfig,
        Nemotron3DiarizationForAudioFrameClassification,
        Nemotron3DiarizationModel,
        Nemotron3DiarizationSpeakerCache,
    )


class Nemotron3DiarizationModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=256,
        is_training=False,
        num_mel_bins=8,
        subsampling_factor=8,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        speaker_hidden_size=8,
        num_speakers=4,
        speaker_cache_length=8,
        fifo_length=4,
        streaming_chunk_length=64,
        chunk_right_context=0,
        speaker_cache_update_period=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.num_mel_bins = num_mel_bins
        self.subsampling_factor = subsampling_factor
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.speaker_hidden_size = speaker_hidden_size
        self.num_speakers = num_speakers
        self.speaker_cache_length = speaker_cache_length
        self.fifo_length = fifo_length
        # named `streaming_chunk_length` because `chunk_length` has a different meaning for `ModelTesterMixin`
        self.streaming_chunk_length = streaming_chunk_length
        self.chunk_right_context = chunk_right_context
        self.speaker_cache_update_period = speaker_cache_update_period
        # A single streaming step: the whole input fits in one chunk, so recorded hidden states and attentions have the
        # encoder frame count as sequence length.
        self.encoder_seq_length = seq_length // subsampling_factor
        self.key_length = self.encoder_seq_length

    def get_config(self):
        return Nemotron3DiarizationConfig(
            encoder_config=Nemotron3DiarizationEncoderConfig(
                num_mel_bins=self.num_mel_bins,
                subsampling_factor=self.subsampling_factor,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                dropout=0.0,
            ),
            speaker_hidden_size=self.speaker_hidden_size,
            num_speakers=self.num_speakers,
            speaker_cache_length=self.speaker_cache_length,
            fifo_length=self.fifo_length,
            chunk_length=self.streaming_chunk_length,
            chunk_right_context=self.chunk_right_context,
            speaker_cache_update_period=self.speaker_cache_update_period,
        )

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.seq_length, self.num_mel_bins])
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        config = self.get_config()
        return config, input_features, attention_mask

    def create_and_check_model(self, config, input_features, attention_mask):
        model = Nemotron3DiarizationForAudioFrameClassification(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_features, attention_mask=attention_mask)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_speakers))

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_features": input_features, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class Nemotron3DiarizationModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Nemotron3DiarizationForAudioFrameClassification,) if is_torch_available() else ()
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = Nemotron3DiarizationModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Nemotron3DiarizationConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_encoder_config_roundtrip(self):
        config = self.model_tester.get_config()
        restored = Nemotron3DiarizationConfig.from_dict(config.to_dict())
        self.assertIsInstance(restored.encoder_config, Nemotron3DiarizationEncoderConfig)
        self.assertEqual(restored.encoder_config.to_dict(), config.encoder_config.to_dict())
        self.assertNotIn("hidden_size", config.to_dict())

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_base_model_precomputed_embeddings(self):
        config, input_features, attention_mask = self.model_tester.prepare_config_and_inputs()
        config.chunk_length = 4
        model = Nemotron3DiarizationModel(config).to(torch_device).eval()
        attention_mask = self._prefix_mask(attention_mask)
        with torch.no_grad():
            expected = model(input_features, attention_mask=attention_mask).logits
            inputs_embeds = model.encoder.feature_stacking(input_features)
            lengths = model._get_feat_extract_output_lengths(attention_mask.sum(-1))
            mask = torch.arange(inputs_embeds.shape[1], device=torch_device)[None, :] < lengths[:, None]
            actual = model(inputs_embeds=inputs_embeds, attention_mask=mask).logits
        self.assertGreater(inputs_embeds.shape[1], config.chunk_length)
        self.assertEqual(
            actual.shape,
            (
                input_features.shape[0],
                inputs_embeds.shape[1] * config.encoder_config.subsampling_factor,
                config.num_speakers,
            ),
        )
        torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            model(input_features, inputs_embeds=inputs_embeds)
        with self.assertRaises(ValueError):
            model()

    def test_base_model_and_legacy_checkpoint_loading(self):
        config = self.model_tester.get_config()
        model = Nemotron3DiarizationForAudioFrameClassification(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            base = AutoModel.from_pretrained(directory)
            self.assertIsInstance(base, Nemotron3DiarizationModel)
            for name, tensor in base.state_dict().items():
                torch.testing.assert_close(tensor, model.model.state_dict()[name])

            legacy_state = {name.removeprefix("model."): tensor for name, tensor in model.state_dict().items()}
            model.save_pretrained(directory, state_dict=legacy_state)
            restored, info = Nemotron3DiarizationForAudioFrameClassification.from_pretrained(
                directory, output_loading_info=True
            )
            self.assertFalse(info["missing_keys"])
            self.assertFalse(info["unexpected_keys"])
            for name, tensor in restored.state_dict().items():
                torch.testing.assert_close(tensor, model.state_dict()[name])

    @unittest.skip(reason="Nemotron3Diarization does not use inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass

    def _prefix_mask(self, attention_mask):
        # Streaming operates on per-sample lengths, so the mask must be a prefix mask.
        lengths = attention_mask.sum(-1)
        return torch.arange(attention_mask.shape[1], device=lengths.device)[None, :] < lengths[:, None]

    def test_streaming_steps_match_offline(self):
        config, input_features, attention_mask = self.model_tester.prepare_config_and_inputs()
        config.chunk_length = 4
        config.chunk_right_context = 1
        attention_mask = self._prefix_mask(attention_mask)
        model = Nemotron3DiarizationForAudioFrameClassification(config).to(torch_device).eval()

        with torch.no_grad():
            offline_logits = model(input_features, attention_mask=attention_mask).logits

        chunk_frames = config.chunk_length * config.encoder_config.subsampling_factor
        lookahead_frames = config.chunk_right_context * config.encoder_config.subsampling_factor
        num_frames = input_features.shape[1]
        step_logits, speaker_cache, start = [], None, 0
        with torch.no_grad():
            while start + chunk_frames + lookahead_frames <= num_frames:
                end = start + chunk_frames + lookahead_frames
                outputs = model(
                    input_features[:, start:end],
                    attention_mask=attention_mask[:, start:end],
                    speaker_cache=speaker_cache,
                    use_cache=True,
                )
                step_logits.append(outputs.logits)
                speaker_cache = outputs.speaker_cache
                start += chunk_frames
            # Final flush: the remaining frames are the last (partial) chunk without look-ahead.
            outputs = model(
                input_features[:, start:],
                attention_mask=attention_mask[:, start:],
                speaker_cache=speaker_cache,
            )
            step_logits.append(outputs.logits)
        self.assertIsNone(outputs.speaker_cache)
        streaming_logits = torch.cat(step_logits, dim=1)
        self.assertEqual(streaming_logits.shape, offline_logits.shape)
        torch.testing.assert_close(streaming_logits, offline_logits, atol=1e-5, rtol=1e-5)

    def test_streaming_rejects_too_many_lookahead_frames(self):
        config, input_features, attention_mask = self.model_tester.prepare_config_and_inputs()
        config.chunk_length = 4
        config.chunk_right_context = 1
        model = Nemotron3DiarizationForAudioFrameClassification(config).to(torch_device).eval()
        too_many = (config.chunk_length + config.chunk_right_context + 1) * config.encoder_config.subsampling_factor
        with self.assertRaises(ValueError):
            model(input_features[:, :too_many], use_cache=True)
        with self.assertRaises(ValueError):
            model(input_features[:, : config.encoder_config.subsampling_factor], use_cache=True)

    @slow
    @require_torch_gpu
    def test_speaker_cache_fullgraph(self):
        config = Nemotron3DiarizationConfig(
            encoder_config=Nemotron3DiarizationEncoderConfig(hidden_size=16, num_attention_heads=4),
            num_speakers=2,
            speaker_cache_length=8,
            speaker_cache_silence_frames_per_speaker=1,
            fifo_length=4,
            speaker_cache_update_period=4,
        )
        eager_cache = Nemotron3DiarizationSpeakerCache(config)
        compiled_cache = Nemotron3DiarizationSpeakerCache(config)
        chunk = torch.randn(2, 4, config.encoder_config.hidden_size, device=torch_device)
        silence = torch.randn(config.encoder_config.hidden_size, device=torch_device)
        for cache in (eager_cache, compiled_cache):
            cache.lazy_initialization(chunk)
        buffers = (compiled_cache.embeds, compiled_cache.probs, compiled_cache.fifo)
        addresses = [buffer.data_ptr() for buffer in buffers]

        def step(cache, chunk, probs, silence):
            embeds = cache.get_embeds(chunk)
            chunk_input_embeds = torch.cat([embeds, chunk], dim=1)
            chunk_logits = probs.logit().repeat_interleave(config.encoder_config.subsampling_factor, dim=1)
            cache.update(chunk_input_embeds, chunk_logits, silence)
            return embeds

        compiled_step = torch.compile(step, fullgraph=True)
        with torch.no_grad():
            # Empty state, FIFO filling, cache filling, first compression, and repeated compression.
            for _ in range(6):
                chunk = torch.randn_like(chunk)
                length = eager_cache.cache_length + eager_cache.fifo_length + chunk.shape[1]
                probs = torch.rand(2, length, config.num_speakers, device=torch_device)
                expected = step(eager_cache, chunk, probs, silence)
                actual = compiled_step(compiled_cache, chunk, probs, silence)
                torch.testing.assert_close(actual, expected)
                for name, address in zip(("embeds", "probs", "fifo"), addresses):
                    actual_buffer = getattr(compiled_cache, name)
                    self.assertEqual(actual_buffer.data_ptr(), address)
                    torch.testing.assert_close(actual_buffer, getattr(eager_cache, name))
                self.assertEqual(compiled_cache.cache_length, eager_cache.cache_length)
                self.assertEqual(compiled_cache.fifo_length, eager_cache.fifo_length)
        self.assertTrue(compiled_cache.is_compressed)


@require_torch
class Nemotron3DiarizationForAudioFrameClassificationIntegrationTest(unittest.TestCase):
    """
    Validate the speaker probabilities against NeMo, offline and streaming.

    reproducer (all cases, uploads the golden to
    ``hf://buckets/hf-internal-testing/nemotron3-diarization-integration-test/expected_probabilities.safetensors``):
        ~/audio-model-work/nemotron3_diarization/reproducers/reproducer_probabilities.py
    gist: https://gist.github.com/eustlb/f20c18d24580e7416697aec5747d1107
    """

    # Model-card "low latency" profile (1.04 s input buffer), in encoder frames of 80 ms.
    LOW_LATENCY_PROFILE = {
        "fifo_length": 264,
        "chunk_length": 9,
        "chunk_right_context": 4,
        "speaker_cache_update_period": 222,
    }
    AUDIO_URL = (
        "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/diarization_example.mp3"
    )
    # Seconds kept from the conversation: the whole 97.6 s recording, and a 30 s excerpt, so that the batched test
    # pads two samples of different lengths.
    SECONDS = {"long": None, "short": 30}

    @classmethod
    def setUp(cls):
        cls.checkpoint_name = "nvidia/Nemotron-3-Diarization-preview"
        cls.revision = "refs/pr/5"
        cls.bucket = "hf-internal-testing/nemotron3-diarization-integration-test"
        cls.dtype = torch.float32
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint_name, revision=cls.revision)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def _load_sample(self, name):
        sampling_rate = self.processor.feature_extractor.sampling_rate
        audio = load_audio(self.AUDIO_URL, sampling_rate=sampling_rate)
        num_seconds = self.SECONDS[name]
        return audio if num_seconds is None else audio[: num_seconds * sampling_rate]

    def _load_expected(self, key):
        remote = "expected_probabilities.safetensors"
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, remote)
            download_bucket_files(self.bucket, files=[(remote, local)])
            return load_file(local)[key]

    def _load_model(self, **config_overrides):
        model = Nemotron3DiarizationForAudioFrameClassification.from_pretrained(
            self.checkpoint_name, revision=self.revision, dtype=self.dtype, **config_overrides
        )
        return model.to(torch_device).eval()

    def _assert_close(self, probabilities, expected):
        """
        Compare with NeMo everywhere but on the last encoder frame.

        That frame sees a neighbour frame NeMo does not have — the zero padding of a batched sample, or the extra
        spectrogram frame the feature extractor emits at the end of a recording because it centers its windows.
        """
        boundary = self.processor.subsampling_factor
        self.assertEqual(probabilities.shape, expected.shape)
        torch.testing.assert_close(probabilities[:-boundary], expected[:-boundary], atol=1e-3, rtol=0.0)

    def _speaker_probabilities(self, model, names):
        """Per-sample speaker probabilities on the valid frames, as in the reproducer."""
        inputs = self.processor(
            [self._load_sample(name) for name in names],
            sampling_rate=self.processor.feature_extractor.sampling_rate,
        )
        inputs = inputs.to(torch_device, dtype=self.dtype)
        with torch.no_grad():
            logits = model(**inputs).logits
        probabilities = logits.sigmoid().cpu()
        num_frames = inputs.attention_mask.sum(-1).cpu()
        return [probabilities[i, : num_frames[i]] for i in range(len(names))]

    @slow
    def test_model_integration(self):
        """reproducer: reproducer_probabilities.py, offline profile on the whole conversation"""
        EXPECTED_PROBABILITIES = self._load_expected("offline_long")

        model = self._load_model()
        probabilities = self._speaker_probabilities(model, ["long"])[0]
        self._assert_close(probabilities, EXPECTED_PROBABILITIES)

    @slow
    def test_model_integration_batched(self):
        """
        reproducer: reproducer_probabilities.py — two samples of different lengths, padding must not change the
        valid frames.
        """
        EXPECTED_PROBABILITIES = [self._load_expected("offline_long"), self._load_expected("offline_short")]

        model = self._load_model()
        all_probabilities = self._speaker_probabilities(model, ["long", "short"])
        for probabilities, expected in zip(all_probabilities, EXPECTED_PROBABILITIES):
            self._assert_close(probabilities, expected)

    @slow
    def test_model_integration_low_latency(self):
        """reproducer: reproducer_probabilities.py — 720 ms chunks, 320 ms look-ahead, speaker-cache compressions"""
        EXPECTED_PROBABILITIES = self._load_expected("low_latency_long")

        model = self._load_model(**self.LOW_LATENCY_PROFILE)
        probabilities = self._speaker_probabilities(model, ["long"])[0]
        self._assert_close(probabilities, EXPECTED_PROBABILITIES)

    @slow
    def test_model_integration_streaming_steps(self):
        """
        reproducer: reproducer_probabilities.py — feeding pre-extracted chunk + look-ahead frames one step at a time
        with `use_cache=True` must reproduce the offline forward and the NeMo reference.
        """
        EXPECTED_PROBABILITIES = self._load_expected("low_latency_long")

        model = self._load_model(**self.LOW_LATENCY_PROFILE)
        inputs = self.processor(
            self._load_sample("long"), sampling_rate=self.processor.feature_extractor.sampling_rate
        ).to(torch_device, dtype=self.dtype)
        input_features, attention_mask = inputs.input_features, inputs.attention_mask
        chunk_frames = model.config.chunk_length * model.config.encoder_config.subsampling_factor
        lookahead_frames = model.config.chunk_right_context * model.config.encoder_config.subsampling_factor

        with torch.no_grad():
            offline_logits = model(**inputs).logits
            step_logits, speaker_cache, start = [], None, 0
            while start + chunk_frames + lookahead_frames <= input_features.shape[1]:
                end = start + chunk_frames + lookahead_frames
                outputs = model(
                    input_features[:, start:end],
                    attention_mask=attention_mask[:, start:end],
                    speaker_cache=speaker_cache,
                    use_cache=True,
                )
                step_logits.append(outputs.logits)
                speaker_cache = outputs.speaker_cache
                start += chunk_frames
            # Final flush: the remaining frames are the last (partial) chunk, without look-ahead.
            outputs = model(
                input_features[:, start:], attention_mask=attention_mask[:, start:], speaker_cache=speaker_cache
            )
            step_logits.append(outputs.logits)
        streaming_logits = torch.cat(step_logits, dim=1)

        torch.testing.assert_close(streaming_logits, offline_logits, atol=5e-4, rtol=0.0)
        num_frames = int(attention_mask.sum())
        self._assert_close(streaming_logits[0, :num_frames].sigmoid().cpu(), EXPECTED_PROBABILITIES)

    @slow
    def test_model_integration_streaming_audio_chunks(self):
        """
        reproducer: reproducer_probabilities.py — real streaming: the processor sizes each chunk and extracts its
        spectrogram (centered windows for the first chunk, uncentered afterwards), as for Nemotron ASR Streaming.
        Mirrors the streaming example of the model documentation.
        """
        EXPECTED_PROBABILITIES = self._load_expected("low_latency_long")

        model = self._load_model(**self.LOW_LATENCY_PROFILE)
        processor = self.processor
        processor.set_streaming_profile(
            chunk_length=self.LOW_LATENCY_PROFILE["chunk_length"],
            chunk_right_context=self.LOW_LATENCY_PROFILE["chunk_right_context"],
        )
        self.assertEqual(processor.streaming_latency_ms, 1040)
        audio = self._load_sample("long")
        sampling_rate = processor.feature_extractor.sampling_rate

        def input_features_generator():
            def extract(start, stop, is_first):
                chunk = audio[start:stop] if stop is not None else audio[start:]
                inputs = processor(
                    chunk, sampling_rate=sampling_rate, is_streaming=True, is_first_audio_chunk=is_first
                )
                return inputs.input_features

            yield extract(0, processor.num_samples_first_audio_chunk, True), False

            mel_frame_idx = processor.num_mel_frames_per_step
            start_idx = processor.audio_chunk_start(mel_frame_idx)
            while (end_idx := start_idx + processor.num_samples_per_audio_chunk) <= audio.shape[0]:
                yield extract(start_idx, end_idx, False), False
                mel_frame_idx += processor.num_mel_frames_per_step
                start_idx = processor.audio_chunk_start(mel_frame_idx)

            yield extract(start_idx, None, False), True

        speaker_cache, step_logits = None, []
        with torch.no_grad():
            for input_features, is_last_chunk in input_features_generator():
                outputs = model(
                    input_features.to(torch_device, dtype=self.dtype),
                    speaker_cache=speaker_cache,
                    use_cache=not is_last_chunk,
                )
                step_logits.append(outputs.logits)
                speaker_cache = outputs.speaker_cache
        probabilities = torch.cat(step_logits, dim=1)[0].sigmoid().cpu()

        # The last frame of a recording is emitted only when its analysis window fits inside the audio, so a streamed
        # run ends on the same frame as the offline path or one 10 ms frame earlier. The final chunk is flushed
        # without look-ahead and its spectrogram stops with the audio instead of being centered on it, so it is
        # excluded like the trailing chunk of the other streaming tests.
        num_emitted = probabilities.shape[0]
        self.assertIn(EXPECTED_PROBABILITIES.shape[0] - num_emitted, (0, 1))
        num_flushed = step_logits[-1].shape[1]
        torch.testing.assert_close(
            probabilities[:-num_flushed], EXPECTED_PROBABILITIES[: num_emitted - num_flushed], atol=1e-3, rtol=0.0
        )
