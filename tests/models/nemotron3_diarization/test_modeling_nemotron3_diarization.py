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
    require_torch,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import (
        AutoProcessor,
        AutoModelForAudioFrameClassification,
        Nemotron3DiarizationAudioConfig,
        Nemotron3DiarizationConfig,
        Nemotron3DiarizationForAudioFrameClassification,
        Nemotron3DiarizationHeadConfig,
        Nemotron3DiarizationSpeakerCache,
        Nemotron3DiarizationStreamingConfig,
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
        # the tiny model uses the same FIFO sizes offline and streaming, so that the two modes can be compared
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
            audio_config=Nemotron3DiarizationAudioConfig(
                num_mel_bins=self.num_mel_bins,
                subsampling_factor=self.subsampling_factor,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                dropout=0.0,
            ),
            head_config=Nemotron3DiarizationHeadConfig(
                hidden_size=self.speaker_hidden_size,
                num_speakers=self.num_speakers,
            ),
            streaming_config=Nemotron3DiarizationStreamingConfig(
                speaker_cache_length=self.speaker_cache_length,
                fifo_length=self.fifo_length,
                speaker_cache_update_period=self.speaker_cache_update_period,
            ),
            chunk_length=self.streaming_chunk_length,
            chunk_right_context=self.chunk_right_context,
            fifo_length=self.fifo_length,
            speaker_cache_update_period=self.speaker_cache_update_period,
        )

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.seq_length, self.num_mel_bins])
        # TODO: @eustlb, really important! we need to standardize models silently expecting right padding
        # right padding, as the processor produces: the model reads the mask as per-sample lengths
        lengths = torch.randint(self.seq_length // 2, self.seq_length + 1, (self.batch_size,), device=torch_device)
        attention_mask = (torch.arange(self.seq_length, device=torch_device)[None, :] < lengths[:, None]).long()
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
    # the base model saves its `audio_config`, from which the head's `Nemotron3DiarizationConfig` cannot be rebuilt
    test_missing_keys = False

    def setUp(self):
        self.model_tester = Nemotron3DiarizationModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Nemotron3DiarizationConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Nemotron3Diarization does not use inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass

    def test_modes(self):
        """
        Offline (neither `num_lookahead_frames` nor `speaker_cache`) chunks the input by the config and returns no
        cache; streaming (either given) takes the input as one chunk and returns a cache sized by `streaming_config`.
        """
        config, input_features, attention_mask = self.model_tester.prepare_config_and_inputs()
        config.chunk_length, config.chunk_right_context = 4, 1
        config.streaming_config.fifo_length, config.streaming_config.speaker_cache_update_period = 6, 3
        model = Nemotron3DiarizationForAudioFrameClassification(config).to(torch_device).eval()
        with torch.no_grad():
            offline = model(input_features, attention_mask=attention_mask)
            first = model(input_features, attention_mask=attention_mask, num_lookahead_frames=1)
            last = model(input_features, attention_mask=attention_mask, speaker_cache=first.speaker_cache)
        self.assertIsNone(offline.speaker_cache)
        self.assertEqual(offline.logits.shape[1], input_features.shape[1])
        # a streaming forward scores everything but its look-ahead, as one chunk
        subsampling_factor = config.audio_config.subsampling_factor
        self.assertEqual(first.logits.shape[1], input_features.shape[1] - subsampling_factor)
        self.assertEqual(first.speaker_cache.fifo_length, 6)
        self.assertEqual(first.speaker_cache.speaker_cache_update_period, 3)
        self.assertTrue(first.speaker_cache.streaming)
        self.assertIs(last.speaker_cache, first.speaker_cache)
        self.assertEqual(last.logits.shape[1], input_features.shape[1])
        self.assertFalse(torch.allclose(first.logits, offline.logits[:, : first.logits.shape[1]]))

        # a cache built by the offline loop cannot be continued
        with self.assertRaises(ValueError):
            model(input_features, speaker_cache=Nemotron3DiarizationSpeakerCache(config, streaming=False))

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_streaming_steps_match_offline(self):
        """With equal FIFO sizes, feeding the offline chunks one forward at a time reproduces the offline forward."""
        config, input_features, attention_mask = self.model_tester.prepare_config_and_inputs()
        config.chunk_length = 4
        config.chunk_right_context = 1
        model = Nemotron3DiarizationForAudioFrameClassification(config).to(torch_device).eval()

        with torch.no_grad():
            offline_logits = model(input_features, attention_mask=attention_mask).logits

        chunk_frames = config.chunk_length * config.audio_config.subsampling_factor
        lookahead_frames = config.chunk_right_context * config.audio_config.subsampling_factor
        num_frames = input_features.shape[1]
        step_logits, speaker_cache, start = [], None, 0
        with torch.no_grad():
            while start + chunk_frames + lookahead_frames <= num_frames:
                end = start + chunk_frames + lookahead_frames
                outputs = model(
                    input_features[:, start:end],
                    attention_mask=attention_mask[:, start:end],
                    speaker_cache=speaker_cache,
                    num_lookahead_frames=config.chunk_right_context,
                )
                step_logits.append(outputs.logits)
                speaker_cache = outputs.speaker_cache
                start += chunk_frames
            # The last call: the remaining frames are the last (partial) chunk, none of them is look-ahead.
            outputs = model(
                input_features[:, start:],
                attention_mask=attention_mask[:, start:],
                speaker_cache=speaker_cache,
            )
            step_logits.append(outputs.logits)
        self.assertIs(outputs.speaker_cache, speaker_cache)
        streaming_logits = torch.cat(step_logits, dim=1)
        self.assertEqual(streaming_logits.shape, offline_logits.shape)
        torch.testing.assert_close(streaming_logits, offline_logits, atol=1e-5, rtol=1e-5)

    def test_streaming_rejects_invalid_lookahead(self):
        config, input_features, _ = self.model_tester.prepare_config_and_inputs()
        model = Nemotron3DiarizationForAudioFrameClassification(config).to(torch_device).eval()
        subsampling_factor = config.audio_config.subsampling_factor
        with self.assertRaises(ValueError):
            model(input_features, num_lookahead_frames=-1)
        # nothing but look-ahead
        with self.assertRaises(ValueError):
            model(input_features[:, :subsampling_factor], num_lookahead_frames=1)


@require_torch
class Nemotron3DiarizationIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    """
    Validate the speaker probabilities against NeMo, offline and streaming.

    reproducer (all cases, uploads the golden to
    ``hf://buckets/hf-internal-testing/nemotron3-diarization-integration-test/expected_probabilities.safetensors``):
        ~/audio-model-work/nemotron3_diarization/reproducers/reproducer_probabilities.py
    gist: https://gist.github.com/eustlb/f20c18d24580e7416697aec5747d1107
    """

    AUDIO_URL = (
        "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/diarization_example.mp3"
    )
    # Seconds kept from the conversation: the whole 97.6 s recording, and a 30 s excerpt, so that the batched test
    # pads two samples of different lengths.
    SECONDS = {"long": None, "short": 30}

    def setUp(self):
        super().setUp()
        self.checkpoint_name = "nvidia/Nemotron-3-Diarization-preview"
        self.revision = "refs/pr/6"
        self.bucket = "hf-internal-testing/nemotron3-diarization-integration-test"
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_name, revision=self.revision)

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
        return AutoModelForAudioFrameClassification.from_pretrained(
            self.checkpoint_name, revision=self.revision, device_map=torch_device, **config_overrides
        )

    def _low_latency_offline_model(self):
        """
        The `low_latency` streaming mode run as one offline forward: chunking and FIFO sizes of the mode moved to the
        offline fields of the config, which reproduces the streaming session on pre-extracted features.
        """
        chunk_length, chunk_right_context = self.processor.streaming_modes["low_latency"]
        return self._load_model(
            chunk_length=chunk_length,
            chunk_right_context=chunk_right_context,
            fifo_length=264,
            speaker_cache_update_period=222,
        )

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
            [self._load_sample(name) for name in names], sampling_rate=self.processor.feature_extractor.sampling_rate
        )
        inputs = inputs.to(torch_device, dtype=model.dtype)
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

        model = self._low_latency_offline_model()
        probabilities = self._speaker_probabilities(model, ["long"])[0]
        self._assert_close(probabilities, EXPECTED_PROBABILITIES)

    @slow
    def test_model_integration_streaming_steps(self):
        """
        reproducer: reproducer_probabilities.py — feeding pre-extracted chunk + look-ahead frames one forward at a
        time, in streaming mode, must reproduce the equivalent offline forward and the NeMo reference.
        """
        EXPECTED_PROBABILITIES = self._load_expected("low_latency_long")

        model = self._load_model()
        offline_model = self._low_latency_offline_model()
        processor = self.processor
        inputs = processor(self._load_sample("long"), sampling_rate=processor.feature_extractor.sampling_rate)
        inputs = inputs.to(torch_device, dtype=model.dtype)
        input_features, attention_mask = inputs.input_features, inputs.attention_mask
        self.assertEqual(processor.streaming_mode, "low_latency")
        _, chunk_right_context = processor.streaming_modes[processor.streaming_mode]
        chunk_frames = processor.num_mel_frames_per_step
        lookahead_frames = processor.num_mel_frames_per_audio_chunk - chunk_frames

        with torch.no_grad():
            offline_logits = offline_model(**inputs).logits
            step_logits, speaker_cache, start = [], None, 0
            while start + chunk_frames + lookahead_frames <= input_features.shape[1]:
                end = start + chunk_frames + lookahead_frames
                outputs = model(
                    input_features[:, start:end],
                    attention_mask=attention_mask[:, start:end],
                    speaker_cache=speaker_cache,
                    num_lookahead_frames=chunk_right_context,
                )
                step_logits.append(outputs.logits)
                speaker_cache = outputs.speaker_cache
                start += chunk_frames
            # The last chunk: the remaining frames, none of them is look-ahead.
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

        model = self._load_model()
        processor = self.processor
        self.assertEqual(processor.streaming_mode, "low_latency")
        self.assertEqual(processor.streaming_latency_ms, 1040)
        audio = self._load_sample("long")
        sampling_rate = processor.feature_extractor.sampling_rate

        def inputs_generator():
            def extract(start, stop, is_first, is_last):
                chunk = audio[start:stop] if stop is not None else audio[start:]
                return processor(
                    chunk,
                    sampling_rate=sampling_rate,
                    is_streaming=True,
                    is_first_audio_chunk=is_first,
                    is_last_audio_chunk=is_last,
                )

            yield extract(0, processor.num_samples_first_audio_chunk, True, False)

            mel_frame_idx = processor.num_mel_frames_per_step
            start_idx = processor.audio_chunk_start(mel_frame_idx)
            while (end_idx := start_idx + processor.num_samples_per_audio_chunk) <= audio.shape[0]:
                yield extract(start_idx, end_idx, False, False)
                mel_frame_idx += processor.num_mel_frames_per_step
                start_idx = processor.audio_chunk_start(mel_frame_idx)

            yield extract(start_idx, None, False, True)

        speaker_cache, step_logits = None, []
        with torch.no_grad():
            for inputs in inputs_generator():
                inputs = inputs.to(torch_device, dtype=model.dtype)
                outputs = model(**inputs, speaker_cache=speaker_cache)
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
