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
        NemotronAsrConfig,
        NemotronAsrEncoder,
        NemotronAsrEncoderConfig,
        NemotronAsrForRNNT,
        TextIteratorStreamer,
    )
    from transformers.audio_utils import load_audio


class NemotronAsrEncoderModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=1024,
        is_training=True,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        hidden_act="silu",
        dropout=0.0,  # so gradient checkpointing doesn't fail
        conv_kernel_size=9,
        subsampling_factor=8,
        subsampling_conv_channels=32,
        attention_bias=True,
        num_mel_bins=80,
        scale_input=True,
    ):
        # testing suite parameters
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_mel_bins = num_mel_bins
        self.is_training = is_training

        # config parameters
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.dropout = dropout
        self.conv_kernel_size = conv_kernel_size
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.attention_bias = attention_bias
        self.num_mel_bins = num_mel_bins
        self.scale_input = scale_input

        # Calculate output sequence length after subsampling
        self.output_seq_length = seq_length // subsampling_factor
        self.encoder_seq_length = self.output_seq_length
        self.key_length = self.output_seq_length

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.seq_length, self.num_mel_bins])
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        config = self.get_config()

        return config, input_features, attention_mask

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

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_features": input_features,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class NemotronAsrEncoderModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (NemotronAsrEncoder,) if is_torch_available() else ()

    test_resize_embeddings = False

    @unittest.skip(reason="No available flash-SDPA kernels for NemotronAsr test shapes on this setup")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def setUp(self):
        self.model_tester = NemotronAsrEncoderModelTester(self)
        self.config_tester = ConfigTester(self, config_class=NemotronAsrEncoderConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="NemotronAsrEncoder does not use inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass


class NemotronAsrForRNNTModelTester:
    def __init__(
        self,
        parent,
        encoder_kwargs=None,
        is_training=True,
        vocab_size=128,
        decoder_hidden_size=32,
        joint_hidden_size=32,
        num_decoder_layers=1,
        hidden_act="relu",
        max_symbols_per_step=5,
        pad_token_id=2,
    ):
        if encoder_kwargs is None:
            encoder_kwargs = {}

        self.parent = parent
        self.encoder_model_tester = NemotronAsrEncoderModelTester(parent, **encoder_kwargs)
        self.is_training = is_training

        self.batch_size = self.encoder_model_tester.batch_size
        self.output_seq_length = self.encoder_model_tester.output_seq_length
        self.num_hidden_layers = self.encoder_model_tester.num_hidden_layers
        self.hidden_size = self.encoder_model_tester.hidden_size
        self.seq_length = self.encoder_model_tester.output_seq_length
        self.encoder_seq_length = self.encoder_model_tester.output_seq_length

        self.vocab_size = vocab_size
        self.decoder_hidden_size = decoder_hidden_size
        self.joint_hidden_size = joint_hidden_size
        self.num_decoder_layers = num_decoder_layers
        self.hidden_act = hidden_act
        self.max_symbols_per_step = max_symbols_per_step
        self.pad_token_id = pad_token_id
        self.blank_token_id = vocab_size - 1

    def prepare_config_and_inputs(self):
        _, input_features, attention_mask = self.encoder_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return config, input_features, attention_mask

    def get_config(self):
        return NemotronAsrConfig(
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

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask = self.prepare_config_and_inputs()
        decoder_input_ids = ids_tensor([self.batch_size, 1], self.vocab_size)
        inputs_dict = {
            "input_features": input_features,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
        }
        return config, inputs_dict


@require_torch
class NemotronAsrForRNNTModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (NemotronAsrForRNNT,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": NemotronAsrEncoder,
            "automatic-speech-recognition": NemotronAsrForRNNT,
        }
        if is_torch_available()
        else {}
    )

    test_attention_outputs = False
    test_resize_embeddings = False
    test_torch_exportable = False
    _is_composite = True

    @unittest.skip(reason="No available flash-SDPA kernels for NemotronAsr test shapes on this setup")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def setUp(self):
        self.model_tester = NemotronAsrForRNNTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=NemotronAsrConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_streaming_state_init(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_streaming_state(*config_and_inputs)

    @unittest.skip(reason="NemotronAsrForRNNT does not use inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(
        reason="NemotronAsrForRNNT is a transducer, not a standard encoder-decoder: no separate text config to set"
    )
    def test_attn_implementation_composite_models(self):
        pass

    @unittest.skip(
        reason="NemotronAsrForRNNT is a transducer with an LSTM prediction network; "
        "it does not expose encoder_hidden_states in the standard encoder-decoder sense"
    )
    def test_hidden_states_output(self):
        pass

    @unittest.skip(
        reason="NemotronAsrForRNNT is a transducer with an LSTM prediction network; "
        "it does not expose encoder_hidden_states in the standard encoder-decoder sense"
    )
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(
        reason="NemotronAsrForRNNT has a custom generate() that is not fully compatible with GenerationTesterMixin"
    )
    def test_generation_tester_mixin_inheritance(self):
        pass

    @unittest.skip(reason="NemotronAsrForRNNT is a flat composite model without a separate base_model sub-module")
    def test_model_base_model_prefix(self):
        pass

    @unittest.skip(reason="NemotronAsrForRNNT decoder is an LSTM prediction network without attention")
    def test_flex_attention_with_grads(self):
        pass

    # Original function assumes vision+text model, so overwrite since NemotronAsr is audio+text
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


# Local conversion of the original NeMo `nvidia/nemotron-speech-streaming-en-0.6b` checkpoint. Replace with the
# public Hub id once the converted weights are published.
NEMOTRON_ASR_CHECKPOINT = "/raid/eustache/nemotron-speech-streaming-en-0.6b-hf"
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
        cls.dtype = torch.float32
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint_name)

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
    def test_rnnt_model_integration(self):
        # NeMo `nvidia/nemotron-speech-streaming-en-0.6b` reference; HF matches it exactly.
        # reproducer: https://gist.github.com/eustlb/a395a94b508dd9f20d405c63b45ab8eb#file-reproducer_single_rnnt-py
        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/nemotron_asr/expected_results_single.json"
        with open(RESULTS_PATH) as f:
            EXPECTED_TRANSCRIPTIONS = json.load(f)["transcriptions"]

        samples = self._load_datasamples(len(EXPECTED_TRANSCRIPTIONS))
        model = NemotronAsrForRNNT.from_pretrained(self.checkpoint_name, dtype=self.dtype, device_map="auto")

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
        model = NemotronAsrForRNNT.from_pretrained(self.checkpoint_name, dtype=self.dtype, device_map="auto")

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
        model = NemotronAsrForRNNT.from_pretrained(self.checkpoint_name, dtype=self.dtype, device_map="auto")

        inputs = self.processor(audio, sampling_rate=self.processor.feature_extractor.sampling_rate)
        inputs.to(model.device, dtype=model.dtype)

        def input_features_generator():
            start_idx, first_chunk_size, chunk_size = 0, 49, 56
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
            "att_context_size": [70, 6],
            "streamer": streamer,
        }
        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()
        streamed_text = "".join(streamer)
        thread.join()

        self.assertEqual(streamed_text, EXPECTED_TRANSCRIPTION)
