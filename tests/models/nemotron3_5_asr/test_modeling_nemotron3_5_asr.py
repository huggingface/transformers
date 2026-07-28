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
        Nemotron3_5AsrForRNNT,
        NemotronAsrStreamingEncoderConfig,
        TextIteratorStreamer,
    )
    from transformers.audio_utils import load_audio


FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures/nemotron3_5_asr"


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
        num_decoder_layers=1,
        hidden_act="relu",
        max_symbols_per_step=5,
        pad_token_id=2,
        num_prompts=8,
        prompt_intermediate_size=16,
        default_prompt_id=3,
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
        self.num_decoder_layers = num_decoder_layers
        self.hidden_act = hidden_act
        self.max_symbols_per_step = max_symbols_per_step
        self.pad_token_id = pad_token_id
        self.num_prompts = num_prompts
        self.default_prompt_id = default_prompt_id
        self.prompt_intermediate_size = prompt_intermediate_size
        self.is_training = is_training

        self.blank_token_id = vocab_size - 1
        self.output_seq_length = seq_length // subsampling_factor
        self.seq_length = seq_length
        self.encoder_seq_length = self.output_seq_length

    def get_config(self):
        encoder_config = NemotronAsrStreamingEncoderConfig(
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
            num_decoder_layers=self.num_decoder_layers,
            hidden_act=self.hidden_act,
            max_symbols_per_step=self.max_symbols_per_step,
            encoder_config=encoder_config.to_dict(),
            pad_token_id=self.pad_token_id,
            blank_token_id=self.blank_token_id,
            num_prompts=self.num_prompts,
            prompt_intermediate_size=self.prompt_intermediate_size,
            default_prompt_id=self.default_prompt_id,
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
        """Without `prompt_ids` the model defaults to `config.default_prompt_id` (with a warning) and still runs."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Nemotron3_5AsrForRNNT(config=config).to(torch_device).eval()
        no_prompt = {k: v.to(torch_device) for k, v in inputs_dict.items() if k != "prompt_ids"}
        default_prompt = torch.full(
            (no_prompt["input_features"].shape[0],), config.default_prompt_id, dtype=torch.long, device=torch_device
        )
        with torch.no_grad():
            default_logits = model(**no_prompt).logits
            explicit_logits = model(**no_prompt, prompt_ids=default_prompt).logits
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


@require_torch
class Nemotron3_5AsrForRNNTIntegrationTest(unittest.TestCase):
    _dataset = None

    @classmethod
    def setUp(cls):
        cls.checkpoint_name = "nvidia/nemotron-3.5-asr-streaming-0.6b"
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
        speech_samples = self._dataset.sort("id")[:num_samples]["audio"]
        return [x["array"] for x in speech_samples]

    @slow
    def test_model_integration(self):
        # reproducer: https://gist.github.com/eustlb/7e28529787b376b909008109be313ed7#file-reproducer_single_rnnt-py
        RESULTS_PATH = FIXTURES_DIR / "expected_results_single.json"
        with open(RESULTS_PATH) as f:
            expected = json.load(f)

        samples = self._load_datasamples(1)
        model = Nemotron3_5AsrForRNNT.from_pretrained(self.checkpoint_name, dtype=self.dtype, device_map="auto")

        # `expected["results"]` maps each prompting mode -> {"transcriptions", "token_ids"}: `"en-US"` forces
        # the language via the prompt, `"auto"` lets the model detect it (it emits the `<en-US>` tag itself).
        for language, reference in expected["results"].items():
            inputs = self.processor(
                samples, sampling_rate=self.processor.feature_extractor.sampling_rate, language=language
            )
            num_lookahead_tokens = inputs.pop("num_lookahead_tokens")
            inputs.to(model.device, dtype=model.dtype)
            output = model.generate(**inputs, num_lookahead_tokens=num_lookahead_tokens, return_dict_in_generate=True)

            # 1. Transcript: stripping the special tokens (blanks + `<lang>` tag) yields the clean text.
            predicted = self.processor.batch_decode(output.sequences, skip_special_tokens=True)
            self.assertListEqual(predicted, reference["transcriptions"], msg=f"transcript mismatch ({language})")

            # 2. Predicted tokens: `reference["token_ids"]` is NeMo's per-frame greedy alignment (the emitted
            # tokens *including blanks*). HF prepends `decoder_start_token_id`, then emits the same alignment,
            # so `sequences[0] == [decoder_start] + token_ids[0]` (no trailing padding for a single sample;
            # `decoder_start == blank == 13087` for this RNN-T checkpoint).
            token_ids = reference["token_ids"][0]
            self.assertListEqual(
                output.sequences[0].tolist(),
                [model.generation_config.decoder_start_token_id] + token_ids,
                msg=f"token mismatch ({language})",
            )

    @slow
    def test_model_integration_batched(self):
        # reproducer: https://gist.github.com/eustlb/7e28529787b376b909008109be313ed7#file-reproducer_batch_rnnt-py
        RESULTS_PATH = FIXTURES_DIR / "expected_results_batch.json"
        with open(RESULTS_PATH) as f:
            expected = json.load(f)

        samples = self._load_datasamples(5)
        model = Nemotron3_5AsrForRNNT.from_pretrained(self.checkpoint_name, dtype=self.dtype, device_map="auto")

        # `expected["results"]` maps each prompting mode -> {"transcriptions", "token_ids"}: `"en-US"` forces
        # the language via the prompt, `"auto"` lets the model detect it (it emits the `<en-US>` tag itself).
        for language, reference in expected["results"].items():
            inputs = self.processor(
                samples, sampling_rate=self.processor.feature_extractor.sampling_rate, language=language
            )
            num_lookahead_tokens = inputs.pop("num_lookahead_tokens")
            inputs.to(model.device, dtype=model.dtype)
            output = model.generate(**inputs, num_lookahead_tokens=num_lookahead_tokens, return_dict_in_generate=True)

            # 1. Transcript: stripping the special tokens (blanks + `<lang>` tag) yields the clean text.
            predicted = self.processor.batch_decode(output.sequences, skip_special_tokens=True)
            self.assertListEqual(predicted, reference["transcriptions"], msg=f"transcript mismatch ({language})")

            # 2. Predicted tokens: `reference["token_ids"][i]` is NeMo's per-frame greedy alignment for sample
            # `i` (the emitted tokens *including blanks*). HF prepends `decoder_start_token_id`, emits the same
            # alignment, then right-fills the batch to the longest row with blanks (`decoder_start == blank ==
            # 13087` here), so each row is `[decoder_start] + token_ids[i] + [blank] * padding`.
            for i, token_ids in enumerate(reference["token_ids"]):
                row = output.sequences[i].tolist()
                blank_padding = [model.config.blank_token_id] * (len(row) - 1 - len(token_ids))
                self.assertListEqual(
                    row,
                    [model.generation_config.decoder_start_token_id] + token_ids + blank_padding,
                    msg=f"token mismatch (sample {i}, {language})",
                )

    @slow
    def test_model_integration_streaming(self):
        """Cache-aware streaming generation from a generator of per-chunk mel features.

        Mirrors the streaming snippet of the multilingual usage: raw audio is fed to the processor chunk by
        chunk in streaming mode — the first chunk with `is_first_audio_chunk=True` (`center=True`) and every
        subsequent chunk with `is_first_audio_chunk=False` (`center=False`), so each per-chunk STFT reproduces
        a single full-utterance pass frame-for-frame. The language prompt rides along on every chunk via the
        processor's `language="en-US"`; the resulting features (and the `prompt_ids`/`num_lookahead_tokens`
        carried by `**first_chunk_inputs`) are yielded to `generate` as a generator at the streaming attention
        context `[56, 6]`. The decoder/encoder caches are threaded across chunks internally and the transcript
        is consumed incrementally via a `TextIteratorStreamer`.

        The expected value is the HF model's own streamed transcript, sanity-checked against the NeMo
        `nvidia/nemotron-3.5-asr-streaming-0.6b` reference (`reproducer_streaming_rnnt.py`): the two agree
        word-for-word except for the final partial chunk the streaming feed drops (NeMo, fed the whole buffer,
        emits a couple more words) and a single `TextIteratorStreamer` sub-word glitch ("goodness" -> "goodnes",
        which the non-streaming `batch_decode` does not exhibit).

        reproducer: https://gist.github.com/eustlb/7e28529787b376b909008109be313ed7#file-reproducer_streaming_rnnt-py
        """
        RESULTS_PATH = FIXTURES_DIR / "expected_results_streaming.json"
        with open(RESULTS_PATH) as f:
            expected = json.load(f)
        right = expected["att_context_size"][1]

        sampling_rate = self.processor.feature_extractor.sampling_rate
        audio = load_audio(
            "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3",
            sampling_rate=sampling_rate,
        )
        model = Nemotron3_5AsrForRNNT.from_pretrained(self.checkpoint_name, dtype=self.dtype, device_map="auto")

        # Select the streaming right attention context (lookahead, in subsampled encoder frames). This sizes
        # the audio/mel chunks the processor emits and must reach `generate` so the forward matches; it travels
        # through `**first_chunk_inputs` below (alongside the language `prompt_ids`).
        self.processor.set_num_lookahead_tokens(right)

        first_chunk_inputs = self.processor(
            audio[: self.processor.num_samples_first_audio_chunk],
            sampling_rate=sampling_rate,
            is_streaming=True,
            is_first_audio_chunk=True,
            language=expected["language"],
            return_tensors="pt",
        )
        first_chunk_inputs = first_chunk_inputs.to(model.device, dtype=model.dtype)

        def input_features_generator():
            yield first_chunk_inputs.input_features[:, : self.processor.num_mel_frames_first_audio_chunk, :]

            mel_frame_idx = self.processor.num_mel_frames_first_audio_chunk
            hop_length = self.processor.feature_extractor.hop_length
            n_fft = self.processor.feature_extractor.n_fft

            start_idx = mel_frame_idx * hop_length - n_fft // 2
            while (end_idx := start_idx + self.processor.num_samples_per_audio_chunk) < audio.shape[0]:
                inputs = self.processor(
                    audio[start_idx:end_idx],
                    sampling_rate=sampling_rate,
                    is_streaming=True,
                    is_first_audio_chunk=False,
                    language=expected["language"],
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device, dtype=model.dtype)
                yield inputs.input_features

                mel_frame_idx += self.processor.num_mel_frames_per_audio_chunk
                start_idx = mel_frame_idx * hop_length - n_fft // 2

        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_special_tokens=True)
        generate_kwargs = {
            **first_chunk_inputs,
            "input_features": input_features_generator(),
            "streamer": streamer,
        }
        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()
        streamed_text = "".join(streamer)
        thread.join()

        self.assertEqual(streamed_text, expected["transcription"])
