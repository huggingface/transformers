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
"""Testing suite for the PyTorch Nemotron3_5Transcribe model."""

import json
import tempfile
import unittest
from pathlib import Path

from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    Nemotron3_5TranscribeAudioConfig,
    Nemotron3_5TranscribeConfig,
    Nemotron3_5TranscribeForConditionalGeneration,
    Nemotron3_5TranscribeModel,
    NemotronHConfig,
    is_torch_available,
)
from transformers.audio_utils import load_audio
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.testing_utils import (
    require_scipy,
    require_torch,
    require_torch_accelerator_memory,
    require_torchaudio,
    slow,
    torch_device,
)

from ...alm_tester import ALMModelTest, ALMModelTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import floats_tensor
from ..nemotron_h import test_modeling_nemotron_h


if is_torch_available():
    import torch
    from safetensors.torch import load_file


class Nemotron3_5TranscribeModelTester(ALMModelTester):
    config_class = Nemotron3_5TranscribeConfig
    base_model_class = Nemotron3_5TranscribeModel
    conditional_generation_class = Nemotron3_5TranscribeForConditionalGeneration
    text_config_class = NemotronHConfig
    audio_config_class = Nemotron3_5TranscribeAudioConfig
    audio_mask_key = "input_features_mask"

    def __init__(self, parent, **kwargs):
        # 64 spectrogram frames (a multiple of `subsampling_factor`, as the processor pads them) are stacked into 8
        # audio tokens
        kwargs.setdefault("feat_seq_length", 64)
        kwargs.setdefault("num_mel_bins", 16)
        kwargs.setdefault("subsampling_factor", 8)
        kwargs.setdefault("num_speakers", 4)
        kwargs.setdefault("num_hidden_layers", 3)
        super().__init__(parent, **kwargs)

    def get_text_config(self):
        return NemotronHConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            layers_block_type=["linear_attention", "moe", "full_attention"],
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=16,
            intermediate_size=self.intermediate_size,
            moe_intermediate_size=16,
            moe_shared_expert_intermediate_size=16,
            mamba_num_heads=4,
            mamba_n_groups=1,
            mamba_head_dim=16,
            ssm_state_size=8,
            mamba_chunk_size=16,
            n_routed_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=self.max_position_embeddings,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )

    def get_audio_config(self):
        return Nemotron3_5TranscribeAudioConfig(
            asr_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_mel_bins": self.num_mel_bins,
                "subsampling_factor": self.subsampling_factor,
            },
            diarization_config={
                "audio_config": {
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_mel_bins": self.num_mel_bins,
                    "subsampling_factor": self.subsampling_factor,
                },
                "head_config": {
                    "hidden_size": 8,
                    "num_speakers": self.num_speakers,
                    "audio_hidden_size": 16,
                    "subsampling_factor": self.subsampling_factor,
                },
                "streaming_config": {"num_speakers": self.num_speakers, "subsampling_factor": self.subsampling_factor},
                "chunk_length": 500,
                "chunk_left_context": 50,
                "chunk_right_context": 50,
            },
        )

    def _build_modality_sub_configs(self):
        word_timestamp_config = {
            "hidden_size": 32,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "vocab_size": 16,
            "blank_token_id": 15,
            "residual_scale": 0.1,
            "max_position_embeddings": 5000,
        }
        return {**super()._build_modality_sub_configs(), "word_timestamp_config": word_timestamp_config}

    def create_audio_features(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return floats_tensor([batch_size, self.feat_seq_length, self.num_mel_bins])

    def create_audio_mask(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        # the model reads the mask as per-sample lengths: padding on the right only
        lengths = torch.linspace(self.feat_seq_length // 2, self.feat_seq_length, batch_size).long()
        positions = torch.arange(self.feat_seq_length)[None, :]
        return (positions < lengths[:, None]).long().to(torch_device)

    def get_audio_embeds_mask(self, audio_mask):
        return audio_mask[:, :: self.subsampling_factor]


@require_torch
class Nemotron3_5TranscribeForConditionalGenerationModelTest(ALMModelTest, unittest.TestCase):
    model_tester_class = Nemotron3_5TranscribeModelTester
    # `MODEL_FOR_MULTIMODAL_LM_MAPPING` serves the `any-to-any` pipeline (`audio-text-to-text` is not a pipeline task)
    pipeline_model_mapping = (
        {"any-to-any": Nemotron3_5TranscribeForConditionalGeneration} if is_torch_available() else {}
    )
    # speaker activities are thresholded into 0/1 values, so no gradient reaches the diarization model, and the word
    # timestamp encoder and head only run when their logits are requested
    test_all_params_have_gradient = False

    # the language model is Nemotron-H: its Mamba2 layers cache convolution and SSM states
    _get_conv_state_shape = test_modeling_nemotron_h.NemotronHModelTest._get_conv_state_shape
    _get_recurrent_state_shape = test_modeling_nemotron_h.NemotronHModelTest._get_recurrent_state_shape
    _check_past_key_values_for_generate = (
        test_modeling_nemotron_h.NemotronHModelTest._check_past_key_values_for_generate
    )

    def _audio_features_get_expected_num_attentions(self, model_tester=None):
        return self.model_tester.get_audio_config().asr_config.num_hidden_layers

    @unittest.skip(
        reason="The speech recognition encoder is a sub-model that records its own layers and reads `output_attentions` "
        "from its own config, `audio_config.asr_config`, which the test does not reach (it sets the direct sub-configs "
        "only). Passing it as argument works."
    )
    def test_get_audio_features_attentions(self):
        pass

    @unittest.skip(
        reason="The speech recognition encoder is a sub-model that records its own layers and reads `output_hidden_states` "
        "from its own config, `audio_config.asr_config`, which the test does not reach (it sets the direct sub-configs "
        "only). Passing it as argument works."
    )
    def test_get_audio_features_hidden_states(self):
        pass

    @unittest.skip(reason="Nemotron-H needs at least 3 layers to test (mamba, moe, attention)")
    def test_num_layers_is_small(self):
        pass

    @unittest.skip(
        reason="As AudioFlamingo3: the audio embeddings replace the audio tokens of `input_ids`, an `inputs_embeds` "
        "input without `input_ids` gets no audio."
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass

    # TODO: @eustlb, unskip once `revert_weight_conversion` lets a root converter win over a scoped sub-model converter
    # targeting the same parameter on save
    @unittest.skip(
        reason="A model not loaded with `from_pretrained` reverts every mapping the traversal finds on save, and the "
        "`nemotron_h` mapping of the language model then takes the experts before the `nemotron3_5_transcribe` one "
        "(library limitation, to fix in `revert_weight_conversion`). `test_save_pretrained_writes_nemo_layout` covers "
        "the round trip of NVIDIA's checkpoint."
    )
    def test_reverse_loading_mapping(self):
        pass

    def test_attention_outputs(self):
        """
        Only the attention layers of the language model return attention weights. Not bound from `NemotronHModelTest`,
        which reads `hybrid_override_pattern` from the top-level config instead of `text_config`.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        num_attention_layers = config.text_config.layer_types.count("full_attention")
        for model_class in self.all_model_classes:
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class), output_attentions=True)
            self.assertEqual(len(outputs.attentions), num_attention_layers)
            self.assertListEqual(
                list(outputs.attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, self.model_tester.seq_length, self.model_tester.seq_length],
            )

    def test_save_pretrained_writes_nemo_layout(self):
        """A model loaded from NVIDIA's checkpoint is saved back in the same layout, which NeMo and vLLM load."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        reference = Nemotron3_5TranscribeForConditionalGeneration(config)
        # a checkpoint in NVIDIA's layout: the conversions `from_pretrained` records for it
        reference._weight_conversions = get_checkpoint_conversion_mapping("nemo_speechlm")
        with tempfile.TemporaryDirectory() as nemo_dir, tempfile.TemporaryDirectory() as tmp_dir:
            reference.save_pretrained(nemo_dir)
            model, loading_info = Nemotron3_5TranscribeForConditionalGeneration.from_pretrained(
                nemo_dir, output_loading_info=True
            )
            model.save_pretrained(tmp_dir)
            nemo_state_dict = load_file(Path(nemo_dir) / "model.safetensors")
            state_dict = load_file(Path(tmp_dir) / "model.safetensors")
        self.assertFalse(any(loading_info.values()), loading_info)
        self.assertEqual(state_dict.keys(), nemo_state_dict.keys())
        self.assertFalse([key for key in state_dict if key.startswith("model.")])
        for name, tensor in reference.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], tensor, rtol=0, atol=0)

    def test_asr_windows(self):
        """
        Windows whose context covers the whole input give the encoder states and the word timestamp logits of the
        whole-recording run, bit for bit; short windows also run the diarization speaker cache.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Nemotron3_5TranscribeForConditionalGeneration(config).to(torch_device).eval()
        input_features, input_features_mask = inputs_dict["input_features"], inputs_dict["input_features_mask"]
        diarization_config = config.audio_config.diarization_config

        def run(chunk_length, chunk_left_context, chunk_right_context):
            diarization_config.chunk_length = chunk_length
            diarization_config.chunk_left_context = chunk_left_context
            diarization_config.chunk_right_context = chunk_right_context
            audio_outputs = model.get_audio_features(input_features, input_features_mask)
            states = audio_outputs.asr_last_hidden_state
            return audio_outputs, states, model.get_word_timestamp_logits(states, input_features_mask)

        with torch.no_grad():
            _, expected_states, expected_logits = run(500, 50, 50)
            # 2 windows of the 8 encoder frames, each with enough context to see all of them
            _, states, logits = run(5, 8, 4)
            audio_outputs, _, _ = run(2, 1, 1)
        torch.testing.assert_close(states, expected_states, rtol=0, atol=0)
        torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
        self.assertEqual(audio_outputs.last_hidden_state.shape[:2], expected_states.shape[:2])

    def test_generate_returns_word_timestamp_logits(self):
        """
        `generate(..., return_word_timestamp_logits=True)` returns the sequences of the plain call with the word
        timestamp and speaker logits of the audio features.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Nemotron3_5TranscribeForConditionalGeneration(config).to(torch_device).eval()
        generation_kwargs = {"max_new_tokens": 3, "do_sample": False}
        sequences = model.generate(**inputs_dict, **generation_kwargs)
        outputs = model.generate(**inputs_dict, **generation_kwargs, return_word_timestamp_logits=True)
        with torch.no_grad():
            audio_outputs = model.get_audio_features(inputs_dict["input_features"], inputs_dict["input_features_mask"])
            logits = model.get_word_timestamp_logits(
                audio_outputs.asr_last_hidden_state, inputs_dict["input_features_mask"]
            )
        torch.testing.assert_close(outputs.sequences, sequences, rtol=0, atol=0)
        torch.testing.assert_close(outputs.word_timestamp_logits, logits, rtol=0, atol=0)
        torch.testing.assert_close(outputs.speaker_logits, audio_outputs.speaker_logits, rtol=0, atol=0)


@require_torch
@require_torch_accelerator_memory(memory=70)
class Nemotron3_5TranscribeForConditionalGenerationIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    """
    Each test runs a usage example of `docs/source/en/model_doc/nemotron3_5_transcribe.md` as written there, sharded
    over the visible accelerators (`device_map="auto"`, as `require_torch_accelerator_memory` counts them). Expected
    values: NeMo direct, bfloat16, greedy, on an A100 (`tests/fixtures/nemotron3_5_transcribe/expected_*.json`,
    reproducers `reproducer_<case>.py` and `reproducer_word_timestamps.py` in the bucket
    `hf-internal-testing/nemotron3_5_transcribe-integration-tests`).
    """

    checkpoint = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
    # TODO: @eustlb, remove once NVIDIA merges the Transformers files (hub PR #5) into the checkpoint's main
    revision = "refs/pr/5"
    samples = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/"
    speaker_attributed_prompt = (
        "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. "
        "Write numbers in their standard written form. Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the "
        "transcription."
    )

    def _load_fixture(self, name):
        path = Path(__file__).parent.parent.parent / f"fixtures/nemotron3_5_transcribe/{name}.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _expected_speaker_attributed_30s(self):
        """The first 30 s of `diarization_example.mp3` (one window of the original) with `speaker_attributed_prompt`."""
        expected = self._load_fixture("expected_results_speaker_attributed")["transcriptions"][0]
        # Verified difference with the original: the texts differ in the last word only, the name of the "Dia" voice
        # ("diera" there), a near-tie at the end of the sequence that the two language-model stacks (fused kernels
        # there, reference code here) resolve differently.
        return expected.replace(" diera voice from", " d-a voice from")

    def _assert_word_timestamps_match(self, timestamps, expected):
        # the fixtures keep NeMo's layout: `{"word", "start", "end", "speaker"}` and `{"speaker", "start", "end"}`
        chunks, expected_words = timestamps["chunks"], expected["words"]
        self.assertListEqual(
            [(chunk["text"], chunk["speaker"]) for chunk in chunks],
            [(word["word"], word["speaker"]) for word in expected_words],
        )
        # one encoder frame (80 ms), plus the rounding of the times in seconds
        for index, key in enumerate(("start", "end")):
            max_difference = max(
                abs(chunk["timestamp"][index] - ref[key]) for chunk, ref in zip(chunks, expected_words)
            )
            self.assertLessEqual(max_difference, 0.08 + 1e-6, key)
        self.assertListEqual(
            [
                (segment["speaker"], round(segment["timestamp"][0], 6), round(segment["timestamp"][1], 6))
                for segment in timestamps["diarization"]
            ],
            [
                (segment["speaker"], round(segment["start"], 6), round(segment["end"], 6))
                for segment in expected["diarization"]
            ],
        )
        self.assertDictEqual(
            timestamps["speaker_tag_to_diarization_speaker"],
            {int(tag): speaker for tag, speaker in expected["speaker_tag_to_diarization_speaker"].items()},
        )
        self.assertListEqual(timestamps["failed_speakers"], expected["failed_speakers"])

    @slow
    def test_transcription(self):
        """Usage example "Transcription". Reproducer: `reproducer_single.py`."""
        processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision)
        model = AutoModelForMultimodalLM.from_pretrained(self.checkpoint, revision=self.revision, device_map="auto")

        inputs = processor.apply_transcription_request(self.samples + "en-Alice_woman.wav").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=256)
        texts = processor.decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)

        expected = self._load_fixture("expected_results_single")
        EXPECTED_TEXTS = processor.decode(expected["token_ids"], skip_special_tokens=True)
        self.assertEqual(texts, EXPECTED_TEXTS)

    @slow
    def test_batched_transcription(self):
        """
        Usage example "Batched transcription". Checked against the per-sample references: `reproducer_single.py` and
        `reproducer_speaker_attributed.py`.
        """
        processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision)
        model = AutoModelForMultimodalLM.from_pretrained(self.checkpoint, revision=self.revision, device_map="auto")

        audio = [
            load_audio(self.samples + "en-Alice_woman.wav", sampling_rate=16000),
            load_audio(self.samples + "diarization_example.mp3", sampling_rate=16000)[: 30 * 16000],
        ]
        prompt = [None, self.speaker_attributed_prompt]

        inputs = processor.apply_transcription_request(audio, prompt=prompt).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=400)
        texts = processor.decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=False)
        texts = [text.replace(processor.tokenizer.pad_token, "") for text in texts]

        EXPECTED_TEXTS = [
            *self._load_fixture("expected_results_single")["transcriptions"],
            self._expected_speaker_attributed_30s(),
        ]
        self.assertEqual(texts, EXPECTED_TEXTS)

    @slow
    def test_speaker_attributed_transcription(self):
        """
        Usage example "Speaker-attributed transcription": the whole conversation (97.6 s, three 40 s windows with the
        speaker cache). Reproducer: `reproducer_long_form.py`.
        """
        processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision)
        model = AutoModelForMultimodalLM.from_pretrained(self.checkpoint, revision=self.revision, device_map="auto")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.speaker_attributed_prompt},
                    {"type": "audio", "path": self.samples + "diarization_example.mp3"},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            enable_thinking=False,
        ).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=400)
        text = processor.decode(outputs[0, inputs.input_ids.shape[1] :], skip_special_tokens=False)

        # Verified difference with the original: the audio embeddings match the original's to 5e-5 and the first 79
        # tokens are identical; at the 80th the two language-model stacks (fused kernels there, reference code here)
        # resolve a near-tie on a proper noun differently ("kokoro" -> "cocoro") and the greedy decodes drift apart.
        # The port's output is validated at the embedding level.
        EXPECTED_TEXT = (
            "<spk:0> to celebrate the release of nemotron three diarization we invited seven more open text to speech "
            "voices to the party i'm chris the nemotron voice from nvidia <spk:1> hi everybody thanks for having me "
            "i'm jane a pocket voice from qti <spk:2> nice to meet you chris and jane i'm serena the queen three "
            "voice <spk:3> hey i'm adam from cocoro just eighty two million parameters and i still made the guest "
            "list <spk:4> hello hello chatterbox here from resemble ai i brought the expressive energy <spk:5> hi i'm "
            "john a d a voice from narilabs great crowd we've got here already <spk:3> what's up everyone great to be "
            "here i'm the mya one voice from mya research <spk:5> hi all i'm piper from raspy okay so now we have a "
            "full house of eight synthetic open voices let's put nemotron three diarization to the test <spk:0> okay "
            "so the new diarization model can tag each of our voices as it transcribes but can it do that while we "
            "all speak together let's find out <spk:3> even when two <spk:4> it still <spk:3> of us talk <spk:4> "
            "names both <spk:3> at once <spk:4> of us even while we speak over each other <spk:2> frame by frame it "
            "labels who spoke when <spk:0> amazing we can all talk at the same time and nemotron three diarization "
            "still separates every speaker <spk:4> talking together <spk:3> dies letting <spk:1> hey robin <spk:2> oh "
            "<spk:3> us speak four <spk:2> yeah <spk:3> at once <spk:0> okay so even the old nemo sort former model "
            "could capture four but eight now that's hard everyone all together go here <spk:1> together <spk:0> we "
            "go all eight <spk:3> every voice <spk:2> all at once <spk:4> all in one room <spk:5> be great too "
            "<spk:1> wow this was incredible let's assemble tomorrow and try it again <spk:4> i'd love to as well "
            "great meeting you all but i've"
        )
        self.assertEqual(text, EXPECTED_TEXT)

    @slow
    @require_scipy
    @require_torchaudio
    def test_word_timestamps(self):
        """
        Usage example "Word timestamps": the first 30 s of the conversation. Reproducer of the text:
        `reproducer_speaker_attributed.py`; the timestamps of a fixed transcript are checked by the next tests.
        """
        processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision)
        model = AutoModelForMultimodalLM.from_pretrained(self.checkpoint, revision=self.revision, device_map="auto")

        audio = load_audio(self.samples + "diarization_example.mp3", sampling_rate=16000)[: 30 * 16000]

        inputs = processor.apply_transcription_request(audio, prompt=self.speaker_attributed_prompt).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=400, return_word_timestamp_logits=True)
        texts = processor.decode(outputs.sequences[:, inputs.input_ids.shape[1] :], skip_special_tokens=False)
        timestamps = processor.align_word_timestamps(
            texts,
            outputs.word_timestamp_logits,
            inputs.input_features_mask,
            outputs.speaker_logits,
        )

        EXPECTED_CHUNKS = [
            {"text": "to", "timestamp": (0.32, 0.35000000000000003), "speaker": 0},
            {"text": "celebrate", "timestamp": (0.35000000000000003, 0.96), "speaker": 0},
            {"text": "the", "timestamp": (0.96, 1.04), "speaker": 0},
        ]
        self.assertEqual(texts, [self._expected_speaker_attributed_30s()])
        self.assertEqual(timestamps[0]["chunks"][:3], EXPECTED_CHUNKS)

    @slow
    @require_scipy
    @require_torchaudio
    def test_word_timestamps_of_a_transcript(self):
        """
        Usage example "Word timestamps", transcript not generated: the first 30 s of the conversation aligned on the
        original's transcript (87 words, 6 diarization segments). Expected: `expected_word_timestamps_speaker_attributed.json`
        (reproducer `reproducer_word_timestamps.py`); measured on an A100: times bit-exact with the original.
        """
        expected = self._load_fixture("expected_word_timestamps_speaker_attributed")
        processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision)
        model = AutoModelForMultimodalLM.from_pretrained(self.checkpoint, revision=self.revision, device_map="auto")

        audio = load_audio(self.samples + "diarization_example.mp3", sampling_rate=16000)[: 30 * 16000]
        inputs = processor.apply_transcription_request(audio, prompt=self.speaker_attributed_prompt).to(model.device)

        input_features_mask = inputs.input_features_mask
        audio_outputs = model.get_audio_features(inputs.input_features, input_features_mask)
        word_timestamp_logits = model.get_word_timestamp_logits(
            audio_outputs.asr_last_hidden_state, input_features_mask
        )
        timestamps = processor.align_word_timestamps(
            [expected["transcription"]],
            word_timestamp_logits,
            input_features_mask,
            audio_outputs.speaker_logits,
        )

        self._assert_word_timestamps_match(timestamps[0], expected)

    @slow
    @require_scipy
    @require_torchaudio
    def test_word_timestamps_of_a_transcript_long_form(self):
        """
        As `test_word_timestamps_of_a_transcript` on the whole conversation (97.6 s: the head runs in the encoder
        windows; the original searches it in a band of states, the port in all of them), aligned on the port's verified
        long-form text (287 words, 29 diarization segments). Expected: `expected_word_timestamps_long_form.json`
        (reproducer `reproducer_word_timestamps.py`); measured on an A100: 3 word boundaries (5 words) shifted by one
        encoder frame (80 ms) where the bfloat16 log-probabilities tie, all other times within 1 ms.
        """
        expected = self._load_fixture("expected_word_timestamps_long_form")
        processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision)
        model = AutoModelForMultimodalLM.from_pretrained(self.checkpoint, revision=self.revision, device_map="auto")

        audio = load_audio(self.samples + "diarization_example.mp3", sampling_rate=16000)
        inputs = processor.apply_transcription_request(audio, prompt=self.speaker_attributed_prompt).to(model.device)

        input_features_mask = inputs.input_features_mask
        audio_outputs = model.get_audio_features(inputs.input_features, input_features_mask)
        word_timestamp_logits = model.get_word_timestamp_logits(
            audio_outputs.asr_last_hidden_state, input_features_mask
        )
        timestamps = processor.align_word_timestamps(
            [expected["transcription"]],
            word_timestamp_logits,
            input_features_mask,
            audio_outputs.speaker_logits,
        )

        self._assert_word_timestamps_match(timestamps[0], expected)
