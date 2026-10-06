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

import tempfile
import unittest

from transformers import (
    AutoProcessor,
    Nemotron3_5TranscribeAsrEncoder,
    Nemotron3_5TranscribeConfig,
    Nemotron3_5TranscribeEncoderConfig,
    Nemotron3_5TranscribeForConditionalGeneration,
    Nemotron3_5TranscribeModel,
    NemotronHConfig,
    is_torch_available,
)
from transformers.audio_utils import load_audio
from transformers.testing_utils import cleanup, require_torch, require_torch_accelerator_memory, slow, torch_device

from ...alm_tester import ALMModelTest, ALMModelTester
from ...test_modeling_common import floats_tensor
from ..nemotron_h import test_modeling_nemotron_h


if is_torch_available():
    import torch


class Nemotron3_5TranscribeModelTester(ALMModelTester):
    config_class = Nemotron3_5TranscribeConfig
    base_model_class = Nemotron3_5TranscribeModel
    conditional_generation_class = Nemotron3_5TranscribeForConditionalGeneration
    text_config_class = NemotronHConfig
    audio_config_class = Nemotron3_5TranscribeEncoderConfig
    audio_mask_key = "input_features_mask"

    def __init__(self, parent, **kwargs):
        # 61 spectrogram frames are stacked into 8 audio tokens
        kwargs.setdefault("feat_seq_length", 61)
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
        return Nemotron3_5TranscribeEncoderConfig(
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
            },
        )

    def create_audio_features(self):
        return floats_tensor([self.batch_size, self.feat_seq_length, self.num_mel_bins])

    def create_audio_mask(self):
        # the model reads the mask as per-sample lengths: padding on the right only
        lengths = torch.linspace(self.feat_seq_length // 2, self.feat_seq_length, self.batch_size).long()
        positions = torch.arange(self.feat_seq_length)[None, :]
        return (positions < lengths[:, None]).long().to(torch_device)

    def get_audio_embeds_mask(self, audio_mask):
        return audio_mask[:, :: self.subsampling_factor]


@require_torch
class Nemotron3_5TranscribeForConditionalGenerationModelTest(ALMModelTest, unittest.TestCase):
    model_tester_class = Nemotron3_5TranscribeModelTester
    # the speaker cache silence embedding is only used for audio longer than one diarization chunk
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
        reason="The speech recognition encoder reads `output_attentions` from its own sub-config "
        "(`audio_config.asr_config`), so setting it on `audio_config` has no effect. Passing it as argument works."
    )
    def test_get_audio_features_attentions(self):
        pass

    @unittest.skip(
        reason="The speech recognition encoder reads `output_hidden_states` from its own sub-config "
        "(`audio_config.asr_config`), so setting it on `audio_config` has no effect. Passing it as argument works."
    )
    def test_get_audio_features_hidden_states(self):
        pass

    @unittest.skip(reason="Nemotron-H needs at least 3 layers to test (mamba, moe, attention)")
    def test_num_layers_is_small(self):
        pass

    def test_attention_outputs(self):
        """Only the attention layers of the language model return attention weights."""
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

    @unittest.skip(
        reason="The inputs_embeds of the audio tokens are replaced by the audio embeddings when input features are given."
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass

    def test_asr_encoder_qk_norm_and_partial_rotary(self):
        """The speech recognition encoder normalizes queries and keys, and rotates part of each head."""
        config = self.model_tester.get_audio_config().asr_config
        model = Nemotron3_5TranscribeAsrEncoder(config)
        head_dim = config.hidden_size // config.num_attention_heads
        rotary_dim = int(head_dim * config.rope_parameters["partial_rotary_factor"])
        self.assertEqual(model.rotary_emb.inv_freq.shape[0], rotary_dim // 2)
        self.assertIsInstance(model.layers[0].self_attn.q_norm, torch.nn.LayerNorm)
        self.assertIsInstance(model.layers[0].self_attn.k_norm, torch.nn.LayerNorm)

    def test_long_audio_uses_diarization_chunks(self):
        """Audio longer than one diarization chunk goes through the speaker cache, with look-back and look-ahead."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        diarization_config = config.audio_config.diarization_config
        diarization_config.chunk_length, diarization_config.chunk_left_context = 2, 1
        diarization_config.chunk_right_context = 1
        encoder = Nemotron3_5TranscribeForConditionalGeneration(config).model.audio_tower.to(torch_device).eval()
        with torch.no_grad():
            outputs = encoder(inputs_dict["input_features"], input_features_mask=inputs_dict["input_features_mask"])
        num_frames = -(-self.model_tester.feat_seq_length // self.model_tester.subsampling_factor)
        self.assertEqual(outputs.last_hidden_state.shape[:2], (self.model_tester.batch_size, num_frames))

    def test_audio_token_outside_of_vocabulary(self):
        """The original audio token id is the vocabulary size, it never goes through the text embeddings."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict["input_ids"].clone()
        input_ids[input_ids == config.audio_token_id] = config.text_config.vocab_size
        config.audio_token_id = config.text_config.vocab_size
        model = Nemotron3_5TranscribeForConditionalGeneration(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model(**{**inputs_dict, "input_ids": input_ids})
        self.assertEqual(outputs.logits.shape[:2], input_ids.shape)

    def test_speaker_embeddings_are_recomputed(self):
        """The fixed speaker embeddings are not saved with the weights."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = Nemotron3_5TranscribeForConditionalGeneration(config)
        expected = model.model.audio_tower.compute_speaker_embeddings(config.audio_config)
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            model = Nemotron3_5TranscribeForConditionalGeneration.from_pretrained(tmpdirname)
        torch.testing.assert_close(model.model.audio_tower.speaker_embeddings, expected, rtol=0, atol=0)


@slow
@require_torch
@require_torch_accelerator_memory(memory=70)
class Nemotron3_5TranscribeForConditionalGenerationIntegrationTest(unittest.TestCase):
    """
    Expected outputs were generated in bfloat16 on 2x A6000, with the converted checkpoint.
    """

    checkpoint = "nvidia/Nemotron-3.5-Transcribe"
    single_speaker_url = (
        "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/en-Alice_woman.wav"
    )
    conversation_url = (
        "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/diarization_example.mp3"
    )
    speaker_attributed_prompt = (
        "Transcribe this audio. Include all spoken words and verbal fillers. Write numbers as spoken. Use no "
        "punctuation and no capitalization. Start every speaker turn with a speaker tag such as <spk:0>, <spk:1>, "
        "<spk:2>, etc.; assign one stable tag per speaker and keep speaker identities consistent. Output only the "
        "speaker-tagged transcript."
    )

    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint)
        cls.model = Nemotron3_5TranscribeForConditionalGeneration.from_pretrained(
            cls.checkpoint, dtype=torch.bfloat16, device_map="auto"
        )

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cleanup(torch_device, gc_collect=True)

    def _generate(self, audio, prompt=None):
        inputs = self.processor.apply_transcription_request(audio, prompt=prompt).to(
            self.model.device, dtype=self.model.dtype
        )
        outputs = self.model.generate(**inputs, max_new_tokens=400, do_sample=False)
        return self.processor.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )

    def _conversation(self):
        # the first 30 seconds, processed by the original implementation in a single window
        return load_audio(self.conversation_url, sampling_rate=16000)[: 30 * 16000]

    def test_transcription(self):
        text = self._generate(load_audio(self.single_speaker_url, sampling_rate=16000))[0]
        expected = (
            "So just to clarify, we've had nineteen to twenty year olds on our podcast so far, and if you don't mind me "
            "asking, how old are you?<|im_end|>"
        )
        self.assertEqual(text, expected)

    def test_speaker_attributed_transcription(self):
        text = self._generate(self._conversation(), prompt=self.speaker_attributed_prompt)[0]
        expected = (
            "<spk:0> to celebrate the release of nemotron three diarization we invited seven more open text to speech "
            "voices to the party i'm chris the nemotron voice from nvidia <spk:1> hi everybody thanks for having me "
            "i'm jane a pocket voice from qt <spk:2> nice to meet you chris and jane i'm serena the queen three voice "
            "<spk:3> hey i'm adam from kokoro just eighty two million parameters and i still made the guest list "
            "<spk:4> hello hello chatterbox here from resemble a i i brought the expressive energy <spk:5> hi i'm john "
            "a dia voice from<|im_end|>"
        )
        self.assertEqual(text, expected)

    def test_transcription_batched(self):
        texts = self._generate([load_audio(self.single_speaker_url, sampling_rate=16000), self._conversation()])
        texts = [text.replace(self.processor.tokenizer.pad_token, "") for text in texts]
        expected = [
            "So just to clarify, we've had nineteen to twenty year olds on our podcast so far, and if you don't mind me "
            "asking, how old are you?<|im_end|>",
            "To celebrate the release of Nematron three diarization, we invited seven more open text to speech voices "
            "to the party. I'm Chris, the Nematron voice from Nvidia. Hi everybody, thanks for having me. I'm Jane, a "
            "Pocket voice from Qt. Nice to meet you, Chris and Jane. I'm Serena, the Queen three voice. Hey, I'm Adam "
            "from Kokoro. Just eighty two million parameters and I still made the guest list. Hello hello, Chatterbox "
            "here from Resemble AI. I brought the expressive energy. Hi, I'm John, a Dia voice from<|im_end|>",
        ]
        self.assertEqual(texts, expected)
