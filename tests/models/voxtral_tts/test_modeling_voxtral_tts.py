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
"""Testing suite for the PyTorch Voxtral TTS model."""

import copy
import unittest
from unittest import mock

from transformers import (
    VoxtralTtsAcousticTransformerConfig,
    VoxtralTtsCodecConfig,
    VoxtralTtsConfig,
)
from transformers.testing_utils import (
    cleanup,
    is_torch_available,
    require_mistral_common,
    require_torch,
    slow,
    torch_device,
)

from ...model_tester_config import ModelTesterConfigMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        AutoProcessor,
        DynamicCache,
        VoxtralTtsCodecModel,
        VoxtralTtsForTextToSpeech,
        VoxtralTtsModel,
    )


@require_torch
class VoxtralTtsModelTester(ModelTesterConfigMixin):
    defaults = {}
    small_config_overrides = {
        "batch_size": 2,
        "seq_length": 9,
        "num_voice_tokens": 3,
        "is_training": False,
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "semantic_codebook_size": 30,
        "acoustic_codebook_size": 5,
        "num_acoustic_codebooks": 4,
        "audio_token_id": 24,
        "begin_audio_token_id": 25,
        "pad_token_id": 11,
    }

    def __init__(self, parent, **kwargs):
        self.parent = parent
        for key, value in self.get_small_config_defaults(**kwargs).items():
            setattr(self, key, value)

    def get_config(self):
        return VoxtralTtsConfig(
            text_config={
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "max_position_embeddings": 512,
                "pad_token_id": self.pad_token_id,
            },
            acoustic_transformer_config=VoxtralTtsAcousticTransformerConfig(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_hidden_layers=1,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                llm_hidden_size=self.hidden_size,
                semantic_codebook_size=self.semantic_codebook_size,
                acoustic_codebook_size=self.acoustic_codebook_size,
                num_acoustic_codebooks=self.num_acoustic_codebooks,
                num_flow_matching_steps=2,
            ),
            codec_config=VoxtralTtsCodecConfig(
                semantic_codebook_size=self.semantic_codebook_size,
                semantic_dim=8,
                acoustic_codebook_size=self.acoustic_codebook_size,
                acoustic_dim=self.num_acoustic_codebooks,
                hidden_size=16,
                intermediate_size=32,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
                patch_size=4,
                patch_proj_kernel_size=3,
                sliding_window=4,
                decoder_num_layers=[1, 1],
                decoder_kernel_sizes=[3, 4],
                decoder_strides=[1, 2],
            ),
            audio_token_id=self.audio_token_id,
            begin_audio_token_id=self.begin_audio_token_id,
            audio_vocab_size=128,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        # text tokens, never the audio placeholder token
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 30) + 30
        input_ids[:, -1] = self.begin_audio_token_id
        attention_mask = torch.ones_like(input_ids)
        return config, input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}

    def prepare_config_and_inputs_with_voice(self):
        """Inputs laid out as the processor does: `[BOS] [BEGIN_AUDIO] [AUDIO] * n text [BEGIN_AUDIO]`."""
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        input_ids[:, 0] = 1
        input_ids[:, 1] = self.begin_audio_token_id
        input_ids[:, 2 : 2 + self.num_voice_tokens] = self.audio_token_id
        voice_embeddings = torch.randn(self.batch_size * self.num_voice_tokens, self.hidden_size, device=torch_device)
        return config, {"input_ids": input_ids, "attention_mask": attention_mask, "voice_embeddings": voice_embeddings}


@require_torch
class VoxtralTtsModelTest(ModelTesterMixin, unittest.TestCase):
    model_tester_class = VoxtralTtsModelTester
    all_model_classes = (VoxtralTtsModel, VoxtralTtsForTextToSpeech) if is_torch_available() else ()
    # `generate` is custom: there are no text logits to sample, every step produces an audio frame with the acoustic
    # transformer (see `test_generate*` below)
    all_generative_model_classes = ()
    _is_composite = True
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = VoxtralTtsModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VoxtralTtsConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def _random_frames(self, config, batch_size, num_frames):
        at = config.acoustic_transformer_config
        semantic = torch.randint(2, 2 + at.semantic_codebook_size, (batch_size, num_frames, 1))
        acoustic = torch.randint(2, 2 + at.acoustic_codebook_size, (batch_size, num_frames, at.num_acoustic_codebooks))
        return torch.cat([semantic, acoustic], dim=-1).to(torch_device)

    def test_voice_embeddings_replace_placeholders(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_with_voice()
        model = VoxtralTtsModel(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model(**inputs)
            inputs_embeds = model.get_input_embeddings()(inputs["input_ids"])
            inputs_embeds[inputs["input_ids"] == config.audio_token_id] = inputs["voice_embeddings"]
            expected = model(inputs_embeds=inputs_embeds, attention_mask=inputs["attention_mask"])
        torch.testing.assert_close(outputs.last_hidden_state, expected.last_hidden_state)

        with self.assertRaisesRegex(ValueError, "voice embedding vectors"):
            model(input_ids=inputs["input_ids"], voice_embeddings=inputs["voice_embeddings"][:-1])

    def test_audio_codes_extend_prompt_attention_mask(self):
        # Appending audio frames to a prompt whose attention mask only covers the prompt (as returned by the
        # processor) must behave as if the mask covered the frames too.
        config, inputs = self.model_tester.prepare_config_and_inputs_with_voice()
        model = VoxtralTtsModel(config).to(torch_device).eval()
        frames = self._random_frames(config, self.model_tester.batch_size, 4)
        full_mask = torch.cat([inputs["attention_mask"], torch.ones_like(frames[..., 0])], dim=-1)
        with torch.no_grad():
            outputs = model(**inputs, audio_codes=frames)
            expected = model(**{**inputs, "attention_mask": full_mask}, audio_codes=frames)
            unmasked = model(
                input_ids=inputs["input_ids"], voice_embeddings=inputs["voice_embeddings"], audio_codes=frames
            )
        torch.testing.assert_close(outputs.last_hidden_state, expected.last_hidden_state)
        torch.testing.assert_close(outputs.last_hidden_state, unmasked.last_hidden_state)

    def test_incremental_decoding_matches_full_forward(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_with_voice()
        model = VoxtralTtsModel(config).to(torch_device).eval()
        frames = self._random_frames(config, self.model_tester.batch_size, 3)
        with torch.no_grad():
            full = model(**inputs, audio_codes=frames).last_hidden_state
            cache = DynamicCache(config=config.text_config)
            prompt = model(**inputs, cache=cache, use_cache=True).last_hidden_state
            steps = []
            attention_mask = inputs["attention_mask"]
            for i in range(frames.shape[1]):
                attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, :1])], dim=-1)
                position_ids = torch.full((frames.shape[0], 1), attention_mask.shape[1] - 1, device=torch_device)
                step = model(
                    audio_codes=frames[:, i : i + 1],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache=cache,
                    use_cache=True,
                )
                steps.append(step.last_hidden_state)
        incremental = torch.cat([prompt] + steps, dim=1)
        torch.testing.assert_close(incremental, full, rtol=1e-4, atol=1e-4)

    def test_left_padding(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_with_voice()
        model = VoxtralTtsModel(config).to(torch_device).eval()
        num_pad = 3
        pad = torch.full((1, num_pad), self.model_tester.pad_token_id, device=torch_device)
        padded_ids = torch.cat([pad, inputs["input_ids"][:1]], dim=-1)
        padded_mask = torch.cat([torch.zeros_like(pad), inputs["attention_mask"][:1]], dim=-1)
        voice = inputs["voice_embeddings"][: self.model_tester.num_voice_tokens]
        with torch.no_grad():
            expected = model(input_ids=inputs["input_ids"][:1], voice_embeddings=voice).last_hidden_state
            padded = model(input_ids=padded_ids, attention_mask=padded_mask, voice_embeddings=voice).last_hidden_state
        torch.testing.assert_close(padded[:, num_pad:], expected, rtol=1e-4, atol=1e-4)

    def test_generate(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_with_voice()
        model = VoxtralTtsForTextToSpeech(config).to(torch_device).eval()
        outputs = model.generate(**inputs, max_new_frames=5, generator=torch.Generator(torch_device).manual_seed(0))
        batch_size, num_codebooks = self.model_tester.batch_size, config.num_codebooks
        self.assertEqual(outputs.audio_codes.shape[0], batch_size)
        self.assertEqual(outputs.audio_codes.shape[2], num_codebooks)
        self.assertLessEqual(outputs.audio_codes.shape[1], 5)
        self.assertEqual(len(outputs.audio), batch_size)
        for audio, length in zip(outputs.audio, outputs.audio_lengths):
            self.assertEqual(audio.shape, (int(length) * config.codec_config.hop_length,))

        # Same seed, same audio
        again = model.generate(**inputs, max_new_frames=5, generator=torch.Generator(torch_device).manual_seed(0))
        torch.testing.assert_close(again.audio_codes, outputs.audio_codes)

    def test_generate_decodes_audio_in_chunks(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_with_voice()
        model = VoxtralTtsForTextToSpeech(config).to(torch_device).eval()
        # Never predict `[END_AUDIO]`, so every frame is decoded
        semantic_head = model.acoustic_transformer.get_semantic_logits

        def get_semantic_logits(hidden_states):
            logits = semantic_head(hidden_states)
            logits[..., 1] = logits.min() - 1
            return logits

        with mock.patch.object(model.acoustic_transformer, "get_semantic_logits", get_semantic_logits):
            outputs = model.generate(
                **inputs, max_new_frames=5, codec_chunk_size=2, generator=torch.Generator(torch_device).manual_seed(0)
            )
        self.assertEqual(outputs.audio_lengths.tolist(), [5] * self.model_tester.batch_size)
        codec_codes = outputs.audio_codes - config.acoustic_transformer_config.num_special_audio_tokens
        with torch.no_grad():
            expected = torch.cat(
                [model.codec_model.decode(codec_codes[:, i : i + 2]).audio_values for i in range(0, 5, 2)], dim=-1
            )
        for audio, expected_audio in zip(outputs.audio, expected):
            torch.testing.assert_close(audio, expected_audio)

    def test_generate_stops_on_end_audio(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_with_voice()
        model = VoxtralTtsForTextToSpeech(config).to(torch_device).eval()
        # All semantic logits equal: `[EMPTY_AUDIO]` is excluded, so the argmax is `[END_AUDIO]`
        model.acoustic_transformer.semantic_head.weight.data.zero_()
        outputs = model.generate(**inputs, max_new_frames=5)
        self.assertEqual(outputs.audio_codes.shape[1], 1)
        self.assertTrue((outputs.audio_codes[:, 0, 0] == 1).all())
        self.assertTrue((outputs.audio_lengths == 0).all())
        self.assertTrue(all(audio.numel() == 0 for audio in outputs.audio))

    def _semantic_logits_forcing(self, model, end_rows_per_call):
        """Patches the semantic head: call `i` forces `[END_AUDIO]` for the rows in `end_rows_per_call[i]` and forbids it
        for every other row. Also records the batch size of every call."""
        semantic_head = model.acoustic_transformer.get_semantic_logits
        batch_sizes = []

        def get_semantic_logits(hidden_states):
            logits = semantic_head(hidden_states)
            end_rows = end_rows_per_call[len(batch_sizes)] if len(batch_sizes) < len(end_rows_per_call) else []
            batch_sizes.append(hidden_states.shape[0])
            logits[..., 1] = logits.min() - 1
            for row in end_rows:
                logits[row, 1] = logits[row].max() + 1
            return logits

        return mock.patch.object(model.acoustic_transformer, "get_semantic_logits", get_semantic_logits), batch_sizes

    def test_generate_with_per_row_generators_is_batch_invariant(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_with_voice()
        model = VoxtralTtsForTextToSpeech(config).to(torch_device).eval()
        num_voice = self.model_tester.num_voice_tokens
        # Row 0 has 2 fewer text tokens, so it is left-padded in the batch
        short_row = torch.cat([inputs["input_ids"][0, : 2 + num_voice], inputs["input_ids"][0, 4 + num_voice :]])
        rows = [short_row, inputs["input_ids"][1]]
        inputs["input_ids"][0] = torch.cat([torch.full((2,), self.model_tester.pad_token_id), short_row.cpu()])
        inputs["attention_mask"][0, :2] = 0

        generators = [torch.Generator(torch_device).manual_seed(seed) for seed in (1, 2)]
        batched = model.generate(**inputs, max_new_frames=4, generator=generators)
        for row, seed in enumerate((1, 2)):
            single = model.generate(
                input_ids=rows[row][None],
                voice_embeddings=inputs["voice_embeddings"][row * num_voice : (row + 1) * num_voice],
                max_new_frames=4,
                generator=[torch.Generator(torch_device).manual_seed(seed)],
            )
            # The batch runs until its longest sequence ends: later frames of a finished row are `[EMPTY_AUDIO]`
            num_frames = single.audio_codes.shape[1]
            torch.testing.assert_close(batched.audio_codes[row : row + 1, :num_frames], single.audio_codes)
            self.assertTrue((batched.audio_codes[row, num_frames:] == 0).all())
            torch.testing.assert_close(batched.audio[row], single.audio[0], rtol=1e-4, atol=1e-4)

        with self.assertRaisesRegex(ValueError, "1 generators for a batch of 2"):
            model.generate(**inputs, max_new_frames=1, generator=generators[:1])

    def test_generate_drops_finished_rows(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_with_voice()
        model = VoxtralTtsForTextToSpeech(config).to(torch_device).eval()
        # Row 0 ends at the first frame, row 1 never ends
        patch, batch_sizes = self._semantic_logits_forcing(model, end_rows_per_call=[[0]])
        with patch:
            outputs = model.generate(
                **inputs, max_new_frames=4, generator=[torch.Generator(torch_device).manual_seed(s) for s in (1, 2)]
            )
        # Once row 0 has ended, only row 1 goes through the model
        self.assertEqual(batch_sizes, [2, 1, 1, 1])
        self.assertEqual(outputs.audio_lengths.tolist(), [0, 4])
        self.assertEqual(int(outputs.audio_codes[0, 0, 0]), 1)
        self.assertTrue((outputs.audio_codes[0, 1:] == 0).all())
        self.assertEqual(outputs.audio[0].numel(), 0)

        # Row 1 is not affected by row 0 ending
        num_voice = self.model_tester.num_voice_tokens
        patch, _ = self._semantic_logits_forcing(model, end_rows_per_call=[])
        with patch:
            single = model.generate(
                input_ids=inputs["input_ids"][1:],
                voice_embeddings=inputs["voice_embeddings"][num_voice:],
                max_new_frames=4,
                generator=[torch.Generator(torch_device).manual_seed(2)],
            )
        torch.testing.assert_close(outputs.audio_codes[1:], single.audio_codes)

    def test_acoustic_transformer_frame_codes(self):
        config = self.model_tester.get_config()
        model = VoxtralTtsForTextToSpeech(config).to(torch_device).eval()
        at_config = config.acoustic_transformer_config
        hidden = torch.randn(3, self.model_tester.hidden_size, device=torch_device)
        noise = torch.randn(3, at_config.num_acoustic_codebooks, device=torch_device)
        frame = model.acoustic_transformer.generate_frame(hidden, noise=noise)
        self.assertEqual(frame.shape, (3, 1 + at_config.num_acoustic_codebooks))
        torch.testing.assert_close(frame, model.acoustic_transformer.generate_frame(hidden, noise=noise))
        # semantic codes are never `[EMPTY_AUDIO]`, acoustic codes are shifted FSQ levels
        self.assertTrue((frame[:, 0] >= 1).all() and (frame[:, 0] < 2 + at_config.semantic_codebook_size).all())
        acoustic = frame[frame[:, 0] != 1, 1:]
        self.assertTrue((acoustic >= 2).all() and (acoustic < 2 + at_config.acoustic_codebook_size).all())

    def test_codec_is_causal(self):
        config = self.model_tester.get_config()
        codec = VoxtralTtsCodecModel(config.codec_config).to(torch_device).eval()
        codec_config = config.codec_config
        codes = torch.cat(
            [
                torch.randint(0, codec_config.semantic_codebook_size, (2, 12, 1)),
                torch.randint(0, codec_config.acoustic_codebook_size, (2, 12, codec_config.acoustic_dim)),
            ],
            dim=-1,
        ).to(torch_device)
        with torch.no_grad():
            full = codec.decode(codes).audio_values
            prefix = codec.decode(codes[:, :7]).audio_values
        self.assertEqual(full.shape, (2, 12 * codec_config.hop_length))
        torch.testing.assert_close(full[:, : prefix.shape[1]], prefix, rtol=1e-4, atol=1e-4)


@require_torch
class VoxtralTtsConversionTest(unittest.TestCase):
    """Converts a tiny checkpoint laid out as the original (Mistral-format) one."""

    # Same structure as the released `params.json`, with tiny dimensions
    params = {
        "dim": 32, "n_layers": 2, "head_dim": 8, "hidden_dim": 64, "n_heads": 4, "n_kv_heads": 2,
        "rope_theta": 1000000.0, "norm_eps": 1e-05, "vocab_size": 64, "tied_embeddings": True,
        "max_position_embeddings": 512,
        "multimodal": {
            "bos_token_id": 1,
            "audio_model_args": {
                "semantic_codebook_size": 30, "acoustic_codebook_size": 5, "n_acoustic_codebook": 4,
                "audio_token_id": 24, "begin_audio_token_id": 25,
                "acoustic_transformer_args": {
                    "input_dim": 32, "dim": 32, "n_layers": 1, "head_dim": 8, "hidden_dim": 64, "n_heads": 4,
                    "n_kv_heads": 2,
                },
            },
            "audio_tokenizer_args": {
                "channels": 1, "sampling_rate": 24000, "pretransform_patch_size": 4, "patch_proj_kernel_size": 3,
                "semantic_codebook_size": 30, "semantic_dim": 8, "acoustic_codebook_size": 5, "acoustic_dim": 4,
                "causal": True, "attn_sliding_window_size": 4, "half_attn_window_upon_downsampling": True,
                "dim": 16, "hidden_dim": 32, "head_dim": 8, "n_heads": 2, "n_kv_heads": 2, "qk_norm_eps": 1e-06,
                "norm_eps": 0.01, "layer_scale_init": 0.01, "decoder_transformer_lengths_str": "1,1",
                "decoder_convs_kernels_str": "3,4", "decoder_convs_strides_str": "1,2",
            },
        },
    }  # fmt: skip

    def original_state_dict(self):
        generator = torch.Generator().manual_seed(0)

        def randn(*shape):
            return torch.randn(*shape, generator=generator)

        p = self.params
        audio_args = p["multimodal"]["audio_model_args"]
        at, codec = audio_args["acoustic_transformer_args"], p["multimodal"]["audio_tokenizer_args"]

        def transformer_layer(prefix, dim, n_heads, n_kv_heads, head_dim, hidden_dim):
            return {
                f"{prefix}.attention.wq.weight": randn(n_heads * head_dim, dim),
                f"{prefix}.attention.wk.weight": randn(n_kv_heads * head_dim, dim),
                f"{prefix}.attention.wv.weight": randn(n_kv_heads * head_dim, dim),
                f"{prefix}.attention.wo.weight": randn(dim, n_heads * head_dim),
                f"{prefix}.attention_norm.weight": randn(dim),
                f"{prefix}.ffn_norm.weight": randn(dim),
                f"{prefix}.feed_forward.w1.weight": randn(hidden_dim, dim),
                f"{prefix}.feed_forward.w2.weight": randn(dim, hidden_dim),
                f"{prefix}.feed_forward.w3.weight": randn(hidden_dim, dim),
            }

        def weight_norm_conv(prefix, out_channels, in_channels, kernel_size, transpose=False):
            # `weight_norm` (dim=0) keeps one gain per output channel of a Conv1d, per input channel of a transpose
            weight_shape = (
                (in_channels, out_channels, kernel_size) if transpose else (out_channels, in_channels, kernel_size)
            )
            return {
                f"{prefix}.conv.parametrizations.weight.original0": randn(weight_shape[0], 1, 1).abs(),
                f"{prefix}.conv.parametrizations.weight.original1": randn(*weight_shape),
            }

        num_special = 2
        audio_vocab = 128 * -(-(audio_args["semantic_codebook_size"] + num_special + (audio_args["acoustic_codebook_size"] + num_special) * audio_args["n_acoustic_codebook"]) // 128)  # fmt: skip
        state_dict = {
            "mm_audio_embeddings.tok_embeddings.weight": randn(p["vocab_size"], p["dim"]),
            "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight": randn(audio_vocab, p["dim"]),
            "norm.weight": randn(p["dim"]),
            "acoustic_transformer.llm_projection.weight": randn(at["dim"], at["input_dim"]),
            "acoustic_transformer.time_projection.weight": randn(at["dim"], at["dim"]),
            "acoustic_transformer.input_projection.weight": randn(at["dim"], audio_args["n_acoustic_codebook"]),
            "acoustic_transformer.semantic_codebook_output.weight": randn(128, at["dim"]),
            "acoustic_transformer.acoustic_codebook_output.weight": randn(audio_args["n_acoustic_codebook"], at["dim"]),
            "acoustic_transformer.norm.weight": randn(at["dim"]),
            "audio_tokenizer.quantizer.semantic_codebook.cluster_usage": randn(codec["semantic_codebook_size"]).abs(),
            "audio_tokenizer.quantizer.semantic_codebook.embedding_sum": randn(codec["semantic_codebook_size"], codec["semantic_dim"]),
        }  # fmt: skip
        for i in range(p["n_layers"]):
            state_dict.update(
                transformer_layer(
                    f"layers.{i}", p["dim"], p["n_heads"], p["n_kv_heads"], p["head_dim"], p["hidden_dim"]
                )
            )
        for i in range(at["n_layers"]):
            state_dict.update(
                transformer_layer(
                    f"acoustic_transformer.layers.{i}", at["dim"], at["n_heads"], at["n_kv_heads"], at["head_dim"], at["hidden_dim"]
                )
            )  # fmt: skip
        # Decoder blocks alternate an (up)sampling convolution and transformer layers
        kernels = [int(k) for k in codec["decoder_convs_kernels_str"].split(",")]
        latent_dim = codec["semantic_dim"] + codec["acoustic_dim"]
        for stage, num_layers in enumerate(int(n) for n in codec["decoder_transformer_lengths_str"].split(",")):
            in_channels = latent_dim if stage == 0 else codec["dim"]
            state_dict.update(
                weight_norm_conv(f"audio_tokenizer.decoder_blocks.{2 * stage}", codec["dim"], in_channels, kernels[stage], transpose=stage > 0)
            )  # fmt: skip
            for j in range(num_layers):
                prefix = f"audio_tokenizer.decoder_blocks.{2 * stage + 1}.layers.{j}"
                state_dict.update(
                    transformer_layer(prefix, codec["dim"], codec["n_heads"], codec["n_kv_heads"], codec["head_dim"], codec["hidden_dim"])
                )  # fmt: skip
                state_dict[f"{prefix}.attention.q_norm.weight"] = randn(codec["n_heads"] * codec["head_dim"])
                state_dict[f"{prefix}.attention.k_norm.weight"] = randn(codec["n_kv_heads"] * codec["head_dim"])
                state_dict[f"{prefix}.attention_scale"] = randn(codec["dim"])
                state_dict[f"{prefix}.ffn_scale"] = randn(codec["dim"])
        state_dict.update(
            weight_norm_conv(
                "audio_tokenizer.output_proj", codec["pretransform_patch_size"], codec["dim"], codec["patch_proj_kernel_size"]
            )
        )  # fmt: skip
        return state_dict

    def convert(self):
        from transformers.models.voxtral_tts.convert_voxtral_tts_weights_to_hf import (
            convert_config,
            convert_state_dict,
        )

        config = convert_config(copy.deepcopy(self.params))
        original_state_dict = self.original_state_dict()
        return config, original_state_dict, convert_state_dict(original_state_dict, config)

    def test_converted_state_dict_loads_strictly(self):
        config, _, state_dict = self.convert()
        self.assertEqual(config.audio_vocab_size, 128)
        self.assertEqual(config.acoustic_transformer_config.num_flow_matching_steps, 7)
        self.assertEqual(config.codec_config.hop_length, 8)
        model = VoxtralTtsForTextToSpeech(config)
        model.load_state_dict(state_dict, strict=True)

    def test_tensors_are_mapped_to_the_right_parameters(self):
        # Tensors of the same shape are easy to swap: every original tensor is distinct random data, so equality pins
        # each mapping rule down
        _, original, state_dict = self.convert()
        expected_mapping = {
            "mm_audio_embeddings.tok_embeddings.weight": "model.language_model.embed_tokens.weight",
            "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight": "model.audio_embeddings.embed_audio_tokens.weight",
            "layers.1.attention.wv.weight": "model.language_model.layers.1.self_attn.v_proj.weight",
            "layers.1.attention.wo.weight": "model.language_model.layers.1.self_attn.o_proj.weight",
            "layers.1.attention_norm.weight": "model.language_model.layers.1.input_layernorm.weight",
            "layers.1.ffn_norm.weight": "model.language_model.layers.1.post_attention_layernorm.weight",
            "layers.1.feed_forward.w1.weight": "model.language_model.layers.1.mlp.gate_proj.weight",
            "layers.1.feed_forward.w3.weight": "model.language_model.layers.1.mlp.up_proj.weight",
            "norm.weight": "model.language_model.norm.weight",
            "acoustic_transformer.layers.0.attention.wq.weight": "acoustic_transformer.layers.0.self_attn.q_proj.weight",
            "acoustic_transformer.layers.0.attention.wk.weight": "acoustic_transformer.layers.0.self_attn.k_proj.weight",
            "acoustic_transformer.llm_projection.weight": "acoustic_transformer.llm_projection.weight",
            "acoustic_transformer.time_projection.weight": "acoustic_transformer.time_projection.weight",
            "acoustic_transformer.semantic_codebook_output.weight": "acoustic_transformer.semantic_head.weight",
            "acoustic_transformer.acoustic_codebook_output.weight": "acoustic_transformer.acoustic_head.weight",
            "audio_tokenizer.decoder_blocks.1.layers.0.attention.q_norm.weight": "codec_model.stages.0.layers.0.self_attn.q_norm.weight",
            "audio_tokenizer.decoder_blocks.1.layers.0.attention.k_norm.weight": "codec_model.stages.0.layers.0.self_attn.k_norm.weight",
            "audio_tokenizer.decoder_blocks.3.layers.0.attention.wq.weight": "codec_model.stages.1.layers.0.self_attn.q_proj.weight",
            "audio_tokenizer.decoder_blocks.3.layers.0.attention_norm.weight": "codec_model.stages.1.layers.0.input_layernorm.weight",
            "audio_tokenizer.decoder_blocks.3.layers.0.attention_scale": "codec_model.stages.1.layers.0.self_attn_layer_scale",
            "audio_tokenizer.decoder_blocks.3.layers.0.ffn_scale": "codec_model.stages.1.layers.0.mlp_layer_scale",
            "audio_tokenizer.decoder_blocks.3.layers.0.feed_forward.w2.weight": "codec_model.stages.1.layers.0.mlp.down_proj.weight",
            "audio_tokenizer.quantizer.semantic_codebook.cluster_usage": "codec_model.quantizer.cluster_usage",
            "audio_tokenizer.quantizer.semantic_codebook.embedding_sum": "codec_model.quantizer.embedding_sum",
        }  # fmt: skip
        for original_key, new_key in expected_mapping.items():
            with self.subTest(original_key):
                torch.testing.assert_close(state_dict[new_key], original[original_key], rtol=0, atol=0)

    def test_rope_permutation_matches_interleaved_rope(self):
        config, original, state_dict = self.convert()
        model = VoxtralTtsForTextToSpeech(config).eval()
        model.load_state_dict(state_dict, strict=True)
        text_config, seq_len = config.text_config, 6
        n_heads, n_kv_heads, head_dim = text_config.num_attention_heads, text_config.num_key_value_heads, 8
        hidden_states = torch.randn(seq_len, text_config.hidden_size, generator=torch.Generator().manual_seed(1))

        # Original formulation: RoPE rotates interleaved pairs of dimensions (mistral-inference)
        freqs = 1.0 / (text_config.rope_parameters["rope_theta"] ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs_cis = torch.polar(torch.ones(seq_len, head_dim // 2), torch.outer(torch.arange(seq_len).float(), freqs))

        def interleaved_rope(x):
            x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
            return torch.view_as_real(x * freqs_cis[:, None]).flatten(-2)

        query = interleaved_rope(
            (hidden_states @ original["layers.0.attention.wq.weight"].T).view(seq_len, n_heads, head_dim)
        )
        key = interleaved_rope(
            (hidden_states @ original["layers.0.attention.wk.weight"].T).view(seq_len, n_kv_heads, head_dim)
        )
        expected_scores = torch.einsum("qhd,khd->hqk", query, key.repeat_interleave(n_heads // n_kv_heads, dim=1))  # fmt: skip

        language_model = model.model.language_model
        attention = language_model.layers[0].self_attn
        position_ids = torch.arange(seq_len)[None]
        cos, sin = language_model.rotary_emb(hidden_states[None], position_ids=position_ids)
        hf_query = attention.q_proj(hidden_states).view(1, seq_len, n_heads, head_dim).transpose(1, 2)
        hf_key = attention.k_proj(hidden_states).view(1, seq_len, n_kv_heads, head_dim).transpose(1, 2)
        rotate_half = lambda x: torch.cat([-x[..., head_dim // 2 :], x[..., : head_dim // 2]], dim=-1)  # noqa: E731
        hf_query = hf_query * cos[:, None] + rotate_half(hf_query) * sin[:, None]
        hf_key = hf_key * cos[:, None] + rotate_half(hf_key) * sin[:, None]
        hf_scores = torch.einsum("hqd,hkd->hqk", hf_query[0], hf_key[0].repeat_interleave(n_heads // n_kv_heads, dim=0))  # fmt: skip
        torch.testing.assert_close(hf_scores, expected_scores, rtol=1e-4, atol=1e-4)

    def test_weight_norm_is_folded(self):
        _, original, state_dict = self.convert()
        for original_prefix, new_key, conv in [
            ("audio_tokenizer.decoder_blocks.0", "codec_model.stages.0.conv.weight", torch.nn.Conv1d(12, 16, 3)),
            ("audio_tokenizer.decoder_blocks.2", "codec_model.stages.1.conv.weight", torch.nn.ConvTranspose1d(16, 16, 4)),
            ("audio_tokenizer.output_proj", "codec_model.output_proj.weight", torch.nn.Conv1d(16, 4, 3)),
        ]:  # fmt: skip
            conv = torch.nn.utils.parametrizations.weight_norm(conv)
            conv.parametrizations.weight.original0.data = original[
                f"{original_prefix}.conv.parametrizations.weight.original0"
            ]
            conv.parametrizations.weight.original1.data = original[
                f"{original_prefix}.conv.parametrizations.weight.original1"
            ]
            torch.testing.assert_close(state_dict[new_key], conv.weight.detach())


@slow
@require_torch
@require_mistral_common
class VoxtralTtsIntegrationTest(unittest.TestCase):
    """
    Expected values come from the original model (a transcription of the vllm-omni reference implementation), which
    generates these frames for `"Paris is a beautiful city!"` with the `casual_male` voice (Whisper transcribes the
    audio as "Paris is a beautiful city."). The free-running bf16 generation is too sensitive to numerics to compare
    across devices, so the model is checked with teacher forcing on the reference frames instead.
    """

    # TODO: point to the Transformers-format files once they are on the Hub
    checkpoint = "mistralai/Voxtral-4B-TTS-2603"
    text = "Paris is a beautiful city!"
    voice = "casual_male"
    # fmt: off
    expected_semantic_codes = [855, 806, 7096, 6114, 1438, 855, 10, 10, 6226, 663, 3950, 2072, 8023, 8023, 8023, 10, 10, 10, 3684, 3684, 7973, 1645, 1]
    expected_first_frames = [
        [855, 2, 15, 6, 6, 16, 10, 6, 6, 18, 5, 18, 5, 7, 20, 13, 11, 9, 6, 17, 10, 16, 7, 8, 14, 9, 9, 18, 21, 16, 20, 4, 6, 6, 17, 3, 4],
        [806, 12, 9, 12, 9, 19, 10, 11, 12, 18, 11, 15, 2, 7, 12, 17, 7, 11, 13, 9, 15, 16, 12, 9, 15, 12, 18, 20, 15, 9, 15, 13, 12, 5, 15, 5, 8],
        [7096, 19, 16, 20, 20, 18, 5, 10, 4, 18, 6, 13, 5, 12, 14, 12, 6, 10, 10, 12, 13, 12, 9, 18, 13, 11, 12, 16, 18, 18, 21, 6, 7, 16, 19, 3, 3],
        [6114, 8, 6, 15, 4, 10, 18, 10, 11, 17, 7, 9, 13, 18, 18, 14, 4, 4, 8, 21, 3, 15, 7, 19, 15, 7, 20, 15, 19, 15, 5, 7, 9, 20, 14, 4, 2],
        [1438, 16, 12, 18, 19, 13, 18, 7, 3, 17, 9, 6, 14, 9, 17, 5, 4, 9, 13, 20, 9, 14, 13, 9, 20, 20, 13, 7, 17, 12, 5, 10, 15, 7, 15, 5, 4],
        [855, 16, 4, 6, 16, 21, 10, 7, 14, 8, 8, 20, 14, 8, 10, 17, 13, 10, 9, 15, 9, 16, 9, 16, 14, 10, 14, 22, 20, 10, 10, 18, 9, 10, 13, 10, 10],
    ]
    # fmt: on

    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint)
        cls.model = VoxtralTtsForTextToSpeech.from_pretrained(
            cls.checkpoint, dtype=torch.bfloat16, device_map=torch_device
        )

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cleanup(torch_device, gc_collect=True)

    def test_teacher_forced_semantic_codes(self):
        inputs = self.processor(self.text, voice=self.voice).to(torch_device)
        frames = torch.tensor(self.expected_first_frames, device=torch_device)
        with torch.no_grad():
            outputs = self.model(**inputs, audio_codes=frames[None])
        prompt_length = inputs["input_ids"].shape[1]
        semantic_logits = outputs.semantic_logits[0, prompt_length - 1 :].float().cpu()

        # Values obtained on CPU
        expected_logits_slice = torch.tensor([-5.38, 1.3, -4.56, 6.56, -4.91, -5.06, 8.94, -7.09])
        torch.testing.assert_close(semantic_logits[0, :8], expected_logits_slice, rtol=0, atol=0.15)

        # Restricted to the valid semantic codes, as in generation
        semantic_logits[:, 0] = -float("inf")
        semantic_logits[:, 2 + self.model.config.acoustic_transformer_config.semantic_codebook_size :] = -float("inf")
        predicted = semantic_logits.argmax(-1).tolist()
        expected = self.expected_semantic_codes[: len(predicted)]
        num_matches = sum(p == e for p, e in zip(predicted, expected))
        # Near-ties of the bf16 logits may flip a code
        self.assertGreaterEqual(num_matches, len(expected) - 1, f"{predicted} != {expected}")

    def test_acoustic_codes_from_reference_noise(self):
        # Seeded flow-matching noise for the first frame, as drawn by `generate` with a CPU generator seeded at 0
        inputs = self.processor(self.text, voice=self.voice).to(torch_device)
        with torch.no_grad():
            hidden_state = self.model(**inputs).last_hidden_state[:, -1]
            noise = torch.randn(1, 36, generator=torch.Generator().manual_seed(0)).to(torch_device)
            frame = self.model.acoustic_transformer.generate_frame(hidden_state, noise=noise)[0].tolist()
        expected = self.expected_first_frames[0]
        self.assertEqual(frame[0], expected[0])
        # Acoustic codes are FSQ levels: bf16 differences may move a few of them by one level
        self.assertLessEqual(max(abs(a - b) for a, b in zip(frame[1:], expected[1:])), 1)
        self.assertGreaterEqual(sum(a == b for a, b in zip(frame[1:], expected[1:])), 30)

    def test_codec_decode(self):
        codes = torch.tensor(self.expected_first_frames, device=torch_device) - 2
        with torch.no_grad():
            audio = self.model.codec_model.decode(codes[None]).audio_values[0].float().cpu()
        self.assertEqual(audio.shape, (6 * self.model.config.codec_config.hop_length,))
        # Values obtained on CPU
        expected_slice = torch.tensor(
            [0.00134, -0.00433, -0.00812, -0.00854, -0.00485, -0.00123, 0.00084, -0.00067, -0.00409, -0.00568]
        )
        torch.testing.assert_close(audio[4000:4010], expected_slice, rtol=0, atol=1e-3)

    def test_generate(self):
        inputs = self.processor(self.text, voice=self.voice).to(torch_device)
        outputs = self.model.generate(**inputs, generator=torch.Generator(torch_device).manual_seed(0))
        # The sentence lasts about 1.8 s (22 frames of 80 ms) and generation stops on `[END_AUDIO]`
        num_frames = int(outputs.audio_lengths[0])
        self.assertTrue(15 <= num_frames <= 30, num_frames)
        self.assertEqual(int(outputs.audio_codes[0, num_frames, 0]), 1)
        self.assertEqual(outputs.audio[0].shape, (num_frames * self.model.config.codec_config.hop_length,))
