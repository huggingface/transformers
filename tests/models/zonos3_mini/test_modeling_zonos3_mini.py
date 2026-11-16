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
"""Testing suite for the PyTorch ZONOS3-Mini model."""

import copy
import json
import unittest
from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download
from parameterized import parameterized

from transformers import Zonos3MiniConfig, is_torch_available
from transformers.testing_utils import require_torch, require_torchaudio, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ..zac.test_modeling_zac import load_reference_clip


if is_torch_available():
    import torch
    from safetensors.torch import load_file

    from transformers import (
        AutoProcessor,
        Zonos3MiniForConditionalGeneration,
        Zonos3MiniSpeakerEncoderModel,
    )


ZONOS3_MINI_CHECKPOINT = "Zyphra/ZONOS3-Mini-transformers"


class Zonos3MiniModelTester:
    def __init__(self, parent, batch_size=2, seq_length=7, is_training=True):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.num_codebooks = 3
        self.codebook_size = 16
        self.vocab_size = 32
        self.num_hidden_layers = 4
        self.hidden_size = 64
        self.num_attention_heads = 4

    def get_config(self):
        return Zonos3MiniConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=96,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=2,
            head_dim=16,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            moe_intermediate_size=32,
            moe_latent_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            num_codebooks=self.num_codebooks,
            codebook_size=self.codebook_size,
            audio_eos_token_id=16,
            audio_pad_token_id=17,
            text_pad_token_id=31,
            condition_bucket_counts=[2, 2, 3],
            max_position_embeddings=64,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=3,
            speaker_encoder_config={
                "in_channels": 4,
                "num_mel_bins": 16,
                "hidden_size": 16,
                "num_attention_heads": 2,
                "num_experts": 2,
                "bottleneck_size": 8,
                "embedding_size": 192,
                "pos_conv_kernel_size": 5,
                "stage_block_specs": [[[1, 4, 1, 1, 2]], [[1, 8, 2, 2, 2]]],
            },
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        audio_ids = ids_tensor([self.batch_size, self.seq_length, self.num_codebooks], self.codebook_size)
        text_ids = ids_tensor([self.batch_size, self.seq_length, 1], self.vocab_size - 1)
        input_ids = torch.cat([audio_ids, text_ids], dim=-1)
        attention_mask = torch.ones(self.batch_size, self.seq_length, dtype=torch.long, device=torch_device)
        return config, input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}


@require_torch
class Zonos3MiniForConditionalGenerationTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (Zonos3MiniForConditionalGeneration,) if is_torch_available() else ()
    test_resize_embeddings = False
    # the speaker encoder has its own model class, tested below
    test_all_params_have_gradient = False

    def setUp(self):
        self.model_tester = Zonos3MiniModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Zonos3MiniConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)
        if return_labels:
            inputs_dict["labels"] = torch.zeros(
                (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.num_codebooks),
                dtype=torch.long,
                device=torch_device,
            )
        return inputs_dict

    def _get_logits_processor_kwargs(self, do_sample=False, config=None):
        # only the per-codebook warpers apply to audio codebook logits
        return {"top_k": 10, "top_p": 0.7, "temperature": 0.7} if do_sample else {}

    def _check_scores(self, batch_size, scores, generated_length, config):
        self.assertEqual(len(scores), generated_length)
        for score in scores:
            self.assertEqual(score.shape, (batch_size, config.num_codebooks, config.audio_vocab_size))

    def _check_logits(self, batch_size, logits, config):
        for logit in logits:
            self.assertEqual(logit.shape, (batch_size, config.num_codebooks, config.audio_vocab_size))

    def _get_conv_state_shape(self, batch_size: int, config):
        key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        return (batch_size, 2 * key_dim + value_dim, config.linear_conv_kernel_dim)

    def _get_recurrent_state_shape(self, batch_size: int, config):
        return (
            batch_size,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        )

    def test_attention_outputs(self):
        """Only the full attention layers return attention weights (the others are gated DeltaNet layers)."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        seq_len = self.model_tester.seq_length
        num_attention_layers = sum(layer == "full_attention" for layer in config.layer_types)
        model = Zonos3MiniForConditionalGeneration._from_config(config, attn_implementation="eager")
        model.to(torch_device).eval()
        with torch.no_grad():
            outputs = model(**inputs_dict, output_attentions=True, output_hidden_states=True)
        self.assertEqual(len(outputs.attentions), num_attention_layers)
        self.assertEqual(list(outputs.attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])
        self.assertEqual(len(outputs.hidden_states), config.num_hidden_layers + 1)

    @unittest.skip(reason="ZONOS3-Mini is a new model: its forward only takes `cache`, not the deprecated names.")
    def test_bc_cache_name_forward(self):
        pass

    def test_speaker_and_condition_rows_replace_embeddings(self):
        """A speaker row takes the projected speaker embedding and a condition row its bucket's embedding."""
        config, input_ids, _ = self.model_tester.prepare_config_and_inputs()
        model = Zonos3MiniForConditionalGeneration(config).to(torch_device).eval()
        speaker_embeddings = floats_tensor([self.model_tester.batch_size, config.speaker_embedding_size])
        speaker_token_mask = torch.zeros(input_ids.shape[:2], dtype=torch.bool, device=torch_device)
        speaker_token_mask[:, 0] = True
        condition_ids = torch.full(input_ids.shape[:2], -1, dtype=torch.long, device=torch_device)
        condition_ids[:, 1] = 2
        embeddings = model.model.embeddings(input_ids, speaker_embeddings, speaker_token_mask, condition_ids)
        torch.testing.assert_close(embeddings[:, 0], model.model.embeddings.speaker_proj(speaker_embeddings))
        torch.testing.assert_close(
            embeddings[:, 1], model.model.embeddings.embed_conditions.weight[2].expand_as(embeddings[:, 1])
        )
        torch.testing.assert_close(embeddings[:, 2:], model.model.embeddings(input_ids)[:, 2:])

    def test_generate_ends_after_end_of_audio(self):
        """An item stops `num_codebooks + 1` rows after its first end-of-audio."""
        config, input_ids, attention_mask = self.model_tester.prepare_config_and_inputs()
        model = Zonos3MiniForConditionalGeneration(config).to(torch_device).eval()
        # every audio column predicts end-of-audio: generation ends after `num_codebooks + 1` rows
        eos_logits = torch.full((config.num_codebooks, config.audio_vocab_size), -10.0, device=torch_device)
        eos_logits[:, config.audio_eos_token_id] = 10.0
        model.lm_head.register_forward_hook(
            lambda module, args, output: eos_logits.flatten().expand(*output.shape[:-1], -1)
        )
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=20, do_sample=False)
        generated = output[:, input_ids.shape[1] :]
        self.assertEqual(generated.shape[1], config.num_codebooks + 1)
        # the item keeps sampling while its delayed codebooks flush, then stops
        self.assertTrue((generated[..., : config.num_codebooks] == config.audio_eos_token_id).all())
        self.assertTrue((generated[..., -1] == config.text_pad_token_id).all())

    def test_generate_cold_start_keeps_delay_triangle_padded(self):
        """Without an audio prefix, codebook `j` of the first `j` generated rows is padding."""
        config, input_ids, attention_mask = self.model_tester.prepare_config_and_inputs()
        input_ids[:, -1, : config.num_codebooks] = config.audio_pad_token_id
        model = Zonos3MiniForConditionalGeneration(config).to(torch_device).eval()
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=4, do_sample=False)
        generated = output[:, input_ids.shape[1] :, : config.num_codebooks]
        for step in range(config.num_codebooks - 1):
            self.assertTrue((generated[:, step, step + 1 :] == config.audio_pad_token_id).all())
            self.assertTrue((generated[:, step, : step + 1] != config.audio_pad_token_id).all())

    def test_speaker_guidance_requires_unconditional_prompt(self):
        config, input_ids, attention_mask = self.model_tester.prepare_config_and_inputs()
        model = Zonos3MiniForConditionalGeneration(config).to(torch_device).eval()
        with self.assertRaises(ValueError):
            model.generate(input_ids=input_ids, guidance_scale=1.5, max_new_tokens=2)
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            uncond_input_ids=input_ids,
            uncond_attention_mask=attention_mask,
            guidance_scale=1.5,
            max_new_tokens=3,
            do_sample=False,
        )
        self.assertEqual(output.shape[0], self.model_tester.batch_size)

    def test_num_layers_is_small(self):
        # 4 layers: three gated DeltaNet layers then one attention layer (the release's period), and dense then sparse
        # feed-forwards
        self.assertLessEqual(self.model_tester.num_hidden_layers, 4)

    def test_forward_with_logits_to_keep(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Zonos3MiniForConditionalGeneration(config).to(torch_device).eval()
        with torch.no_grad():
            all_logits = model(**inputs_dict, logits_to_keep=0).logits
            last_logits = model(**inputs_dict, logits_to_keep=1).logits
        self.assertEqual(last_logits.shape, (*all_logits.shape[:1], 1, *all_logits.shape[2:]))
        torch.testing.assert_close(all_logits[:, -1:], last_logits, rtol=1e-5, atol=1e-5)

    @pytest.mark.generate
    @unittest.skip(reason="ZONOS3-Mini only generates audio rows: the text logits processors do not apply.")
    def test_bc_cache_name_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="ZONOS3-Mini needs prompt rows: it cannot start from a BOS token alone.")
    def test_generate_without_input_ids(self):
        pass

    @pytest.mark.generate
    @unittest.skip(
        reason="The common test pads the attention mask to `sequences.shape[-1]`, the column count of ZONOS3-Mini's "
        "rows; continuation from a cache is covered by `test_generate_continues_from_cache`."
    )
    def test_generate_continue_from_cache(self):
        pass

    def test_generate_continues_from_cache(self):
        config, input_ids, attention_mask = self.model_tester.prepare_config_and_inputs()
        model = Zonos3MiniForConditionalGeneration(config).to(torch_device).eval()
        kwargs = {"do_sample": False, "return_dict_in_generate": True}
        full = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=4, **kwargs)
        first = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=3, **kwargs)
        attention_mask = torch.nn.functional.pad(
            attention_mask, (0, first.sequences.shape[1] - input_ids.shape[1]), value=1
        )
        continued = model.generate(
            input_ids=first.sequences, attention_mask=attention_mask, cache=first.cache, max_new_tokens=1, **kwargs
        )
        torch.testing.assert_close(continued.sequences, full.sequences)

    @unittest.skip(
        reason="The common test splits the inputs on their last dimension, the columns of ZONOS3-Mini's rows."
    )
    def test_recurrent_layers_mask_padding_on_continued_forward(self):
        pass

    def test_custom_4d_attention_mask(self):
        """Three sequences sharing a 3-row prefix give the same last-row logits when packed under a 4D mask."""
        config = self.model_tester.get_config()
        # attention layers only: a packed prefix cannot be shared through the gated DeltaNet's recurrent state
        config.layer_types = ["full_attention"] * config.num_hidden_layers
        model = Zonos3MiniForConditionalGeneration(config).to(torch_device, dtype=torch.float32).eval()
        input_ids = ids_tensor([3, 4, config.num_codebooks + 1], config.codebook_size)
        input_ids[:, :3] = input_ids[0, :3]
        position_ids = torch.arange(4, device=torch_device).expand(3, -1)
        packed_ids = torch.cat([input_ids[0, :3], input_ids[:, 3]])[None]
        mask = torch.tril(torch.ones(6, 6, device=torch_device))
        mask[3:, 3:] = torch.eye(3, device=torch_device)
        mask = (1.0 - mask)[None, None] * torch.finfo(torch.float32).min
        packed_position_ids = torch.tensor([[0, 1, 2, 3, 3, 3]], device=torch_device)
        with torch.no_grad():
            logits = model(input_ids, position_ids=position_ids).logits[:, -1]
            packed_logits = model(packed_ids, attention_mask=mask, position_ids=packed_position_ids).logits[0, -3:]
        torch.testing.assert_close(packed_logits.softmax(-1), logits.softmax(-1), rtol=1e-3, atol=1e-3)

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    @unittest.skip(reason="The common test builds 2D token inputs; ZONOS3-Mini takes rows of audio and text columns.")
    def test_model_rope_scaling_from_config(self, scaling_type):
        pass

    @unittest.skip(reason="The row embedder also takes the speaker and condition inputs: covered by its own test.")
    def test_inputs_embeds(self):
        pass

    def test_inputs_embeds_matches_input_ids(self):
        config, input_ids, attention_mask = self.model_tester.prepare_config_and_inputs()
        model = Zonos3MiniForConditionalGeneration(config).to(torch_device).eval()
        with torch.no_grad():
            from_ids = model(input_ids=input_ids, attention_mask=attention_mask).logits
            inputs_embeds = model.get_input_embeddings()(input_ids)
            from_embeds = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
        torch.testing.assert_close(from_embeds, from_ids)

    def test_speaker_encoder_runs_each_clip_at_its_length(self):
        """A padded clip gets the embedding of the clip alone."""
        config = self.model_tester.get_config()
        encoder = Zonos3MiniSpeakerEncoderModel(config.speaker_encoder_config).to(torch_device).eval()
        features = floats_tensor([2, 40, config.speaker_encoder_config.num_mel_bins])
        mask = torch.ones(2, 40, dtype=torch.long, device=torch_device)
        mask[1, 30:] = 0
        batched = encoder(features, attention_mask=mask).speaker_embeddings
        alone = encoder(features[1:, :30]).speaker_embeddings
        torch.testing.assert_close(batched[1:], alone)

    @pytest.mark.generate
    @unittest.skip(reason="ZONOS3-Mini does not support assisted decoding.")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="ZONOS3-Mini does not support assisted decoding.")
    def test_assisted_decoding_sample(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="ZONOS3-Mini does not support beam search.")
    def test_beam_sample_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="ZONOS3-Mini does not support beam search.")
    def test_beam_search_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="ZONOS3-Mini does not support beam search.")
    def test_beam_search_generate_dict_output(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="ZONOS3-Mini does not support beam search.")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="ZONOS3-Mini does not support beam search.")
    def test_beam_sample_generate_dict_output(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="ZONOS3-Mini does not support beam search.")
    def test_generate_from_inputs_embeds_1_beam_search(self, _, num_beams):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="ZONOS3-Mini does not support beam search.")
    def test_model_parallel_beam_search(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="ZONOS3-Mini does not support prompt lookup decoding.")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="ZONOS3-Mini does not support prompt lookup decoding.")
    def test_prompt_lookup_decoding_stops_at_eos(self):
        pass

    @unittest.skip(reason="ZONOS3-Mini embeds rows of audio and text columns, not a single token sequence.")
    def test_model_get_set_embeddings(self):
        pass


@slow
@require_torch
@require_torchaudio
class Zonos3MiniForConditionalGenerationIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    fixtures_path = Path(__file__).parent.parent.parent / "fixtures/zonos3_mini"

    def _load(self, name):
        with open(self.fixtures_path / name, encoding="utf-8") as f:
            return json.load(f)

    def _reference_voice(self):
        return load_file(hf_hub_download(ZONOS3_MINI_CHECKPOINT, "voices/Savannah.safetensors"))

    def test_logits(self):
        """Reproducer: zonos3_repro/reproducer.py `prompt` and `logits` (the remote code of Zyphra/ZONOS3-Mini-hf@e39d30d)."""
        expected_prompt, expected = self._load("expected_prompt.json"), self._load("expected_logits.json")
        processor = AutoProcessor.from_pretrained(ZONOS3_MINI_CHECKPOINT)
        model = Zonos3MiniForConditionalGeneration.from_pretrained(ZONOS3_MINI_CHECKPOINT, dtype=torch.float32)
        model = model.to(torch_device).eval()
        inputs = processor(
            text=expected_prompt["text"],
            speaker_reference_codes=[self._reference_voice()["codes"]],
            text_normalization=False,
            return_tensors="pt",
        ).to(torch_device)
        torch.testing.assert_close(
            inputs["input_ids"][0].cpu(), torch.tensor(expected_prompt["reference"]["input_ids"]), atol=0, rtol=0
        )
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].cpu()
        torch.testing.assert_close(logits[:, :16], torch.tensor(expected["logits"]), atol=2e-3, rtol=2e-3)
        self.assertEqual(logits.argmax(-1).tolist(), expected["argmax"])

    def test_generate_reference_voice(self):
        """Reproducer: zonos3_repro/reproducer.py `generate` (the remote code of Zyphra/ZONOS3-Mini-hf@e39d30d)."""
        expected_prompt, expected = self._load("expected_prompt.json"), self._load("expected_generate.json")
        processor = AutoProcessor.from_pretrained(ZONOS3_MINI_CHECKPOINT)
        model = Zonos3MiniForConditionalGeneration.from_pretrained(ZONOS3_MINI_CHECKPOINT, dtype=torch.float32)
        model = model.to(torch_device).eval()
        inputs = processor(
            text=expected_prompt["text"],
            speaker_reference_codes=[self._reference_voice()["codes"]],
            text_normalization=False,
            return_tensors="pt",
        ).to(torch_device)
        # the reference samples with a generator seeded with 0: on CPU, the draws match the global RNG's
        torch.manual_seed(0)
        sequences = model.generate(**inputs, do_sample=True, max_new_tokens=expected["max_new_frames"])
        generated = sequences[0, inputs["input_ids"].shape[1] :].cpu()
        torch.testing.assert_close(generated[:6], torch.tensor(expected["generated_rows"])[:6], atol=0, rtol=0)

        audio = processor.decode(sequences)[0]
        self.assertEqual(audio.shape[-1], expected["num_samples"])
        torch.testing.assert_close(audio[:4000], torch.tensor(expected["waveform"]), atol=1e-3, rtol=1e-3)

    def test_speaker_embedding(self):
        """Reproducer: zonos3_repro/reproducer.py `speaker` (the remote code of Zyphra/ZONOS3-Mini-hf@e39d30d)."""
        expected = self._load("expected_speaker_embedding.json")
        processor = AutoProcessor.from_pretrained(ZONOS3_MINI_CHECKPOINT)
        model = Zonos3MiniForConditionalGeneration.from_pretrained(ZONOS3_MINI_CHECKPOINT).to(torch_device).eval()
        features = processor.feature_extractor(load_reference_clip().numpy(), sampling_rate=44100, return_tensors="pt")
        with torch.no_grad():
            embedding = model.speaker_encoder(**features.to(torch_device)).speaker_embeddings[0].cpu()
        torch.testing.assert_close(embedding, torch.tensor(expected["speaker_embedding"]), atol=1e-4, rtol=1e-4)
