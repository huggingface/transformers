# coding=utf-8
# Copyright 2025 Boson AI and The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch HiggsAudio model."""

import copy
import pathlib
import tempfile
import unittest

import numpy as np
import pytest

from transformers import AutoTokenizer
from transformers.models.higgs_audio import HiggsAudioConfig
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)
from transformers.utils import (
    is_soundfile_available,
    is_torch_available,
    is_torchaudio_available,
)
from transformers.utils.import_utils import is_datasets_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_datasets_available():
    from datasets import Audio, load_dataset

if is_torch_available():
    import torch

    from transformers import (
        HiggsAudioForConditionalGeneration,
        HiggsAudioProcessor,
        LlamaConfig,
        PretrainedConfig,
        PreTrainedModel,
    )
    from transformers.cache_utils import (
        Cache,
        StaticCache,
    )

if is_torchaudio_available():
    import torchaudio

if is_soundfile_available():
    import soundfile as sf


@require_torch
class HiggsAudioModelTester:
    def __init__(
        self,
        parent,
        batch_size=1,  # Generation only supports a single sample at the current stage
        seq_length=40,
        max_length=100,
        is_training=True,
        vocab_size=100,
        num_audio_in=2,
        num_audio_out=2,
        hidden_size=16,
        intermediate_size=37,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        audio_decoder_proj_num_layers=1,
        audio_dual_ffn_layers=[0],
        audio_adapter_type="dual_ffn_fast_forward",
        encode_audio_in_tokens=True,
        num_quantizers=4,
        num_channels=1,
        codebook_size=20,
        sample_rate=16000,
        audio_in_token_idx=98,
        audio_out_token_idx=99,
        pad_token_id=95,
        audio_out_bos_token_id=96,
        audio_eos_token_id=97,
        audio_length=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.max_length = max_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.num_audio_in = num_audio_in
        self.num_audio_out = num_audio_out
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.audio_decoder_proj_num_layers = audio_decoder_proj_num_layers
        self.audio_dual_ffn_layers = audio_dual_ffn_layers
        self.audio_adapter_type = audio_adapter_type
        self.encode_audio_in_tokens = encode_audio_in_tokens
        self.num_quantizers = num_quantizers
        self.num_channels = num_channels
        self.codebook_size = codebook_size
        self.sample_rate = sample_rate
        self.audio_in_token_idx = audio_in_token_idx
        self.audio_out_token_idx = audio_out_token_idx
        self.audio_out_bos_token_id = audio_out_bos_token_id
        self.audio_eos_token_id = audio_eos_token_id
        self.pad_token_id = pad_token_id
        self.audio_length = audio_length

    def get_config(self):
        llm_config = LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
        )

        audio_num_codebooks = self.num_quantizers
        audio_codebook_size = self.codebook_size

        higgs_audio_config = HiggsAudioConfig(
            llm_config,
            audio_adapter_type=self.audio_adapter_type,
            audio_dual_ffn_layers=self.audio_dual_ffn_layers,
            audio_decoder_proj_num_layers=self.audio_decoder_proj_num_layers,
            encode_audio_in_tokens=self.encode_audio_in_tokens,
            audio_num_codebooks=audio_num_codebooks,
            audio_codebook_size=audio_codebook_size,
            audio_stream_bos_id=audio_codebook_size,
            audio_stream_eos_id=audio_codebook_size + 1,
            pad_token_id=self.pad_token_id,
            audio_in_token_idx=self.audio_in_token_idx,
            audio_out_token_idx=self.audio_out_token_idx,
            audio_out_bos_token_id=self.audio_out_bos_token_id,
            audio_eos_token_id=self.audio_eos_token_id,
        )
        return higgs_audio_config

    def prepare_config_and_inputs(self) -> tuple[HiggsAudioConfig, dict]:
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 5)
        attention_mask = input_ids.ne(self.pad_token_id)

        rng = np.random.RandomState(0)
        random_indices = rng.choice(self.seq_length, self.num_audio_in + self.num_audio_out, replace=False)
        seq_ids = random_indices % self.seq_length
        input_ids[0, seq_ids[: self.num_audio_in]] = self.audio_in_token_idx
        input_ids[0, seq_ids[self.num_audio_in :]] = self.audio_out_token_idx

        audio_in_lengths = torch.full((self.num_audio_in,), self.audio_length, device=torch_device)
        audio_in_ids = ids_tensor([self.num_quantizers, audio_in_lengths.sum()], self.codebook_size)
        audio_in_ids_start = torch.cumsum(audio_in_lengths, dim=0) - audio_in_lengths

        audio_out_lengths = torch.full((self.num_audio_out,), self.audio_length, device=torch_device)
        audio_out_ids = ids_tensor([self.num_quantizers, audio_out_lengths.sum()], self.codebook_size)
        audio_out_ids_start = torch.cumsum(audio_out_lengths, dim=0) - audio_out_lengths

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_in_ids": audio_in_ids,
            "audio_in_ids_start": audio_in_ids_start,
            "audio_out_ids": audio_out_ids,
            "audio_out_ids_start": audio_out_ids_start,
        }

        config = self.get_config()

        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self) -> tuple[HiggsAudioConfig, dict]:
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_model_forward(self, config, inputs_dict):
        model = HiggsAudioForConditionalGeneration(config=config).to(torch_device).eval()

        # first forward pass
        last_hidden_states = model(**inputs_dict).last_hidden_states

        self.parent.assertTrue(
            last_hidden_states.shape,
            (
                self.batch_size,
                self.seq_length + (self.audio_length - 1) * (self.num_audio_in + self.num_audio_out),
                config.text_config.hidden_size,
            ),
        )


class HiggsAudioGenerationTesterMixin(GenerationTesterMixin):
    def prepare_config_and_inputs_for_generate(self, batch_size=1):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # We don't want a few model inputs in our model input dictionary for generation tests
        input_keys_to_ignore = [
            # we don't want to mask attention heads
            "head_mask",
            # we'll set cache use in each test differently
            "use_cache",
            # Ignore labels if it is in the input dict
            "labels",
            # model-specific exceptions should overload/overwrite this function
        ]
        filtered_inputs_dict = {
            k: (v[:batch_size, ...] if isinstance(v, torch.Tensor) and k in ["input_ids", "attention_mask"] else v)
            for k, v in inputs_dict.items()
            if k not in input_keys_to_ignore
        }

        # It is important set `eos_token_id` to `None` to avoid early stopping (would break for length-based checks)
        text_gen_config = config.get_text_config(decoder=True)
        if text_gen_config.eos_token_id is not None and text_gen_config.pad_token_id is None:
            text_gen_config.pad_token_id = (
                text_gen_config.eos_token_id
                if isinstance(text_gen_config.eos_token_id, int)
                else text_gen_config.eos_token_id[0]
            )
        text_gen_config.eos_token_id = None
        text_gen_config.forced_eos_token_id = None

        return config, filtered_inputs_dict

    @pytest.mark.generate
    def test_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).to(torch_device).eval()
            output_generate = self._sample_generate(model=model, inputs_dict=inputs_dict, num_return_sequences=1)

            self.assertTrue(output_generate[0].shape[1] == self.max_new_tokens + inputs_dict["input_ids"].shape[1])

    @pytest.mark.generate
    def test_greedy_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).to(torch_device).eval()
            output_generate = self._greedy_generate(model=model, inputs_dict=inputs_dict)

            self.assertTrue(output_generate[0].shape[1] == self.max_new_tokens + inputs_dict["input_ids"].shape[1])


@require_torch
class HiggsAudioForConditionalGenerationTest(
    ModelTesterMixin,
    HiggsAudioGenerationTesterMixin,
    PipelineTesterMixin,
    unittest.TestCase,
):
    all_model_classes = (HiggsAudioForConditionalGeneration,) if is_torch_available() else ()
    # We only allow greedy search / sampling with one sequence; see `skip_non_greedy_generate`
    all_generative_model_classes = (HiggsAudioForConditionalGeneration,)
    # TODO: support new pipeline behavior in tests
    pipeline_model_mapping = {}
    # pipeline_model_mapping = {"text-to-audio": HiggsAudioForConditionalGeneration} if is_torch_available() else {}
    test_pruning = False
    test_head_masking = False
    test_resize_embeddings = False
    is_encoder_decoder = False
    _is_composite = False
    has_attentions = True

    def setUp(self):
        self.model_tester = HiggsAudioModelTester(self)
        # Skipping `has_text_modality` but manually testing down below
        self.config_tester = ConfigTester(self, has_text_modality=False, config_class=HiggsAudioConfig)
        self.skip_non_greedy_generate()

    def skip_non_greedy_generate(self):
        skippable_tests = [
            "test_sample_generate_dict_output",
            "test_beam",
            "test_group_beam",
            "test_constrained_beam",
            "test_contrastive",
            "test_assisted",
            "test_dola",
            "test_prompt_lookup",
            "test_model_parallel_beam_search",
            "test_generate_without_input_ids",
            "test_generate_with_head_masking",
            "test_generate_from_random_inputs_embeds",
            "test_generate_from_inputs_embeds_1_beam_search",
        ]

        for test in skippable_tests:
            if self._testMethodName.startswith(test):
                self.skipTest(reason="HiggsAudio only supports greedy search / sampling with one sequence.")

    @pytest.mark.generate
    @unittest.skip(reason="HiggsAudio does not support input_embeds.")
    def test_generate_from_inputs_embeds_0_greedy(self):
        pass

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        """Overriden to account for the 2D flattened structure"""
        inputs_dict = copy.deepcopy(inputs_dict)

        if return_labels:
            inputs_dict["label_ids"] = torch.ones(
                (
                    self.model_tester.batch_size * self.model_tester.num_channels,
                    self.model_tester.seq_length,
                ),
                dtype=torch.long,
                device=torch_device,
            )
            inputs_dict["label_audio_ids"] = torch.ones(
                (
                    self.model_tester.num_quantizers,
                    self.model_tester.audio_length * self.model_tester.num_audio_out,
                ),
                dtype=torch.long,
                device=torch_device,
            )

        return inputs_dict

    def _get_logits_processor_kwargs(self, do_sample=False, config=None):
        # HiggsAudio does not support repetition_penalty
        logits_processor_kwargs = {
            "bad_words_ids": [[1, 0]],
            "remove_invalid_values": True,
        }
        if do_sample:
            logits_processor_kwargs.update(
                {
                    "top_k": 10,
                    "top_p": 0.7,
                    "temperature": 0.7,
                }
            )
        # TODO (joao, raushan): see this comment for a long-term fix
        # https://github.com/huggingface/transformers/pull/33593#issuecomment-2361824264)
        # This is a band-aid for VLM models, to ensure they don't generate image/video tokens which would cause them
        # to crash. On pretrained models this isn't a risk, as they are trained to not generate these tokens.
        if config is not None:
            for key in [
                "image_token_id",
                "video_token_id",
                "audio_token_id",
                "vision_start_token_id",
                "audio_start_token_id",
                "audio_end_token_id",
                "vision_end_token_id",
            ]:
                token_index = getattr(config, key, None)
                if token_index is None and hasattr(self, "model_tester"):
                    token_index = getattr(self.model_tester, key, None)
                if token_index is not None and token_index < config.get_text_config().vocab_size:
                    logits_processor_kwargs["bad_words_ids"].append([token_index])

        return logits_processor_kwargs

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_from_and_save_pretrained_subfolder()
        self.config_tester.create_and_test_config_from_and_save_pretrained_composite()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

        # Manual testing because of composite configs
        config = self.model_tester.prepare_config_and_inputs()[0]
        self.assertTrue(
            hasattr(config.text_config, "vocab_size"),
            msg="LLM backbone `vocab_size` does not exist",
        )

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def _check_logits(self, batch_size, logits, config):
        vocab_size = config.vocab_size
        expected_audio_logit_shape = (
            batch_size * self.model_tester.num_quantizers,
            self.model_tester.codebook_size,
        )
        expected_text_logit_shape = (batch_size, vocab_size)
        self.assertIsInstance(logits, tuple)
        for iter_logit in logits:
            if iter_logit.shape[0] == batch_size:
                self.assertEqual(iter_logit.shape, expected_text_logit_shape)
            else:
                self.assertEqual(iter_logit.shape, expected_audio_logit_shape)

        vocab_diff = vocab_size - logits[0].shape[-1]
        self.assertTrue(vocab_diff in [0, 1])
        self.assertListEqual(
            [vocab_size - score.shape[-1] for score in logits],
            [vocab_diff] * len(logits),
        )

    def _check_attentions_for_generate(
        self,
        batch_size,
        attentions,
        prompt_length,
        output_length,
        config,
        decoder_past_key_values,
    ):
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [isinstance(iter_attentions, tuple) for iter_attentions in attentions],
            [True] * len(attentions),
        )
        self.assertEqual(len(attentions), (output_length - prompt_length))

        use_cache = decoder_past_key_values is not None
        has_static_cache = isinstance(decoder_past_key_values, StaticCache)

        # When `output_attentions=True`, each iteration of generate appends the attentions corresponding to the new
        # token(s)
        for generated_length, iter_attentions in enumerate(attentions):
            # regardless of using cache, the first forward pass will have the full prompt as input
            if use_cache and generated_length > 0:
                model_input_length = 1
            else:
                model_input_length = prompt_length + generated_length
                # HiggsAudio embeds audio features and tokens inside its forward
                model_input_length += (self.model_tester.num_audio_in + self.model_tester.num_audio_out) * (
                    self.model_tester.audio_length - 1
                )
            query_length = (
                prompt_length
                + generated_length
                + (self.model_tester.num_audio_in + self.model_tester.num_audio_out)
                * (self.model_tester.audio_length - 1)
                if not has_static_cache
                else decoder_past_key_values.get_max_cache_shape()
            )

            expected_shape = (
                batch_size,
                config.num_attention_heads,  # Decoder config
                model_input_length,
                query_length,
            )
            # check attn size
            self.assertListEqual(
                [layer_attention.shape for layer_attention in iter_attentions],
                [expected_shape] * len(iter_attentions),
            )

    def _check_encoder_attention_for_generate(self, attentions, batch_size, config, prompt_length):
        # Encoder config
        encoder_expected_shape = (
            batch_size,
            config.encoder_config.num_attention_heads,
            prompt_length,
            prompt_length,
        )
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [layer_attentions.shape for layer_attentions in attentions],
            [encoder_expected_shape] * len(attentions),
        )

    def _check_hidden_states_for_generate(
        self,
        batch_size,
        hidden_states,
        prompt_length,
        output_length,
        config,
        use_cache=False,
    ):
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [isinstance(iter_hidden_states, tuple) for iter_hidden_states in hidden_states],
            [True] * len(hidden_states),
        )
        self.assertEqual(len(hidden_states), (output_length - prompt_length))

        # When `output_hidden_states=True`, each iteration of generate appends the hidden states corresponding to the
        # new token(s)
        for generated_length, iter_hidden_states in enumerate(hidden_states):
            # regardless of using cache, the first forward pass will have the full prompt as input
            if use_cache and generated_length > 0:
                model_input_length = 1
            else:
                model_input_length = prompt_length + generated_length
                # HiggsAudio embeds audio features and tokens inside its forward
                model_input_length += (self.model_tester.num_audio_in + self.model_tester.num_audio_out) * (
                    self.model_tester.audio_length - 1
                )

            # check hidden size
            expected_shape = (batch_size, model_input_length, config.hidden_size)
            self.assertTrue(
                [layer_hidden_states.shape for layer_hidden_states in iter_hidden_states]
                == [expected_shape] * len(iter_hidden_states)
            )

    def _check_past_key_values_for_generate(self, batch_size, decoder_past_key_values, cache_length, config):
        self.assertIsInstance(decoder_past_key_values, (tuple, Cache))
        cache_length += (self.model_tester.num_audio_in + self.model_tester.num_audio_out) * (
            self.model_tester.audio_length - 1
        )

        # (batch, head, seq_length, head_features)
        expected_shape = (
            batch_size,
            (config.num_key_value_heads if hasattr(config, "num_key_value_heads") else config.num_attention_heads),
            cache_length,
            (config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads),
        )

        if isinstance(decoder_past_key_values, Cache):
            self.assertListEqual(
                [key_tensor.shape for key_tensor in decoder_past_key_values.key_cache],
                [expected_shape] * len(decoder_past_key_values.key_cache),
            )
            self.assertListEqual(
                [value_tensor.shape for value_tensor in decoder_past_key_values.value_cache],
                [expected_shape] * len(decoder_past_key_values.value_cache),
            )

    def _check_scores(self, batch_size, scores, generated_length, config):
        # Special case where HiggsAudio keeps score in a 2D mesh of (bsz * num_quantizers, codebook_size) for audio scores and (bsz, vocab) for text tokens
        vocab_size = config.vocab_size
        expected_audio_score_shape = (
            batch_size * self.model_tester.num_quantizers,
            self.model_tester.codebook_size,
        )
        expected_text_score_shape = (batch_size, vocab_size)
        self.assertIsInstance(scores, tuple)
        self.assertEqual(len(scores), generated_length)
        for iter_score in scores:
            if iter_score.shape[0] == batch_size:
                self.assertEqual(iter_score.shape, expected_text_score_shape)
            else:
                self.assertEqual(iter_score.shape, expected_audio_score_shape)

    def test_sdpa_can_dispatch_composite_models(self):
        """
        Overwritten as it relies on hardcoded namings atm - checking for our case here specifically
        """
        for model_class in self.all_model_classes:
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname)

                sub_models_supporting_sdpa = [
                    (module._supports_sdpa or module._supports_attention_backend)
                    for name, module in model.named_modules()
                    if isinstance(module, PreTrainedModel) and name != ""
                ]
                supports_sdpa_all_modules = (
                    all(sub_models_supporting_sdpa)
                    if len(sub_models_supporting_sdpa) > 0
                    else (model._supports_sdpa or model._supports_attention_backend)
                )

                if not supports_sdpa_all_modules:
                    with self.assertRaises(ValueError):
                        model_sdpa = model_class.from_pretrained(tmpdirname, attn_implementation="sdpa")
                else:
                    model_sdpa = model_class.from_pretrained(tmpdirname, attn_implementation="sdpa")
                    for key in model_sdpa.config:
                        if isinstance(getattr(model_sdpa.config, key), PretrainedConfig):
                            sub_config = getattr(model_sdpa.config, key)
                            self.assertTrue(sub_config._attn_implementation == "sdpa")

    def test_attention_outputs(self):
        self.model_tester.encoder_seq_length = self.model_tester.seq_length + (
            self.model_tester.num_audio_in + self.model_tester.num_audio_out
        ) * (self.model_tester.audio_length - 1)
        super().test_attention_outputs()
        self.model_tester.encoder_seq_length = None

    @pytest.mark.generate
    @unittest.skip("HiggsAudio has complicated attention mask schemes and doesn't support continue from past kv")
    def test_generate_continue_from_past_key_values(self):
        pass

    @pytest.mark.generate
    @unittest.skip("HiggsAudio has to DynamicCache, and DynamicCache object has no attribute key_cache")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="Indirectly checked in HiggsAudio through the generate methods.")
    def test_past_key_values_format(self, custom_all_cache_shapes=None):
        pass

    @unittest.skip(reason="Indirectly checked in HiggsAudio through the generate methods.")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(
        reason="HiggsAudio has too many mixed embedding types which would cause unintentional side effects, e.g. attempts at tying embeddings"
    )
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Theoretically works but kernel library causes issues.")
    def test_torchscript_output_hidden_state(self):
        pass

    @unittest.skip(reason="Theoretically works but kernel library causes issues.")
    def test_torchscript_simple(self):
        pass

    @unittest.skip(reason="NotImplementedError: Cannot copy out of meta tensor; no data!")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="NotImplementedError: Cannot copy out of meta tensor; no data!")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(reason="NotImplementedError: Cannot copy out of meta tensor; no data!")
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip(reason="HiggsAudio does not support left-padding")
    def test_left_padding_compatibility(self):
        pass


class HiggsAudioForConditionalGenerationIntegrationTest(unittest.TestCase):
    """
    See https://gist.github.com/szhengac/22060ca87e654d85886a6dec161fe01e for generating the integration tests

    NOTE: We add a single `eos` line for the last channel which is skipped in the original HiggsAudio
    (It doesn't change the behaviour as we cut by the eos token position)
    """

    def setUp(self):
        # it's a dummy ckpt but should suffice for testing purposes
        self.model_checkpoint = "szhengac25/higgs-audio-v2-generation-3B-base"
        self.sampling_rate = 24000

        # prepare audio
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        audio_sample = librispeech_dummy[-1]["audio"]["array"]
        # 10 and 5 codebooks as prefix - saved as files as we need wav files for the original HiggsAudio
        dac_chunk_len = 512
        self.audio_prompt_path = "/tmp/higgs_audio_test_sample.mp3"
        sf.write(
            self.audio_prompt_path,
            audio_sample[: (dac_chunk_len * 10)],
            self.sampling_rate,
        )

    def tearDown(self):
        pathlib.Path(self.audio_prompt_path).unlink()
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_torch_accelerator
    def test_higgs_audio_model_integration_generate_tts(self):
        conversation = [
            {
                "role": "system",
                "content": "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>",
            },
            {
                "role": "user",
                "content": "I hear that you can understand what people say and even know their age and gender, so can you guess my age and gender from my voice?",
            },
        ]
        processor = HiggsAudioProcessor.from_pretrained(self.model_checkpoint, torch_dtype="auto")
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, padding=True, return_tensors="pt").to(torch_device)

        model = HiggsAudioForConditionalGeneration.from_pretrained(self.model_checkpoint, torch_dtype="auto").to(
            torch_device
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            tokenizer=tokenizer,
            do_sample=False,
            ras_win_len=None,
        )

        # fmt: off
        EXPECTED_OUTPUT_TEXT_TOKENS = torch.tensor(
            [[128000, 128006, 9125, 128007, 271, 32215, 7855, 2768, 7754,
              382, 128018, 198, 15097, 374, 12715, 505, 264, 11594,
              3130, 627, 128019, 128009, 128006, 882, 128007, 271, 40,
              6865, 430, 499, 649, 3619, 1148, 1274, 2019, 323,
              1524, 1440, 872, 4325, 323, 10026, 11, 779, 649,
              499, 8101, 856, 4325, 323, 10026, 505, 856, 7899,
              30, 128009, 128006, 78191, 128007, 271, 128013, 128016]]
        )

        EXPECTED_OUTPUT_AUDIO_TOKENS = torch.tensor(
            [[1024, 244, 563, 949, 810, 675, 195, 813, 813, 988, 447, 178,
              785, 834, 287, 916, 259, 154, 758, 707, 524, 539, 532, 456,
              338, 338, 95, 219, 878, 908, 885],
             [1024, 1024, 717, 928, 385, 579, 467, 881, 156, 762, 846, 856,
              471, 803, 99, 918, 193, 729, 5, 112, 785, 358, 258, 629,
              522, 311, 854, 908, 385, 682, 329],
             [1024, 1024, 1024, 623, 207, 523, 329, 941, 931, 878, 728, 359,
              54, 18, 507, 798, 256, 991, 153, 563, 256, 613, 38, 991,
              604, 711, 585, 852, 685, 930, 965],
             [1024, 1024, 1024, 1024, 171, 975, 230, 8, 218, 857, 260, 483,
              562, 95, 305, 110, 736, 575, 800, 775, 601, 924, 32, 915,
              114, 639, 287, 997, 722, 965, 666],
             [1024, 1024, 1024, 1024, 1024, 609, 686, 315, 1017, 67, 174, 834,
              733, 905, 205, 242, 203, 29, 207, 314, 829, 265, 300, 367,
              448, 568, 747, 227, 348, 587, 502],
             [1024, 1024, 1024, 1024, 1024, 1024, 318, 171, 102, 574, 849, 848,
              593, 49, 512, 976, 823, 507, 158, 919, 260, 359, 794, 971,
              684, 574, 913, 177, 851, 980, 809],
             [1024, 1024, 1024, 1024, 1024, 1024, 1024, 579, 256, 382, 613, 801,
              104, 265, 20, 720, 982, 250, 416, 838, 712, 153, 13, 297,
              79, 679, 57, 386, 693, 297, 344],
             [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 165, 939, 210, 865,
              593, 533, 431, 524, 856, 484, 341, 991, 657, 830, 344, 416,
              888, 379, 1, 100, 787, 593, 41]]
        )
        # fmt: on

        torch.testing.assert_close(outputs[0].cpu(), EXPECTED_OUTPUT_TEXT_TOKENS)
        torch.testing.assert_close(outputs[1][0].cpu(), EXPECTED_OUTPUT_AUDIO_TOKENS)

    @slow
    @require_torch_accelerator
    def test_higgs_audio_model_integration_generate_audio_context(self):
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Generate audio following instruction with the same voice.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>",
                    },
                    {"type": "audio"},
                ],
            },
            {
                "role": "user",
                "content": "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.",
            },
        ]
        audio_sample = (
            torchaudio.load(self.audio_prompt_path, channels_first=True, backend="soundfile")[0].squeeze().numpy()
        )
        audio = [audio_sample]
        processor = HiggsAudioProcessor.from_pretrained(self.model_checkpoint, torch_dtype="auto")
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audio=audio, padding=True, return_tensors="pt").to(torch_device)

        model = HiggsAudioForConditionalGeneration.from_pretrained(self.model_checkpoint, torch_dtype="auto").to(
            torch_device
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            tokenizer=tokenizer,
            do_sample=False,
            ras_win_len=None,
        )

        # fmt: off
        EXPECTED_OUTPUT_TEXT_TOKENS = torch.tensor(
            [[128000, 128006, 9125, 128007, 271, 32215, 7855, 2768, 7754,
              449, 279, 1890, 7899, 382, 128018, 198, 15097, 374,
              12715, 505, 264, 11594, 3130, 627, 128019, 128011, 128015,
              128012, 128009, 128006, 882, 128007, 271, 791, 7160, 38268,
              304, 279, 11226, 323, 7437, 304, 279, 9909, 13,
              1115, 4382, 2144, 706, 1027, 13468, 555, 12966, 369,
              9214, 315, 1667, 13, 128009, 128006, 78191, 128007, 271,
              128013, 128016]]
        )

        EXPECTED_OUTPUT_AUDIO_TOKENS = torch.tensor(
            [[1024, 244, 578, 609, 950, 196, 937, 599, 603, 486, 885, 705,
              988, 934, 460, 460, 291, 99, 701, 99, 949, 343, 809, 949,
              610, 287, 330, 796, 186, 287, 287],
             [1024, 1024, 537, 710, 972, 863, 394, 464, 89, 103, 28, 543,
              618, 618, 276, 89, 695, 705, 66, 750, 504, 315, 397, 602,
              25, 358, 26, 675, 526, 533, 353],
             [1024, 1024, 1024, 998, 136, 771, 654, 358, 163, 115, 989, 111,
              406, 111, 135, 564, 760, 608, 1002, 871, 207, 604, 7, 787,
              62, 684, 684, 684, 657, 636, 153],
             [1024, 1024, 1024, 1024, 485, 310, 332, 675, 96, 116, 980, 338,
              495, 852, 322, 960, 160, 722, 258, 354, 321, 666, 878, 465,
              394, 322, 1021, 530, 948, 531, 70],
             [1024, 1024, 1024, 1024, 1024, 799, 406, 335, 748, 609, 517, 754,
              870, 849, 12, 845, 223, 375, 82, 370, 900, 486, 748, 253,
              152, 461, 574, 549, 549, 253, 671],
             [1024, 1024, 1024, 1024, 1024, 1024, 342, 826, 760, 318, 907, 139,
              171, 528, 590, 603, 758, 1005, 760, 477, 679, 316, 235, 77,
              318, 64, 739, 830, 848, 258, 565],
             [1024, 1024, 1024, 1024, 1024, 1024, 1024, 53, 53, 927, 296, 798,
              106, 359, 733, 946, 565, 352, 463, 668, 790, 603, 414, 868,
              975, 347, 170, 255, 385, 578, 359],
             [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 669, 228, 1, 544,
              518, 620, 10, 842, 114, 487, 579, 648, 666, 512, 10, 423,
              900, 617, 64, 45, 356, 676, 166]]
        )
        # fmt: on

        torch.testing.assert_close(outputs[0].cpu(), EXPECTED_OUTPUT_TEXT_TOKENS)
        torch.testing.assert_close(outputs[1][0].cpu(), EXPECTED_OUTPUT_AUDIO_TOKENS)
