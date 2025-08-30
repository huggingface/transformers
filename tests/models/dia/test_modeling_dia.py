# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Dia model."""

import copy
import pathlib
import tempfile
import unittest

import pytest

from transformers.models.dia import DiaConfig, DiaDecoderConfig, DiaEncoderConfig
from transformers.testing_utils import (
    cleanup,
    is_flaky,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)
from transformers.utils import is_soundfile_available, is_torch_available, is_torchaudio_available
from transformers.utils.import_utils import is_datasets_available

from ...generation.test_utils import GenerationTesterMixin, has_similar_generate_outputs
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_datasets_available():
    from datasets import Audio, load_dataset

if is_torch_available():
    import torch

    from transformers import (
        DiaForConditionalGeneration,
        DiaModel,
        DiaProcessor,
        PretrainedConfig,
        PreTrainedModel,
    )
    from transformers.cache_utils import (
        Cache,
        StaticCache,
    )
    from transformers.models.dia.modeling_dia import DiaDecoder, DiaEncoder

if is_torchaudio_available():
    import torchaudio

if is_soundfile_available():
    import soundfile as sf


@require_torch
class DiaModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,  # need batch_size != num_hidden_layers
        seq_length=7,
        max_length=50,
        is_training=True,
        vocab_size=100,
        hidden_size=16,
        intermediate_size=37,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=8,
        decoder_hidden_size=32,  # typically larger than encoder
        hidden_act="silu",
        eos_token_id=97,  # special tokens all occur after eos
        pad_token_id=98,
        bos_token_id=99,
        delay_pattern=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.max_length = max_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.decoder_hidden_size = decoder_hidden_size
        self.hidden_act = hidden_act
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        # Set default delay pattern if not provided
        self.delay_pattern = delay_pattern if delay_pattern is not None else [0, 1, 2]
        self.num_channels = len(self.delay_pattern)

    def get_config(self):
        encoder_config = DiaEncoderConfig(
            max_position_embeddings=self.max_length,
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_attention_heads,  # same as num_attention_heads for testing
            head_dim=self.head_dim,
            intermediate_size=self.intermediate_size,
            vocab_size=self.vocab_size,
            hidden_act=self.hidden_act,
        )

        decoder_config = DiaDecoderConfig(
            max_position_embeddings=self.max_length,
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.decoder_hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=1,  # GQA
            head_dim=self.head_dim,
            cross_num_attention_heads=self.num_attention_heads,
            cross_head_dim=self.head_dim,
            cross_num_key_value_heads=1,  # GQA
            cross_hidden_size=self.hidden_size,  # match encoder hidden size
            vocab_size=self.vocab_size,
            hidden_act=self.hidden_act,
            num_channels=self.num_channels,
        )

        config = DiaConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            delay_pattern=self.delay_pattern,
        )

        return config

    def prepare_config_and_inputs(self) -> tuple[DiaConfig, dict]:
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = input_ids.ne(self.pad_token_id)

        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length, self.num_channels], self.vocab_size)
        decoder_attention_mask = decoder_input_ids[..., 0].ne(self.pad_token_id)

        config = self.get_config()
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self) -> tuple[DiaConfig, dict]:
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_model_forward(self, config, inputs_dict):
        model = DiaModel(config=config).to(torch_device).eval()

        input_ids = inputs_dict["input_ids"]
        decoder_input_ids = inputs_dict["decoder_input_ids"]

        # first forward pass
        last_hidden_state = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).last_hidden_state

        self.parent.assertTrue(
            last_hidden_state.shape, (self.batch_size, self.seq_length, config.decoder_config.hidden_size)
        )

    def check_encoder_decoder_model_standalone(self, config, inputs_dict):
        model = DiaModel(config=config).to(torch_device).eval()
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = DiaEncoder.from_pretrained(tmpdirname).to(torch_device)

        encoder_last_hidden_state_2 = encoder(
            input_ids=inputs_dict["input_ids"], attention_mask=inputs_dict["attention_mask"]
        )[0]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 3e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = DiaDecoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            input_ids=inputs_dict["decoder_input_ids"],
            attention_mask=inputs_dict["decoder_attention_mask"],
            encoder_hidden_states=encoder_last_hidden_state,
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 3e-3)


@require_torch
class DiaModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (DiaModel, DiaForConditionalGeneration) if is_torch_available() else ()
    # We only allow greedy search / sampling with one sequence; see `skip_non_greedy_generate`
    all_generative_model_classes = (DiaForConditionalGeneration,)
    # TODO: support new pipeline behavior in tests
    pipeline_model_mapping = {}
    # pipeline_model_mapping = {"text-to-audio": DiaForConditionalGeneration} if is_torch_available() else {}
    test_pruning = False
    test_head_masking = False
    test_resize_embeddings = False
    is_encoder_decoder = True
    # Indicates VLMs usually but there are many audio models which are also composite
    _is_composite = True

    def setUp(self):
        self.model_tester = DiaModelTester(self)
        # Skipping `has_text_modality` but manually testing down below
        self.config_tester = ConfigTester(self, has_text_modality=False, config_class=DiaConfig)
        self.skip_non_greedy_generate()

    def skip_non_greedy_generate(self):
        skippable_tests = [
            "test_sample_generate_dict_output",  # return sequences > 1
            "test_beam",
            "test_contrastive",
            "test_assisted",
            "test_prompt_lookup",
            "test_model_parallel_beam_search",
            "test_generate_without_input_ids",
            "test_generate_with_head_masking",
        ]

        for test in skippable_tests:
            if self._testMethodName.startswith(test):
                self.skipTest(reason="Dia only supports greedy search / sampling with one sequence.")

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        """Overriden to account for the 2D flattened structure"""
        inputs_dict = copy.deepcopy(inputs_dict)

        if return_labels:
            inputs_dict["labels"] = torch.ones(
                (
                    self.model_tester.batch_size * self.model_tester.num_channels,
                    self.model_tester.seq_length,
                ),
                dtype=torch.long,
                device=torch_device,
            )

        return inputs_dict

    def test_config(self):
        self.config_tester.run_common_tests()

        # Manual testing because of composite configs
        config = self.model_tester.prepare_config_and_inputs()[0]
        self.assertTrue(hasattr(config.encoder_config, "vocab_size"), msg="Encoder `vocab_size` does not exist")
        self.assertTrue(hasattr(config.decoder_config, "vocab_size"), msg="Decoder `vocab_size` does not exist")

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    @is_flaky
    def test_encoder_decoder_model_standalone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.check_encoder_decoder_model_standalone(*config_and_inputs)

    # Overriding shape checks as Dia has different shapes on encoder/decoder using a composite config
    # + additional special cases where 3D x 2D meshes confuse the expected shape
    def _check_logits(self, batch_size, logits, config):
        batch_size *= len(config.delay_pattern)  # Account for flattening
        vocab_size = config.decoder_config.vocab_size
        self.assertIsInstance(logits, tuple)
        self.assertListEqual([iter_logits.shape[0] for iter_logits in logits], [batch_size] * len(logits))
        # vocabulary difference equal to one (imagegptmodel?) or zero (all other models)
        vocab_diff = vocab_size - logits[0].shape[-1]
        self.assertTrue(vocab_diff in [0, 1])
        self.assertListEqual([vocab_size - score.shape[-1] for score in logits], [vocab_diff] * len(logits))

    def _check_attentions_for_generate(
        self, batch_size, attentions, prompt_length, output_length, config, decoder_past_key_values
    ):
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [isinstance(iter_attentions, tuple) for iter_attentions in attentions], [True] * len(attentions)
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
            query_length = (
                prompt_length + generated_length
                if not has_static_cache
                else decoder_past_key_values.get_max_cache_shape()
            )

            expected_shape = (
                batch_size,
                config.decoder_config.num_attention_heads,  # Decoder config
                model_input_length,
                query_length,
            )
            # check attn size
            self.assertListEqual(
                [layer_attention.shape for layer_attention in iter_attentions], [expected_shape] * len(iter_attentions)
            )

    def _check_encoder_attention_for_generate(self, attentions, batch_size, config, prompt_length):
        # Encoder config
        encoder_expected_shape = (batch_size, config.encoder_config.num_attention_heads, prompt_length, prompt_length)
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [layer_attentions.shape for layer_attentions in attentions],
            [encoder_expected_shape] * len(attentions),
        )

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, prompt_length, output_length, config, use_cache=False
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

            # check hidden size
            # we can have different hidden sizes between encoder and decoder --> check both
            expected_shape_encoder = (batch_size, model_input_length, config.encoder_config.hidden_size)
            expected_shape_decoder = (batch_size, model_input_length, config.decoder_config.hidden_size)
            self.assertTrue(
                [layer_hidden_states.shape for layer_hidden_states in iter_hidden_states]
                == [expected_shape_encoder] * len(iter_hidden_states)
                or [layer_hidden_states.shape for layer_hidden_states in iter_hidden_states]
                == [expected_shape_decoder] * len(iter_hidden_states)
            )

    def _check_encoder_hidden_states_for_generate(self, hidden_states, batch_size, config, prompt_length):
        # Encoder config
        encoder_expected_shape = (batch_size, prompt_length, config.encoder_config.hidden_size)
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [layer_hidden_states.shape for layer_hidden_states in hidden_states],
            [encoder_expected_shape] * len(hidden_states),
        )

    def _check_past_key_values_for_generate(self, batch_size, decoder_past_key_values, cache_length, config):
        self.assertIsInstance(decoder_past_key_values, (tuple, Cache))

        # we need the decoder config here
        config = config.decoder_config

        # (batch, head, seq_length, head_features)
        expected_shape = (
            batch_size,
            config.num_key_value_heads if hasattr(config, "num_key_value_heads") else config.num_attention_heads,
            cache_length,
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads,
        )

        if isinstance(decoder_past_key_values, Cache):
            self.assertListEqual(
                [layer.keys.shape for layer in decoder_past_key_values.layers],
                [expected_shape] * len(decoder_past_key_values.layers),
            )
            self.assertListEqual(
                [layer.values.shape for layer in decoder_past_key_values.layers],
                [expected_shape] * len(decoder_past_key_values.layers),
            )

    def _check_scores(self, batch_size, scores, generated_length, config):
        # Special case where Dia keeps score in a 2D mesh of (bsz * channels, vocab)
        vocab_size = config.decoder_config.vocab_size
        expected_shape = (batch_size * len(config.delay_pattern), vocab_size)
        self.assertIsInstance(scores, tuple)
        self.assertEqual(len(scores), generated_length)
        self.assertListEqual([iter_scores.shape for iter_scores in scores], [expected_shape] * len(scores))

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

    @pytest.mark.generate
    @unittest.skip(reason="Custom processor `DiaEOSDelayPatternLogitsProcessor` forces eos token.")
    def test_generate_continue_from_past_key_values(self):
        """Only a small change due to the expected shapes"""
        # Tests that we can continue generating from past key values, returned from a previous `generate` call
        for model_class in self.all_generative_model_classes:
            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

            # Let's make it always:
            # 1. use cache (for obvious reasons)
            # 2. generate to max length (which can be achieved by setting the eos token to an invalid value), which
            #    would make the test flaky (e.g. EOS is generated on iteration 1 on both generations, but the
            #    continuation would force it to generate beyond an EOS token)
            # 3. ignore `token_type_ids` for simplicity
            # 4. ignore `forced_eos_token_id`, which requires further manipulation of the continuation inputs and is
            #    active by default on some models
            # 5. ignore `encoder_no_repeat_ngram_size`, which is set by default in some encoder-decoder models. When
            #    we use their decoder as a stand-alone model, `encoder_no_repeat_ngram_size` actually prevents
            #    repetition exclusively from the prompt. This test relies on comparing one call vs 2 calls
            #    with cache, what is considered a prompt is different in the two cases.

            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]

            model = model_class(config).to(torch_device)
            model.eval()

            generate_kwargs = {
                "pad_token_id": -1,
                "eos_token_id": -1,
                "forced_eos_token_id": None,
                "encoder_no_repeat_ngram_size": 0,
                "use_cache": True,
                "do_sample": False,
                "return_dict_in_generate": True,
                "output_scores": True,
            }

            # Traditional way of generating text, with `return_dict_in_generate` to return the past key values
            outputs = model.generate(**inputs, **generate_kwargs, max_new_tokens=4)

            # Let's generate again, but passing the past key values in between (3 + 1 = 4 tokens). Note that the
            # inputs may need to be tweaked across `generate` calls (like the attention mask).
            outputs_cached = model.generate(**inputs, **generate_kwargs, max_new_tokens=3)

            # Continue from the tokens generated above, preparing the inputs accordingly
            inputs["past_key_values"] = outputs_cached.past_key_values
            new_attention_len = outputs_cached.sequences.shape[1]  # the only real modification in this test
            inputs["decoder_input_ids"] = outputs_cached.sequences
            if "decoder_attention_mask" in inputs:
                inputs["decoder_attention_mask"] = torch.nn.functional.pad(
                    inputs["decoder_attention_mask"],
                    (0, new_attention_len - inputs["decoder_attention_mask"].shape[1]),
                    mode="constant",
                    value=1,
                )

            first_caches_scores = outputs_cached.scores
            outputs_cached = model.generate(**inputs, **generate_kwargs, max_new_tokens=1)
            full_cached_scores = first_caches_scores + outputs_cached.scores
            outputs_cached.scores = full_cached_scores

            # The two sets of generated text and past kv should be equal to each other
            self.assertTrue(has_similar_generate_outputs(outputs, outputs_cached))
            for layer_idx in range(len(outputs_cached.past_key_values)):
                for kv_idx in range(len(outputs_cached.past_key_values[layer_idx])):
                    self.assertTrue(
                        torch.allclose(
                            outputs.past_key_values[layer_idx][kv_idx],
                            outputs_cached.past_key_values[layer_idx][kv_idx],
                        )
                    )

    @unittest.skip(reason="Indirectly checked in Dia through the generate methods.")
    def test_past_key_values_format(self, custom_all_cache_shapes=None):
        pass

    @unittest.skip(reason="Indirectly checked in Dia through the generate methods.")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(
        reason="Dia has too many mixed embedding types which would cause unintentional side effects, e.g. attempts at tying embeddings"
    )
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Theoretically works but kernel library causes issues.")
    def test_torchscript_output_hidden_state(self):
        pass

    @unittest.skip(reason="Theoretically works but kernel library causes issues.")
    def test_torchscript_simple(self):
        pass

    @unittest.skip(reason="Encoder-Decoder cache can not be initialized.")
    def test_multi_gpu_data_parallel_forward(self):
        pass


class DiaForConditionalGenerationIntegrationTest(unittest.TestCase):
    """
    See https://gist.github.com/vasqu/0e3b06360373a4e612aa3b9a7c09185e for generating the integration tests

    NOTE: We add a single `eos` line for the last channel which is skipped in the original Dia
    (It doesn't change the behaviour as we cut by the eos token position)
    """

    def setUp(self):
        # it's a dummy ckpt but should suffice for testing purposes
        self.model_checkpoint = "AntonV/Dia-1.6B"
        self.sampling_rate = 44100

        # prepare audio
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        audio_sample_1 = librispeech_dummy[-1]["audio"]["array"]
        audio_sample_2 = librispeech_dummy[-2]["audio"]["array"]
        # 10 and 5 codebooks as prefix - saved as files as we need wav files for the original Dia
        dac_chunk_len = 512
        self.audio_prompt_1_path = "/tmp/dia_test_sample_1.mp3"
        self.audio_prompt_2_path = "/tmp/dia_test_sample_2.mp3"
        sf.write(self.audio_prompt_1_path, audio_sample_1[: (dac_chunk_len * 10)], self.sampling_rate)
        sf.write(self.audio_prompt_2_path, audio_sample_2[: (dac_chunk_len * 5)], self.sampling_rate)

    def tearDown(self):
        pathlib.Path(self.audio_prompt_1_path).unlink()
        pathlib.Path(self.audio_prompt_2_path).unlink()
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_torch_accelerator
    def test_dia_model_integration_generate_tts(self):
        text = ["[S1] Dia is an open weights text to dialogue model.", "This is a test"]
        processor = DiaProcessor.from_pretrained(self.model_checkpoint)
        inputs = processor(text=text, padding=True, return_tensors="pt").to(torch_device)

        model = DiaForConditionalGeneration.from_pretrained(self.model_checkpoint).to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)

        # fmt: off
        EXPECTED_OUTPUT_TOKENS = torch.tensor([[[1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 568, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 568, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 568, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 568, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 568, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 568, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 568, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 568, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 568,  778, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 568,  778,  338, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 568,  804,   10,  524, 1026, 1026, 1026, 1026, 1026],
         [ 568,  804,   10,  674,  967, 1026, 1026, 1026, 1026],
         [ 568,  804,   10,  674,  364,  360, 1026, 1026, 1026],
         [ 568,  804,   10,  674,  364,  981,  728, 1026, 1026],
         [ 568,  804,   10,  674,  364,  981,  741,  550, 1026],
         [ 568,  804,   10,  674,  364,  981,  568,  378,   90],
         [1024,  804,   10,  674,  364,  981,  568,  378,  731],
         [1025,  804,   10,  674,  364,  981,  568,  378,  731],
         [1025,  804,   10,  674,  364,  981,  568,  378,  731],
         [1025,  804,   10,  674,  364,  981,  568,  378,  731],
         [1025,  804,   10,  674,  364,  981,  568,  378,  731],
         [1025,  804,   10,  674,  364,  981,  568,  378,  731],
         [1025,  804,   10,  674,  364,  981,  568,  378,  731],
         [1025,  804,   10,  674,  364,  981,  568,  378,  731],
         [1025, 1024,   10,  674,  364,  981,  568,  378,  731],
         [1025, 1025, 1024,  674,  364,  981,  568,  378,  731],
         [1025, 1025, 1025, 1024,  364,  981,  568,  378,  731],
         [1025, 1025, 1025, 1025, 1024,  981,  568,  378,  731],
         [1025, 1025, 1025, 1025, 1025, 1024,  568,  378,  731],
         [1025, 1025, 1025, 1025, 1025, 1025, 1024,  378,  731],
         [1025, 1025, 1025, 1025, 1025, 1025, 1025, 1024,  731],
         [1025, 1025, 1025, 1025, 1025, 1025, 1025, 1025, 1024]],

        [[1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 568, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 568, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 698, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 592, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 592, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 592, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 592, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 592, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 592,  778, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 592,  778,  338, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 592,  697,   10,  524, 1026, 1026, 1026, 1026, 1026],
         [ 592,  288,  476,  649,  967, 1026, 1026, 1026, 1026],
         [ 592,  740,  386,  674,  364,  360, 1026, 1026, 1026],
         [ 592,  402,  386,  347,  362,  981,  728, 1026, 1026],
         [ 592,  402,  721,  728,  327,  981,  741,  550, 1026],
         [ 592,  402,  721,  728,  460,   62,  676,  378,   90],
         [1024,  402,  721,  728,  837,  595,  195,  982,  784],
         [1025,  402,  721,  677,  497,  102,  692,   24,  330],
         [1025,  402,  721,  677,  511,  102,  503,  871,  609],
         [1025,  402,  721,  677,  511,   96,  801,  871,  894],
         [1025,  402,  721,  677,  511,  745,  314,  498,  775],
         [1025,  402,  721,  677,  511,  745,  314,  498,  105],
         [1025,  402,  721,  677,  511,  745,  314,  861,  889],
         [1025,  893,  721,  677,  511,  744,  314,  871,  353],
         [1025, 1024,  888,  677,  511,  744,  314,  871,  332],
         [1025, 1025, 1024,  518,  511,  744,  314,  871,  366],
         [1025, 1025, 1025, 1024,  611,  744,  314,  871,  366],
         [1025, 1025, 1025, 1025, 1024,  980,  314,  871,  366],
         [1025, 1025, 1025, 1025, 1025, 1024,   45,  124,  366],
         [1025, 1025, 1025, 1025, 1025, 1025, 1024,  871,  366],
         [1025, 1025, 1025, 1025, 1025, 1025, 1025, 1024,  719],
         [1025, 1025, 1025, 1025, 1025, 1025, 1025, 1025, 1024]]])
        # fmt: on

        torch.testing.assert_close(outputs.cpu(), EXPECTED_OUTPUT_TOKENS)

    @slow
    @require_torch_accelerator
    def test_dia_model_integration_generate_audio_context(self):
        text = ["[S1] Dia is an open weights text to dialogue model.", "This is a test"]
        audio_sample_1 = (
            torchaudio.load(self.audio_prompt_1_path, channels_first=True, backend="soundfile")[0].squeeze().numpy()
        )
        audio_sample_2 = (
            torchaudio.load(self.audio_prompt_2_path, channels_first=True, backend="soundfile")[0].squeeze().numpy()
        )
        audio = [audio_sample_1, audio_sample_2]

        processor = DiaProcessor.from_pretrained(self.model_checkpoint)
        inputs = processor(text=text, audio=audio, padding=True, return_tensors="pt").to(torch_device)

        model = DiaForConditionalGeneration.from_pretrained(self.model_checkpoint).to(torch_device)
        # dia has right padding while we have left padding (for faster prefill)
        # additionally we have new tokens vs dia's max tokens (hence we compare each in the respective settings)
        outputs_1 = model.generate(**inputs, max_new_tokens=22, do_sample=False)
        outputs_2 = model.generate(**inputs, max_new_tokens=27, do_sample=False)

        # fmt: off
        EXPECTED_OUTPUT_TOKENS_1 = torch.tensor([[1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 578, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 592, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 494, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 330, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 330, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 330, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 330, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 330, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 330,  501, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 330,  204,   34, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 330,  254,  915,  863, 1026, 1026, 1026, 1026, 1026],
         [ 330,  215,  458,  313,   50, 1026, 1026, 1026, 1026],
         [ 330,  615,  529,  216,  801,  237, 1026, 1026, 1026],
         [ 330,  580,  563,  233,  337,   37, 1018, 1026, 1026],
         [ 330,  567,  530,  753,  607,  179,  954,  242, 1026],
         [ 330,  627,    6, 1010,  500,  189,  598,  858,  247],
         [1024,  432,  480,  530,  122,    3,  788,  149,  814],
         [1025,  875,  826,  458,   98,  540,  181,  122,  608],
         [1025,  495,  840,  413,  337,  784,  591,  150, 1017],
         [1025,  808,  189,  137,  445,    0,  227,  658,  345],
         [1025,  397,   89,  753, 1016,  173,  984,    0,  910],
         [1025,  875,  460,  934,   50,  335,  670,  818,  722],
         [1025,  875,  460,  762,  119,  372,  503,  858,  584],
         [1025,  348,  555,  475,  469,  458,  963,   41,  664],
         [1025, 1024,  852,  683,  761,  193,  595,  895,  885],
         [1025, 1025, 1024,  135,  761,  902,  163,  623,  385],
         [1025, 1025, 1025, 1024,  852,  282,  581,  623,   70],
         [1025, 1025, 1025, 1025, 1024,   41,  661,  790,  977],
         [1025, 1025, 1025, 1025, 1025, 1024,  580,  401,  464],
         [1025, 1025, 1025, 1025, 1025, 1025, 1024,  756,   61],
         [1025, 1025, 1025, 1025, 1025, 1025, 1025, 1024,  752],
         [1025, 1025, 1025, 1025, 1025, 1025, 1025, 1025, 1024]])

        EXPECTED_OUTPUT_TOKENS_2 = torch.tensor([[1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 619, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 315, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 315, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 315, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 315, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 315, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 315, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 315, 1026, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 315,  968, 1026, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 315, 1007,  458, 1026, 1026, 1026, 1026, 1026, 1026],
         [ 315,   35,  266,   68, 1026, 1026, 1026, 1026, 1026],
         [ 315,  359,  285,  811,  154, 1026, 1026, 1026, 1026],
         [ 315,  906,  407,  297,  785,  649, 1026, 1026, 1026],
         [ 315,  249,  678,  868,  899,  257,  950, 1026, 1026],
         [ 315,  249,  217,  471,  292,  908,  196,  469, 1026],
         [ 315,  249,  825,  771,  839,  802,  633,  590,  531],
         [1024,  249,  150,   53,  126,   76,  794,  626,  442],
         [1025,  249,  825,  218,  359,  864,  526,  626,  770],
         [1025,  249,  150,  137,  530,  845,  877,  600,  111],
         [1025,  249,  150,  287,  730,  991,  135,  259,   39],
         [1025,  249,  825,  104,  198, 1020,  719,  625,  208],
         [1025,  249,  825,  997,  602,  256,  859,  322,  518],
         [1025,  668,  825,  979,  584,  256,   98,  665,  589],
         [1025,  954,  458,   54,  206,   52,  244,  822,  599],
         [1025, 1024,  104,  914,  435,  579,  860,   92,  661],
         [1025, 1025, 1024,  848,  126,   74,  304,   92,  753],
         [1025, 1025, 1025, 1024,  362,  376,  304,  586,  753],
         [1025, 1025, 1025, 1025, 1024,  633,  996,  586,   83],
         [1025, 1025, 1025, 1025, 1025, 1024,  179,  898,  928],
         [1025, 1025, 1025, 1025, 1025, 1025, 1024,  506,  102],
         [1025, 1025, 1025, 1025, 1025, 1025, 1025, 1024,   79],
         [1025, 1025, 1025, 1025, 1025, 1025, 1025, 1025, 1024]])
        # fmt: on

        torch.testing.assert_close(outputs_1[0].cpu(), EXPECTED_OUTPUT_TOKENS_1)
        torch.testing.assert_close(outputs_2[1, 5:].cpu(), EXPECTED_OUTPUT_TOKENS_2)  # left padding
