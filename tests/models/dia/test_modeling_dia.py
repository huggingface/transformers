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
import tempfile
import unittest
from typing import Tuple

from transformers.models.dia import DiaConfig, DiaDecoderConfig, DiaEncoderConfig
from transformers.testing_utils import (
    require_torch,
    require_torch_sdpa,
    torch_device,
)
from transformers.utils import is_torch_available, is_torchaudio_available
from transformers.utils.import_utils import is_datasets_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_datasets_available():
    pass

if is_torch_available():
    import torch

    from transformers import (
        DiaForConditionalGeneration,
        DiaModel,
        PretrainedConfig,
        PreTrainedModel,
    )
    from transformers.cache_utils import (
        Cache,
        StaticCache,
    )
    from transformers.models.dia.modeling_dia import DiaDecoder, DiaEncoder


if is_torchaudio_available():
    pass


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

    def prepare_config_and_inputs(self) -> Tuple[DiaConfig, dict]:
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

    def prepare_config_and_inputs_for_common(self) -> Tuple[DiaConfig, dict]:
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

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 2e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = DiaDecoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            input_ids=inputs_dict["decoder_input_ids"],
            attention_mask=inputs_dict["decoder_attention_mask"],
            encoder_hidden_states=encoder_last_hidden_state,
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 2e-3)


@require_torch
class DiaModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (DiaModel, DiaForConditionalGeneration) if is_torch_available() else ()
    # We only allow greedy search / sampling with one sequence; see `skip_non_greedy_generate`
    all_generative_model_classes = (DiaForConditionalGeneration,)
    # TODO: needs processor for pipeline / do we allow pipeline here (see csm?)
    # pipeline_model_mapping = {"text-to-audio": DiaForConditionalGeneration} if is_torch_available() else {}
    pipeline_model_mapping = {}
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
            "test_group_beam",
            "test_constrained_beam",
            "test_contrastive",
            "test_assisted",
            "test_dola",
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
                [key_tensor.shape for key_tensor in decoder_past_key_values.key_cache],
                [expected_shape] * len(decoder_past_key_values.key_cache),
            )
            self.assertListEqual(
                [value_tensor.shape for value_tensor in decoder_past_key_values.value_cache],
                [expected_shape] * len(decoder_past_key_values.value_cache),
            )

    def _check_scores(self, batch_size, scores, generated_length, config):
        # Special case where Dia keeps score in a 2D mesh of (bsz * channels, vocab)
        vocab_size = config.decoder_config.vocab_size
        expected_shape = (batch_size * len(config.delay_pattern), vocab_size)
        self.assertIsInstance(scores, tuple)
        self.assertEqual(len(scores), generated_length)
        self.assertListEqual([iter_scores.shape for iter_scores in scores], [expected_shape] * len(scores))

    @require_torch_sdpa
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

    @unittest.skip(reason="Decoder preparation in Dia is currently not designed around cache continuation.")
    def test_generate_continue_from_past_key_values(self):
        pass

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


# TODO: integration tests
