# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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
import os
import unittest

import pytest
from packaging import version

from transformers import AutoTokenizer, ModernBertConfig, is_torch_available
from transformers.models.auto import get_values
from transformers.testing_utils import (
    CaptureLogger,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        MODEL_FOR_PRETRAINING_MAPPING,
        ModernBertForCausalLM,
        ModernBertForMaskedLM,
        ModernBertForSequenceClassification,
        ModernBertForTokenClassification,
        ModernBertModel,
        logging,
    )


class ModernBertModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        pad_token_id=0,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_activation="gelu",
        mlp_dropout=0.0,
        attention_dropout=0.0,
        embedding_dropout=0.0,
        classifier_dropout=0.0,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        bos_token_id=1,
        eos_token_id=2,
        mask_token_id=3,
        sep_token_id=4,
        is_causal=False,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_activation = hidden_activation
        self.mlp_dropout = mlp_dropout
        self.attention_dropout = attention_dropout
        self.embedding_dropout = embedding_dropout
        self.classifier_dropout = classifier_dropout
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id
        self.sep_token_id = sep_token_id
        self.is_causal = is_causal
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        """
        Returns a tiny configuration by default.
        """
        config = ModernBertConfig(
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_activation=self.hidden_activation,
            mlp_dropout=self.mlp_dropout,
            attention_dropout=self.attention_dropout,
            embedding_dropout=self.embedding_dropout,
            classifier_dropout=self.classifier_dropout,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            mask_token_id=self.mask_token_id,
            sep_token_id=self.sep_token_id,
        )
        if test := os.environ.get("PYTEST_CURRENT_TEST", False):
            test_name = test.split(":")[-1].split(" ")[0]

            # tests from the generation mixin that use CausalLM
            # we should default to the causal lm approach bc
            # the pesudo-causal approach only works with greedy search
            # and requires adding special tokens
            generation_tests = [
                "test_batching_equivalence",
                "test_greedy_generate",
                "test_greedy_generate_dict_outputs",
                "test_greedy_generate_dict_outputs_use_cache",
                "test_sample_generate",
                "test_sample_generate_dict_output",
                "test_beam_search_generate",
                "test_beam_search_generate_dict_output",
                "test_beam_search_generate_dict_outputs_use_cache",
                "test_constrained_beam_search_generate_dict_output",
                "test_beam_sample_generate",
                "test_group_beam_search_generate",
                "test_constrained_beam_search_generate",
                "test_contrastive_generate",
                "test_contrastive_generate_dict_outputs_use_cache",
                "test_contrastive_generate_low_memory",
                "test_generate_with_head_masking",
                "test_generate_with_static_cache",
                "test_generate_with_static_cache_multi_gpu",
                "test_init_static_cache_multi_gpu",
                "test_left_padding_compatibility",
                "test_model_parallel_beam_search",
                "test_past_key_values_format",
                "test_generate_from_inputs_embeds",
                "test_prompt_lookup_decoding_matches_greedy_search",
                "test_prompt_lookup_decoding_stops_at_eos",
                "test_assisted_decoding_matches_greedy_search_0_random",
                "test_assisted_decoding_matches_greedy_search_1_same",
                "test_assisted_decoding_sample",
                "test_dola_decoding_sample",
                "test_model_kwarg_assisted_decoding_decoder_only",
                "test_model_kwarg_assisted_decoding_encoder_decoder",
                "test_assisted_decoding_encoder_decoder_shared_encoder",
                "test_assisted_decoding_in_different_gpu",
                "test_assisted_decoding_model_in_gpu_assistant_in_cpu",
                "test_attention_outputs",
                "test_beam_sample_generate_dict_output",
                "test_beam_search_low_memory",
                "test_cpu_offload",
                "test_determinism",
                "test_disk_offload_bin",
                "test_disk_offload_safetensors",
                "test_dola_decoding_sample",
                "test_for_causal_lm",
                "test_inputs_embeds_matches_input_ids",
                "test_retain_grad_hidden_states_attentions",
                "test_eager_matches_sdpa_inference_0_float16",
                "test_eager_matches_sdpa_inference_1_bfloat16",
                "test_eager_matches_sdpa_inference_2_float32",
                "test_feed_forward_chunking",
                "test_group_beam_search_generate_dict_output",
                "test_generate_from_inputs_embeds_0_greedy",
                "test_generate_from_inputs_embeds_1_beam_search",
                "test_assisted_decoding_matches_greedy_search_1",
                "test_generate_from_inputs_embeds_2_group_beam_search",
                "test_generate_from_inputs_embeds_3_group_beam_search_dict_output",
                "test_generate_from_inputs_embeds_4_group_beam_search_dict_output_use_cache",
                "test_generate_from_inputs_embeds_5_group_beam_search_dict_output_use_cache",
                "test_generate_continue_from_past_key_values",
                "test_generate_without_input_ids",
                "test_save_load",
                "test_hidden_states_output",
                "test_model_outputs_equivalence",
                "test_resize_embeddings_untied",
                "test_resize_tokens_embeddings",
                "test_inputs_embeds",
                # should not try training pseudo-causal as it's not supported
                "test_training",
                "test_training_gradient_checkpointing",
                "test_training_gradient_checkpointing_use_reentrant",
                "test_training_gradient_checkpointing_use_reentrant_false",
            ]

            if test_name in generation_tests:
                config.is_causal = True

            # If we're testing `test_retain_grad_hidden_states_attentions`, we normally get an error
            # that compilation doesn't work. Users can then set compile=False when loading the model,
            # much like here. We're testing whether it works once they've done that.

            # If we're testing `test_inputs_embeds_matches_input_ids`, then we'd like to test with `reference_compile`
            # set to False, otherwise the input_ids with compiled input embeddings will not match the inputs_embeds
            # with atol=1e-8 and rtol=1e-5
            turn_off_compile_tests = [
                "test_retain_grad_hidden_states_attentions",
                "test_inputs_embeds_matches_input_ids",
                "test_training_gradient_checkpointing_use_reentrant",
                "test_training_gradient_checkpointing_use_reentrant_false",
                "test_left_padding_compatibility",
            ]
            if test_name in turn_off_compile_tests:
                config.reference_compile = False

            if test_name in (
                # Some tests require attentions to be outputted, in that case we'll set the attention implementation to eager
                # as the others don't support outputted attentions
                "test_attention_outputs",
                "test_beam_sample_generate_dict_output",
                "test_assisted_decoding_matches_greedy_search_1",
                "test_assisted_decoding_matches_greedy_search_1_same",
                "test_assisted_decoding_matches_greedy_search_0_random",
                "test_hidden_states_output",
                "test_retain_grad_hidden_states_attentions",
                # those that `use_cache=True` have to use eager
                # they don't work with unpadding
                "test_sample_generate",
                "test_greedy_generate",
                "test_greedy_generate_dict_outputs_use_cache",
                "test_beam_search_generate",
                "test_beam_sample_generate",
                "test_beam_search_generate_dict_output",
                "test_beam_search_generate_dict_outputs_use_cache",
                "test_constrained_beam_search_generate",
                "test_constrained_beam_search_generate_dict_output",
                "test_group_beam_search_generate",
                "test_group_beam_search_generate_dict_output",
                "test_generate_from_inputs_embeds_0_greedy",
                "test_beam_search_low_memory",
                "test_contrastive_generate",
                "test_contrastive_generate_dict_outputs_use_cache",
                "test_contrastive_generate_low_memory",
                "test_dola_decoding_sample",
                "test_assisted_decoding_sample",
                "test_generate_from_inputs_embeds_1_beam_search",
                "test_generate_from_inputs_embeds_2_group_beam_search",
                "test_generate_from_inputs_embeds_3_group_beam_search_dict_output",
                "test_generate_from_inputs_embeds_4_group_beam_search_dict_output_use_cache",
                "test_generate_from_inputs_embeds_5_group_beam_search_dict_output_use_cache",
                "test_generate_continue_from_past_key_values",
                "test_prompt_lookup_decoding_matches_greedy_search",
                "test_past_key_values_format",
                "test_sample_generate_dict_output",
                "test_greedy_generate_dict_outputs",
            ):
                config._attn_implementation = "eager"
        return config

    def create_and_check_model(self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels):
        model = ModernBertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_masked_lm(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = ModernBertForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = ModernBertForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = ModernBertForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_causal_lm(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = ModernBertForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class ModernBertModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    test_torchscript = False

    all_model_classes = (
        (
            ModernBertModel,
            ModernBertForCausalLM,
            ModernBertForMaskedLM,
            ModernBertForSequenceClassification,
            ModernBertForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (ModernBertForCausalLM,)
    pipeline_model_mapping = (
        {
            "feature-extraction": ModernBertModel,
            "fill-mask": ModernBertForMaskedLM,
            "text-classification": ModernBertForSequenceClassification,
            "text-generation": ModernBertForCausalLM,
            "token-classification": ModernBertForTokenClassification,
            "zero-shot": ModernBertForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    model_split_percents = [0.5, 0.8, 0.9]

    # special case for ForPreTraining model
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if inputs_dict.get("output_attentions", False):
            inputs_dict["output_attentions"] = True

        if return_labels:
            if model_class in get_values(MODEL_FOR_PRETRAINING_MAPPING):
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )
                inputs_dict["next_sentence_label"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
        return inputs_dict

    def setUp(self):
        self.model_tester = ModernBertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ModernBertConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                # The classifier.weight from ModernBertForSequenceClassification and ModernBertForTokenClassification
                # are initialized without `initializer_range`, so they're not set to ~0 via the _config_zero_init
                if param.requires_grad and not (
                    (name == "classifier.weight" or name == "head.weight")
                    and model_class
                    in [ModernBertForSequenceClassification, ModernBertForTokenClassification, ModernBertForCausalLM]
                ):
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_for_warning_if_padding_and_no_attention_mask(self):
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.model_tester.prepare_config_and_inputs()

        # Set pad tokens in the input_ids
        input_ids[0, 0] = config.pad_token_id

        # Check for warnings if the attention_mask is missing.
        logger = logging.get_logger("transformers.modeling_utils")
        # clear cache so we can test the warning is emitted (from `warning_once`).
        logger.warning_once.cache_clear()

        with CaptureLogger(logger) as cl:
            model = ModernBertModel(config=config)
            model.to(torch_device)
            model.eval()
            model(input_ids, attention_mask=None)
        self.assertIn("We strongly recommend passing in an `attention_mask`", cl.out)

    @unittest.skip("ModernBert doesn't use separate classes for SDPA, but a function instead.")
    def test_sdpa_can_dispatch_non_composite_models(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "google-bert/bert-base-uncased"
        model = ModernBertModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        self.skipTest(reason="ModernBert flash attention does not support right padding")

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_conversion(self):
        self.skipTest(reason="ModernBert doesn't use the ModernBertFlashAttention2 class method.")


@require_torch
class ModernBertModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        model = ModernBertForMaskedLM.from_pretrained(
            "answerdotai/ModernBERT-base", reference_compile=False, attn_implementation="sdpa"
        )
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

        inputs = tokenizer("Hello World!", return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)[0]
        expected_shape = torch.Size((1, 5, 50368))
        self.assertEqual(output.shape, expected_shape)

        # compare the actual values for a slice.
        expected_slice = torch.tensor(
            [[[3.8387, -0.2017, 12.2839], [3.6300, 0.6869, 14.7123], [-5.1137, -3.8122, 11.9874]]]
        )
        torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_causal_lm(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        model = ModernBertForCausalLM.from_pretrained(
            "blab-jhu/test-32m-dec", reference_compile=False, attn_implementation="sdpa"
        )
        tokenizer = AutoTokenizer.from_pretrained("blab-jhu/test-32m-dec")

        inputs = tokenizer("Paris is the capital of", return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)[0]
        expected_shape = torch.Size((1, 6, 50368))
        self.assertEqual(output.shape, expected_shape)

        # compare the actual values for a slice.
        expected_slice = torch.tensor(
            [[[-8.0183, -7.1578, -0.4453], [-6.2909, -6.1557, 4.9063], [-6.7689, -5.8068, 6.1078]]]
        )
        torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_no_head(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        model = ModernBertModel.from_pretrained(
            "answerdotai/ModernBERT-base", reference_compile=False, attn_implementation="sdpa"
        )
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

        inputs = tokenizer("Hello World!", return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)[0]
        expected_shape = torch.Size((1, 5, 768))
        self.assertEqual(output.shape, expected_shape)

        # compare the actual values for a slice.
        expected_slice = torch.tensor(
            [[[0.3151, -0.6417, -0.7027], [-0.7834, -1.5810, 0.4576], [1.0614, -0.7268, -0.0871]]]
        )
        torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_token_classification(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        model = ModernBertForTokenClassification.from_pretrained(
            "hf-internal-testing/tiny-random-ModernBertForTokenClassification",
            reference_compile=False,
            attn_implementation="sdpa",
        )
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-ModernBertForTokenClassification")

        inputs = tokenizer("Hello World!", return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)[0]
        expected_shape = torch.Size((1, 5, 2))
        self.assertEqual(output.shape, expected_shape)

        expected = torch.tensor(
            [[[2.0159, 4.6569], [-0.9430, 3.1595], [-3.8770, 3.2653], [1.5752, 4.5167], [-1.6939, 1.2524]]]
        )
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_sequence_classification(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        model = ModernBertForSequenceClassification.from_pretrained(
            "hf-internal-testing/tiny-random-ModernBertForSequenceClassification",
            reference_compile=False,
            attn_implementation="sdpa",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-ModernBertForSequenceClassification"
        )

        inputs = tokenizer("Hello World!", return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)[0]
        expected_shape = torch.Size((1, 2))
        self.assertEqual(output.shape, expected_shape)

        expected = torch.tensor([[1.6466, 4.5662]])
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    @slow
    def test_export(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        bert_model = "answerdotai/ModernBERT-base"
        device = "cpu"
        attn_implementation = "sdpa"
        max_length = 512

        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        inputs = tokenizer(
            "the man worked as a [MASK].",
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
        )

        model = ModernBertForMaskedLM.from_pretrained(
            bert_model,
            device_map=device,
            attn_implementation=attn_implementation,
        )

        logits = model(**inputs).logits
        eg_predicted_mask = tokenizer.decode(logits[0, 6].topk(5).indices)
        self.assertEqual(eg_predicted_mask.split(), ["lawyer", "mechanic", "teacher", "doctor", "waiter"])

        exported_program = torch.export.export(
            model,
            args=(inputs["input_ids"],),
            kwargs={"attention_mask": inputs["attention_mask"]},
            strict=True,
        )

        result = exported_program.module().forward(inputs["input_ids"], inputs["attention_mask"])
        ep_predicted_mask = tokenizer.decode(result.logits[0, 6].topk(5).indices)
        self.assertEqual(eg_predicted_mask, ep_predicted_mask)
