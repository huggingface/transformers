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
import copy
import json
import os
import tempfile
import unittest

import pytest
from packaging import version

from transformers import AutoTokenizer, ModernBertConfig, PreTrainedModel, is_torch_available
from transformers.models.auto import get_values
from transformers.testing_utils import (
    CaptureLogger,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        MODEL_FOR_PRETRAINING_MAPPING,
        ModernBertForMaskedLM,
        ModernBertForMultipleChoice,
        ModernBertForQuestionAnswering,
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
        )
        if test := os.environ.get("PYTEST_CURRENT_TEST", None):
            test_name = test.split(":")[-1].split(" ")[0]

            # If we're testing `test_retain_grad_hidden_states_attentions`, we normally get an error
            # that compilation doesn't work. Users can then set compile=False when loading the model,
            # much like here. We're testing whether it works once they've done that.

            # If we're testing `test_inputs_embeds_matches_input_ids`, then we'd like to test with `reference_compile`
            # set to False, otherwise the input_ids with compiled input embeddings will not match the inputs_embeds
            # with atol=1e-8 and rtol=1e-5
            if test_name in ("test_retain_grad_hidden_states_attentions", "test_inputs_embeds_matches_input_ids"):
                config.reference_compile = False
            # Some tests require attentions to be outputted, in that case we'll set the attention implementation to eager
            # as the others don't support outputted attentions
            if test_name in (
                "test_attention_outputs",
                "test_hidden_states_output",
                "test_retain_grad_hidden_states_attentions",
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

    def create_and_check_for_multiple_choice(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = ModernBertForMultipleChoice(config=config)
        model.to(torch_device)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        result = model(
            multiple_choice_inputs_ids,
            attention_mask=multiple_choice_input_mask,
            labels=choice_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))

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
class ModernBertModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            ModernBertModel,
            ModernBertForMaskedLM,
            ModernBertForSequenceClassification,
            ModernBertForTokenClassification,
            ModernBertForQuestionAnswering,
            ModernBertForMultipleChoice,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": ModernBertModel,
            "fill-mask": ModernBertForMaskedLM,
            "text-classification": ModernBertForSequenceClassification,
            "token-classification": ModernBertForTokenClassification,
            "zero-shot": ModernBertForSequenceClassification,
            "question-answering": ModernBertForQuestionAnswering,
        }
        if is_torch_available()
        else {}
    )

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

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

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

    @pytest.mark.torch_compile_test
    def test_saved_config_excludes_reference_compile(self):
        config = ModernBertConfig(reference_compile=True)
        with tempfile.TemporaryDirectory() as tmpdirname:
            config.save_pretrained(tmpdirname)
            with open(os.path.join(tmpdirname, "config.json")) as f:
                config_dict = json.load(f)
            self.assertNotIn("reference_compile", config_dict)

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    def test_flash_attention_dispatches_by_default(self):
        "ModernBert should dispatch to FA2 by default, not SDPA"
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config=config)
            self.assertTrue(model.config._attn_implementation == "flash_attention_2")

    # This is overloaded because the model handles padding / unpadding on its own, thus ModernBertForMultipleChoice has
    # a different hidden states shape when using FA2.
    def flash_attn_inference_equivalence(
        self, attn_implementation: str, padding_side: str, atol: float = 4e-2, rtol: float = 4e-2
    ):
        r"""
        Tests the equivalence between the eager and flash attention implementations.
        This test is only for inference and runs with `dtype=torch.bfloat16`.
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        # This flag is used to know if the test was skipped for all `self.all_model_classes` or not
        _has_run_at_least_one_model = False

        for model_class in self.all_model_classes:
            # Custom kernel which needs the mask interface to be properly usable on these models
            if not model_class._supports_attention_backend and not attn_implementation.startswith("flash_attention"):
                continue

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            # flash attention variants does not always support arbitrary headim
            config = self._prepare_config_headdim(config, 16)

            # forcing the prefill size to go over sliding window size to check for SWA correctness
            if getattr(config, "sliding_window", None):
                config.sliding_window = 2

            model = model_class(config)
            if not all(
                submodel._supports_flash_attn for submodel in model.modules() if isinstance(submodel, PreTrainedModel)
            ):
                continue

            # If we end up here, at least one model class was not skipped
            _has_run_at_least_one_model = True
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Save the model so we can reload with correct attention
                model.save_pretrained(tmpdirname)

                # Create first inputs without attention mask
                main_input = inputs_dict[model.main_input_name]
                # Only keep first batch sequence
                if isinstance(main_input, torch.Tensor):
                    main_input = main_input[:1]
                    # Fix the dtype
                    if torch.is_floating_point(main_input):
                        main_input = main_input.to(torch.bfloat16)
                first_inputs = {model.main_input_name: main_input, "output_hidden_states": True}
                # Some models have main input name which is different from input_ids, but require input_ids... e.g. BarkFine
                if model.main_input_name != "input_ids" and "input_ids" in inputs_dict:
                    first_inputs["input_ids"] = inputs_dict["input_ids"][:1]
                # If we have some pixel values, use them as well
                if model.main_input_name != "pixel_values" and "pixel_values" in inputs_dict:
                    # NOTE: this fixes qwen2_5_vl/omni because test break w/ pixel values
                    if "image_grid_thw" in inputs_dict:
                        continue
                    first_inputs["pixel_values"] = inputs_dict["pixel_values"][:1].to(torch.bfloat16)
                if model.config.is_encoder_decoder:
                    decoder_input_ids = inputs_dict.get("decoder_input_ids", first_inputs.get("input_ids"))
                    if decoder_input_ids is not None:
                        first_inputs["decoder_input_ids"] = decoder_input_ids[:1]

                # Create attention mask with padding
                dummy_attention_mask = inputs_dict.get("attention_mask", None)
                if dummy_attention_mask is not None:
                    dummy_attention_mask = dummy_attention_mask[:1]
                    if padding_side == "left":
                        dummy_attention_mask[:, 1:] = 1
                        dummy_attention_mask[:, 0] = 0
                    else:
                        dummy_attention_mask[:, :-1] = 1
                        dummy_attention_mask[:, -1] = 0

                # Create second inputs with attention mask and padding
                second_inputs = copy.deepcopy(first_inputs)
                if dummy_attention_mask is not None:
                    second_inputs["attention_mask"] = dummy_attention_mask
                    if model.config.is_encoder_decoder:
                        second_inputs["decoder_attention_mask"] = dummy_attention_mask

                # Use prepare for class to account for special attributes (e.g. in QnA models)
                first_inputs = self._prepare_for_class(first_inputs, model_class)
                first_inputs = {
                    k: v.to(torch_device) if isinstance(v, torch.Tensor) else v for k, v in first_inputs.items()
                }
                second_inputs = self._prepare_for_class(second_inputs, model_class)
                second_inputs = {
                    k: v.to(torch_device) if isinstance(v, torch.Tensor) else v for k, v in second_inputs.items()
                }

                model = model_class.from_pretrained(
                    tmpdirname, dtype=torch.bfloat16, attn_implementation="eager", device_map=torch_device
                )

                # First run without attention mask
                outputs = model(**first_inputs)
                retrieve_logits = model_class == ModernBertForMultipleChoice
                logits_1_eager = outputs.logits if retrieve_logits else outputs.hidden_states[-1]
                # Second run with attention mask and padding
                outputs = model(**second_inputs)
                logits_2_eager = outputs.logits if retrieve_logits else outputs.hidden_states[-1]

                # Switch to FA
                del model
                model = model_class.from_pretrained(
                    tmpdirname, dtype=torch.bfloat16, attn_implementation=attn_implementation, device_map=torch_device
                )
                outputs = model(**first_inputs)
                logits_1_fa = outputs.logits if retrieve_logits else outputs.hidden_states[-1]
                # Second run with attention mask and padding
                outputs = model(**second_inputs)
                logits_2_fa = outputs.logits if retrieve_logits else outputs.hidden_states[-1]

                # Check the results
                torch.testing.assert_close(logits_1_eager, logits_1_fa, atol=atol, rtol=rtol)
                if padding_side == "left":
                    torch.testing.assert_close(logits_2_eager[1:], logits_2_fa[1:], atol=atol, rtol=rtol)
                    # Check it can run in training mode
                    model.train()
                    _ = model(**second_inputs)
                else:
                    torch.testing.assert_close(logits_2_eager[:-1], logits_2_fa[:-1], atol=atol, rtol=rtol)

        # In this case, the test should appear as skipped, not successful
        if not _has_run_at_least_one_model:
            self.skipTest(
                f"Model architecture does not support {attn_implementation}, or setting its attention dynamically"
            )


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

    @pytest.mark.torch_export_test
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

    @slow
    def test_inference_multiple_choice(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        model = (
            ModernBertForMultipleChoice.from_pretrained(
                "netique/ModernBertForMultipleChoice",
                reference_compile=False,
                attn_implementation="sdpa",
            )
            .eval()
            .to(torch_device)
        )

        prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        choices = [
            "It is eaten with a fork and a knife.",
            "It is eaten while held in the hand.",
            "It also walks on the sidewalks.",
            "It is a common drink.",
        ]
        labels = torch.tensor([0], device=torch_device)

        encoding = tokenizer([prompt for _ in choices], choices, return_tensors="pt", padding=True)
        outputs = model(**{k: v.unsqueeze(0).to(torch_device) for k, v in encoding.items()}, labels=labels)

        expected_logits = torch.tensor([[0.1973, 0.2041, 0.1835, 0.1896]])
        logits = outputs.logits.to("cpu")

        self.assertTrue(
            torch.allclose(logits, expected_logits, atol=1e-4, rtol=1e-4),
            f"Logits: {logits.tolist()}\nExpected: {expected_logits.tolist()}",
        )
