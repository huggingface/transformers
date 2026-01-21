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
import os
import tempfile
import unittest

import pytest
from packaging import version
from pytest import mark

from transformers import AutoTokenizer, ModernBertConfig, is_torch_available
from transformers.modeling_utils import FLASH_ATTN_KERNEL_FALLBACK
from transformers.models.auto import get_values
from transformers.testing_utils import (
    CaptureLogger,
    force_serialization_as_bin_files,
    require_accelerate,
    require_flash_attn,
    require_non_hpu,
    require_torch,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)
from transformers.utils import CONFIG_NAME, GENERATION_CONFIG_NAME

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        MODEL_FOR_PRETRAINING_MAPPING,
        AutoModel,
        AutoModelForSequenceClassification,
        ModernBertForMaskedLM,
        ModernBertForMultipleChoice,
        ModernBertForQuestionAnswering,
        ModernBertForSequenceClassification,
        ModernBertForTokenClassification,
        ModernBertModel,
        logging,
    )
    from transformers.integrations.accelerate import compute_module_sizes
    from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES


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
            # Use SDPA as the default attention implementation for testing
            attn_implementation="sdpa",
        )
        if test := os.environ.get("PYTEST_CURRENT_TEST", None):
            test_name = test.split(":")[-1].split(" ")[0]

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

    @slow
    def test_model_from_pretrained(self):
        model_name = "google-bert/bert-base-uncased"
        model = ModernBertModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @require_flash_attn
    @require_torch_accelerator
    @pytest.mark.flash_attn_test
    def test_flash_attention_dispatches_by_default(self):
        """ModernBert should dispatch to FA2 by default, not SDPA"""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config._attn_implementation = None  # Let the model choose the default attention implementation
        for model_class in self.all_model_classes:
            model = model_class(config=config)
            # If flash_attn is not available, fallback to kernels loading mechanism
            expected_implementations = ["flash_attention_2", FLASH_ATTN_KERNEL_FALLBACK.get("flash_attention_2")]
            self.assertIn(model.config._attn_implementation, expected_implementations)

    @unittest.skip("ModernBert dispatches to flash_attention on default")
    def test_sdpa_can_dispatch_non_composite_models(self):
        pass

    # Override tests(from test_save_load to test_model_parallelism) that use from_pretrained to ensure SDPA attention is used instead of FlashAttention.
    # ModernBERT defaults to FlashAttention when available, but FA only supports fp16 and bf16 data types,
    # so these tests would fail with fp32. We force SDPA here for dtype compatibility.
    def test_save_load(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # the config file (and the generation config file, if it can generate) should be saved
                self.assertTrue(os.path.exists(os.path.join(tmpdirname, CONFIG_NAME)))
                self.assertEqual(
                    model.can_generate(), os.path.exists(os.path.join(tmpdirname, GENERATION_CONFIG_NAME))
                )

                # Force SDPA attention for FA dtype compatibility (FA only supports fp16/bf16)
                model = model_class.from_pretrained(tmpdirname, attn_implementation="sdpa")
                model.to(torch_device)
                with torch.no_grad():
                    second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

                # Save and load second time because `from_pretrained` adds a bunch of new config fields
                # so we need to make sure those fields can be loaded back after saving
                # Simply init as `model(config)` doesn't add those fields
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname, attn_implementation="sdpa")

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    torch.testing.assert_close(
                        tensor1, tensor2, msg="Running save/load and forward yields different results"
                    )
            else:
                torch.testing.assert_close(first, second, msg="Running save/load and forward yields different results")

    def test_load_with_mismatched_shapes(self):
        if not self.test_mismatched_shapes:
            self.skipTest(reason="test_mismatched_shapes is set to False")
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class.__name__ not in get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES):
                continue

            with self.subTest(msg=f"Testing {model_class}"):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    model = model_class(config)
                    model.save_pretrained(tmp_dir)
                    # Fails when we don't set ignore_mismatched_sizes=True
                    # Force SDPA attention for FA dtype compatibility (FA only supports fp16/bf16)
                    with self.assertRaises(RuntimeError):
                        new_model = AutoModelForSequenceClassification.from_pretrained(
                            tmp_dir, num_labels=42, attn_implementation="sdpa"
                        )
                    with self.assertRaises(RuntimeError):
                        new_model_without_prefix = AutoModel.from_pretrained(
                            tmp_dir, vocab_size=10, attn_implementation="sdpa"
                        )

                    logger = logging.get_logger("transformers.modeling_utils")

                    with CaptureLogger(logger) as cl:
                        new_model = AutoModelForSequenceClassification.from_pretrained(
                            tmp_dir, num_labels=42, ignore_mismatched_sizes=True, attn_implementation="sdpa"
                        )
                    self.assertIn("Reinit due to size mismatch", cl.out)
                    new_model.to(torch_device)
                    inputs = self._prepare_for_class(inputs_dict, model_class)
                    logits = new_model(**inputs).logits
                    self.assertEqual(logits.shape[1], 42)

                    with CaptureLogger(logger) as cl:
                        new_model_without_prefix = AutoModel.from_pretrained(
                            tmp_dir, vocab_size=10, ignore_mismatched_sizes=True, attn_implementation="sdpa"
                        )
                    self.assertIn("Reinit due to size mismatch", cl.out)
                    input_ids = ids_tensor((2, 8), 10)
                    new_model_without_prefix.to(torch_device)
                    if self.is_encoder_decoder:
                        new_model_without_prefix(input_ids, decoder_input_ids=input_ids)
                    else:
                        new_model_without_prefix(input_ids)

    @require_accelerate
    @mark.accelerate_tests
    @require_torch_accelerator
    def test_cpu_offload(self, rtol=1e-5, atol=1e-5):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class._no_split_modules is None:
                continue

            inputs_dict_class = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(copy.deepcopy(config)).eval()
            model = model.to(torch_device)

            torch.manual_seed(0)
            base_output = model(**inputs_dict_class)

            model_size = compute_module_sizes(model)[0][""]
            # We test several splits of sizes to make sure it works.
            max_gpu_sizes = [int(p * model_size) for p in self.model_split_percents[1:]]
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.cpu().save_pretrained(tmp_dir)

                for max_size in max_gpu_sizes:
                    max_memory = {0: max_size, "cpu": model_size * 2}
                    # Force SDPA attention for FA dtype compatibility (FA only supports fp16/bf16)
                    new_model = model_class.from_pretrained(
                        tmp_dir, device_map="auto", max_memory=max_memory, attn_implementation="sdpa"
                    )
                    # Making sure part of the model will actually end up offloaded
                    self.assertSetEqual(set(new_model.hf_device_map.values()), {0, "cpu"})

                    self.check_device_map_is_respected(new_model, new_model.hf_device_map)

                    torch.manual_seed(0)
                    new_output = new_model(**inputs_dict_class)

                    if isinstance(base_output[0], tuple) and isinstance(new_output[0], tuple):
                        [
                            torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
                            for a, b in zip(base_output[0], new_output[0])
                        ]
                    else:
                        torch.testing.assert_close(base_output[0], new_output[0], rtol=rtol, atol=atol)

    @require_accelerate
    @mark.accelerate_tests
    @require_torch_accelerator
    def test_disk_offload_bin(self, rtol=1e-5, atol=1e-5):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class._no_split_modules is None:
                continue

            inputs_dict_class = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(copy.deepcopy(config)).eval()
            model = model.to(torch_device)
            torch.manual_seed(0)
            base_output = model(**inputs_dict_class)

            model_size = compute_module_sizes(model)[0][""]
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Since we don't support saving with bins files anymore, but still support loading we use this context
                # to easily create the bins files and try to load them
                with force_serialization_as_bin_files():
                    model.cpu().save_pretrained(tmp_dir)

                with self.assertRaises(ValueError):
                    max_size = int(self.model_split_percents[0] * model_size)
                    max_memory = {0: max_size, "cpu": max_size}
                    # This errors out cause it's missing an offload folder
                    # Force SDPA attention for FA dtype compatibility (FA only supports fp16/bf16)
                    new_model = model_class.from_pretrained(
                        tmp_dir,
                        device_map="auto",
                        max_memory=max_memory,
                        use_safetensors=False,
                        attn_implementation="sdpa",
                    )

                max_size = int(self.model_split_percents[1] * model_size)
                max_memory = {0: max_size, "cpu": max_size}
                # Force SDPA attention for FA dtype compatibility (FA only supports fp16/bf16)
                new_model = model_class.from_pretrained(
                    tmp_dir,
                    device_map="auto",
                    max_memory=max_memory,
                    offload_folder=tmp_dir,
                    use_safetensors=False,
                    attn_implementation="sdpa",
                )

                self.check_device_map_is_respected(new_model, new_model.hf_device_map)
                torch.manual_seed(0)
                new_output = new_model(**inputs_dict_class)

                if isinstance(base_output[0], tuple) and isinstance(new_output[0], tuple):
                    [
                        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
                        for a, b in zip(base_output[0], new_output[0])
                    ]
                else:
                    torch.testing.assert_close(base_output[0], new_output[0], rtol=rtol, atol=atol)

    @require_accelerate
    @mark.accelerate_tests
    @require_torch_accelerator
    def test_disk_offload_safetensors(self, rtol=1e-5, atol=1e-5):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class._no_split_modules is None:
                continue

            inputs_dict_class = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(copy.deepcopy(config)).eval()
            model = model.to(torch_device)
            torch.manual_seed(0)
            base_output = model(**inputs_dict_class)

            model_size = compute_module_sizes(model)[0][""]
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.cpu().save_pretrained(tmp_dir)

                max_size = int(self.model_split_percents[1] * model_size)
                max_memory = {0: max_size, "cpu": max_size}

                # This doesn't error out as it's in safetensors and doesn't need an offload folder
                # Force SDPA attention for FA dtype compatibility (FA only supports fp16/bf16)
                new_model = model_class.from_pretrained(
                    tmp_dir,
                    device_map="auto",
                    max_memory=max_memory,
                    offload_folder=tmp_dir,
                    attn_implementation="sdpa",
                )

                self.check_device_map_is_respected(new_model, new_model.hf_device_map)
                torch.manual_seed(0)
                new_output = new_model(**inputs_dict_class)

                if isinstance(base_output[0], tuple) and isinstance(new_output[0], tuple):
                    [
                        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
                        for a, b in zip(base_output[0], new_output[0])
                    ]
                else:
                    torch.testing.assert_close(base_output[0], new_output[0], rtol=rtol, atol=atol)

    @require_non_hpu
    @require_accelerate
    @mark.accelerate_tests
    @require_torch_multi_accelerator
    def test_model_parallelism(self, rtol=1e-5, atol=1e-5):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class._no_split_modules is None:
                continue

            inputs_dict_class = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(config).eval()
            model = model.to(torch_device)

            torch.manual_seed(0)
            base_output = model(**inputs_dict_class)

            model_size = compute_module_sizes(model)[0][""]
            # We test several splits of sizes to make sure it works.
            max_gpu_sizes = [int(p * model_size) for p in self.model_split_percents[1:]]
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.cpu().save_pretrained(tmp_dir)

                for max_size in max_gpu_sizes:
                    max_memory = {0: max_size, 1: model_size * 2, "cpu": model_size * 2}
                    # Force SDPA attention for FA dtype compatibility (FA only supports fp16/bf16)
                    new_model = model_class.from_pretrained(
                        tmp_dir, device_map="auto", max_memory=max_memory, attn_implementation="sdpa"
                    )
                    # Making sure part of the model will actually end up offloaded
                    self.assertSetEqual(set(new_model.hf_device_map.values()), {0, 1})
                    self.check_device_map_is_respected(new_model, new_model.hf_device_map)

                    torch.manual_seed(0)
                    new_output = new_model(**inputs_dict_class)

                    if isinstance(base_output[0], tuple) and isinstance(new_output[0], tuple):
                        [
                            torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
                            for a, b in zip(base_output[0], new_output[0])
                        ]
                    else:
                        torch.testing.assert_close(base_output[0], new_output[0], rtol=rtol, atol=atol)

@require_torch
class ModernBertModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        model = ModernBertForMaskedLM.from_pretrained(
            "answerdotai/ModernBERT-base", attn_implementation="sdpa"
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
            "answerdotai/ModernBERT-base", attn_implementation="sdpa"
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

    @require_flash_attn
    @require_torch_accelerator
    @pytest.mark.flash_attn_test
    @slow
    def test_inference_masked_lm_flash_attention_2(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        model = ModernBertForMaskedLM.from_pretrained("answerdotai/ModernBERT-base", dtype=torch.float16).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

        inputs = tokenizer("Hello World!", return_tensors="pt").to(torch_device)
        with torch.no_grad():
            output = model(**inputs)[0]
        expected_shape = torch.Size((1, 5, 50368))
        self.assertEqual(output.shape, expected_shape)

        # compare the actual values for a slice.
        expected_slice = torch.tensor(
            [[[3.8203, -0.2125, 12.2812], [3.6055, 0.6797, 14.6875], [-5.1094, -3.8105, 11.9922]]],
            dtype=torch.float16
        )
        torch.testing.assert_close(output[:, :3, :3].cpu(), expected_slice, rtol=1e-4, atol=1e-4)
