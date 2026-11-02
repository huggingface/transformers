# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from transformers import AutoTokenizer, UltrabertConfig, is_torch_available
from transformers.testing_utils import Expectations, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        AutoModel,
        AutoModelForMaskedLM,
        AutoModelForSequenceClassification,
        UltrabertForMaskedLM,
        UltrabertForQuestionAnswering,
        UltrabertForSequenceClassification,
        UltrabertForTokenClassification,
        UltrabertModel,
    )


class UltrabertModelTester:
    if is_torch_available():
        base_model_class = UltrabertModel

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        hidden_act="silu",
        attention_dropout=0.0,
        max_position_embeddings=512,
        initializer_range=0.02,
        num_labels=3,
        pad_token_id=0,
        sliding_window=2,
        prediction_head_size=32,
        layer_types=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id
        self.sliding_window = sliding_window
        self.prediction_head_size = prediction_head_size
        self.layer_types = (
            layer_types
            if layer_types is not None
            else ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(num_hidden_layers)]
        )

    def get_config(self, **kwargs):
        options = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "attention_dropout": self.attention_dropout,
            "max_position_embeddings": self.max_position_embeddings,
            "initializer_range": self.initializer_range,
            "pad_token_id": self.pad_token_id,
            "sliding_window": self.sliding_window,
            "prediction_head_size": self.prediction_head_size,
            "layer_types": self.layer_types,
        }
        options.update(kwargs)
        return UltrabertConfig(**options)

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length]) if self.use_input_mask else None
        sequence_labels = ids_tensor([self.batch_size], self.num_labels)
        token_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        return self.get_config(), input_ids, attention_mask, sequence_labels, token_labels

    def create_and_check_model(self, config, input_ids, attention_mask, sequence_labels, token_labels):
        model = UltrabertModel(config).to(torch_device).eval()
        result = model(input_ids, attention_mask=attention_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_masked_lm(self, config, input_ids, attention_mask, sequence_labels, token_labels):
        model = UltrabertForMaskedLM(config).to(torch_device).eval()
        result = model(input_ids, attention_mask=attention_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, attention_mask, sequence_labels, token_labels
    ):
        config.num_labels = self.num_labels
        model = UltrabertForSequenceClassification(config).to(torch_device).eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self, config, input_ids, attention_mask, sequence_labels, token_labels
    ):
        config.num_labels = self.num_labels
        labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
        labels[:, -1] = -100
        model = UltrabertForTokenClassification(config).to(torch_device).eval()
        result = model(input_ids, attention_mask=attention_mask, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_question_answering(
        self, config, input_ids, attention_mask, sequence_labels, token_labels
    ):
        model = UltrabertForQuestionAnswering(config).to(torch_device).eval()
        start_positions = ids_tensor([self.batch_size], self.seq_length)
        end_positions = start_positions + 1
        result = model(
            input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions
        )
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask, _, _ = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}


@require_torch
class UltrabertModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            UltrabertModel,
            UltrabertForMaskedLM,
            UltrabertForSequenceClassification,
            UltrabertForTokenClassification,
            UltrabertForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": UltrabertModel,
            "fill-mask": UltrabertForMaskedLM,
            "text-classification": UltrabertForSequenceClassification,
            "token-classification": UltrabertForTokenClassification,
            "zero-shot": UltrabertForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = UltrabertModelTester

    def setUp(self):
        self.model_tester = UltrabertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=UltrabertConfig, hidden_size=32, num_attention_heads=4)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="SDPA's flash backend doesn't support the mandatory sliding-window attention masks.")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def test_model(self):
        self.model_tester.create_and_check_model(*self.model_tester.prepare_config_and_inputs())

    def test_for_masked_lm(self):
        self.model_tester.create_and_check_for_masked_lm(*self.model_tester.prepare_config_and_inputs())

    def test_for_sequence_classification(self):
        self.model_tester.create_and_check_for_sequence_classification(*self.model_tester.prepare_config_and_inputs())

    def test_for_token_classification(self):
        self.model_tester.create_and_check_for_token_classification(*self.model_tester.prepare_config_and_inputs())

    def test_for_question_answering(self):
        self.model_tester.create_and_check_for_question_answering(*self.model_tester.prepare_config_and_inputs())

    def test_legacy_config(self):
        attention_windows = [2, 2, None]
        rope_theta = [10000.0, 10000.0, 2560000.0]
        config = self.model_tester.get_config(
            num_hidden_layers=3, layer_types=None, attention_windows=attention_windows, rope_theta=rope_theta
        )
        self.assertEqual(config.sliding_window, 2)
        self.assertEqual(
            config.layer_types, ["sliding_attention" if window else "full_attention" for window in attention_windows]
        )
        for layer_type, base in zip(config.layer_types, rope_theta):
            self.assertEqual(config.rope_parameters[layer_type], {"rope_type": "default", "rope_theta": base})
        self.assertNotIn("attention_windows", config.to_dict())
        self.assertNotIn("rope_theta", config.to_dict())

    def test_rope_parameters_override_legacy_theta(self):
        rope_parameters = {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "linear", "rope_theta": 2560000.0, "factor": 2.0},
        }
        config = self.model_tester.get_config(
            rope_parameters=rope_parameters, rope_theta=[12345.0] * self.model_tester.num_hidden_layers
        )
        self.assertEqual(config.rope_parameters, rope_parameters)

    def test_attention_window_boundaries(self):
        config = self.model_tester.get_config(attn_implementation="eager")
        model = UltrabertModel(config).to(torch_device).eval()
        input_ids = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.vocab_size)
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        for attention, layer_type in zip(outputs.attentions, model.config.layer_types):
            if layer_type == "sliding_attention":
                window = model.config.sliding_window
                self.assertTrue(torch.all(attention[:, :, 0, window] > 0))
                self.assertTrue(torch.all(attention[:, :, 0, window + 1] == 0))
                self.assertTrue(torch.all(attention[:, :, window, 0] > 0))
                self.assertTrue(torch.all(attention[:, :, window + 1, 0] == 0))
            else:
                self.assertTrue(torch.all(attention[:, :, 0, -1] > 0))


@require_torch
@slow
class UltrabertIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    """Check native loading against float32 outputs from the original checkpoint code."""

    def load_checkpoint(self, auto_class, checkpoint):
        model = auto_class.from_pretrained(
            checkpoint,
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        return model, tokenizer

    def test_inference_embedding(self):
        model, tokenizer = self.load_checkpoint(AutoModel, "dmis-lab/UltraBERT-Embed-150M")
        inputs = tokenizer(
            ["The quick brown fox jumps over the lazy dog. " * 30, "Hello world!"],
            padding=True,
            return_tensors="pt",
        )
        expected_slice = Expectations(
            {
                (None, None): torch.tensor(
                    [
                        [3.815864, 1.319243, -2.833093],
                        [0.084467, -0.133652, -0.315864],
                        [3.821545, -4.447476, -2.473820],
                        [1.938878, -0.788768, 6.423543],
                        [0.796432, -5.030339, -2.032893],
                    ]
                )
            }
        ).get_expectation()
        expected_padded_cls = Expectations(
            {(None, None): torch.tensor([0.821952, -1.553840, 1.073606])}
        ).get_expectation()
        with torch.no_grad():
            output = model(**inputs).last_hidden_state
        self.assertEqual(output.shape, (2, 302, 768))
        torch.testing.assert_close(output[0, [0, 127, 128, 129, 256], :3], expected_slice, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(output[1, 0, :3], expected_padded_cls, rtol=1e-4, atol=1e-4)

    def test_inference_masked_lm_150m(self):
        model, tokenizer = self.load_checkpoint(AutoModelForMaskedLM, "dmis-lab/UltraBERT-150M")
        inputs = tokenizer(
            ["The capital of France is <|mask|>.", "The capital of Germany is <|mask|>."],
            padding=True,
            return_tensors="pt",
        )
        expected_slice = Expectations(
            {
                (None, None): torch.tensor(
                    [
                        [-173.299500, -172.092453, -172.083084, -166.659607, -168.356415],
                        [-166.776337, -165.526764, -165.573578, -160.336792, -161.950073],
                    ]
                )
            }
        ).get_expectation()
        with torch.no_grad():
            logits = model(**inputs).logits
        self.assertEqual(logits.shape, (2, 8, 49152))
        masked_logits = logits[inputs.input_ids == tokenizer.mask_token_id]
        torch.testing.assert_close(masked_logits[:, :5], expected_slice, rtol=1e-4, atol=1e-4)
        self.assertEqual(tokenizer.batch_decode(masked_logits.argmax(-1)[:, None]), [" Paris", " Berlin"])

    def test_inference_masked_lm_380m(self):
        model, tokenizer = self.load_checkpoint(AutoModelForMaskedLM, "dmis-lab/UltraBERT-380M")
        inputs = tokenizer(
            ["The capital of France is <|mask|>.", "The capital of Germany is <|mask|>."],
            padding=True,
            return_tensors="pt",
        )
        expected_slice = Expectations(
            {
                (None, None): torch.tensor(
                    [
                        [-195.506729, -193.607315, -193.664520, -189.095596, -189.571594],
                        [-189.560608, -187.724716, -187.760635, -183.072098, -183.630432],
                    ]
                )
            }
        ).get_expectation()
        with torch.no_grad():
            logits = model(**inputs).logits
        self.assertEqual(logits.shape, (2, 8, 49152))
        masked_logits = logits[inputs.input_ids == tokenizer.mask_token_id]
        torch.testing.assert_close(masked_logits[:, :5], expected_slice, rtol=1e-4, atol=1e-4)
        self.assertEqual(tokenizer.batch_decode(masked_logits.argmax(-1)[:, None]), [" Paris", " Berlin"])

    def test_inference_sequence_classification(self):
        model, tokenizer = self.load_checkpoint(
            AutoModelForSequenceClassification,
            "cross-encoder-testing/ultrabert-150m-reranker-test",
        )
        inputs = tokenizer(
            [
                ["What is the capital of France?", "Paris is the capital of France."],
                ["What is the capital of France?", "Berlin is the capital of Germany and its largest city."],
            ],
            padding=True,
            return_tensors="pt",
        )
        expected = Expectations({(None, None): torch.tensor([[3.867966175], [-4.749768734]])}).get_expectation()
        with torch.no_grad():
            logits = model(**inputs).logits
        self.assertEqual(logits.shape, (2, 1))
        torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)
