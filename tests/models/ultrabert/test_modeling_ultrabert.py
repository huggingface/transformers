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

import tempfile
import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError
from parameterized import parameterized

from transformers import AutoConfig, AutoTokenizer, UltrabertConfig, is_torch_available
from transformers.testing_utils import require_torch, require_torch_gpu, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        AutoModel,
        AutoModelForMaskedLM,
        AutoModelForQuestionAnswering,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        EuroBertConfig,
        EuroBertModel,
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

    def get_config(self):
        return UltrabertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            attention_windows=[2 if i % 2 == 0 else None for i in range(self.num_hidden_layers)],
            rope_theta=[10000.0 if i % 2 == 0 else 2560000.0 for i in range(self.num_hidden_layers)],
            prediction_head_size=self.hidden_size,
        )

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
        self.parent.assertTrue(torch.isfinite(result.loss))

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
        self.parent.assertTrue(torch.isfinite(result.loss))

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
    _torch_compile_train_cls = UltrabertForMaskedLM if is_torch_available() else None

    def setUp(self):
        self.model_tester = UltrabertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=UltrabertConfig, hidden_size=32, num_attention_heads=4)

    def test_config(self):
        self.config_tester.run_common_tests()

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

    def get_config(self, **kwargs):
        options = {
            "vocab_size": 41,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 12,
            "attention_windows": [2, 3, None],
            "rope_theta": [10000.0, 12345.0, 2560000.0],
            "prediction_head_size": 20,
            "attn_implementation": "eager",
        }
        options.update(kwargs)
        return UltrabertConfig(**options)

    def test_config_scalars_and_roundtrip(self):
        config = self.get_config(attention_windows=2, rope_theta=10000.0)
        self.assertEqual(config.attention_windows, [2, 2, 2])
        self.assertEqual(config.rope_theta, [10000.0] * 3)
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            restored = AutoConfig.from_pretrained(directory)
        self.assertIsInstance(restored, UltrabertConfig)
        self.assertEqual(restored.attention_windows, config.attention_windows)
        self.assertEqual(restored.rope_theta, config.rope_theta)
        with self.assertRaises(StrictDataclassClassValidationError):
            self.get_config(attention_windows=[2])
        with self.assertRaises(StrictDataclassClassValidationError):
            self.get_config(rope_theta=[10000.0])

    def test_global_encoder_matches_eurobert(self):
        config = self.get_config(attention_windows=None, rope_theta=10000.0, head_dim=8)
        eurobert_config = EuroBertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            pad_token_id=config.pad_token_id,
            bos_token_id=0,
            eos_token_id=0,
            mask_token_id=1,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            attn_implementation="eager",
        )
        model = UltrabertModel(config).eval()
        reference = EuroBertModel(eurobert_config).eval()
        reference.load_state_dict(model.state_dict(), strict=True)
        input_ids = torch.randint(1, config.vocab_size, (2, 9))
        with torch.no_grad():
            torch.testing.assert_close(model(input_ids).last_hidden_state, reference(input_ids).last_hidden_state)

    def test_attention_window_boundaries_and_padding(self):
        model = UltrabertModel(self.get_config()).eval()
        input_ids = torch.randint(1, model.config.vocab_size, (2, 9))
        attention_mask = torch.ones_like(input_ids)
        attention_mask[:, -1] = 0
        with torch.no_grad():
            outputs = model(
                input_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states=True
            )
        self.assertEqual(len(outputs.hidden_states), 4)
        self.assertEqual(len(outputs.attentions), 3)
        for attention, window in zip(outputs.attentions, model.config.attention_windows):
            self.assertTrue(torch.all(attention[..., -1] == 0))
            if window is not None:
                self.assertTrue(torch.all(attention[:, :, 0, window] > 0))
                self.assertTrue(torch.all(attention[:, :, 0, window + 1] == 0))
                self.assertTrue(torch.all(attention[:, :, window, 0] > 0))
                self.assertTrue(torch.all(attention[:, :, window + 1, 0] == 0))
            else:
                self.assertTrue(torch.all(attention[:, :, 0, -2] > 0))

    def test_sdpa_matches_eager_on_non_padding_tokens(self):
        model = UltrabertModel(self.get_config()).eval()
        input_ids = torch.randint(1, model.config.vocab_size, (2, 11))
        attention_mask = torch.ones_like(input_ids)
        attention_mask[0, -5:] = 0
        attention_mask[1, :5] = 0
        position_ids = torch.arange(11)[None] + 100
        with torch.no_grad():
            expected = model(input_ids, attention_mask=attention_mask, position_ids=position_ids).last_hidden_state
            model.set_attn_implementation("sdpa")
            actual = model(input_ids, attention_mask=attention_mask, position_ids=position_ids).last_hidden_state
        valid_tokens = attention_mask.bool()
        torch.testing.assert_close(actual[valid_tokens], expected[valid_tokens], atol=2e-6, rtol=2e-5)

    @require_torch_gpu
    def test_sdpa_unpadded_matches_explicit_mask(self):
        torch.manual_seed(42)
        config = self.get_config(attention_windows=None, attn_implementation="sdpa", head_dim=16)
        input_ids = torch.randint(1, config.vocab_size, (1, 139), device=torch_device)
        for dtype, atol, rtol in ((torch.float16, 5e-3, 5e-3), (torch.bfloat16, 1e-2, 3e-2)):
            with self.subTest(dtype=dtype):
                model = UltrabertModel(config).to(device=torch_device, dtype=dtype).eval()
                explicit_mask = torch.zeros((1, 1, 139, 139), device=torch_device, dtype=dtype)
                with torch.no_grad():
                    expected = model(input_ids, attention_mask=explicit_mask).last_hidden_state
                    for mask in (None, torch.ones_like(input_ids)):
                        actual = model(input_ids, attention_mask=mask).last_hidden_state
                        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_masked_lm_and_gradients(self):
        input_ids = torch.randint(1, 41, (2, 9))
        model = AutoModelForMaskedLM.from_config(self.get_config())
        outputs = model(input_ids, labels=input_ids)
        self.assertTrue(torch.isfinite(outputs.loss))
        outputs.loss.backward()
        self.assertIsNotNone(model.ultrabert.embed_tokens.weight.grad)
        model.eval()
        with torch.no_grad():
            self.assertIsInstance(model(input_ids, return_dict=False), tuple)

    def test_sequence_classification_losses_and_gradients(self):
        input_ids = torch.randint(1, 41, (2, 9))
        cases = [
            (1, None, torch.tensor([0.2, 0.8]), "regression"),
            (3, None, torch.tensor([0, 2]), "single_label_classification"),
            (3, None, torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), "multi_label_classification"),
            (1, "multi_label_classification", torch.tensor([[1.0], [0.0]]), "multi_label_classification"),
        ]
        for num_labels, problem_type, labels, expected_problem_type in cases:
            with self.subTest(num_labels=num_labels, problem_type=expected_problem_type):
                model = AutoModelForSequenceClassification.from_config(
                    self.get_config(num_labels=num_labels, problem_type=problem_type, classifier_dropout=0.1)
                )
                self.assertIsInstance(model, UltrabertForSequenceClassification)
                outputs = model(input_ids, labels=labels)
                self.assertEqual(outputs.logits.shape, (2, num_labels))
                self.assertEqual(model.config.problem_type, expected_problem_type)
                self.assertTrue(torch.isfinite(outputs.loss))
                outputs.loss.backward()
                self.assertGreater(model.classifier.weight.grad.abs().sum().item(), 0)
                self.assertGreater(model.ultrabert.embed_tokens.weight.grad.abs().sum().item(), 0)

    @parameterized.expand([("sequence_classification", 1), ("token_classification", 3), ("question_answering", 2)])
    def test_task_head_from_encoder_and_roundtrip(self, task, num_labels):
        auto_class = {
            "sequence_classification": AutoModelForSequenceClassification,
            "token_classification": AutoModelForTokenClassification,
            "question_answering": AutoModelForQuestionAnswering,
        }[task]
        head_name = "qa_outputs" if task == "question_answering" else "classifier"
        output_names = ("start_logits", "end_logits") if task == "question_answering" else ("logits",)
        encoder = UltrabertModel(self.get_config()).eval()
        input_ids = torch.randint(1, encoder.config.vocab_size, (2, 9))
        with tempfile.TemporaryDirectory() as directory:
            encoder.save_pretrained(directory)
            model, info = auto_class.from_pretrained(
                directory, num_labels=num_labels, attn_implementation="eager", output_loading_info=True
            )
            self.assertEqual(set(info["missing_keys"]), {f"{head_name}.weight", f"{head_name}.bias"})
            self.assertFalse(info["unexpected_keys"])
            with torch.no_grad():
                torch.testing.assert_close(
                    model.ultrabert(input_ids).last_hidden_state, encoder(input_ids).last_hidden_state
                )
                outputs = model(input_ids, output_hidden_states=True, output_attentions=True)
            self.assertEqual(len(outputs.hidden_states), 4)
            self.assertEqual(len(outputs.attentions), 3)
            model.save_pretrained(directory)
            restored = auto_class.from_pretrained(directory)
        with torch.no_grad():
            restored_outputs = restored(input_ids)
            tuple_outputs = restored(input_ids, return_dict=False)
        for index, name in enumerate(output_names):
            torch.testing.assert_close(getattr(restored_outputs, name), getattr(outputs, name))
            torch.testing.assert_close(tuple_outputs[index], getattr(outputs, name))

    def test_checkpoint_roundtrip_and_tied_embeddings(self):
        model = UltrabertModel(self.get_config()).eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored, info = AutoModel.from_pretrained(directory, output_loading_info=True)
        self.assertFalse(info["missing_keys"])
        self.assertFalse(info["unexpected_keys"])
        for name, buffer in model.named_buffers():
            torch.testing.assert_close(buffer, restored.get_buffer(name))
        input_ids = torch.randint(1, model.config.vocab_size, (2, 9))
        with torch.no_grad():
            torch.testing.assert_close(model(input_ids).last_hidden_state, restored(input_ids).last_hidden_state)
        mlm = UltrabertForMaskedLM(self.get_config(tie_word_embeddings=True))
        self.assertIs(mlm.cls.weight, mlm.ultrabert.embed_tokens.weight)


@require_torch
@slow
class UltrabertIntegrationTest(unittest.TestCase):
    """Check native loading against float32 outputs from the original checkpoint code."""

    def load_checkpoint(self, auto_class, checkpoint, revision):
        model, loading_info = auto_class.from_pretrained(
            checkpoint,
            revision=revision,
            trust_remote_code=False,
            dtype=torch.float32,
            attn_implementation="eager",
            output_loading_info=True,
        )
        self.assertFalse(loading_info["missing_keys"])
        self.assertFalse(loading_info["unexpected_keys"])
        self.assertFalse(loading_info.get("mismatched_keys"))
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision, trust_remote_code=False)
        return model.eval(), tokenizer

    def test_inference_embedding(self):
        model, tokenizer = self.load_checkpoint(
            AutoModel, "dmis-lab/UltraBERT-Embed-150M", "367d5343b5a0164d57b810cedf5de4b8983d5984"
        )
        inputs = tokenizer(
            ["The quick brown fox jumps over the lazy dog. " * 30, "Hello world!"],
            padding=True,
            return_tensors="pt",
        )
        expected_slice = torch.tensor(
            [
                [3.815864, 1.319243, -2.833093],
                [0.084467, -0.133652, -0.315864],
                [3.821545, -4.447476, -2.473820],
                [1.938878, -0.788768, 6.423543],
                [0.796432, -5.030339, -2.032893],
            ]
        )
        expected_padded_cls = torch.tensor([0.821952, -1.553840, 1.073606])
        for backend in ("eager", "sdpa"):
            with self.subTest(backend=backend), torch.no_grad():
                model.set_attn_implementation(backend)
                output = model(**inputs).last_hidden_state
                self.assertEqual(output.shape, (2, 302, 768))
                torch.testing.assert_close(
                    output[0, [0, 127, 128, 129, 256], :3], expected_slice, rtol=1e-4, atol=1e-4
                )
                torch.testing.assert_close(output[1, 0, :3], expected_padded_cls, rtol=1e-4, atol=1e-4)

    @parameterized.expand(
        [
            (
                "150M",
                "f821af159246ab4ac84669d42533281366bc55ab",
                [
                    [-173.299500, -172.092453, -172.083084, -166.659607, -168.356415],
                    [-166.776337, -165.526764, -165.573578, -160.336792, -161.950073],
                ],
            ),
            (
                "380M",
                "7789071ca161cb99e03ec1006655d24ad8c1b0e1",
                [
                    [-195.506729, -193.607315, -193.664520, -189.095596, -189.571594],
                    [-189.560608, -187.724716, -187.760635, -183.072098, -183.630432],
                ],
            ),
        ]
    )
    def test_inference_masked_lm(self, size, revision, expected_slice):
        model, tokenizer = self.load_checkpoint(AutoModelForMaskedLM, f"dmis-lab/UltraBERT-{size}", revision)
        inputs = tokenizer(
            ["The capital of France is <|mask|>.", "The capital of Germany is <|mask|>."],
            padding=True,
            return_tensors="pt",
        )
        for backend in ("eager", "sdpa"):
            with self.subTest(backend=backend), torch.no_grad():
                model.set_attn_implementation(backend)
                logits = model(**inputs).logits
                self.assertEqual(logits.shape, (2, 8, 49152))
                masked_logits = logits[inputs.input_ids == tokenizer.mask_token_id]
                torch.testing.assert_close(masked_logits[:, :5], torch.tensor(expected_slice), rtol=1e-4, atol=1e-4)
                self.assertEqual(tokenizer.batch_decode(masked_logits.argmax(-1)[:, None]), [" Paris", " Berlin"])

    def test_inference_sequence_classification(self):
        model, tokenizer = self.load_checkpoint(
            AutoModelForSequenceClassification,
            "tomaarsen/ultrabert-150m-reranker-msmarco32k-3e-5-20261014",
            "b8d94a0187a6b14a3ba6707602a346fdb4c4d7d6",
        )
        inputs = tokenizer(
            [
                ["What is the capital of France?", "Paris is the capital of France."],
                ["What is the capital of France?", "Berlin is the capital of Germany and its largest city."],
            ],
            padding=True,
            return_tensors="pt",
        )
        expected = torch.tensor([[1.918486238], [-3.480727434]])
        for backend in ("eager", "sdpa"):
            with self.subTest(backend=backend), torch.no_grad():
                model.set_attn_implementation(backend)
                logits = model(**inputs).logits
                self.assertEqual(logits.shape, (2, 1))
                torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
