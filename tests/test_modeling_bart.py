# coding=utf-8
# Copyright 2020 Huggingface
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


import tempfile
import unittest

from transformers import is_torch_available

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor
from .utils import CACHE_DIR, require_torch, slow, torch_device


if is_torch_available():
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        BartModel,
        BartForMaskedLM,
        BartForSequenceClassification,
        BartConfig,
    )
    from transformers.modeling_bart import (
        BART_PRETRAINED_MODEL_ARCHIVE_MAP,
        shift_tokens_right,
        _prepare_bart_decoder_inputs,
    )
    from transformers.tokenization_bart import BartTokenizer


@require_torch
class ModelTester:
    def __init__(
        self, parent,
    ):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_labels = False
        self.vocab_size = 99
        self.hidden_size = 32
        self.num_hidden_layers = 5
        self.num_attention_heads = 4
        self.intermediate_size = 37
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 12
        torch.manual_seed(0)

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).clamp(3,)
        input_ids[:, -1] = 2  # Eos Token

        config = BartConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
        )
        inputs_dict = prepare_bart_inputs_dict(config, input_ids)
        return config, inputs_dict


def prepare_bart_inputs_dict(
    config, input_ids, attention_mask=None,
):
    if attention_mask is None:
        attention_mask = input_ids.ne(config.pad_token_id)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


@require_torch
class BARTModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (BartModel, BartForMaskedLM, BartForSequenceClassification) if is_torch_available() else ()
    is_encoder_decoder = True
    # TODO(SS): fix the below in a separate PR
    test_pruning = False
    test_torchscript = False
    test_head_masking = False
    test_resize_embeddings = False  # This requires inputs_dict['input_ids']

    def setUp(self):
        self.model_tester = ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BartConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_advanced_inputs(self):
        # (config, input_ids, token_type_ids, input_mask, *unused) = \
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        decoder_input_ids, decoder_attn_mask = _prepare_bart_decoder_inputs(config, inputs_dict["input_ids"])
        model = BartModel(config)
        model.to(torch_device)
        model.eval()
        # test init
        self.assertTrue((model.encoder.embed_tokens.weight == model.shared.weight).all().item())

        def _check_var(module):
            """Check that we initialized various parameters from N(0, config.init_std)."""
            self.assertAlmostEqual(torch.std(module.weight).item(), config.init_std, 2)

        _check_var(model.encoder.embed_tokens)
        _check_var(model.encoder.layers[0].self_attn.k_proj)
        _check_var(model.encoder.layers[0].fc1)
        _check_var(model.encoder.embed_positions)

        decoder_features_with_created_mask = model.forward(**inputs_dict)[0]
        decoder_features_with_passed_mask = model.forward(
            decoder_attention_mask=decoder_attn_mask, decoder_input_ids=decoder_input_ids, **inputs_dict
        )[0]
        _assert_tensors_equal(decoder_features_with_passed_mask, decoder_features_with_created_mask)
        useless_mask = torch.zeros_like(decoder_attn_mask)
        decoder_features = model.forward(decoder_attention_mask=useless_mask, **inputs_dict)[0]
        self.assertTrue(isinstance(decoder_features, torch.Tensor))  # no hidden states or attentions
        self.assertEqual(
            decoder_features.size(), (self.model_tester.batch_size, self.model_tester.seq_length, config.d_model)
        )
        if decoder_attn_mask.min().item() < -1e3:  # some tokens were masked
            self.assertFalse((decoder_features_with_created_mask == decoder_features).all().item())

        # Test different encoder attention masks
        decoder_features_with_long_encoder_mask = model.forward(
            inputs_dict["input_ids"], attention_mask=inputs_dict["attention_mask"].long()
        )[0]
        _assert_tensors_equal(decoder_features_with_long_encoder_mask, decoder_features_with_created_mask)

    def test_save_load_strict(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model2, info = model_class.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertEqual(info["missing_keys"], [])

    @unittest.skip("Passing inputs_embeds not implemented for Bart.")
    def test_inputs_embeds(self):
        pass


@require_torch
class BartHeadTests(unittest.TestCase):

    vocab_size = 99

    def test_lm_forward(self):
        input_ids = torch.Tensor(
            [
                [71, 82, 18, 33, 46, 91, 2],
                [68, 34, 26, 58, 30, 82, 2],
                [5, 97, 17, 39, 94, 40, 2],
                [76, 83, 94, 25, 70, 78, 2],
                [87, 59, 41, 35, 48, 66, 2],
                [55, 13, 16, 58, 5, 2, 1],  # note padding
                [64, 27, 31, 51, 12, 75, 2],
                [52, 64, 86, 17, 83, 39, 2],
                [48, 61, 9, 24, 71, 82, 2],
                [26, 1, 60, 48, 22, 13, 2],
                [21, 5, 62, 28, 14, 76, 2],
                [45, 98, 37, 86, 59, 48, 2],
                [70, 70, 50, 9, 28, 0, 2],
            ]
        ).long()
        batch_size = input_ids.shape[0]
        decoder_lm_labels = ids_tensor([batch_size, input_ids.shape[1]], self.vocab_size)

        config = BartConfig(
            vocab_size=self.vocab_size,
            d_model=24,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_position_embeddings=48,
        )
        model = BartForSequenceClassification(config)
        outputs = model.forward(input_ids=input_ids, decoder_input_ids=input_ids)
        logits = outputs[0]
        expected_shape = torch.Size((batch_size, config.num_labels))
        self.assertEqual(logits.shape, expected_shape)

        lm_model = BartForMaskedLM(config)
        loss, logits, enc_features = lm_model.forward(
            input_ids=input_ids, lm_labels=decoder_lm_labels, decoder_input_ids=input_ids
        )
        expected_shape = (batch_size, input_ids.shape[1], config.vocab_size)
        self.assertEqual(logits.shape, expected_shape)
        self.assertIsInstance(loss.item(), float)

    def test_lm_uneven_forward(self):
        config = BartConfig(
            vocab_size=self.vocab_size,
            d_model=24,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_position_embeddings=48,
        )
        lm_model = BartForMaskedLM(config)
        context = torch.Tensor([[71, 82, 18, 33, 46, 91, 2], [68, 34, 26, 58, 30, 2, 1]]).long()
        summary = torch.Tensor([[82, 71, 82, 18, 2], [58, 68, 2, 1, 1]]).long()
        logits, enc_features = lm_model.forward(input_ids=context, decoder_input_ids=summary)
        expected_shape = (*summary.shape, config.vocab_size)
        self.assertEqual(logits.shape, expected_shape)

    def test_generate(self):
        input_ids = torch.Tensor([[71, 82, 2], [68, 34, 2]]).long()
        config = BartConfig(
            vocab_size=self.vocab_size,
            d_model=24,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_position_embeddings=48,
            output_past=True,
        )
        lm_model = BartForMaskedLM(config)
        lm_model.eval()
        new_input_ids = lm_model.generate(input_ids)
        self.assertEqual(new_input_ids.shape, (input_ids.shape[0], 20))

    def test_shift_tokens_right(self):
        input_ids = torch.Tensor([[71, 82, 18, 33, 2, 1, 1], [68, 34, 26, 58, 30, 82, 2]]).long()
        shifted = shift_tokens_right(input_ids, 1)
        n_pad_before = input_ids.eq(1).float().sum()
        n_pad_after = shifted.eq(1).float().sum()
        self.assertEqual(shifted.shape, input_ids.shape)
        self.assertEqual(n_pad_after, n_pad_before - 1)
        self.assertTrue(torch.eq(shifted[:, 0], 2).all())

    @slow
    def test_tokenization(self):
        tokenizer = BartTokenizer.from_pretrained("bart-large")
        examples = [" Hello world", " DomDramg"]  # need leading spaces for equality
        fairseq_results = [
            torch.Tensor([0, 20920, 232, 2]),
            torch.Tensor([0, 11349, 495, 4040, 571, 2]),
        ]
        for ex, desired_result in zip(examples, fairseq_results):
            bart_toks = tokenizer.encode(ex, return_tensors="pt")
            _assert_tensors_equal(desired_result.long(), bart_toks, prefix=ex)


def _assert_tensors_equal(a, b, atol=1e-12, prefix=""):
    """If tensors not close, or a and b arent both tensors, raise a nice Assertion error."""
    if a is None and b is None:
        return True
    try:
        if torch.allclose(a, b, atol=atol):
            return True
        raise
    except Exception:
        msg = "{} != {}".format(a, b)
        if prefix:
            msg = prefix + ": " + msg
        raise AssertionError(msg)


TOLERANCE = 1e-4


@require_torch
class BartModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head(self):
        model = BartModel.from_pretrained("bart-large")
        input_ids = torch.Tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]]).long()
        inputs_dict = prepare_bart_inputs_dict(model.config, input_ids)
        with torch.no_grad():
            output = model.forward(**inputs_dict)[0]
        expected_shape = torch.Size((1, 11, 1024))
        self.assertEqual(output.shape, expected_shape)
        expected_slice = torch.Tensor(
            [[0.7144, 0.8143, -1.2813], [0.7144, 0.8143, -1.2813], [-0.0467, 2.5911, -2.1845]]
        )
        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=TOLERANCE))

    @slow
    def test_mnli_inference(self):

        example_b = [0, 31414, 232, 328, 740, 1140, 69, 46078, 1588, 2, 1]
        input_ids = torch.Tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2], example_b]).long()

        model = AutoModelForSequenceClassification.from_pretrained("bart-large-mnli")  # eval called in from_pre
        inputs_dict = prepare_bart_inputs_dict(model.config, input_ids)
        # Test that model hasn't changed
        with torch.no_grad():
            batched_logits, features = model.forward(**inputs_dict)
        expected_shape = torch.Size((2, 3))
        self.assertEqual(batched_logits.shape, expected_shape)
        expected_slice = torch.Tensor([[0.1907, 1.4342, -1.0289]])
        logits_arr = batched_logits[0].detach()

        # Test that padding does not change results
        input_ids_no_pad = torch.Tensor([example_b[:-1]]).long()

        inputs_dict = prepare_bart_inputs_dict(model.config, input_ids=input_ids_no_pad)
        with torch.no_grad():
            logits2 = model.forward(**inputs_dict)[0]
        _assert_tensors_equal(batched_logits[1], logits2, atol=TOLERANCE)
        _assert_tensors_equal(expected_slice, logits_arr, atol=TOLERANCE)

    @unittest.skip("This is just too slow")
    def test_model_from_pretrained(self):
        # Forces 1.6GB download from S3 for each model
        for model_name in list(BART_PRETRAINED_MODEL_ARCHIVE_MAP.keys()):
            model = BartModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
            self.assertIsNotNone(model)
