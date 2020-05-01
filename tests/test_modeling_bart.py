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

import timeout_decorator  # noqa

from transformers import is_torch_available

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor
from .utils import require_torch, slow, torch_device


if is_torch_available():
    import torch
    from transformers import (
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BartModel,
        BartForConditionalGeneration,
        BartForSequenceClassification,
        BartConfig,
        BartTokenizer,
        MBartTokenizer,
    )
    from transformers.modeling_bart import (
        BART_PRETRAINED_MODEL_ARCHIVE_MAP,
        shift_tokens_right,
        invert_mask,
        _prepare_bart_decoder_inputs,
        SinusoidalPositionalEmbedding,
    )


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
        self.hidden_size = 16
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.intermediate_size = 4
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 20
        self.eos_token_id = 2
        self.pad_token_id = 1
        self.bos_token_id = 0
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
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
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
    all_model_classes = (
        (BartModel, BartForConditionalGeneration, BartForSequenceClassification) if is_torch_available() else ()
    )
    all_generative_model_classes = (BartForConditionalGeneration,) if is_torch_available() else ()
    is_encoder_decoder = True
    # TODO(SS): fix the below in a separate PR
    test_pruning = False
    test_torchscript = False
    test_head_masking = False
    test_resize_embeddings = True  # This requires inputs_dict['input_ids']
    test_missing_keys = False  # because BartForConditionalGeneration and BartModel now have identical state_dict

    def setUp(self):
        self.model_tester = ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BartConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_initialization_more(self):
        # (config, input_ids, token_type_ids, input_mask, *unused) = \
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
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

    def test_advanced_inputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict["input_ids"][:, -2:] = config.pad_token_id
        decoder_input_ids, decoder_attn_mask, causal_mask = _prepare_bart_decoder_inputs(
            config, inputs_dict["input_ids"]
        )
        model = BartModel(config).to(torch_device).eval()

        decoder_features_with_created_mask = model(**inputs_dict)[0]
        decoder_features_with_passed_mask = model(
            decoder_attention_mask=invert_mask(decoder_attn_mask), decoder_input_ids=decoder_input_ids, **inputs_dict
        )[0]
        _assert_tensors_equal(decoder_features_with_passed_mask, decoder_features_with_created_mask)
        useless_mask = torch.zeros_like(decoder_attn_mask)
        decoder_features = model(decoder_attention_mask=useless_mask, **inputs_dict)[0]
        self.assertTrue(isinstance(decoder_features, torch.Tensor))  # no hidden states or attentions
        self.assertEqual(
            decoder_features.size(), (self.model_tester.batch_size, self.model_tester.seq_length, config.d_model)
        )
        if decoder_attn_mask.min().item() < -1e3:  # some tokens were masked
            self.assertFalse((decoder_features_with_created_mask == decoder_features).all().item())

        # Test different encoder attention masks
        decoder_features_with_long_encoder_mask = model(
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

    def test_tiny_model(self):
        model_name = "sshleifer/bart-tiny-random"
        tiny = AutoModel.from_pretrained(model_name)  # same vocab size
        tok = AutoTokenizer.from_pretrained(model_name)  # same tokenizer
        inputs_dict = tok.batch_encode_plus(["Hello my friends"], return_tensors="pt")

        with torch.no_grad():
            tiny(**inputs_dict)


@require_torch
class BartTranslationTests(unittest.TestCase):
    _model = None

    @classmethod
    def setUpClass(cls):
        checkpoint_name = "mbart-large-en-ro"
        cls.tokenizer = MBartTokenizer.from_pretrained(checkpoint_name)
        cls.pad_token_id = 1
        net_input = {
            "input_ids": _long_tensor(
                [
                    [3493, 3060, 621, 104064, 1810, 100, 142, 566, 13158, 6889, 5, 2, 250004],
                    [64511, 7, 765, 2837, 45188, 297, 4049, 237, 10, 122122, 5, 2, 250004],
                ]
            ),
            "decoder_input_ids": _long_tensor(
                [
                    [250020, 31952, 144, 9019, 242307, 21980, 55749, 11, 5, 2, 1, 1],
                    [250020, 884, 9019, 96, 9, 916, 86792, 36, 18743, 15596, 5, 2],
                ]
            ),
            "generation_mode": False,
        }
        net_input["attention_mask"] = net_input["input_ids"].ne(cls.pad_token_id)
        cls.net_input = net_input

        return cls

    @property
    def model(self):
        """Only load the model if needed."""
        if self._model is None:
            model = BartForConditionalGeneration.from_pretrained("mbart-large-en-ro")
            self._model = model
        return self._model

    @slow
    def test_enro_forward(self):
        model = self.model
        with torch.no_grad():
            logits, *other_stuff = model(**self.net_input)

        expected_slice = torch.tensor([9.0078, 10.1113, 14.4787])
        result_slice = logits[0][0][:3]
        self.assertTrue(torch.allclose(expected_slice, result_slice, atol=TOLERANCE))

    @slow
    def test_enro_generate(self):
        model = self.model
        # example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
        # inputs: dict = tokenizer.batch_encode_plus([example_english_phrase], return_tensors="pt",)
        expected_translation_romanian = "Şeful ONU declară că nu există o soluţie militară în Siria"

        inputs = {
            "input_ids": torch.LongTensor(
                [[8274, 127873, 25916, 7, 8622, 2071, 438, 67485, 53, 187895, 23, 51712, 2]]  # 250004
            )
        }
        translated_tokens = model.generate(input_ids=inputs["input_ids"].to(torch_device), num_beams=5,)
        decoded = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for g in translated_tokens
        ]
        self.assertEqual(expected_translation_romanian, decoded[0])

    def test_mbart_enro_config(self):
        mbart_models = ["mbart-large-en-ro"]
        expected = {"scale_embedding": True, "output_past": True}
        for name in mbart_models:
            config = BartConfig.from_pretrained(name)
            self.assertTrue(config.is_valid_mbart())
            for k, v in expected.items():
                try:
                    self.assertEqual(v, getattr(config, k))
                except AssertionError as e:
                    e.args += (name, k)
                    raise

    def test_enro_tokenizer(self):
        raw = "UN Chief Says There Is No Military Solution in Syria"
        ids = self.tokenizer.batch_encode_plus([raw])["input_ids"][0]
        expected_result = [0, 8274, 127873, 25916, 7, 8622, 2071, 438, 67485, 53, 187895, 23, 51712, 2]
        # TODO(SS): should be  [8274, ..., 2, 250020]
        self.assertListEqual(expected_result, ids)

    def test_mbart_fast_forward(self):
        config = BartConfig(
            vocab_size=99,
            d_model=24,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_position_embeddings=48,
            add_final_layer_norm=True,
        )
        lm_model = BartForConditionalGeneration(config).to(torch_device)
        context = torch.Tensor([[71, 82, 18, 33, 46, 91, 2], [68, 34, 26, 58, 30, 2, 1]]).long().to(torch_device)
        summary = torch.Tensor([[82, 71, 82, 18, 2], [58, 68, 2, 1, 1]]).long().to(torch_device)
        loss, logits, enc_features = lm_model(input_ids=context, decoder_input_ids=summary, lm_labels=summary)
        expected_shape = (*summary.shape, config.vocab_size)
        self.assertEqual(logits.shape, expected_shape)


@require_torch
class BartHeadTests(unittest.TestCase):
    vocab_size = 99

    def _get_config_and_data(self):
        input_ids = torch.tensor(
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
            ],
            dtype=torch.long,
            device=torch_device,
        )

        batch_size = input_ids.shape[0]
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
            eos_token_id=2,
            pad_token_id=1,
            bos_token_id=0,
        )
        return config, input_ids, batch_size

    def test_sequence_classification_forward(self):
        config, input_ids, batch_size = self._get_config_and_data()
        labels = _long_tensor([2] * batch_size).to(torch_device)
        model = BartForSequenceClassification(config)
        model.to(torch_device)
        outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=labels)
        logits = outputs[1]
        expected_shape = torch.Size((batch_size, config.num_labels))
        self.assertEqual(logits.shape, expected_shape)
        loss = outputs[0]
        self.assertIsInstance(loss.item(), float)

    @timeout_decorator.timeout(1)
    def test_lm_forward(self):
        config, input_ids, batch_size = self._get_config_and_data()
        lm_labels = ids_tensor([batch_size, input_ids.shape[1]], self.vocab_size).to(torch_device)
        lm_model = BartForConditionalGeneration(config)
        lm_model.to(torch_device)
        loss, logits, enc_features = lm_model(input_ids=input_ids, lm_labels=lm_labels)
        expected_shape = (batch_size, input_ids.shape[1], config.vocab_size)
        self.assertEqual(logits.shape, expected_shape)
        self.assertIsInstance(loss.item(), float)

    def test_lm_uneven_forward(self):
        config = BartConfig(
            vocab_size=self.vocab_size,
            d_model=14,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=8,
            decoder_ffn_dim=8,
            max_position_embeddings=48,
        )
        lm_model = BartForConditionalGeneration(config).to(torch_device)
        context = torch.Tensor([[71, 82, 18, 33, 46, 91, 2], [68, 34, 26, 58, 30, 2, 1]]).long().to(torch_device)
        summary = torch.Tensor([[82, 71, 82, 18, 2], [58, 68, 2, 1, 1]]).long().to(torch_device)
        loss, logits, enc_features = lm_model(input_ids=context, decoder_input_ids=summary, lm_labels=summary)
        expected_shape = (*summary.shape, config.vocab_size)
        self.assertEqual(logits.shape, expected_shape)

    def test_generate_beam_search(self):
        input_ids = torch.Tensor([[71, 82, 2], [68, 34, 2]]).long().to(torch_device)
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
            eos_token_id=2,
            pad_token_id=1,
            bos_token_id=0,
        )
        lm_model = BartForConditionalGeneration(config).to(torch_device)
        lm_model.eval()

        max_length = 5
        new_input_ids = lm_model.generate(
            input_ids.clone(),
            do_sample=True,
            num_return_sequences=1,
            num_beams=2,
            no_repeat_ngram_size=3,
            max_length=max_length,
        )
        self.assertEqual(new_input_ids.shape, (input_ids.shape[0], max_length))
        # TODO(SS): uneven length batches, empty inputs

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

    @unittest.skipIf(torch_device == "cpu", "Cant do half precision")
    def test_generate_fp16(self):
        config, input_ids, batch_size = self._get_config_and_data()
        attention_mask = input_ids.ne(1).to(torch_device)
        model = BartForConditionalGeneration(config).eval().to(torch_device).half()
        model.generate(input_ids, attention_mask=attention_mask, do_sample=False, early_stopping=True)

    @unittest.skipIf(torch_device == "cpu", "Cant do half precision")
    def test_base_model_fp16(self):
        config, input_ids, batch_size = self._get_config_and_data()
        attention_mask = input_ids.ne(1).to(torch_device)
        lm_model = BartForConditionalGeneration(config).eval().to(torch_device).half()
        lm_model(input_ids, attention_mask=attention_mask)

    def test_default_generate_kwargs(self):
        config, input_ids, _ = self._get_config_and_data()
        model = BartForConditionalGeneration(config).eval().to(torch_device)
        model.generate(input_ids)
        model.generate(num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)

    def test_dummy_inputs(self):
        config, *_ = self._get_config_and_data()
        model = BartForConditionalGeneration(config).eval().to(torch_device)
        model(**model.dummy_inputs)

    def test_prepare_bart_decoder_inputs(self):
        config, *_ = self._get_config_and_data()
        input_ids = _long_tensor(([4, 4, 2]))
        decoder_input_ids = _long_tensor([[26388, 2, config.pad_token_id]])
        ignore = float("-inf")
        decoder_input_ids, decoder_attn_mask, causal_mask = _prepare_bart_decoder_inputs(
            config, input_ids, decoder_input_ids
        )
        expected_causal_mask = torch.tensor(
            [[0, ignore, ignore], [0, 0, ignore], [0, 0, 0]]  # never attend to the final token, because its pad
        ).to(input_ids.device)
        self.assertEqual(decoder_attn_mask.size(), decoder_input_ids.size())
        self.assertTrue(torch.eq(expected_causal_mask, causal_mask).all())

    def test_resize_tokens_embeddings_more(self):
        config, input_ids, _ = self._get_config_and_data()

        def _get_embs(m):
            return (m.get_input_embeddings().weight.data.clone(), m.get_output_embeddings().weight.data.clone())

        model = BartForConditionalGeneration(config).eval().to(torch_device)
        input, output = _get_embs(model)
        self.assertTrue(torch.eq(input, output).all())
        new_vocab_size = 45
        model.resize_token_embeddings(new_vocab_size)
        input_new, output_new = _get_embs(model)
        self.assertEqual(input_new.shape, (new_vocab_size, config.d_model))
        self.assertEqual(output_new.shape, (new_vocab_size, config.d_model))
        self.assertTrue(torch.eq(input_new, output_new).all())


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


def _long_tensor(tok_lst):
    return torch.tensor(tok_lst, dtype=torch.long, device=torch_device,)


TOLERANCE = 1e-4


@require_torch
class BartModelIntegrationTests(unittest.TestCase):
    @slow
    def test_inference_no_head(self):
        model = BartModel.from_pretrained("bart-large").to(torch_device)
        input_ids = _long_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        inputs_dict = prepare_bart_inputs_dict(model.config, input_ids)
        with torch.no_grad():
            output = model(**inputs_dict)[0]
        expected_shape = torch.Size((1, 11, 1024))
        self.assertEqual(output.shape, expected_shape)
        expected_slice = torch.tensor(
            [[0.7144, 0.8143, -1.2813], [0.7144, 0.8143, -1.2813], [-0.0467, 2.5911, -2.1845]], device=torch_device
        )
        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=TOLERANCE))

    @slow
    def test_mnli_inference(self):

        example_b = [0, 31414, 232, 328, 740, 1140, 69, 46078, 1588, 2, 1]
        input_ids = _long_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2], example_b])

        model = AutoModelForSequenceClassification.from_pretrained("bart-large-mnli").to(
            torch_device
        )  # eval called in from_pre
        inputs_dict = prepare_bart_inputs_dict(model.config, input_ids)
        # Test that model hasn't changed
        with torch.no_grad():
            batched_logits, features = model(**inputs_dict)
        expected_shape = torch.Size((2, 3))
        self.assertEqual(batched_logits.shape, expected_shape)
        expected_slice = torch.Tensor([[0.1907, 1.4342, -1.0289]]).to(torch_device)
        logits_arr = batched_logits[0].detach()

        # Test that padding does not change results
        input_ids_no_pad = _long_tensor([example_b[:-1]])

        inputs_dict = prepare_bart_inputs_dict(model.config, input_ids=input_ids_no_pad)
        with torch.no_grad():
            logits2 = model(**inputs_dict)[0]
        _assert_tensors_equal(batched_logits[1], logits2, atol=TOLERANCE)
        _assert_tensors_equal(expected_slice, logits_arr, atol=TOLERANCE)

    @unittest.skip("This is just too slow")
    def test_model_from_pretrained(self):
        # Forces 1.6GB download from S3 for each model
        for model_name in list(BART_PRETRAINED_MODEL_ARCHIVE_MAP.keys()):
            model = BartModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @slow
    def test_xsum_summarization_same_as_fairseq(self):
        model = BartForConditionalGeneration.from_pretrained("bart-large-xsum").to(torch_device)
        self.assertFalse(model.config.is_valid_mbart())
        tok = BartTokenizer.from_pretrained("bart-large")

        PGE_ARTICLE = """ PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."""
        EXPECTED_SUMMARY = "California's largest power company has begun shutting off power to tens of thousands of homes and businesses in the state."
        dct = tok.batch_encode_plus([PGE_ARTICLE], max_length=1024, pad_to_max_length=True, return_tensors="pt",)

        hypotheses_batch = model.generate(
            input_ids=dct["input_ids"].to(torch_device),
            attention_mask=dct["attention_mask"].to(torch_device),
            num_beams=2,
            max_length=62,
            min_length=11,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
            decoder_start_token_id=model.config.eos_token_id,
        )

        decoded = [
            tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in hypotheses_batch
        ]
        self.assertEqual(EXPECTED_SUMMARY, decoded[0])

    def test_xsum_config_generation_params(self):
        config = BartConfig.from_pretrained("bart-large-xsum")
        expected_params = dict(num_beams=6, do_sample=False, early_stopping=True, length_penalty=1.0)
        config_params = {k: getattr(config, k, "MISSING") for k, v in expected_params.items()}
        self.assertDictEqual(expected_params, config_params)

    @slow
    def test_cnn_summarization_same_as_fairseq(self):
        hf = BartForConditionalGeneration.from_pretrained("bart-large-cnn").to(torch_device)
        tok = BartTokenizer.from_pretrained("bart-large")

        FRANCE_ARTICLE = ' Marseille, France (CNN)The French prosecutor leading an investigation into the crash of Germanwings Flight 9525 insisted Wednesday that he was not aware of any video footage from on board the plane. Marseille prosecutor Brice Robin told CNN that "so far no videos were used in the crash investigation." He added, "A person who has such a video needs to immediately give it to the investigators." Robin\'s comments follow claims by two magazines, German daily Bild and French Paris Match, of a cell phone video showing the harrowing final seconds from on board Germanwings Flight 9525 as it crashed into the French Alps. All 150 on board were killed. Paris Match and Bild reported that the video was recovered from a phone at the wreckage site. The two publications described the supposed video, but did not post it on their websites. The publications said that they watched the video, which was found by a source close to the investigation. "One can hear cries of \'My God\' in several languages," Paris Match reported. "Metallic banging can also be heard more than three times, perhaps of the pilot trying to open the cockpit door with a heavy object.  Towards the end, after a heavy shake, stronger than the others, the screaming intensifies. Then nothing." "It is a very disturbing scene," said Julian Reichelt, editor-in-chief of Bild online. An official with France\'s accident investigation agency, the BEA, said the agency is not aware of any such video. Lt. Col. Jean-Marc Menichini, a French Gendarmerie spokesman in charge of communications on rescue efforts around the Germanwings crash site, told CNN that the reports were "completely wrong" and "unwarranted." Cell phones have been collected at the site, he said, but that they "hadn\'t been exploited yet." Menichini said he believed the cell phones would need to be sent to the Criminal Research Institute in Rosny sous-Bois, near Paris, in order to be analyzed by specialized technicians working hand-in-hand with investigators. But none of the cell phones found so far have been sent to the institute, Menichini said. Asked whether staff involved in the search could have leaked a memory card to the media, Menichini answered with a categorical "no." Reichelt told "Erin Burnett: Outfront" that he had watched the video and stood by the report, saying Bild and Paris Match are "very confident" that the clip is real. He noted that investigators only revealed they\'d recovered cell phones from the crash site after Bild and Paris Match published their reports. "That is something we did not know before. ... Overall we can say many things of the investigation weren\'t revealed by the investigation at the beginning," he said. What was mental state of Germanwings co-pilot? German airline Lufthansa confirmed Tuesday that co-pilot Andreas Lubitz had battled depression years before he took the controls of Germanwings Flight 9525, which he\'s accused of deliberately crashing last week in the French Alps. Lubitz told his Lufthansa flight training school in 2009 that he had a "previous episode of severe depression," the airline said Tuesday. Email correspondence between Lubitz and the school discovered in an internal investigation, Lufthansa said, included medical documents he submitted in connection with resuming his flight training. The announcement indicates that Lufthansa, the parent company of Germanwings, knew of Lubitz\'s battle with depression, allowed him to continue training and ultimately put him in the cockpit. Lufthansa, whose CEO Carsten Spohr previously said Lubitz was 100% fit to fly, described its statement Tuesday as a "swift and seamless clarification" and said it was sharing the information and documents -- including training and medical records -- with public prosecutors. Spohr traveled to the crash site Wednesday, where recovery teams have been working for the past week to recover human remains and plane debris scattered across a steep mountainside. He saw the crisis center set up in Seyne-les-Alpes, laid a wreath in the village of Le Vernet, closer to the crash site, where grieving families have left flowers at a simple stone memorial. Menichini told CNN late Tuesday that no visible human remains were left at the site but recovery teams would keep searching. French President Francois Hollande, speaking Tuesday, said that it should be possible to identify all the victims using DNA analysis by the end of the week, sooner than authorities had previously suggested. In the meantime, the recovery of the victims\' personal belongings will start Wednesday, Menichini said. Among those personal belongings could be more cell phones belonging to the 144 passengers and six crew on board. Check out the latest from our correspondents . The details about Lubitz\'s correspondence with the flight school during his training were among several developments as investigators continued to delve into what caused the crash and Lubitz\'s possible motive for downing the jet. A Lufthansa spokesperson told CNN on Tuesday that Lubitz had a valid medical certificate, had passed all his examinations and "held all the licenses required." Earlier, a spokesman for the prosecutor\'s office in Dusseldorf, Christoph Kumpa, said medical records reveal Lubitz suffered from suicidal tendencies at some point before his aviation career and underwent psychotherapy before he got his pilot\'s license. Kumpa emphasized there\'s no evidence suggesting Lubitz was suicidal or acting aggressively before the crash. Investigators are looking into whether Lubitz feared his medical condition would cause him to lose his pilot\'s license, a European government official briefed on the investigation told CNN on Tuesday. While flying was "a big part of his life," the source said, it\'s only one theory being considered. Another source, a law enforcement official briefed on the investigation, also told CNN that authorities believe the primary motive for Lubitz to bring down the plane was that he feared he would not be allowed to fly because of his medical problems. Lubitz\'s girlfriend told investigators he had seen an eye doctor and a neuropsychologist, both of whom deemed him unfit to work recently and concluded he had psychological issues, the European government official said. But no matter what details emerge about his previous mental health struggles, there\'s more to the story, said Brian Russell, a forensic psychologist. "Psychology can explain why somebody would turn rage inward on themselves about the fact that maybe they weren\'t going to keep doing their job and they\'re upset about that and so they\'re suicidal," he said. "But there is no mental illness that explains why somebody then feels entitled to also take that rage and turn it outward on 149 other people who had nothing to do with the person\'s problems." Germanwings crash compensation: What we know . Who was the captain of Germanwings Flight 9525? CNN\'s Margot Haddad reported from Marseille and Pamela Brown from Dusseldorf, while Laura Smith-Spark wrote from London. CNN\'s Frederik Pleitgen, Pamela Boykoff, Antonia Mortensen, Sandrine Amiel and Anna-Maja Rappard contributed to this report.'  # @noqa
        EXPECTED_SUMMARY_FRANCE = 'French prosecutor says he\'s not aware of any video footage from on board the plane. German daily Bild and French Paris Match claim to have found a cell phone video of the crash. A French Gendarmerie spokesman calls the reports "completely wrong" and "unwarranted" German airline Lufthansa confirms co-pilot Andreas Lubitz had battled depression.'

        SHORTER_ARTICLE = ' (CNN)The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based. The Palestinians signed the ICC\'s founding Rome Statute in January, when they also accepted its jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including East Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination into the situation in Palestinian territories, paving the way for possible war crimes investigations against Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and the United States, neither of which is an ICC member, opposed the Palestinians\' efforts to join the body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday\'s ceremony, said it was a move toward greater justice. "As Palestine formally becomes a State Party to the Rome Statute today, the world is also a step closer to ending a long era of impunity and injustice," he said, according to an ICC news release. "Indeed, today brings us closer to our shared goals of justice and peace." Judge Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the Palestinians. "As the Rome Statute today enters into force for the State of Palestine, Palestine acquires all the rights as well as responsibilities that come with being a State Party to the Statute. These are substantive commitments, which cannot be taken lightly," she said. Rights group Human Rights Watch welcomed the development. "Governments seeking to penalize Palestine for joining the ICC should immediately end their pressure, and countries that support universal acceptance of the court\'s treaty should speak out to welcome its membership," said Balkees Jarrah, international justice counsel for the group. "What\'s objectionable is the attempts to undermine international justice, not Palestine\'s decision to join a treaty to which over 100 countries around the world are members." In January, when the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described it as an outrage, saying the court was overstepping its boundaries. The United States also said it "strongly" disagreed with the court\'s decision. "As we have said repeatedly, we do not believe that Palestine is a state and therefore we do not believe that it is eligible to join the ICC," the State Department said in a statement. It urged the warring sides to resolve their differences through direct negotiations. "We will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace," it said. But the ICC begs to differ with the definition of a state for its purposes and refers to the territories as "Palestine." While a preliminary examination is not a formal investigation, it allows the court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou Bensouda said her office would "conduct its analysis in full independence and impartiality." The war between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The inquiry will include alleged war crimes committed since June. The International Criminal Court was set up in 2002 to prosecute genocide, crimes against humanity and war crimes. CNN\'s Vasco Cotovio, Kareem Khadder and Faith Karimi contributed to this report.'
        EXPECTED_SUMMARY_SHORTER = "The Palestinian Authority becomes the 123rd member of the International Criminal Court. The move gives the court jurisdiction over alleged crimes in Palestinian territories. Israel and the United States opposed the Palestinians' efforts to join the body. But Palestinian Foreign Minister Riad al-Malki said it was a move toward greater justice."

        # The below article tests that we don't add any hypotheses outside of the top n_beams
        IRAN_ARTICLE = " (CNN)The United States and its negotiating partners reached a very strong framework agreement with Iran in Lausanne, Switzerland, on Thursday that limits Iran's nuclear program in such a way as to effectively block it from building a nuclear weapon. Expect pushback anyway, if the recent past is any harbinger. Just last month, in an attempt to head off such an agreement, House Speaker John Boehner invited Israeli Prime Minister Benjamin Netanyahu to preemptively blast it before Congress, and 47 senators sent a letter to the Iranian leadership warning them away from a deal. The debate that has already begun since the announcement of the new framework will likely result in more heat than light. It will not be helped by the gathering swirl of dubious assumptions and doubtful assertions. Let us address some of these: . The most misleading assertion, despite universal rejection by experts, is that the negotiations' objective at the outset was the total elimination of any nuclear program in Iran. That is the position of Netanyahu and his acolytes in the U.S. Congress. But that is not and never was the objective. If it had been, there would have been no Iranian team at the negotiating table. Rather, the objective has always been to structure an agreement or series of agreements so that Iran could not covertly develop a nuclear arsenal before the United States and its allies could respond. The new framework has exceeded expectations in achieving that goal. It would reduce Iran's low-enriched uranium stockpile, cut by two-thirds its number of installed centrifuges and implement a rigorous inspection regime. Another dubious assumption of opponents is that the Iranian nuclear program is a covert weapons program. Despite sharp accusations by some in the United States and its allies, Iran denies having such a program, and U.S. intelligence contends that Iran has not yet made the decision to build a nuclear weapon. Iran's continued cooperation with International Atomic Energy Agency inspections is further evidence on this point, and we'll know even more about Iran's program in the coming months and years because of the deal. In fact, the inspections provisions that are part of this agreement are designed to protect against any covert action by the Iranians. What's more, the rhetoric of some members of Congress has implied that the negotiations have been between only the United States and Iran (i.e., the 47 senators' letter warning that a deal might be killed by Congress or a future president). This of course is not the case. The talks were between Iran and the five permanent members of the U.N. Security Council (United States, United Kingdom, France, China and Russia) plus Germany, dubbed the P5+1. While the United States has played a leading role in the effort, it negotiated the terms alongside its partners. If the agreement reached by the P5+1 is rejected by Congress, it could result in an unraveling of the sanctions on Iran and threaten NATO cohesion in other areas. Another questionable assertion is that this agreement contains a sunset clause, after which Iran will be free to do as it pleases. Again, this is not the case. Some of the restrictions on Iran's nuclear activities, such as uranium enrichment, will be eased or eliminated over time, as long as 15 years. But most importantly, the framework agreement includes Iran's ratification of the Additional Protocol, which allows IAEA inspectors expanded access to nuclear sites both declared and nondeclared. This provision will be permanent. It does not sunset. Thus, going forward, if Iran decides to enrich uranium to weapons-grade levels, monitors will be able to detect such a move in a matter of days and alert the U.N. Security Council. Many in Congress have said that the agreement should be a formal treaty requiring the Senate to \"advise and consent.\" But the issue is not suited for a treaty. Treaties impose equivalent obligations on all signatories. For example, the New START treaty limits Russia and the United States to 1,550 deployed strategic warheads. But any agreement with Iran will not be so balanced.  The restrictions and obligations in the final framework agreement will be imposed almost exclusively on Iran. The P5+1 are obligated only to ease and eventually remove most but not all economic sanctions, which were imposed as leverage to gain this final deal. Finally some insist that any agreement must address Iranian missile programs, human rights violations or support for Hamas or Hezbollah.  As important as these issues are, and they must indeed be addressed, they are unrelated to the most important aim of a nuclear deal: preventing a nuclear Iran.  To include them in the negotiations would be a poison pill. This agreement should be judged on its merits and on how it affects the security of our negotiating partners and allies, including Israel. Those judgments should be fact-based, not based on questionable assertions or dubious assumptions."
        EXPECTED_SUMMARY_IRAN = "The U.S. and its negotiating partners reached a very strong framework agreement with Iran. Peter Bergen: The debate that has already begun will likely result in more heat than light. He says the agreement limits Iran's nuclear program in such a way as to effectively block it from building a nuclear weapon. Bergen says the most important aim of a nuclear deal is preventing a nuclear Iran."

        ARTICLE_SUBWAY = ' New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.  Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other. In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage. Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the 2010 marriage license application, according to court documents. Prosecutors said the marriages were part of an immigration scam. On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further. After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.  All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say. Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.  Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted. The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali. Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force. If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.'
        EXPECTED_SUMMARY_SUBWAY = "Liana Barrientos has been married 10 times, sometimes within two weeks of each other. Prosecutors say the marriages were part of an immigration scam. On Friday, she pleaded not guilty at State Supreme Court in the Bronx. She was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the subway."

        dct = tok.batch_encode_plus(
            [FRANCE_ARTICLE, SHORTER_ARTICLE, IRAN_ARTICLE, ARTICLE_SUBWAY],
            max_length=1024,
            pad_to_max_length=True,
            return_tensors="pt",
        )

        max_length = 140
        min_length = 55

        self.assertEqual(1024, dct["input_ids"].shape[1])
        hypotheses_batch = hf.generate(
            input_ids=dct["input_ids"].to(torch_device),
            attention_mask=dct["attention_mask"].to(torch_device),
            num_beams=4,
            length_penalty=2.0,
            max_length=max_length + 2,
            min_length=min_length + 1,
            no_repeat_ngram_size=3,
            do_sample=False,
            early_stopping=True,
            decoder_start_token_id=hf.config.eos_token_id,
        )

        decoded = [
            tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in hypotheses_batch
        ]

        self.assertListEqual(
            [EXPECTED_SUMMARY_FRANCE, EXPECTED_SUMMARY_SHORTER, EXPECTED_SUMMARY_IRAN, EXPECTED_SUMMARY_SUBWAY],
            decoded,
        )
        # TODO(SS): run fairseq again with num_beams=2, min_len=20.
        # TODO(SS): add test case that hits max_length


@require_torch
class TestSinusoidalPositionalEmbeddings(unittest.TestCase):
    desired_weights = [
        [0, 0, 0, 0, 0],
        [0.84147096, 0.82177866, 0.80180490, 0.78165019, 0.76140374],
        [0.90929741, 0.93651021, 0.95829457, 0.97505713, 0.98720258],
    ]

    def test_positional_emb_cache_logic(self):
        pad = 1
        input_ids = torch.tensor([[4, 10]], dtype=torch.long, device=torch_device)
        emb1 = SinusoidalPositionalEmbedding(num_positions=32, embedding_dim=6, padding_idx=pad).to(torch_device)
        no_cache = emb1(input_ids, use_cache=False)
        yes_cache = emb1(input_ids, use_cache=True)
        self.assertEqual((1, 1, 6), yes_cache.shape)  # extra dim to allow broadcasting, feel free to delete!
        self.assertListEqual(no_cache[-1].tolist(), yes_cache[0][0].tolist())

    def test_odd_embed_dim(self):
        with self.assertRaises(NotImplementedError):
            SinusoidalPositionalEmbedding(num_positions=4, embedding_dim=5, padding_idx=0).to(torch_device)

        # odd num_positions is allowed
        SinusoidalPositionalEmbedding(num_positions=5, embedding_dim=4, padding_idx=0).to(torch_device)

    def test_positional_emb_weights_against_marian(self):
        pad = 1
        emb1 = SinusoidalPositionalEmbedding(num_positions=512, embedding_dim=512, padding_idx=pad).to(torch_device)
        weights = emb1.weight.data[:3, :5].tolist()
        for i, (expected_weight, actual_weight) in enumerate(zip(self.desired_weights, weights)):
            for j in range(5):
                self.assertAlmostEqual(expected_weight[j], actual_weight[j], places=3)

        # test that forward pass is just a lookup, there is no ignore padding logic
        input_ids = torch.tensor([[4, 10, pad, pad, pad]], dtype=torch.long, device=torch_device)
        no_cache_pad_zero = emb1(input_ids)
        self.assertTrue(torch.allclose(torch.Tensor(self.desired_weights), no_cache_pad_zero[:3, :5], atol=1e-3))
