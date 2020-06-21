import unittest

import torch

from transformers import BartConfig, BartForConditionalGeneration, BatchEncoding, MBartTokenizer
from transformers.file_utils import cached_property

from .test_modeling_bart import TOLERANCE, _long_tensor
from .utils import require_torch, slow, torch_device


EN_CODE = 250004


@require_torch
class MBartIntegrationTests(unittest.TestCase):

    checkpoint_name = None

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = MBartTokenizer.from_pretrained(cls.checkpoint_name)
        cls.pad_token_id = 1
        return cls

    @cached_property
    def model(self):
        """Only load the model if needed."""
        model = BartForConditionalGeneration.from_pretrained(self.checkpoint_name).to(torch_device)
        if "cuda" in torch_device:
            model = model.half()
        return model


@require_torch
class MBartEnroIntegrationTest(MBartIntegrationTests):
    checkpoint_name = "sshleifer/mbart-large-en-ro"
    src_text = [
        " UN Chief Says There Is No Military Solution in Syria",
        " I ate lunch twice yesterday",
    ]
    tgt_text = ["Şeful ONU declară că nu există o soluţie militară în Siria", "to be padded"]
    expected_src_tokens = [8274, 127873, 25916, 7, 8622, 2071, 438, 67485, 53, 187895, 23, 51712, 2, EN_CODE]

    @slow
    def test_enro_forward(self):
        model = self.model
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
        net_input["attention_mask"] = net_input["input_ids"].ne(self.pad_token_id)
        with torch.no_grad():
            logits, *other_stuff = model(**net_input)

        expected_slice = torch.tensor([9.0078, 10.1113, 14.4787], device=torch_device, dtype=model.dtype)
        result_slice = logits[0][0][:3]
        self.assertTrue(torch.allclose(expected_slice, result_slice, atol=TOLERANCE))

    @slow
    def test_enro_generate(self):
        inputs: dict = self.tokenizer.prepare_translation_batch([self.src_text[0]]).to(torch_device)
        translated_tokens = self.model.generate(input_ids=inputs["input_ids"].to(torch_device))
        decoded = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        self.assertEqual(self.tgt_text[0], decoded[0])

    def test_mbart_enro_config(self):
        mbart_models = ["facebook/mbart-large-en-ro"]
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
        loss, logits, enc_features = lm_model(input_ids=context, decoder_input_ids=summary, labels=summary)
        expected_shape = (*summary.shape, config.vocab_size)
        self.assertEqual(logits.shape, expected_shape)

    def test_enro_tokenizer_prepare_translation_batch(self):
        batch = self.tokenizer.prepare_translation_batch(
            self.src_text, tgt_texts=self.tgt_text, max_length=len(self.expected_src_tokens),
        )
        self.assertIsInstance(batch, BatchEncoding)

        self.assertEqual((2, 14), batch.input_ids.shape)
        self.assertEqual((2, 14), batch.attention_mask.shape)
        result = batch.input_ids.tolist()[0]
        self.assertListEqual(self.expected_src_tokens, result)
        self.assertEqual(2, batch.decoder_input_ids[0, -2])  # EOS

    def test_enro_tokenizer_batch_encode_plus(self):
        ids = self.tokenizer.batch_encode_plus(self.src_text).input_ids[0]
        self.assertListEqual(self.expected_src_tokens, ids)

    def test_enro_tokenizer_truncation(self):
        src_text = ["this is gunna be a long sentence " * 20]
        assert isinstance(src_text[0], str)
        desired_max_length = 10
        ids = self.tokenizer.prepare_translation_batch(
            src_text, return_tensors=None, max_length=desired_max_length
        ).input_ids[0]
        self.assertEqual(ids[-2], 2)
        self.assertEqual(ids[-1], EN_CODE)
        self.assertEqual(len(ids), desired_max_length)


class MBartCC25IntegrationTest(MBartIntegrationTests):
    checkpoint_name = "sshleifer/mbart-large-cc25"
    src_text = [
        " UN Chief Says There Is No Military Solution in Syria",
        " I ate lunch twice yesterday",
    ]
    tgt_text = ["Şeful ONU declară că nu există o soluţie militară în Siria", "to be padded"]

    @slow
    def test_cc25_generate(self):
        inputs = self.tokenizer.prepare_translation_batch([self.src_text[0]]).to(torch_device)
        translated_tokens = self.model.generate(
            input_ids=inputs["input_ids"].to(torch_device),
            decoder_start_token_id=self.tokenizer.lang_code_to_id["ro_RO"],
        )
        decoded = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        self.assertEqual(self.tgt_text[0], decoded[0])
