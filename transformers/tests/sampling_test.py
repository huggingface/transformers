# coding=utf-8
import sys
import unittest

import numpy as np
import pytest

from transformers import is_torch_available

if is_torch_available():
    import torch

    from transformers import (
        BertConfig,
        BertModel,
        GPT2Config,
        GPT2LMHeadModel,
        OpenAIGPTConfig,
        OpenAIGPTLMHeadModel,
        TransfoXLConfig,
        TransfoXLLMHeadModel,
        XLMConfig,
        XLMWithLMHeadModel,
        XLNetConfig,
        XLNetLMHeadModel,
        Model2Model,
    )
    from transformers.modeling_utils import Sampler
else:
    pytestmark = pytest.mark.skip("Require Torch")


class SamplerTest(unittest.TestCase):
    def test_nucleus_sampling(self):
        inf = -float("Inf")
        test_cases = (
            {
                "p": 0,
                "logits": torch.tensor([0.3, 0.1, 0.2]),
                "expected": torch.tensor([0.3, 0.1, 0.2]),
            },
            {
                "p": 0.01,
                "logits": torch.tensor([0.3, 0.1, 0.2]),
                "expected": torch.tensor([0.3, inf, inf]),
            },
            {
                "p": 1,
                "logits": torch.tensor([0.3, 0.1, 0.2]),
                "expected": torch.tensor([0.3, 0.1, 0.2]),
            },
            {
                "p": 0.2,
                "logits": torch.tensor([0.7, 0.1, 0.2]),
                "expected": torch.tensor([0.7, inf, inf]),
            },
            {
                "p": 0.71,
                "logits": torch.tensor([0.7, 0.1, 0.2]),
                "expected": torch.tensor([0.7, inf, 0.2]),
            },
            {
                "p": 0.71,
                "logits": torch.tensor([0.1, 0.7, 0.2]),
                "expected": torch.tensor([inf, 0.7, 0.2]),
            },
            {
                "p": 0.71,
                "logits": torch.tensor([0.7, 0.2, 0.1]),
                "expected": torch.tensor([0.7, 0.2, inf]),
            },
            {
                "p": 0.91,
                "logits": torch.tensor([0.7, 0.1, 0.2]),
                "expected": torch.tensor([0.7, 0.1, 0.2]),
            },
        )
        for case in test_cases:
            config = {
                "do_sample": True,
                "temperature": 1.0,
                "k": 0,
                "p": case["p"],
                "repetition_penalty": 1.0,
            }
            sampler = Sampler(**config)
            filtered_logits = sampler.apply_nucleus_filter(case["logits"])
            np.testing.assert_array_equal(case["expected"].numpy(), filtered_logits.numpy())

    def test_top_k_filter(self):
        inf = -float("Inf")
        test_cases = (
            {
                "k": 0,
                "logits": torch.tensor([0.7, 0.1, 0.2]),
                "expected": torch.tensor([0.7, 0.1, 0.2]),
            },
            {
                "k": 1,
                "logits": torch.tensor([0.7, 0.1, 0.2]),
                "expected": torch.tensor([0.7, inf, inf]),
            },
            {
                "k": 2,
                "logits": torch.tensor([0.7, 0.1, 0.2]),
                "expected": torch.tensor([0.7, inf, 0.2]),
            },
            {
                "k": 3,
                "logits": torch.tensor([0.7, 0.1, 0.2]),
                "expected": torch.tensor([0.7, 0.1, 0.2]),
            },
        )
        for case in test_cases:
            config = {
                "do_sample": True,
                "temperature": 1.0,
                "k": case["k"],
                "p": 0,
                "repetition_penalty": 1.0,
            }
            sampler = Sampler(**config)
            filtered_logits = sampler.apply_top_k_filter(case["logits"])
            np.testing.assert_array_equal(case["expected"].numpy(), filtered_logits.numpy())

    @pytest.mark.skipif(sys.version_info < (3, 2), reason="assertWarns() requires Python >= 3.2")
    def test_wrong_k_value(self):
        case = {"k": 10, "vocab_size": 5}
        config = {
            "do_sample": True,
            "temperature": 1.0,
            "k": case["k"],
            "p": 0,
            "repetition_penalty": 1.0,
        }
        sampler = Sampler(**config)
        next_token_logits = torch.rand(case["vocab_size"]).unsqueeze(0)
        past_sequence = torch.tensor([])
        with self.assertWarns(UserWarning):
            _ = sampler.get_one_token(next_token_logits, past_sequence)

    def test_zero_temperature(self):
        temperature = 0
        config = {
            "do_sample": True,
            "temperature": temperature,
            "k": 0,
            "p": 0,
            "repetition_penalty": 1.0,
        }
        sampler = Sampler(**config)
        next_token_logits = torch.rand(10).unsqueeze(0)
        past_sequence = torch.tensor([])
        with self.assertRaises(ZeroDivisionError):
            _ = sampler.get_one_token(next_token_logits, past_sequence)


class SamplerSingleStackTest(unittest.TestCase):
    def test_raises_exception_when_no_LM_head(self):
        models = [BertModel(BertConfig())]
        for model in models:
            with self.assertRaises(AttributeError):
                model.decode()

    @pytest.mark.slow
    def test_forward_pass_and_output_length(self):
        models = {
            "XLNet": XLNetLMHeadModel(XLNetConfig()),
            "XLM": XLMWithLMHeadModel(XLMConfig()),
            "TransfoXL": TransfoXLLMHeadModel(TransfoXLConfig()),
            "GPT2": GPT2LMHeadModel(GPT2Config()),
            "GPT": OpenAIGPTLMHeadModel(OpenAIGPTConfig()),
        }
        kwargs = {
            "XLNet": {},
            "XLM": {"mask_token": 0},
            "TransfoXL": {},
            "GPT2": {},
            "GPT": {},
        }
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        generated_length = 5
        expected_length = 8

        for name, model in models.items():
            kwargs_model = kwargs[name]
            output = model.decode(prompt_ids=prompt, length=generated_length, **kwargs_model)
            self.assertEqual(len(output), expected_length)


class SamplerEncoderDecoderTest(unittest.TestCase):
    @pytest.mark.slow
    def test_forward_pass_and_output_length(self):
        model = Model2Model.from_pretrained("bert-base-uncased")

        encoder_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        generated_length = 5
        expected_length = 8

        output = model.decode(
            encoder_input_ids,
            decoder_prompt_ids=prompt,
            k=2,
            p=0.5,
            repetition_penalty=2,
            length=generated_length,
        )
        self.assertEqual(len(output), expected_length)


if __name__ == "__main__":
    unittest.main()
