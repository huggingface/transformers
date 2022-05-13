# coding=utf-8
# Copyright 2021, The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch OPT model. """


import copy
import tempfile
import unittest

import timeout_decorator  # noqa

from transformers import OPTConfig, is_torch_available, pipeline
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device
from transformers.utils import cached_property

from ...generation.test_generation_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch

    from transformers import GPT2Tokenizer, OPTForCausalLM, OPTModel


def prepare_opt_inputs_dict(
    config,
    input_ids,
    decoder_input_ids=None,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
):
    if attention_mask is None:
        attention_mask = input_ids.ne(config.pad_token_id)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "head_mask": head_mask,
    }


class OPTModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
        embed_dim=16,
        word_embed_proj_dim=16,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.embed_dim = embed_dim
        self.word_embed_proj_dim = word_embed_proj_dim
        self.is_encoder_decoder = False

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).clamp(
            3,
        )
        input_ids[:, -1] = self.eos_token_id  # Eos Token

        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()
        inputs_dict = prepare_opt_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def get_config(self):
        return OPTConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            embed_dim=self.embed_dim,
            is_encoder_decoder=False,
            word_embed_proj_dim=self.word_embed_proj_dim,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        config.max_position_embeddings = 100
        return config

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = OPTModel(config=config).to(torch_device).eval()

        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]
        head_mask = inputs_dict["head_mask"]

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, head_mask=head_mask, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([attention_mask, next_attn_mask], dim=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))


@require_torch
class OPTModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (OPTModel, OPTForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (OPTForCausalLM,) if is_torch_available() else ()
    is_encoder_decoder = False
    test_pruning = False
    test_missing_keys = False

    def setUp(self):
        self.model_tester = OPTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=OPTConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_save_load_strict(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model2, info = model_class.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertEqual(info["missing_keys"], [])

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in (OPTModel,):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            if not self.is_encoder_decoder:
                input_ids = inputs["input_ids"]
                del inputs["input_ids"]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
                del inputs["input_ids"]
                inputs.pop("decoder_input_ids", None)

            wte = model.get_input_embeddings()
            if not self.is_encoder_decoder:
                inputs["inputs_embeds"] = wte(input_ids)
            else:
                inputs["inputs_embeds"] = wte(encoder_input_ids)
                inputs["decoder_inputs_embeds"] = wte(decoder_input_ids)

            with torch.no_grad():
                model(**inputs)[0]

    def test_generate_fp16(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = OPTForCausalLM(config).eval().to(torch_device)
        if torch_device == "cuda":
            model.half()
        model.generate(input_ids, attention_mask=attention_mask)
        model.generate(num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)


def assert_tensors_close(a, b, atol=1e-12, prefix=""):
    """If tensors have different shapes, different values or a and b are not both tensors, raise a nice Assertion error."""
    if a is None and b is None:
        return True
    try:
        if torch.allclose(a, b, atol=atol):
            return True
        raise
    except Exception:
        pct_different = (torch.gt((a - b).abs(), atol)).float().mean().item()
        if a.numel() > 100:
            msg = f"tensor values are {pct_different:.1%} percent different."
        else:
            msg = f"{a} != {b}"
        if prefix:
            msg = prefix + ": " + msg
        raise AssertionError(msg)


def _long_tensor(tok_lst):
    return torch.tensor(tok_lst, dtype=torch.long, device=torch_device)


@require_torch
@require_sentencepiece
@require_tokenizers
class OPTModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_tokenizer(self):
        return GPT2Tokenizer.from_pretrained("patrickvonplaten/opt_gpt2_tokenizer")

    @slow
    def test_inference_no_head(self):
        model = OPTModel.from_pretrained("facebook/opt-350m").to(torch_device)
        input_ids = _long_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        attention_mask = input_ids.ne(model.config.pad_token_id)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        expected_shape = torch.Size((1, 11, 1024))
        self.assertEqual(output.shape, expected_shape)
        expected_slice = torch.tensor(
            [[0.7144, 0.8143, -1.2813], [0.7144, 0.8143, -1.2813], [-0.0467, 2.5911, -2.1845]], device=torch_device
        )
        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=1e-3))


@require_tokenizers
@require_torch
@slow
class OPTEmbeddingsTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.path_model = "facebook/opt-350m"

    def test_load_model(self):
        try:
            _ = OPTForCausalLM.from_pretrained(self.path_model)
        except BaseException:
            self.fail("Failed loading model")

    def test_logits(self):
        model = OPTForCausalLM.from_pretrained(self.path_model)
        model = model.eval()
        tokenizer = GPT2Tokenizer.from_pretrained("patrickvonplaten/opt_gpt2_tokenizer")
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

        prompts = [
            "Today is a beautiful day and I want to",
            "In the city of",
            "Paris is the capital of France and",
            "Computers and mobile phones have taken",
        ]
        input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids
        logits = model(input_ids)[0].mean(dim=-1)
        # logits_meta = torch.load(self.path_logits_meta)
        logits_meta = torch.Tensor(
            [
                [1.3851, -13.8923, -10.5229, -10.7533, -0.2309, -10.2384, -0.5365, -9.0947, -5.1670],
                [-4.7073, -10.6276, -3.9415, -21.5242, -0.2822, -0.2822, -0.2822, -0.2822, -0.2822],
                [0.6247, -3.4229, -8.9179, -1.4297, -14.1650, 1.4146, -9.0218, -0.2703, -0.2703],
                [6.4783, -1.9913, -10.7926, -2.3336, 1.5092, -0.9974, -6.8213, 1.3477, 1.3477],
            ]
        )

        assert torch.allclose(logits, logits_meta, atol=1e-4)


@require_tokenizers
@slow
class OPTGenerationTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.all_model_path = ["facebook/opt-125m", "facebook/opt-350m"]

    def test_generation(self):
        prompts = [
            "Today is a beautiful day and I want to",
            "In the city of",
            "Paris is the capital of France and",
            "Computers and mobile phones have taken",
        ]
        NEXT_TOKENS = [3392, 764, 5, 81]
        GEN_OUTPUT = []

        tokenizer = GPT2Tokenizer.from_pretrained("patrickvonplaten/opt_gpt2_tokenizer")
        for model in self.all_model_path:
            model = OPTForCausalLM.from_pretrained(self.path_model)
            model = model.eval()
            model.config.eos_token_id = tokenizer.eos_token_id

            gen = pipeline("text-generation", model=model, tokenizer=tokenizer, return_tensors=True)

            for prompt in prompts:
                len_input_sentence = len(tokenizer.tokenize(prompt))
                predicted_next_token = gen(prompt)[0]["generated_token_ids"][len_input_sentence]
                GEN_OUTPUT.append(predicted_next_token)
            self.assertListEqual(GEN_OUTPUT, NEXT_TOKENS)
