# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch NllbMoe model. """


import copy
import tempfile
import unittest

from transformers import NllbMoeConfig, is_torch_available, set_seed
from transformers.testing_utils import (
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)
from transformers.utils import cached_property

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import NllbMoeForConditionalGeneration, NllbMoeModel, NllbTokenizer
    from transformers.models.nllb_moe.modeling_nllb_moe import (
        NllbMoeDecoder,
        NllbMoeEncoder,
        NllbMoeTop2Router,
    )


def prepare_nllb_moe_inputs_dict(
    config,
    input_ids,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
):
    if attention_mask is None:
        attention_mask = input_ids.ne(config.pad_token_id)
    if decoder_attention_mask is None:
        decoder_attention_mask = decoder_input_ids.ne(config.pad_token_id)
    if head_mask is None:
        head_mask = torch.ones(config.encoder_layers, config.encoder_attention_heads, device=torch_device)
    if decoder_head_mask is None:
        decoder_head_mask = torch.ones(config.decoder_layers, config.decoder_attention_heads, device=torch_device)
    if cross_attn_head_mask is None:
        cross_attn_head_mask = torch.ones(config.decoder_layers, config.decoder_attention_heads, device=torch_device)
    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
    }


class NllbMoeModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="relu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        max_position_embeddings=20,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
        num_experts=4,
        use_attention_mask=True,
        encoder_sparse_step=2,
        decoder_sparse_step=1,
        expert_capacity=100,
        router_jitter_noise=0.0,
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
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.max_position_embeddings = max_position_embeddings
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.encoder_sparse_step = encoder_sparse_step
        self.decoder_sparse_step = decoder_sparse_step
        self.expert_capacity = expert_capacity
        self.router_jitter_noise = router_jitter_noise
        self.num_experts = num_experts

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_ids[:, -1] = self.eos_token_id  # Eos Token
        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        # we need to clamp the input ids here to avoid having pad token in between
        # this is because for NllbMoe the position_ids are prepared such that
        # all pad tokens have pos id = 2 and rest are between 2..seq_length
        # and the seq_length here is seq_length - num_pad_tokens
        # but when using past, there is no way of knowing if the past input ids had
        # pad tokens in them, which results in incorrect seq_lenth and which in turn results in
        # position_ids being off by num_pad_tokens in past input
        input_ids = input_ids.clamp(self.pad_token_id + 1)
        decoder_input_ids = decoder_input_ids.clamp(self.pad_token_id + 1)

        config = self.get_config()
        inputs_dict = prepare_nllb_moe_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def get_config(self):
        return NllbMoeConfig(
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
            encoder_layerdrop=self.encoder_layerdrop,
            decoder_layerdrop=self.decoder_layerdrop,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            expert_capacity=self.expert_capacity,
            router_jitter_noise=self.router_jitter_noise,
            decoder_sparse_step=self.decoder_sparse_step,
            encoder_sparse_step=self.encoder_sparse_step,
            num_experts=self.num_experts,
        )

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = NllbMoeModel(config=config).get_decoder().to(torch_device).eval()
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
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-2))

    def check_encoder_decoder_model_standalone(self, config, inputs_dict):
        model = NllbMoeModel(config=config).to(torch_device).eval()
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = NllbMoeEncoder.from_pretrained(tmpdirname).to(torch_device)

        encoder_last_hidden_state_2 = encoder(inputs_dict["input_ids"], attention_mask=inputs_dict["attention_mask"])[
            0
        ]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = NllbMoeDecoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            input_ids=inputs_dict["decoder_input_ids"],
            attention_mask=inputs_dict["decoder_attention_mask"],
            encoder_hidden_states=encoder_last_hidden_state,
            encoder_attention_mask=inputs_dict["attention_mask"],
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)


@require_torch
class NllbMoeModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            NllbMoeModel,
            NllbMoeForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (NllbMoeForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "conversational": NllbMoeForConditionalGeneration,
            "feature-extraction": NllbMoeModel,
            "summarization": NllbMoeForConditionalGeneration,
            "text2text-generation": NllbMoeForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    is_encoder_decoder = True
    fx_compatible = False  # TODO should this be supported out of the box
    test_pruning = False
    test_missing_keys = True

    def setUp(self):
        self.model_tester = NllbMoeModelTester(self)
        self.config_tester = ConfigTester(self, config_class=NllbMoeConfig)

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

    def test_encoder_decoder_model_standalone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.check_encoder_decoder_model_standalone(*config_and_inputs)

    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in (NllbMoeModel, NllbMoeForConditionalGeneration):
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
        model = NllbMoeForConditionalGeneration(config).eval().to(torch_device)
        if torch_device == "cuda":
            model.half()
        model.generate(input_ids, attention_mask=attention_mask)
        model.generate(num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)


def _long_tensor(tok_lst):
    return torch.tensor(tok_lst, dtype=torch.long, device=torch_device)


TOLERANCE = 1e-4


@require_torch
@require_sentencepiece
@require_tokenizers
@slow
class NllbMoeModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_tokenizer(self):
        return NllbTokenizer.from_pretrained("ArthurZ/dummy-nllb-moe-2-experts")

    @require_torch_gpu
    def test_small_logits(self):
        r"""
        Logits testing to check implementation consistency between `t5x` implementation
        and `transformers` implementation of Switch-C transformers. We only check the logits
        of the first batch.
        """
        model = NllbMoeModel.from_pretrained("ArthurZ/random-nllb-moe-2-experts").eval()
        tokenizer = NllbTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="fra_Latn"
        )

        src_text = "Life is like a box of chocolates."
        tgt_text = "La vie est comme une boîte de chocolat."

        model_inputs = tokenizer(
            [src_text, "I just want to code."], text_target=tgt_text, return_tensors="pt", padding=True
        )
        model_inputs.pop("labels")
        with torch.no_grad():
            hf_outputs = model(
                **model_inputs, decoder_input_ids=torch.tensor([[2, tokenizer.lang_code_to_id["fra_Latn"]]] * 2)
            )
        hf_outputs.last_hidden_state[1, 0, :30]

        # fmt: off
        EXPECTED_ENCODET_LAST_HIDDEN = torch.Tensor([ 0.3920, -0.1974, -0.0279,  0.3463, -0.8306, -1.0629, -0.4643,  2.0563,
         1.1123,  0.3566, -0.9291, -0.3840, -0.2527, -0.9858,  1.5185, -1.1346,
         0.0323, -0.9103, -0.3647, -0.4462, -0.9720, -0.3541,  0.1777, -0.4647,
         1.6970, -0.9062,  0.2727, -1.0737,  0.8785,  0.4324])
        # fmt: on

        torch.testing.assert_allclose(
            hf_outputs.encoder_last_hidden_state[1, 0, :30], EXPECTED_ENCODET_LAST_HIDDEN, rtol=6e-3, atol=9e-3
        )

        # fmt: off
        torch.Tensor([ 0.3920, -0.1974, -0.0279,  0.3463, -0.8306, -1.0629, -0.4643,  2.0563,
         1.1123,  0.3566, -0.9291, -0.3840, -0.2527, -0.9858,  1.5185, -1.1346,
         0.0323, -0.9103, -0.3647, -0.4462, -0.9720, -0.3541,  0.1777, -0.4647,
         1.6970, -0.9062,  0.2727, -1.0737,  0.8785,  0.4324])
        # fmt: on

    def test_inference_no_head(self):
        model = NllbMoeModel.from_pretrained("ArthurZ/dummy-nllb-moe-2-experts").to(torch_device)
        input_ids = _long_tensor([[128028, 98, 12, 30527, 2732, 159, 7755, 61904, 39144, 38, 2]])
        decoder_input_ids = _long_tensor([[2, 128028, 98, 12, 30527, 2732, 159, 7755, 61904, 39144, 38]])
        inputs_dict = prepare_nllb_moe_inputs_dict(model.config, input_ids, decoder_input_ids)
        with torch.no_grad():
            output = model(**inputs_dict)[0]
        expected_shape = torch.Size((1, 11, 1024))
        self.assertEqual(output.shape, expected_shape)
        # change to expected output here
        expected_slice = torch.tensor(
            [[-0.7780, -0.1676, 0.1038], [-6.7556, -1.3992, 0.0567], [-7.5383, -0.5920, -0.2779]], device=torch_device
        )
        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=TOLERANCE))

    def test_inference_head(self):
        model = NllbMoeForConditionalGeneration.from_pretrained("ArthurZ/dummy-nllb-moe-2-experts").to(torch_device)

        # change to intended input
        input_ids = _long_tensor([[128028, 98, 12, 30527, 2732, 159, 7755, 61904, 39144, 38, 2]])
        decoder_input_ids = _long_tensor([[2, 128028, 98, 12, 30527, 2732, 159, 7755, 61904, 39144, 38]])
        inputs_dict = prepare_nllb_moe_inputs_dict(model.config, input_ids, decoder_input_ids)
        with torch.no_grad():
            output = model(**inputs_dict)[0]
        expected_shape = torch.Size((1, 11, model.config.vocab_size))
        self.assertEqual(output.shape, expected_shape)
        # change to expected output here
        expected_slice = torch.tensor(
            [[-1.0448, -1.0411, 3.7992], [-3.2191, -3.2386, -1.3451], [-3.6210, -3.5993, 0.4925]], device=torch_device
        )
        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=TOLERANCE))

    def test_seq_to_seq_generation(self):
        model = NllbMoeForConditionalGeneration.from_pretrained("ArthurZ/dummy-nllb-moe-2-experts").to(torch_device)
        tokenizer = NllbTokenizer.from_pretrained("ArthurZ/dummy-nllb-moe-2-experts", src_lang="fr", tgt_lang="en")

        src_fr = [
            "L'affaire NSA souligne l'absence totale de débat sur le renseignement",
            "Selon moi, il y a deux niveaux de réponse de la part du gouvernement français.",
            "Lorsque François Hollande téléphone à Barack Obama ou quand le ministre des affaires étrangères Laurent"
            " Fabius convoque l'ambassadeur des Etats-Unis, ils réagissent à une vraie découverte, qui est celle de"
            " l'ampleur de la surveillance américaine sur l'ensemble des communications en France.",
        ]

        # The below article tests that we don't add any hypotheses outside of the top n_beams
        dct = tokenizer(src_fr, padding=True, return_tensors="pt")

        hypotheses_batch = model.generate(
            input_ids=dct["input_ids"].to(torch_device),
            attention_mask=dct["attention_mask"].to(torch_device),
            num_beams=5,
            forced_bos_token_id=tokenizer.get_lang_id("en"),
        )

        expected_en = [
            "The NSA case highlights the total absence of intelligence debate",
            "I think there are two levels of response from the French government.",
            "When François Hollande calls Barack Obama or when Foreign Minister Laurent Fabius calls the U.S."
            " Ambassador, they respond to a real discovery, which is that of the scale of U.S. surveillance on all"
            " communications in France.",
        ]

        generated = tokenizer.batch_decode(
            hypotheses_batch.tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True
        )
        assert generated == expected_en


@require_torch
class NllbMoeRouterTest(unittest.TestCase):
    r"""
    Switch Transformers has different blocks from classic transformer based models.
    The Swift MLP contains a Router class, that has to be tested to check if it is correctly implemented

    Original implementation of the routers here:

    """
    config = NllbMoeConfig(
        num_experts=4,
        hidden_size=32,
        d_ff=16,
        expert_capacity=4,
    )

    def test_top_2_routing(self):
        # test routing with minimal reproduction
        batch_size = 2
        sequence_length = 20  # exceeds the expert capacity
        mask = torch.ones((batch_size, sequence_length), dtype=torch.bool)
        mask[0][0] = False
        mask[1][0] = False
        mask = mask.reshape(-1)
        set_seed(0)
        # test routing with minimal reproduction
        hidden_states = torch.rand((batch_size, sequence_length, self.config.hidden_size))
        classfier = torch.nn.Linear(self.config.hidden_size, self.config.num_experts)

        hf_router = NllbMoeTop2Router(self.config)
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        logits = classfier(hidden_states.reshape((batch_size * sequence_length), hidden_dim))
        dispatch_mask, router_probs, router_logits = hf_router.route_tokens(
            logits, sequence_length=sequence_length, padding_mask=mask
        )
        dispatch_mask = dispatch_mask.reshape((batch_size, sequence_length, self.config.num_experts))
        set_seed(0)
        experts = [
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
        ]

        expert_outputs = []
        for idx, expert in enumerate(experts):
            token_indices = dispatch_mask[:, :, idx].bool()
            expert_outputs.append(expert(hidden_states[token_indices]))
        expert_outputs = torch.cat(expert_outputs, dim=0)

        self.training = False
        if self.config.moe_token_dropout > 0:
            if self.training:
                expert_outputs = torch.nn.Dropout2d(self.config.moe_token_dropout)(expert_outputs)
            else:
                expert_outputs *= 1 - self.config.moe_token_dropout

        combined_output = router_probs.mm(expert_outputs.reshape(self.config.num_experts, hidden_dim))
        combined_output = combined_output.reshape(batch_size, sequence_length, hidden_dim)[:, 0]
        # Now test that the next states are correct
        # fmt: off
        EXPECTED_MEAN_FAIRSEQ_HIDDEN_STATES = torch.Tensor(
            [
                [
                    -1.1620e-01, 1.7170e-01, -2.0238e-01, -1.0510e-01, -1.0427e-02, -6.7572e-02, 1.5150e-01, 5.2377e-02, 1.2958e-01, -5.8975e-01, -2.1205e-01, 1.6905e-01, -2.9141e-01, -3.5747e-01, 7.0752e-02, -1.5030e-01, -6.7922e-02, 1.8135e-01, -1.5806e-01, -2.8438e-02, 1.7408e-01, 4.8298e-02, -1.1792e-02, -3.3192e-01, 1.5980e-01, 1.6762e-01, 2.0164e-01, -1.5490e-01, 9.5776e-02, 2.2973e-01, -1.4285e-02, -1.0743e-01,
                ],
                [
                    -8.9075e-02, 6.2051e-02, 2.8243e-03, -9.6243e-02, -1.1256e-01, 1.4499e-01, 7.9538e-02, 2.2229e-01, -3.1332e-02, -4.6955e-01, 1.7454e-01, 2.6892e-01, -5.7691e-04, -3.4575e-01, 1.7524e-01, -9.0632e-02, -2.4328e-01, 1.1492e-01, -9.1722e-02, -1.8781e-01, 5.9616e-01, 1.7841e-02, -1.5246e-02, -2.8347e-01, 6.0415e-02, 3.0621e-01, 1.9543e-01, 6.2028e-02, 1.5608e-01, 1.6039e-01, 6.9567e-02, -1.4346e-01,
                ],
            ],
        )
        # fmt: on
        self.assertTrue(torch.allclose(combined_output, EXPECTED_MEAN_FAIRSEQ_HIDDEN_STATES, 1e-4))
