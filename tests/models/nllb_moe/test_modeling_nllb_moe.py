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

from transformers import NllbMoeConfig, is_torch_available
from transformers.testing_utils import (
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    slow,
    torch_device,
    require_torch_gpu,
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
        NllbMoeTop1Router,
        load_balancing_loss_func,
        router_z_loss_func,
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
        num_hidden_layers=2,
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
        sparse_step=1,
        use_attention_mask=True,
        num_sparse_decoder_layers=2,
        num_sparse_encoder_layers=2,
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
        self.sparse_step = sparse_step
        self.num_sparse_decoder_layers = num_sparse_decoder_layers
        self.num_sparse_encoder_layers = num_sparse_encoder_layers
        self.expert_capacity = expert_capacity
        self.router_jitter_noise = router_jitter_noise

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
            sparse_step=self.sparse_step,
            num_sparse_encoder_layers=self.num_sparse_encoder_layers,
            num_sparse_decoder_layers=self.num_sparse_decoder_layers,
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
    fx_compatible = True
    test_pruning = False
    test_missing_keys = False

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
        return NllbTokenizer.from_pretrained("facebook/nllb-moe_418M")

    @require_torch_gpu
    def test_small_logits(self):
        r"""
        Logits testing to check implementation consistency between `t5x` implementation
        and `transformers` implementation of Switch-C transformers. We only check the logits
        of the first batch.
        """
        model = NllbMoeModel.from_pretrained("google/switch-base-8", torch_dtype=torch.bfloat16).to(torch_device)
        input_ids = torch.ones((32, 64), dtype=torch.long).to(torch_device)
        decoder_input_ids = torch.ones((32, 64), dtype=torch.long).to(torch_device)

        # fmt: off
        EXPECTED_MEAN_LOGITS = torch.Tensor(
            [
                -0.204102, -0.193359, 0.523438, -0.296875, 0.108887,
                0.0211182, 0.605469, -0.100586, -0.0551758, 0.296875,
                0.0090332, 0.174805, 0.139648, -0.170898, -0.0981445,
                0.0245361, 0.0373535, 0.050293, -0.212891, 0.129883,
                0.390625, -0.203125, -0.122559, -0.180664, 0.0437012,
                -0.349609, -0.0250244, -0.104004, -0.15918, -0.133789
            ]
        ).to(torch.bfloat16)
        # fmt: on
        hf_logits = model(input_ids, decoder_input_ids=decoder_input_ids).last_hidden_state.cpu()
        hf_logits = hf_logits[0, 0, :30]

        torch.testing.assert_allclose(hf_logits, EXPECTED_MEAN_LOGITS, rtol=6e-3, atol=9e-3)

    def test_inference_no_head(self):
        model = NllbMoeModel.from_pretrained("facebook/nllb-moe_418M").to(torch_device)
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
        model = NllbMoeForConditionalGeneration.from_pretrained("facebook/nllb-moe_418M").to(torch_device)

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
        model = NllbMoeForConditionalGeneration.from_pretrained("facebook/nllb-moe_418M").to(torch_device)
        tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-moe_418M", src_lang="fr", tgt_lang="en")

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
        num_experts=2,
        hidden_size=8,
        d_ff=16,
        router_jitter_noise=0,
        expert_capacity=4,
    )

    def test_equivalency_balancy_loss(self):
        r"""
        This test checks if the balancy loss is correctly implemented
        as in the original implementation of the Switch Transformer .
        """
        router_probs = torch.Tensor(
            [
                [0.35490513, 0.60419905],
                [0.4275843, 0.23061597],
                [0.32985854, 0.43953657],
                [0.25099766, 0.27730572],
                [0.7678207, 0.71474564],
            ]
        )

        expert_indices = torch.Tensor([[0], [1], [1], [0], [0]]).to(torch.int32)

        loss = load_balancing_loss_func(router_probs, expert_indices)
        self.assertAlmostEqual(loss.item(), 0.8741045, places=5)

    def test_equivalency_router_z_loss(self):
        r"""
        This test checks if the router z loss is correctly implemented
        as in the original implementation of the Switch Transformer .
        """
        logits = torch.Tensor(
            [
                [
                    [-4.2124424, 3.891939, -3.6481273, 1.8849981],
                    [0.32625437, 2.918651, 0.84758997, -4.556842],
                    [-3.32062, 4.6977115, -0.15439987, 0.44086337],
                    [3.4467149, 4.3436565, -4.7224274, -4.264637],
                    [-2.224406, -2.5318158, -1.3832569, 1.1891162],
                    [-2.320062, -0.44705987, 4.289819, -0.00662684],
                ],
                [
                    [0.99470854, -0.6992364, 0.25503993, 4.2952085],
                    [3.5937333, -3.2408535, -4.298278, 4.426601],
                    [0.7669008, 2.6588762, 2.4505413, 4.6051874],
                    [0.23330331, -3.0845237, 0.6262374, -2.9865491],
                    [0.7595146, -2.1099675, -4.155346, -2.8326452],
                    [2.3771453, 1.004138, -3.1781673, 0.7581556],
                ],
            ]
        )

        loss = router_z_loss_func(logits)
        self.assertAlmostEqual(loss.item(), 13.786719, places=5)

    def test_equivalency_token_chose_masked_router(self):
        r"""
        This test tests the equivalency between the `NllbMoeTop1Router`
        originally implemented from here: TODO: provide link
        """

        input_tokens = torch.Tensor(
            [
                [
                    [0.6433916, 0.18188512, 0.02240455, 0.563781],
                    [0.5526401, 0.0958724, 0.34253013, 0.03644359],
                    [0.08744538, 0.7909105, 0.35205448, 0.53364205],
                ],
                [
                    [0.02900076, 0.4168595, 0.5802449, 0.91486526],
                    [0.27414513, 0.14991808, 0.9383501, 0.5209162],
                    [0.51207185, 0.90618336, 0.7309413, 0.95533276],
                ],
            ]
        )

        model = NllbMoeTop1Router(self.config)

        model.classifier.weight = torch.nn.Parameter(
            torch.Tensor(
                [
                    [0.02008116, 0.00620062],
                    [-0.00811031, -0.00031623],
                    [-0.03542127, 0.02703803],
                    [0.02335377, -0.02971946],
                ],
            ).t()
        )

        expert_index, _, router_logits = model(input_tokens)
        router_probs = torch.softmax(router_logits, dim=-1)

        router_z_loss = router_z_loss_func(router_logits)
        auxiliary_loss = load_balancing_loss_func(router_probs, torch.argmax(expert_index, dim=-1))

        self.assertAlmostEqual(auxiliary_loss.item(), 1.000308, places=5)
        self.assertAlmostEqual(router_z_loss.item(), 0.4789799, places=5)

        # self.assertTrue(torch.allclose(expert_index.bool().unsqueeze(-1), expected_dispatch_mask))

    def test_max_routing_capacity(self):
        model = NllbMoeTop1Router(self.config)
        seq_len = 128
        batch_size = 4
        hidden_states = torch.stack(batch_size * [torch.rand((seq_len, self.config.hidden_size))])

        router_probs, router_logits = model._compute_router_probabilities(hidden_states)
        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.config.num_experts)

        token_priority = torch.cumsum(expert_index, dim=-2)
        expert_capacity_mask = token_priority <= self.config.expert_capacity
        expert_index = expert_index * expert_capacity_mask

        assert torch.sum(expert_index) <= batch_size * self.config.num_experts * self.config.expert_capacity
