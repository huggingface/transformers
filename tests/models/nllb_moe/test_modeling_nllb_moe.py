# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch NLLB-MoE model."""

import copy
import tempfile
import unittest
from functools import cached_property

from transformers import NllbMoeConfig, is_torch_available, set_seed
from transformers.testing_utils import (
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    require_torch_fp16,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import NllbMoeForConditionalGeneration, NllbMoeModel, NllbTokenizer
    from transformers.models.nllb_moe.modeling_nllb_moe import NllbMoeDecoder, NllbMoeEncoder, NllbMoeTop2Router


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
        num_experts=4,
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

    def prepare_nllb_moe_inputs_dict(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask=None,
        decoder_attention_mask=None,
    ):
        if attention_mask is None:
            attention_mask = input_ids.ne(config.pad_token_id)
        if decoder_attention_mask is None:
            decoder_attention_mask = decoder_input_ids.ne(config.pad_token_id)
        return {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": attention_mask,
        }

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
        inputs_dict = self.prepare_nllb_moe_inputs_dict(config, input_ids, decoder_input_ids)
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

    @require_torch
    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = NllbMoeModel(config=config).get_decoder().to(torch_device).eval()
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

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
    all_model_classes = (NllbMoeModel, NllbMoeForConditionalGeneration) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": NllbMoeModel,
            "summarization": NllbMoeForConditionalGeneration,
            "text2text-generation": NllbMoeForConditionalGeneration,
            "translation": NllbMoeForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    is_encoder_decoder = True

    test_missing_keys = True

    # TODO: Fix the failed tests when this model gets more usage
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        # Saving the slow tokenizer after saving the fast tokenizer causes the loading of the later hanging forever.
        return True

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
            self.assertEqual(info["missing_keys"], set())

    def test_decoder_model_past_with_large_inputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        config.decoder_sparse_step = 0
        self.model_tester.create_and_check_decoder_model_past_large_inputs(config, inputs_dict)

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

    @require_torch_fp16
    def test_generate_fp16(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = NllbMoeForConditionalGeneration(config).eval().to(torch_device)
        model.half()
        model.generate(input_ids, attention_mask=attention_mask)
        model.generate(num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)

    def test_get_loss(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()
        input_dict["output_router_logits"] = True
        input_dict["labels"] = input_dict["input_ids"]
        model = NllbMoeForConditionalGeneration(config).eval().to(torch_device)
        out = model(**input_dict)
        self.assertIsNotNone(out.loss)
        self.assertIsNotNone(model(**input_dict)["encoder_router_logits"])
        self.assertIsNotNone(model(**input_dict)["decoder_router_logits"])

    @unittest.skip(
        reason="This architecture has tied weights by default and there is no way to remove it, check: https://github.com/huggingface/transformers/pull/31771#issuecomment-2210915245"
    )
    def test_load_save_without_tied_weights(self):
        pass

    @unittest.skip(reason="This is broken now, no idea why")
    def test_generate_continue_from_past_key_values(self):
        pass


@require_torch
@require_sentencepiece
@require_tokenizers
@slow
class NllbMoeModelIntegrationTests(unittest.TestCase):
    @require_torch
    @cached_property
    def model_inputs(self):
        return {
            "input_ids": torch.LongTensor(
                [
                    [28768, 248, 6399, 9, 65972, 452, 1925, 629, 123543, 248075, 2, 256047],
                    [117, 7027, 7195, 202, 44778, 248075, 2, 256047, 1, 1, 1, 1],
                ]
            ),
            "attention_mask": torch.Tensor(
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]
            ),
            "decoder_input_ids": torch.LongTensor([[2, 256057], [2, 256057]]),
        }

    @cached_property
    def tokenizer(self):
        return NllbTokenizer.from_pretrained("hf-internal-testing/random-nllb-moe-2-experts")

    @cached_property
    def big_model(self):
        return NllbMoeForConditionalGeneration.from_pretrained("facebook/nllb-moe-54b")

    def inference_no_head(self):
        model = NllbMoeModel.from_pretrained("hf-internal-testing/random-nllb-moe-2-experts").eval()
        with torch.no_grad():
            output = model(**self.model_inputs)
        # fmt: off
        EXPECTED_ENCODER_STATE = torch.Tensor([ 0.3920, -0.1974, -0.0279,  0.3463, -0.8306, -1.0629, -0.4643,  2.0563, 1.1123,  0.3566, -0.9291, -0.3840, -0.2527, -0.9858,  1.5185, -1.1346, 0.0323, -0.9103, -0.3647, -0.4462, -0.9720, -0.3541,  0.1777, -0.4647, 1.6970, -0.9062,  0.2727, -1.0737,  0.8785,  0.4324])
        EXPECTED_DECODER_STATE = torch.Tensor([-6.0425e-02, -2.0015e-01,  6.0575e-02, -8.6366e-01, -1.1310e+00, 6.8369e-01,  7.5615e-01,  7.3555e-01,  2.3071e-01,  1.5954e+00, -7.0728e-01, -2.2647e-01, -1.3292e+00,  4.8246e-01, -6.9153e-01, -1.8199e-02, -7.3664e-01,  1.5902e-03,  1.0760e-01,  1.0298e-01, -9.3933e-01, -4.6567e-01,  8.0417e-01,  1.5243e+00,  5.5844e-01, -9.9239e-02,  1.4885e+00,  7.1527e-02, -5.2612e-01,  9.4435e-02])
        # fmt: on

        torch.testing.assert_close(
            output.encoder_last_hidden_state[1, 0, :30], EXPECTED_ENCODER_STATE, rtol=6e-3, atol=9e-3
        )
        torch.testing.assert_close(output.last_hidden_state[1, 0, :30], EXPECTED_DECODER_STATE, rtol=6e-3, atol=9e-3)

    def test_inference_logits(self):
        r"""
        Logits testing to check implementation consistency between `fairseq` implementation
        and `transformers` implementation of NLLB-MoE transformers. We only check the logits
        of the second sample of the batch, as it is padded.
        """
        model = NllbMoeForConditionalGeneration.from_pretrained("hf-internal-testing/random-nllb-moe-2-experts").eval()
        with torch.no_grad():
            output = model(**self.model_inputs)

        EXPECTED_LOGTIS = torch.Tensor([-0.3059, 0.0000, 9.3029, 0.6456, -0.9148, 1.7836, 0.6478, 0.9438, -0.5272, -0.6617, -1.2717, 0.4564, 0.1345, -0.2301, -1.0140, 1.1427, -1.5535, 0.1337, 0.2082, -0.8112, -0.3842, -0.3377, 0.1256, 0.6450, -0.0452, 0.0219, 1.4274, -0.4991, -0.2063, -0.4409,])  # fmt: skip
        torch.testing.assert_close(output.logits[1, 0, :30], EXPECTED_LOGTIS, rtol=6e-3, atol=9e-3)

    @unittest.skip(reason="This requires 300GB of RAM")
    def test_large_logits(self):
        model = self.big_model
        with torch.no_grad():
            output = model(**self.model_inputs)

        # fmt: off
        EXPECTED_ENCODER_STATE = torch.Tensor([ 0.1696, -0.0059,  0.0489,  0.0479, -0.4222, -0.2178, -0.1372, -0.0860, -0.4249, -0.0081, -0.1186,  0.6678,  0.0160,  0.4140,  0.1799,  0.0672, -0.4941,  0.0173, -0.0740,  0.0845, -0.2197,  0.4465,  0.2268, -0.1752, -0.0562,  0.1033, -0.0869, -0.5490,  0.0582,  0.2165])
        EXPECTED_DECODER_STATE = torch.Tensor([ 0.0374, -0.1055, -0.1060, -0.1711, -0.0540, -0.1183, -0.0779,  0.0610, -0.0279, -0.0848,  0.0222,  0.0372, -0.0298, -0.0861, -0.0354, -0.0103,  0.0538, -0.0148, -0.0105,  0.0224,  0.0629, -0.0291, -0.0671,  0.0173, -0.0066, -0.0245, -0.0499,  0.0760, -0.0067,  0.0086])
        EXPECTED_LOGTIS = torch.Tensor([ 0.3834,  0.2057,  4.5399,  0.8301,  0.4810,  0.9325,  0.9928,  0.9574,  0.5517,  0.9156,  0.2698,  0.6728,  0.7121,  0.3080,  0.4693,  0.5756,  1.0407,  0.2219,  0.3714,  0.5699,  0.5547,  0.8472,  0.3178,  0.1286,  0.1791,  0.9391,  0.5153, -0.2146,  0.1689,  0.6816])
        # fmt: on

        torch.testing.assert_close(
            output.encoder_last_hidden_state[1, 0, :30], EXPECTED_ENCODER_STATE, rtol=6e-3, atol=9e-3
        )
        torch.testing.assert_close(output.last_hidden_state[1, 0, :30], EXPECTED_DECODER_STATE, rtol=6e-3, atol=9e-3)
        torch.testing.assert_close(output.logits[1, 0, :30], EXPECTED_LOGTIS, rtol=6e-3, atol=9e-3)

    @unittest.skip(reason="This requires 300GB of RAM")
    def test_seq_to_seq_generation(self):
        model = self.big_model
        tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-moe-54b")

        # first 6 samples of load_dataset("facebook/flores", "eng_Latn-fra_Latn"), devtest. Truth are very similar to the fairseq translation files
        FIRST_6_FLORES_200 = [
            'We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added.',
            "Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia and chair of the clinical and scientific division of the Canadian Diabetes Association cautioned that the research is still in its early days.",
            "Like some other experts, he is skeptical about whether diabetes can be cured, noting that these findings have no relevance to people who already have Type 1 diabetes.",
            "On Monday, Sara Danius, permanent secretary of the Nobel Committee for Literature at the Swedish Academy, publicly announced during a radio program on Sveriges Radio in Sweden the committee, unable to reach Bob Dylan directly about winning the 2016 Nobel Prize in Literature, had abandoned its efforts to reach him.",
            'Danius said, "Right now we are doing nothing. I have called and sent emails to his closest collaborator and received very friendly replies. For now, that is certainly enough."',
            "Previously, Ring's CEO, Jamie Siminoff, remarked the company started when his doorbell wasn't audible from his shop in his garage.",
        ]
        inputs = tokenizer(FIRST_6_FLORES_200, padding=True, return_tensors="pt").to(torch_device)
        batch_translation = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["fra_Latn"])

        EXPECTED_FAIRSEQ_TRANSLATION = [
            '"Nous avons maintenant des souris de 4 mois non diabétiques qui étaient diabétiques", a-t-il ajouté.',
            "Le docteur Ehud Ur, professeur de médecine à l'université Dalhousie, à Halifax, en Nouvelle-Écosse, et président de la division clinique et scientifique de l'Association canadienne du diabète, prévient que la recherche n'en est qu'à ses débuts.",
            "Comme d'autres spécialistes, il est sceptique quant à la guérison du diabète.",
            "Lundi, Sara Danius, secrétaire permanente du Comité Nobel de littérature à l'Académie suédoise, a annoncé publiquement lors d'une émission de radio sur Sveriges Radio en Suède que le comité, incapable de joindre Bob Dylan directement pour lui annoncer le prix Nobel de littérature 2016, avait abandonné ses efforts pour le joindre.",
            "Danius a déclaré: \"Pour l'instant, nous ne faisons rien. J'ai appelé et envoyé des courriels à son plus proche collaborateur et j'ai reçu des réponses très amicales. Pour l'instant, c'est certainement suffisant\".",
            "Auparavant, le PDG de Ring, Jamie Siminoff, a fait remarquer que la société avait commencé lorsque sa sonnette n'était pas audible depuis son magasin dans son garage.",
        ]

        translation = tokenizer.batch_decode(
            batch_translation.tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True
        )
        assert translation == EXPECTED_FAIRSEQ_TRANSLATION


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
    batch_size = 2
    sequence_length = 20

    def test_top_2_routing(self):
        # test routing with minimal reproduction
        mask = torch.ones((self.batch_size, self.sequence_length), dtype=torch.bool)
        mask[0][0] = False
        mask[1][0] = False
        mask = mask.reshape(-1)
        set_seed(42)
        hidden_states = torch.rand((self.batch_size, self.sequence_length, self.config.hidden_size))
        classifier = torch.nn.Linear(self.config.hidden_size, self.config.num_experts)
        hf_router = NllbMoeTop2Router(self.config)

        _, _, hidden_dim = hidden_states.shape
        logits = classifier(hidden_states.reshape((self.batch_size * self.sequence_length), hidden_dim))
        top_1_mask, router_probs = hf_router.route_tokens(logits, padding_mask=mask)
        torch.argmax(top_1_mask, dim=-1)
        router_mask = router_probs.bool()
        set_seed(42)
        experts = [
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
        ]
        hidden_states = hidden_states.reshape((self.batch_size * self.sequence_length), hidden_dim)
        masked_hidden_states = torch.einsum("bm,be->ebm", hidden_states, router_mask)
        for idx, expert in enumerate(experts):
            token_indices = router_mask[:, idx]
            combining_weights = router_probs[token_indices, idx]
            expert_output = expert(masked_hidden_states[idx, token_indices])
            expert_output *= 1 - self.config.moe_token_dropout
            masked_hidden_states[idx, token_indices] = torch.einsum("b,be->be", combining_weights, expert_output)
        hidden_states = masked_hidden_states.sum(dim=0).reshape(self.batch_size, self.sequence_length, hidden_dim)

        EXPECTED_MEAN_FAIRSEQ_HIDDEN_STATES = torch.Tensor([[ 0.03915622,  0.02865297,  0.00022152, -0.02593718, -0.00860747, -0.00445788,  0.00158017, -0.00622981, -0.01272568, -0.00443411, -0.03017339,  0.00102607, -0.00048952, -0.01215746, -0.00431588,  0.00832375, -0.01586730,  0.01831601, -0.02038063, -0.01010270,  0.01012242,  0.00816753,  0.01245080,  0.00132014,  0.00115318,  0.01677979,  0.00506316,  0.02593341, -0.01275316, -0.01998429,  0.00706401, -0.01330973], [ 0.01176139,  0.01778850, -0.01071202, -0.02497797,  0.00654299,  0.00279202, -0.00208195, -0.00590398,  0.00158412, -0.00482685, -0.01441989,  0.00275030, -0.00831994, -0.00507957,  0.01014271,  0.00146595, -0.00417354, -0.00280298, -0.02198826, -0.00942392,  0.02605285,  0.01864457, -0.00344483, -0.00184672,  0.01180858,  0.01278269,  0.00773582,  0.00552372, -0.00114356, -0.02552024,  0.00967451, -0.01226149]])  # fmt: skip
        torch.testing.assert_close(hidden_states.mean(1), EXPECTED_MEAN_FAIRSEQ_HIDDEN_STATES, atol=1e-4, rtol=1e-4)

    def test_batch_prioritized_routing(self):
        set_seed(42)
        config = NllbMoeConfig(
            num_experts=4, hidden_size=32, d_ff=16, expert_capacity=4, second_expert_policy="random"
        )
        mask = torch.zeros((self.batch_size * self.sequence_length), dtype=torch.bool)
        logits = torch.rand((self.batch_size * self.sequence_length, 4))
        config.batch_prioritized_routing = True
        router = NllbMoeTop2Router(config)
        top_1_mask, _ = router.route_tokens(logits, padding_mask=mask)
        # check that the routing is batch first. One of the last token is routed while expert capacity is very small
        # this means that it had a greater probability of being routed
        assert top_1_mask[-1, 0] == 0

    def test_second_expert_policy(self):
        config = NllbMoeConfig(
            num_experts=4,
            hidden_size=32,
            d_ff=16,
            expert_capacity=40,
        )
        set_seed(42)
        mask = torch.zeros((self.batch_size * self.sequence_length), dtype=torch.bool)
        logits = torch.rand((self.batch_size * self.sequence_length, 4))

        set_seed(42)
        config.second_expert_policy = "random"
        router = NllbMoeTop2Router(config)
        top_1_mask, router_probs = router.route_tokens(logits, padding_mask=mask)

        set_seed(42)
        config.second_expert_policy = "sampling"
        router = NllbMoeTop2Router(config)
        top_1_mask_sp, router_probs_sp = router.route_tokens(logits, padding_mask=mask)

        set_seed(42)
        config.second_expert_policy = "all"
        router = NllbMoeTop2Router(config)
        top_1_mask_all, router_probs_all = router.route_tokens(logits, padding_mask=mask)

        # fmt: off
        EXPECTED_ROUTER_ALL = torch.tensor([[0.58605188, 0.00000000, 0.41394812, 0.00000000], [0.00000000, 0.85299969, 0.00000000, 0.14700034], [0.00000000, 0.00000000, 0.74955785, 0.25044215], [0.30080634, 0.00000000, 0.69919366, 0.00000000], [0.00000000, 0.00000000, 0.20320441, 0.79679561], [0.90920591, 0.00000000, 0.09079411, 0.00000000], [0.04189575, 0.00000000, 0.95810419, 0.00000000], [0.50911492, 0.00000000, 0.49088508, 0.00000000], [0.93492341, 0.06507659, 0.00000000, 0.00000000], [0.21269986, 0.00000000, 0.00000000, 0.78730011], [0.52476716, 0.00000000, 0.00000000, 0.47523284], [0.84478962, 0.00000000, 0.00000000, 0.15521035], [0.00000000, 0.27197742, 0.72802258, 0.00000000], [0.00000000, 0.71292859, 0.28707141, 0.00000000], [0.52122664, 0.00000000, 0.47877336, 0.00000000], [0.00000000, 0.47759300, 0.52240700, 0.00000000], [0.32156345, 0.67843658, 0.00000000, 0.00000000], [0.00000000, 0.00000000, 0.55345929, 0.44654068], [0.00000000, 0.00000000, 0.79953444, 0.20046560], [0.55898255, 0.44101751, 0.00000000, 0.00000000], [0.00000000, 0.00000000, 0.71453607, 0.28546396], [0.00000000, 0.00000000, 0.03601381, 0.96398622], [0.30293706, 0.00000000, 0.69706291, 0.00000000], [0.00000000, 0.92059422, 0.00000000, 0.07940583], [0.00000000, 0.41886434, 0.58113569, 0.00000000], [0.07215069, 0.00000000, 0.00000000, 0.92784929], [0.28082481, 0.71917516, 0.00000000, 0.00000000], [0.59494448, 0.00000000, 0.40505546, 0.00000000], [0.80946648, 0.00000000, 0.19053355, 0.00000000], [0.39879149, 0.00000000, 0.00000000, 0.60120851], [0.19673465, 0.80326533, 0.00000000, 0.00000000], [0.50106758, 0.00000000, 0.49893236, 0.00000000], [0.53200942, 0.00000000, 0.46799061, 0.00000000], [0.42457944, 0.00000000, 0.57542056, 0.00000000], [0.08162964, 0.91837037, 0.00000000, 0.00000000], [0.20909494, 0.00000000, 0.79090512, 0.00000000], [0.00000000, 0.14923605, 0.85076392, 0.00000000], [0.00000000, 0.23456344, 0.76543659, 0.00000000], [0.33135825, 0.00000000, 0.66864175, 0.00000000], [0.00000000, 0.12020690, 0.00000000, 0.87979311]])
        EXPECTED_ROUTER_SP = torch.tensor([[0.48075044, 0.00000000, 0.00000000, 0.51924956], [0.00000000, 0.45196214, 0.00000000, 0.54803789], [0.50154328, 0.00000000, 0.49845666, 0.00000000], [0.53203368, 0.00000000, 0.46796635, 0.00000000], [0.56414306, 0.00000000, 0.00000000, 0.43585697], [0.36307240, 0.00000000, 0.00000000, 0.63692760], [0.43696275, 0.00000000, 0.56303722, 0.00000000], [0.40026453, 0.00000000, 0.59973544, 0.00000000], [0.57512784, 0.42487213, 0.00000000, 0.00000000], [0.00000000, 0.63683844, 0.00000000, 0.36316153], [0.42260975, 0.00000000, 0.00000000, 0.57739025], [0.54972672, 0.00000000, 0.00000000, 0.45027325], [0.67893279, 0.00000000, 0.32106718, 0.00000000], [0.00000000, 0.56617302, 0.43382704, 0.00000000], [0.44196928, 0.00000000, 0.55803066, 0.00000000], [0.00000000, 0.37861145, 0.62138855, 0.00000000], [0.00000000, 0.37311727, 0.62688273, 0.00000000], [0.00000000, 0.00000000, 0.57030505, 0.42969498], [0.72652513, 0.00000000, 0.27347484, 0.00000000], [0.32042775, 0.00000000, 0.00000000, 0.67957222], [0.00000000, 0.00000000, 0.61476612, 0.38523385], [0.66639966, 0.00000000, 0.00000000, 0.33360034], [0.00000000, 0.51168150, 0.48831853, 0.00000000], [0.00000000, 0.64721936, 0.00000000, 0.35278070], [0.65364254, 0.00000000, 0.34635746, 0.00000000], [0.00000000, 0.61959863, 0.00000000, 0.38040143], [0.58741009, 0.41258991, 0.00000000, 0.00000000], [0.28142369, 0.00000000, 0.71857625, 0.00000000], [0.61833024, 0.00000000, 0.38166985, 0.00000000], [0.59327477, 0.00000000, 0.00000000, 0.40672523], [0.54672259, 0.45327741, 0.00000000, 0.00000000], [0.45709357, 0.00000000, 0.54290640, 0.00000000], [0.49725920, 0.00000000, 0.50274086, 0.00000000], [0.00000000, 0.61859524, 0.38140479, 0.00000000], [0.00000000, 0.34521589, 0.65478414, 0.00000000], [0.45597604, 0.00000000, 0.54402399, 0.00000000], [0.00000000, 0.00000000, 0.46368176, 0.53631824], [0.58167320, 0.00000000, 0.41832677, 0.00000000], [0.37514153, 0.00000000, 0.62485844, 0.00000000], [0.63124204, 0.00000000, 0.00000000, 0.36875802]])
        EXPECTED_ROUTER = torch.tensor([[0.00000000, 0.00000000, 0.00000000, 1.00000000], [0.00000000, 0.45196214, 0.00000000, 0.54803789], [1.00000000, 0.00000000, 0.00000000, 0.00000000], [1.00000000, 0.00000000, 0.00000000, 0.00000000], [0.56414306, 0.00000000, 0.00000000, 0.43585697], [0.00000000, 0.00000000, 0.00000000, 1.00000000], [0.00000000, 0.47768536, 0.52231461, 0.00000000], [0.40026453, 0.00000000, 0.59973544, 0.00000000], [0.51925015, 0.00000000, 0.00000000, 0.48074985], [0.41920051, 0.58079952, 0.00000000, 0.00000000], [0.00000000, 0.00000000, 0.00000000, 1.00000000], [0.50009936, 0.00000000, 0.49990064, 0.00000000], [1.00000000, 0.00000000, 0.00000000, 0.00000000], [0.00000000, 0.55061668, 0.00000000, 0.44938332], [0.00000000, 0.00000000, 1.00000000, 0.00000000], [0.46936488, 0.00000000, 0.53063512, 0.00000000], [0.43499801, 0.00000000, 0.56500196, 0.00000000], [0.00000000, 0.00000000, 1.00000000, 0.00000000], [1.00000000, 0.00000000, 0.00000000, 0.00000000], [0.00000000, 0.00000000, 0.00000000, 1.00000000], [0.00000000, 0.00000000, 1.00000000, 0.00000000], [0.50435966, 0.00000000, 0.49564037, 0.00000000], [0.00000000, 1.00000000, 0.00000000, 0.00000000], [0.48953408, 0.51046598, 0.00000000, 0.00000000], [1.00000000, 0.00000000, 0.00000000, 0.00000000], [0.00000000, 0.54758018, 0.45241979, 0.00000000], [1.00000000, 0.00000000, 0.00000000, 0.00000000], [0.00000000, 0.00000000, 0.59997642, 0.40002355], [1.00000000, 0.00000000, 0.00000000, 0.00000000], [1.00000000, 0.00000000, 0.00000000, 0.00000000], [1.00000000, 0.00000000, 0.00000000, 0.00000000], [0.00000000, 0.46989843, 0.53010160, 0.00000000], [0.00000000, 0.00000000, 1.00000000, 0.00000000], [0.00000000, 1.00000000, 0.00000000, 0.00000000], [0.00000000, 0.00000000, 1.00000000, 0.00000000], [0.45597604, 0.00000000, 0.54402399, 0.00000000], [0.00000000, 0.00000000, 0.00000000, 1.00000000], [0.58167320, 0.00000000, 0.41832677, 0.00000000], [0.37514153, 0.00000000, 0.62485844, 0.00000000], [1.00000000, 0.00000000, 0.00000000, 0.00000000]])

        EXPECTED_TOP_1_ALL = torch.LongTensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        EXPECTED_TOP_1_SP = torch.LongTensor([[0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]])
        # `sampling` and `random` do not affect the mask of the top_1 router
        # fmt: on

        torch.testing.assert_close(router_probs_all, EXPECTED_ROUTER_ALL, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(router_probs_sp, EXPECTED_ROUTER_SP, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(router_probs, EXPECTED_ROUTER, rtol=1e-4, atol=1e-4)

        torch.testing.assert_close(top_1_mask_all, EXPECTED_TOP_1_ALL, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(top_1_mask_sp, EXPECTED_TOP_1_SP, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(top_1_mask, EXPECTED_TOP_1_SP, rtol=1e-4, atol=1e-4)
