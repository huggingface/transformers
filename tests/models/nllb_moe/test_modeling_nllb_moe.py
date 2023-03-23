# coding=utf-8
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
""" Testing suite for the PyTorch NLLB-MoE model. """


import copy
import tempfile
import unittest

from transformers import NllbMoeConfig, is_torch_available, set_seed
from transformers.testing_utils import (
    require_accelerate,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    require_torch_gpu,
    slow,
    tooslow,
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


@require_torch
@require_sentencepiece
@require_tokenizers
@slow
class NllbMoeModelIntegrationTests(unittest.TestCase):
    @cached_property
    def tokenizer(self):
        return NllbTokenizer.from_pretrained("ArthurZ/random-nllb-moe-2-experts")
    
    @cached_property
    def big_model(self):
        return  NllbMoeForConditionalGeneration.from_pretrained("/home/arthur_huggingface_co/fairseq/weights/checkpoints/hf-converted-moe-54b")

    #@require_torch_gpu
    def test_inference_logits(self):
        r"""
        Logits testing to check implementation consistency between `fairseq` implementation
        and `transformers` implementation of NLLB-MoE transformers. We only check the logits
        of the second sample of the batch, as it is padded.
        """
        model = NllbMoeForConditionalGeneration.from_pretrained("ArthurZ/random-nllb-moe-2-experts")
        model = model.eval().to(torch_device)
        src_text = ["Life is like a box of chocolates.", "I just want to code."]
        model_inputs = self.tokenizer(src_text, return_tensors="pt", padding=True).to(torch_device)
        decoder_input_ids = torch.tensor([[2, 256057], [2, 256057]], device=torch_device)

        with torch.no_grad():
            output = model(**model_inputs, decoder_input_ids=decoder_input_ids)

        # fmt: off
        EXPECTED_ENCODER_STATE = torch.Tensor([ 0.3920, -0.1974, -0.0279,  0.3463, -0.8306, -1.0629, -0.4643,  2.0563, 1.1123,  0.3566, -0.9291, -0.3840, -0.2527, -0.9858,  1.5185, -1.1346, 0.0323, -0.9103, -0.3647, -0.4462, -0.9720, -0.3541,  0.1777, -0.4647, 1.6970, -0.9062,  0.2727, -1.0737,  0.8785,  0.4324])
        EXPECTED_DECODER_STATE = torch.Tensor([-6.0425e-02, -2.0015e-01,  6.0575e-02, -8.6366e-01, -1.1310e+00, 6.8369e-01,  7.5615e-01,  7.3555e-01,  2.3071e-01,  1.5954e+00, -7.0728e-01, -2.2647e-01, -1.3292e+00,  4.8246e-01, -6.9153e-01, -1.8199e-02, -7.3664e-01,  1.5902e-03,  1.0760e-01,  1.0298e-01, -9.3933e-01, -4.6567e-01,  8.0417e-01,  1.5243e+00,  5.5844e-01, -9.9239e-02,  1.4885e+00,  7.1527e-02, -5.2612e-01,  9.4435e-02])
        EXPECTED_LOGTIS = torch.Tensor([-0.3059, 0.0000, 9.3029, 0.6456, -0.9148, 1.7836, 0.6478, 0.9438, -0.5272, -0.6617, -1.2717, 0.4564, 0.1345, -0.2301, -1.0140, 1.1427, -1.5535, 0.1337, 0.2082, -0.8112, -0.3842, -0.3377, 0.1256, 0.6450, -0.0452, 0.0219, 1.4274, -0.4991, -0.2063, -0.4409,])
        # fmt: on

        torch.testing.assert_allclose(
            output.encoder_last_hidden_state[1, 0, :30].cpu(), EXPECTED_ENCODER_STATE, rtol=6e-3, atol=9e-3
        )
        torch.testing.assert_allclose(
            output.last_hidden_state[1, 0, :30].cpu(), EXPECTED_DECODER_STATE, rtol=6e-3, atol=9e-3
        )
        torch.testing.assert_allclose(output.logits[1, 0, :30].cpu(), EXPECTED_LOGTIS, rtol=6e-3, atol=9e-3)

    # @tooslow("This test requires at least 340GB of RAM.")
    def test_large_logits(self):
        model = self.big_model
        tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-moe-54b")
        model = model.eval().to(torch_device)
        src_text = ["Life is like a box of chocolates.", "I just want to code."]
        model_inputs = tokenizer(src_text, return_tensors="pt", padding=True).to(torch_device)
        decoder_input_ids = torch.tensor([[2, 256057], [2, 256057]], device=torch_device)

        with torch.no_grad():
            output = model(**model_inputs, decoder_input_ids=decoder_input_ids)

        # fmt: off
        EXPECTED_ENCODER_STATE = torch.Tensor([ 0.1696, -0.0059,  0.0489,  0.0479, -0.4222, -0.2178, -0.1372, -0.0860, -0.4249, -0.0081, -0.1186,  0.6678,  0.0160,  0.4140,  0.1799,  0.0672, -0.4941,  0.0173, -0.0740,  0.0845, -0.2197,  0.4465,  0.2268, -0.1752, -0.0562,  0.1033, -0.0869, -0.5490,  0.0582,  0.2165])
        EXPECTED_DECODER_STATE = torch.Tensor([ 0.0374, -0.1055, -0.1060, -0.1711, -0.0540, -0.1183, -0.0779,  0.0610, -0.0279, -0.0848,  0.0222,  0.0372, -0.0298, -0.0861, -0.0354, -0.0103,  0.0538, -0.0148, -0.0105,  0.0224,  0.0629, -0.0291, -0.0671,  0.0173, -0.0066, -0.0245, -0.0499,  0.0760, -0.0067,  0.0086])
        EXPECTED_LOGTIS = torch.Tensor([ 0.3834,  0.2057,  4.5399,  0.8301,  0.4810,  0.9325,  0.9928,  0.9574,  0.5517,  0.9156,  0.2698,  0.6728,  0.7121,  0.3080,  0.4693,  0.5756,  1.0407,  0.2219,  0.3714,  0.5699,  0.5547,  0.8472,  0.3178,  0.1286,  0.1791,  0.9391,  0.5153, -0.2146,  0.1689,  0.6816])
        # fmt: on

        torch.testing.assert_allclose(
            output.encoder_last_hidden_state[1, 0, :30].cpu(), EXPECTED_ENCODER_STATE, rtol=6e-3, atol=9e-3
        )
        torch.testing.assert_allclose(
            output.last_hidden_state[1, 0, :30].cpu(), EXPECTED_DECODER_STATE, rtol=6e-3, atol=9e-3
        )
        torch.testing.assert_allclose(output.logits[1, 0, :30].cpu(), EXPECTED_LOGTIS, rtol=6e-3, atol=9e-3)

    # @tooslow("This test requires at least 340GB of RAM.")
    def test_seq_to_seq_generation(self):
        model = self.big_model
        tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-moe-54b")

        # first 6 samples of load_dataset("facebook/flores", "eng_Latn-fra_Latn"), devtest. Truth from the fairseq translation files
        FIRST_6_FLORES_200 = [
            'We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added.',
            "Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia and chair of the clinical and scientific division of the Canadian Diabetes Association cautioned that the research is still in its early days.",
            "Like some other experts, he is skeptical about whether diabetes can be cured, noting that these findings have no relevance to people who already have Type 1 diabetes.",
            "On Monday, Sara Danius, permanent secretary of the Nobel Committee for Literature at the Swedish Academy, publicly announced during a radio program on Sveriges Radio in Sweden the committee, unable to reach Bob Dylan directly about winning the 2016 Nobel Prize in Literature, had abandoned its efforts to reach him.",
            'Danius said, "Right now we are doing nothing. I have called and sent emails to his closest collaborator and received very friendly replies. For now, that is certainly enough."',
            "Previously, Ring's CEO, Jamie Siminoff, remarked the company started when his doorbell wasn't audible from his shop in his garage.",
        ]
        inputs = tokenizer(FIRST_6_FLORES_200, padding=True, return_tensors="pt").to(torch_device)

        # takes about 38.176 seconds
        hypotheses_batch = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["fra_Latn"])

        EXPECTED_FAIRSEQ_TRANSLATION = [
            '"Nous avons maintenant des souris de 4 mois non diabétiques qui étaient diabétiques", a-t-il ajouté.',
            "Le docteur Ehud Ur, professeur de médecine à l'université Dalhousie, à Halifax, en Nouvelle-Écosse, et président de la division clinique et scientifique de l'Association canadienne du diabète, prévient que la recherche n'en est qu'à ses débuts.",
            "Comme d'autres spécialistes, il est sceptique quant à la guérison du diabète, notant que ces résultats ne sont pas pertinents pour les personnes atteintes de diabète de type 1.",
            "Lundi, Sara Danius, secrétaire permanente du Comité Nobel de littérature à l'Académie suédoise, a annoncé publiquement lors d'une émission de radio sur Sveriges Radio en Suède que le comité, incapable de contacter Bob Dylan directement au sujet du prix Nobel de littérature 2016, avait abandonné ses efforts pour le joindre.",
            "Danius a déclaré: \"Pour le moment, nous ne faisons rien. J'ai appelé et envoyé des courriels à son plus proche collaborateur et j'ai reçu des réponses très amicales. Pour l'instant, c'est certainement suffisant\".",
            "Auparavant, le PDG de Ring, Jamie Siminoff, a fait remarquer que la société avait commencé lorsque sa sonnette n'était pas audible depuis son magasin dans son garage.",
        ]

        translation = tokenizer.batch_decode(
            hypotheses_batch.tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True
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

    def test_top_2_routing(self):
        # test routing with minimal reproduction
        batch_size = 2
        sequence_length = 20  # exceeds the expert capacity
        mask = torch.ones((batch_size, sequence_length), dtype=torch.bool)
        mask[0][0] = False
        mask[1][0] = False
        mask = mask.reshape(-1)
        set_seed(0)
        hidden_states = torch.rand((batch_size, sequence_length, self.config.hidden_size))
        classfier = torch.nn.Linear(self.config.hidden_size, self.config.num_experts)
        hf_router = NllbMoeTop2Router(self.config)

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        logits = classfier(hidden_states.reshape((batch_size * sequence_length), hidden_dim))
        top_1_mask, router_probs, router_logits = hf_router.route_tokens(logits, padding_mask=mask)
        torch.argmax(top_1_mask, dim=-1)
        router_mask = router_probs.bool()
        set_seed(0)
        experts = [
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
        ]
        hidden_states = hidden_states.reshape((batch_size * sequence_length), hidden_dim)
        masked_hidden_states = torch.einsum("bm,be->ebm", hidden_states, router_mask)
        for idx, expert in enumerate(experts):
            token_indices = router_mask[:, idx]
            combining_weights = router_probs[token_indices, idx]
            expert_output = expert(masked_hidden_states[idx, token_indices])
            expert_output *= 1 - self.config.moe_token_dropout
            masked_hidden_states[idx, token_indices] = torch.einsum("b,be->be", combining_weights, expert_output)
        hidden_states = masked_hidden_states.sum(dim=0).reshape(batch_size, sequence_length, hidden_dim)

        # fmt: off
        EXPECTED_MEAN_FAIRSEQ_HIDDEN_STATES = torch.Tensor([[ 7.0340e-04,  2.7997e-03, -1.3351e-02, -7.6705e-03, -3.5089e-03,3.9773e-03,  7.4593e-03,  1.2566e-02,  3.5860e-03, -2.7448e-02,-1.3731e-02, -1.0534e-02, -1.3606e-02, -1.5048e-02, -2.8914e-03,-5.0371e-03, -1.3963e-03,  6.0076e-03, -1.1380e-02, -1.4620e-02, 5.2401e-03,  8.4660e-04, -1.5319e-03, -1.6735e-02,  1.1302e-02, 3.6119e-03,  4.6084e-03, -1.3458e-02,  7.7792e-05,  1.4312e-02, 4.9107e-03, -5.0936e-03], [-4.4538e-03,  3.1026e-03,  1.4121e-04, -4.8121e-03, -5.6279e-03, 7.2493e-03,  3.9769e-03,  1.1114e-02, -1.5666e-03, -2.3477e-02, 8.7268e-03,  1.3446e-02, -2.8845e-05, -1.7287e-02,  8.7619e-03, -4.5316e-03, -1.2164e-02,  5.7461e-03, -4.5861e-03, -9.3907e-03, 2.9808e-02,  8.9206e-04, -7.6232e-04, -1.4173e-02,  3.0208e-03, 1.5310e-02,  9.7717e-03,  3.1014e-03,  7.8042e-03,  8.0197e-03, 3.4784e-03, -7.1728e-03]])
        # fmt: on
        self.assertTrue(torch.allclose(hidden_states.mean(1), EXPECTED_MEAN_FAIRSEQ_HIDDEN_STATES, 1e-4))

    def test_batch_prioritized_routing(self):
        self.config.batch_prioritized_routing=True
        NllbMoeTop2Router(self.config)

    def test_seconde_expert_policy(self):
        pass
