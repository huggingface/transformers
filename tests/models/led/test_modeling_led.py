# Copyright 2021 Iz Beltagy, Matthew E. Peters, Arman Cohan and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch LED model."""

import copy
import tempfile
import unittest

from transformers import LEDConfig, is_torch_available
from transformers.models.auto import get_values
from transformers.testing_utils import (
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    require_torch_fp16,
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

    from transformers import (
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        LEDForConditionalGeneration,
        LEDForQuestionAnswering,
        LEDForSequenceClassification,
        LEDModel,
        LEDTokenizer,
    )
    from transformers.models.led.modeling_led import LEDDecoder, LEDEncoder


def prepare_led_inputs_dict(
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
        "decoder_attention_mask": decoder_attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
    }


class LEDModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=11,
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
        max_position_embeddings=32,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
        attention_window=4,
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
        self.attention_window = attention_window

        # `ModelTesterMixin.test_attention_outputs` is expecting attention tensors to be of size
        # [num_attention_heads, encoder_seq_length, encoder_key_length], but LongformerSelfAttention
        # returns attention of shape [num_attention_heads, encoder_seq_length, self.attention_window + 1]
        # because its local attention only attends to `self.attention_window + 1` locations
        # (assuming no token with global attention, otherwise the last dimension of attentions
        # is x + self.attention_window + 1, where x is the number of tokens with global attention)
        # x is set to 1
        self.encoder_key_length = self.attention_window + 2

        # because of padding `encoder_seq_length`, is different from `seq_length`. Relevant for
        # the `test_attention_outputs` and `test_hidden_states_output` tests
        self.encoder_seq_length = self.seq_length

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).clamp(
            3,
        )
        input_ids[:, -1] = self.eos_token_id  # Eos Token

        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()
        inputs_dict = prepare_led_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def get_config(self):
        return LEDConfig(
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
            attention_window=self.attention_window,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        config.max_position_embeddings = 100
        config.vocab_size = 300
        return config

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        global_attention_mask = torch.zeros_like(inputs_dict["input_ids"])
        global_attention_mask[:, -1] = 1
        inputs_dict["global_attention_mask"] = global_attention_mask

        return config, inputs_dict

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = LEDModel(config=config).get_decoder().to(torch_device).eval()
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
        model = LEDModel(config=config).to(torch_device).eval()
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = LEDEncoder.from_pretrained(tmpdirname).to(torch_device)

        encoder_last_hidden_state_2 = encoder(
            inputs_dict["input_ids"],
            attention_mask=inputs_dict["attention_mask"],
            global_attention_mask=inputs_dict["global_attention_mask"],
        )[0]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = LEDDecoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            input_ids=inputs_dict["decoder_input_ids"],
            attention_mask=inputs_dict["decoder_attention_mask"],
            encoder_hidden_states=encoder_last_hidden_state,
            encoder_attention_mask=inputs_dict["attention_mask"],
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)

    def check_global_attention(self, config, inputs_dict):
        model = LEDModel(config=config).to(torch_device).eval()
        model.config.output_attentions = True
        attention_mask = ids_tensor(inputs_dict["input_ids"].shape, vocab_size=2)
        global_attention_mask = torch.zeros_like(attention_mask)

        # set some tokens to global_attention
        num_tokens_with_global_attention = 2

        attention_mask[:, 2 : 2 + num_tokens_with_global_attention] = 1
        global_attention_mask[:, 2 : 2 + num_tokens_with_global_attention] = 1
        inputs_dict["attention_mask"] = attention_mask
        inputs_dict["global_attention_mask"] = global_attention_mask

        outputs = model(**inputs_dict)
        self.parent.assertIsNotNone(outputs.encoder_global_attentions)

        # setting `num_tokens_with_global_attention` to global_attentions yields
        # makes last dim to be of `num_tokens_with_global_attention`
        self.parent.assertTrue(
            outputs.encoder_global_attentions[0].shape,
            (self.batch_size, self.num_attention_heads, self.encoder_seq_length, num_tokens_with_global_attention),
        )


@require_torch
class LEDModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (LEDModel, LEDForConditionalGeneration, LEDForSequenceClassification, LEDForQuestionAnswering)
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": LEDModel,
            "question-answering": LEDForQuestionAnswering,
            "summarization": LEDForConditionalGeneration,
            "text-classification": LEDForSequenceClassification,
            "text2text-generation": LEDForConditionalGeneration,
            "translation": LEDForConditionalGeneration,
            "zero-shot": LEDForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    is_encoder_decoder = True
    test_pruning = False
    test_missing_keys = False
    test_torchscript = False

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
        if pipeline_test_case_name == "QAPipelineTests" and not tokenizer_name.endswith("Fast"):
            return True

        return False

    def setUp(self):
        self.model_tester = LEDModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LEDConfig)

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

    def test_global_attention(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.check_global_attention(*config_and_inputs)

    def prepare_config_and_inputs_for_generate(self, *args, **kwargs):
        config, inputs_dict = super().prepare_config_and_inputs_for_generate(*args, **kwargs)
        # LED computes attention scores based on mask indices if `is_global`
        inputs_dict.pop("global_attention_mask")
        return config, inputs_dict

    # LEDForSequenceClassification does not support inputs_embeds
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in (LEDModel, LEDForConditionalGeneration, LEDForQuestionAnswering):
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
        model = LEDForConditionalGeneration(config).eval().to(torch_device)
        model.half()
        model.generate(input_ids, attention_mask=attention_mask)
        model.generate(num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)

    @unittest.skip(reason="Longformer cannot keep gradients in attentions or hidden states")
    def test_retain_grad_hidden_states_attentions(self):
        return

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_length = self.model_tester.seq_length
        encoder_seq_length = self.model_tester.encoder_seq_length
        encoder_key_length = self.model_tester.encoder_key_length

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )
            out_len = len(outputs)

            # global attention outputs are added as well => so +1 here
            correct_outlen = 6

            # loss is at first position
            if "labels" in inputs_dict:
                correct_outlen += 1  # loss is added to beginning
            # Question Answering model returns start_logits and end_logits
            if model_class in get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING):
                correct_outlen += 1  # start_logits and end_logits instead of only 1 output
            if "past_key_values" in outputs:
                correct_outlen += 1  # past_key_values have been returned

            self.assertEqual(out_len, correct_outlen)

            # decoder attentions
            decoder_attentions = outputs.decoder_attentions
            self.assertIsInstance(decoder_attentions, (list, tuple))
            self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(decoder_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, seq_length, seq_length],
            )

            # cross attentions
            cross_attentions = outputs.cross_attentions
            self.assertIsInstance(cross_attentions, (list, tuple))
            self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(cross_attentions[0].shape[-3:]),
                [
                    self.model_tester.num_attention_heads,
                    seq_length,
                    seq_length,
                ],
            )

    def _check_encoder_attention_for_generate(self, attentions, batch_size, config, prompt_length):
        # overwrite because LED does not have (bs, num_heads, seq_len, seq_len) shape
        encoder_expected_shape = (
            batch_size,
            config.num_attention_heads,
            prompt_length,
            self.model_tester.attention_window // 2 * 2 + 1,
        )
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [layer_attentions.shape for layer_attentions in attentions],
            [encoder_expected_shape] * len(attentions),
        )


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


TOLERANCE = 1e-4


@require_torch
@require_sentencepiece
@require_tokenizers
@slow
class LEDModelIntegrationTests(unittest.TestCase):
    """All the below results were obtained with the original checkpoints and code
    base from https://github.com/allenai/longformer.
    IMPORTANT: Note that the original checkpoints include a `postion_embeddings` "hack"
    and have to be cut to have the correct shape.
    See: https://github.com/huggingface/transformers/pull/9278#issue-544709661.
    """

    @cached_property
    def default_tokenizer(self):
        return LEDTokenizer.from_pretrained("allenai/led-base-16384")

    def test_inference_no_head(self):
        model = LEDModel.from_pretrained("allenai/led-base-16384").to(torch_device)

        # change to intended input
        input_ids = _long_tensor([512 * [0, 31414, 232, 328, 740, 1140, 12695, 69]])
        decoder_input_ids = _long_tensor([128 * [0, 31414, 232, 328, 740, 1140, 12695, 69]])
        inputs_dict = prepare_led_inputs_dict(model.config, input_ids, decoder_input_ids)
        with torch.no_grad():
            output = model(**inputs_dict).last_hidden_state
        expected_shape = torch.Size((1, 1024, 768))
        self.assertEqual(output.shape, expected_shape)
        # change to expected output here
        expected_slice = torch.tensor(
            [[2.3050, 2.8279, 0.6531], [-1.8457, -0.1455, -3.5661], [-1.0186, 0.4586, -2.2043]], device=torch_device
        )
        torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=TOLERANCE, atol=TOLERANCE)

    def test_inference_head(self):
        model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384").to(torch_device)

        # change to intended input
        input_ids = _long_tensor([512 * [0, 31414, 232, 328, 740, 1140, 12695, 69]])
        decoder_input_ids = _long_tensor([128 * [0, 31414, 232, 328, 740, 1140, 12695, 69]])
        inputs_dict = prepare_led_inputs_dict(model.config, input_ids, decoder_input_ids)
        with torch.no_grad():
            output = model(**inputs_dict, use_cache=False).logits
        expected_shape = torch.Size((1, 1024, model.config.vocab_size))
        self.assertEqual(output.shape, expected_shape)
        # change to expected output here
        expected_slice = torch.tensor(
            [[33.6507, 6.4572, 16.8089], [5.8739, -2.4238, 11.2902], [-3.2139, -4.3149, 4.2783]], device=torch_device
        )
        torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=TOLERANCE, atol=TOLERANCE)

    def test_seq_to_seq_generation(self):
        # this test requires 16GB of RAM
        hf = LEDForConditionalGeneration.from_pretrained("allenai/led-large-16384-arxiv").to(torch_device)
        tok = LEDTokenizer.from_pretrained("allenai/led-large-16384-arxiv")

        ARTICLE_LEP = r"""the lep experiments at the resonance of @xmath1-boson have tested the standard model ( sm ) at quantum level , measuring the @xmath1-decay into fermion pairs with an accuracy of one part in ten thousands . the good agreement of the lep data with the sm predictions have severely constrained the behavior of new physics at the @xmath1-pole . taking these achievements into account one can imagine that the physics of @xmath1-boson will again play the central role in the frontier of particle physics if the next generation @xmath1 factory comes true with the generated @xmath1 events several orders of magnitude higher than that of the lep . this factory can be realized in the gigaz option of the international linear collider ( ilc)@xcite . the ilc is a proposed electron - positron collider with tunable energy ranging from @xmath12 to @xmath13 and polarized beams in its first phase , and the gigaz option corresponds to its operation on top of the resonance of @xmath1 boson by adding a bypass to its main beam line . given the high luminosity , @xmath14 , and the cross section at the resonance of @xmath1 boson , @xmath15 , about @xmath16 @xmath1 events can be generated in an operational year of @xmath17 of gigaz , which implies that the expected sensitivity to the branching ratio of @xmath1-decay can be improved from @xmath18 at the lep to @xmath19 at the gigaz@xcite . in light of this , the @xmath1-boson properties , especially its exotic or rare decays which are widely believed to be sensitive to new physics , should be investigated comprehensively to evaluate their potential in probing new physics .    among the rare @xmath1-decays , the flavor changing ( fc ) processes were most extensively studied to explore the flavor texture in new physics @xcite , and it was found that , although these processes are severely suppressed in the sm , their branching ratios in new physics models can be greatly enhanced to @xmath19 for lepton flavor violation decays @xcite and @xmath20 for quark flavor violation decays @xcite . besides the fc processes , the @xmath1-decay into light higgs boson(s ) is another type of rare process that was widely studied , e.g. the decay @xmath21 ( @xmath22 ) with the particle @xmath0 denoting a light higgs boson was studied in @xcite , the decay @xmath23 was studied in the two higgs doublet model ( 2hdm)@xcite and the minimal supersymmetric standard model ( mssm)@xcite , and the decay @xmath4 was studied in a model independent way @xcite , in 2hdm@xcite and also in mssm@xcite . these studies indicate that , in contrast with the kinematic forbidden of these decays in the sm , the rates of these decays can be as large as @xmath18 in new physics models , which lie within the expected sensitivity of the gigaz . in this work , we extend the previous studies of these decays to some new models and investigate these decays altogether . we are motivated by some recent studies on the singlet extension of the mssm , such as the next - to - minimal supersymmetric standard model ( nmssm ) @xcite and the nearly minimal supersymmetric standard model ( nmssm ) @xcite , where a light cp - odd higgs boson @xmath0 with singlet - dominant component may naturally arise from the spontaneous breaking of some approximate global symmetry like @xmath24 or peccei - quuin symmetry @xcite . these non - minimal supersymmetric models can not only avoid the @xmath25-problem , but also alleviate the little hierarchy by having such a light higgs boson @xmath0 @xcite . we are also motivated by that , with the latest experiments , the properties of the light higgs boson are more stringently constrained than before . so it is worth updating the previous studies . so far there is no model - independent lower bound on the lightest higgs boson mass . in the sm , it must be heavier than @xmath26 gev , obtained from the null observation of the higgs boson at lep experiments . however , due to the more complex structure of the higgs sector in the extensions of the sm , this lower bound can be significantly relaxed according to recent studies , e.g. , for the cp - odd higgs boson @xmath0 we have @xmath27 gev in the nmssm @xcite , @xmath28 gev in the nmssm @xcite , and @xmath29 gev in the lepton - specific 2hdm ( l2hdm ) @xcite . with such a light cp - odd higgs boson , the z - decay into one or more @xmath0 is open up . noting that the decay @xmath30 is forbidden due to bose symmetry , we in this work study the rare @xmath1-decays @xmath6 ( @xmath22 ) , @xmath31 and @xmath4 in a comparative way for four models , namely the type - ii 2hdm@xcite , the l2hdm @xcite , the nmssm and the nmssm . in our study , we examine carefully the constraints on the light @xmath0 from many latest experimental results . this work is organized as follows . in sec . ii we briefly describe the four new physics models . in sec . iii we present the calculations of the rare @xmath1-decays . in sec . iv we list the constraints on the four new physics models . in sec . v we show the numerical results for the branching ratios of the rare @xmath1-decays in various models . finally , the conclusion is given in sec . as the most economical way , the sm utilizes one higgs doublet to break the electroweak symmetry . as a result , the sm predicts only one physical higgs boson with its properties totally determined by two free parameters . in new physics models , the higgs sector is usually extended by adding higgs doublets and/or singlets , and consequently , more physical higgs bosons are predicted along with more free parameters involved in . the general 2hdm contains two @xmath32 doublet higgs fields @xmath33 and @xmath34 , and with the assumption of cp - conserving , its scalar potential can be parameterized as@xcite : @xmath35,\end{aligned}\ ] ] where @xmath36 ( @xmath37 ) are free dimensionless parameters , and @xmath38 ( @xmath39 ) are the parameters with mass dimension . after the electroweak symmetry breaking , the spectrum of this higgs sector includes three massless goldstone modes , which become the longitudinal modes of @xmath40 and @xmath1 bosons , and five massive physical states : two cp - even higgs bosons @xmath41 and @xmath42 , one neutral cp - odd higgs particle @xmath0 and a pair of charged higgs bosons @xmath43 . noting the constraint @xmath44 with @xmath45 and @xmath46 denoting the vacuum expectation values ( vev ) of @xmath33 and @xmath34 respectively , we choose @xmath47 as the input parameters with @xmath48 , and @xmath49 being the mixing angle that diagonalizes the mass matrix of the cp - even higgs fields . the difference between the type - ii 2hdm and the l2hdm comes from the yukawa coupling of the higgs bosons to quark / lepton . in the type - ii 2hdm , one higgs doublet @xmath34 generates the masses of up - type quarks and the other doublet @xmath33 generates the masses of down - type quarks and charged leptons ; while in the l2hdm one higgs doublet @xmath33 couples only to leptons and the other doublet @xmath34 couples only to quarks . so the yukawa interactions of @xmath0 to fermions in these two models are given by @xcite @xmath50 with @xmath51 denoting generation index . obviously , in the type - ii 2hdm the @xmath52 coupling and the @xmath53 coupling can be simultaneously enhanced by @xmath54 , while in the l2hdm only the @xmath53 coupling is enhanced by @xmath55 . the structures of the nmssm and the nmssm are described by their superpotentials and corresponding soft - breaking terms , which are given by @xcite @xmath56 where @xmath57 is the superpotential of the mssm without the @xmath25 term , @xmath58 and @xmath59 are higgs doublet and singlet superfields with @xmath60 and @xmath61 being their scalar component respectively , @xmath62 , @xmath63 , @xmath64 , @xmath65 , @xmath66 and @xmath67 are soft breaking parameters , and @xmath68 and @xmath69 are coefficients of the higgs self interactions .    with the superpotentials and the soft - breaking terms , one can get the higgs potentials of the nmssm and the nmssm respectively . like the 2hdm , the higgs bosons with same cp property will mix and the mass eigenstates are obtained by diagonalizing the corresponding mass matrices : @xmath70 where the fields on the right hands of the equations are component fields of @xmath71 , @xmath72 and @xmath61 defined by @xmath73 @xmath74 and @xmath75 are respectively the cp - even and cp - odd neutral higgs bosons , @xmath76 and @xmath77 are goldstone bosons eaten by @xmath1 and @xmath78 , and @xmath79 is the charged higgs boson . so both the nmssm and nmssm predict three cp - even higgs bosons , two cp - odd higgs bosons and one pair of charged higgs bosons . in general , the lighter cp - odd higgs @xmath0 in these model is the mixture of the singlet field @xmath80 and the doublet field combination , @xmath81 , i.e. @xmath82 and its couplings to down - type quarks are then proportional to @xmath83 . so for singlet dominated @xmath0 , @xmath84 is small and the couplings are suppressed . as a comparison , the interactions of @xmath0 with the squarks are given by@xcite @xmath85 i.e. the interaction does not vanish when @xmath86 approaches zero . just like the 2hdm where we use the vevs of the higgs fields as fundamental parameters , we choose @xmath68 , @xmath69 , @xmath87 , @xmath88 , @xmath66 and @xmath89 as input parameters for the nmssm@xcite and @xmath68 , @xmath54 , @xmath88 , @xmath65 , @xmath90 and @xmath91 as input parameters for the nmssm@xcite . about the nmssm and the nmssm , three points should be noted . the first is for the two models , there is no explicit @xmath92term , and the effective @xmath25 parameter ( @xmath93 ) is generated when the scalar component of @xmath59 develops a vev . the second is , the nmssm is actually same as the nmssm with @xmath94@xcite , because the tadpole terms @xmath95 and its soft breaking term @xmath96 in the nmssm do not induce any interactions , except for the tree - level higgs boson masses and the minimization conditions . and the last is despite of the similarities , the nmssm has its own peculiarity , which comes from its neutralino sector . in the basis @xmath97 , its neutralino mass matrix is given by @xcite @xmath98 where @xmath99 and @xmath100 are @xmath101 and @xmath102 gaugino masses respectively , @xmath103 , @xmath104 , @xmath105 and @xmath106 . after diagonalizing this matrix one can get the mass eigenstate of the lightest neutralino @xmath107 with mass taking the following form @xcite @xmath108 this expression implies that @xmath107 must be lighter than about @xmath109 gev for @xmath110 ( from lower bound on chargnio mass ) and @xmath111 ( perturbativity bound ) . like the other supersymmetric models , @xmath107 as the lightest sparticle acts as the dark matter in the universe , but due to its singlino - dominated nature , it is difficult to annihilate sufficiently to get the correct density in the current universe . so the relic density of @xmath107 plays a crucial way in selecting the model parameters . for example , as shown in @xcite , for @xmath112 , there is no way to get the correct relic density , and for the other cases , @xmath107 mainly annihilates by exchanging @xmath1 boson for @xmath113 , or by exchanging a light cp - odd higgs boson @xmath0 with mass satisfying the relation @xmath114 for @xmath115 . for the annihilation , @xmath54 and @xmath25 are required to be less than 10 and @xmath116 respectively because through eq.([mass - exp ] ) a large @xmath87 or @xmath25 will suppress @xmath117 to make the annihilation more difficult . the properties of the lightest cp - odd higgs boson @xmath0 , such as its mass and couplings , are also limited tightly since @xmath0 plays an important role in @xmath107 annihilation . the phenomenology of the nmssm is also rather special , and this was discussed in detail in @xcite . in the type - ii 2hdm , l2hdm , nmssm and nmssm , the rare @xmath1-decays @xmath118 ( @xmath22 ) , @xmath3 and @xmath4 may proceed by the feynman diagrams shown in fig.[fig1 ] , fig.[fig2 ] and fig.[fig3 ] respectively . for these diagrams , the intermediate state @xmath119 represents all possible cp - even higgs bosons in the corresponding model , i.e. @xmath41 and @xmath42 in type - ii 2hdm and l2hdm and @xmath41 , @xmath42 and @xmath120 in nmssm and nmssm . in order to take into account the possible resonance effects of @xmath119 in fig.[fig1](c ) for @xmath2 and fig.[fig3 ] ( a ) for @xmath11 , we have calculated all the decay modes of @xmath119 and properly included the width effect in its propagator . as to the decay @xmath121 , two points should be noted . one is , unlike the decays @xmath6 and @xmath11 , this process proceeds only through loops mediated by quarks / leptons in the type - ii 2hdm and l2hdm , and additionally by sparticles in the nmssm and nmssm . so in most cases its rate should be much smaller than the other two . the other is due to cp - invariance , loops mediated by squarks / sleptons give no contribution to the decay@xcite . in actual calculation , this is reflected by the fact that the coupling coefficient of @xmath122 differs from that of @xmath123 by a minus sign ( see eq.([asqsq ] ) ) , and as a result , the squark - mediated contributions to @xmath121 are completely canceled out .    with regard to the rare decay @xmath11 , we have more explanations . in the lowest order , this decay proceeds by the diagram shown in fig.[fig3 ] ( a ) , and hence one may think that , as a rough estimate , it is enough to only consider the contributions from fig.[fig3](a ) . however , we note that in some cases of the type - ii 2hdm and l2hdm , due to the cancelation of the contributions from different @xmath119 in fig.[fig3 ] ( a ) and also due to the potentially largeness of @xmath124 couplings ( i.e. larger than the electroweak scale @xmath125 ) , the radiative correction from the higgs - mediated loops may dominate over the tree level contribution even when the tree level prediction of the rate , @xmath126 , exceeds @xmath20 . on the other hand , we find the contribution from quark / lepton - mediated loops can be safely neglected if @xmath127 in the type - ii 2hdm and the l2hdm . in the nmssm and the nmssm , besides the corrections from the higgs- and quark / lepton - mediated loops , loops involving sparticles such as squarks , charginos and neutralinos can also contribute to the decay . we numerically checked that the contributions from squarks and charginos can be safely neglected if @xmath127 . we also calculated part of potentially large neutralino correction ( note that there are totally about @xmath128 diagrams for such correction ! ) and found they can be neglected too . since considering all the radiative corrections will make our numerical calculation rather slow , we only include the most important correction , namely that from higgs - mediated loops , in presenting our results for the four models . one can intuitively understand the relative smallness of the sparticle contribution to @xmath11 as follows . first consider the squark contribution which is induced by the @xmath129 interaction ( @xmath130 denotes the squark in chirality state ) and the @xmath131 interaction through box diagrams . because the @xmath132 interaction conserves the chirality of the squarks while the @xmath133 interaction violates the chirality , to get non - zero contribution to @xmath11 from the squark loops , at least four chiral flippings are needed , with three of them provided by @xmath131 interaction and the rest provided by the left - right squark mixing . this means that , if one calculates the amplitude in the chirality basis with the mass insertion method , the amplitude is suppressed by the mixing factor @xmath134 with @xmath135 being the off diagonal element in squark mass matrix . next consider the chargino / neutralino contributions . since for a light @xmath0 , its doublet component , parameterized by @xmath84 in eq.([mixing ] ) , is usually small , the couplings of @xmath0 with the sparticles will never be tremendously large@xcite . so the chargino / neutralino contributions are not important too . in our calculation of the decays , we work in the mass eigenstates of sparticles instead of in the chirality basis . for the type - ii 2hdm and the l2hdm , we consider the following constraints @xcite :    * theoretical constraints on @xmath136 from perturbativity , unitarity and requirements that the scalar potential is finit at large field values and contains no flat directions @xcite , which imply that @xmath137 * the constraints from the lep search for neutral higgs bosons . we compute the signals from the higgs - strahlung production @xmath138 ( @xmath139 ) with @xmath140 @xcite and from the associated production @xmath141 with @xmath142 @xcite , and compare them with the corresponding lep data which have been inputted into our code . we also consider the constraints from @xmath138 by looking for a peak of @xmath143 recoil mass distribution of @xmath1-boson @xcite and the constraint of @xmath144 mev when @xmath145 @xcite . + these constraints limit the quantities such as @xmath146 \times br ( h_i \to \bar{b } b ) $ ] on the @xmath147 plane with the the subscript @xmath148 denoting the coupling coefficient of the @xmath149 interaction . they also impose a model - dependent lower bound on @xmath150 , e.g. , @xmath151 for the type - ii 2hdm ( from our scan results ) , @xmath152 for the l2hdm@xcite , and @xmath153 for the nmssm @xcite . these bounds are significantly lower than that of the sm , i.e. @xmath154 , partially because in new physics models , unconventional decay modes of @xmath155 such as @xmath156 are open up . as to the nmssm , another specific reason for allowing a significantly lighter cp - even higgs boson is that the boson may be singlet - dominated in this model . + with regard to the lightest cp - odd higgs boson @xmath0 , we checked that there is no lower bound on its mass so long as the @xmath157 interaction is weak or @xmath155 is sufficiently heavy . * the constraints from the lep search for a light higgs boson via the yukawa process @xmath158 with @xmath22 and @xmath61 denoting a scalar @xcite . these constraints can limit the @xmath159 coupling versus @xmath160 in new physics models . * the constraints from the cleo - iii limit on @xmath161 and the latest babar limits on @xmath162 . these constraints will put very tight constraints on the @xmath163 coupling for @xmath164 . in our analysis , we use the results of fig.8 in the second paper of @xcite to excluded the unfavored points . * the constraints from @xmath165 couplings . since the higgs sector can give sizable higher order corrections to @xmath165 couplings , we calculate them to one loop level and require the corrected @xmath165 couplings to lie within the @xmath166 range of their fitted value . the sm predictions for the couplings at @xmath1-pole are given by @xmath167 and @xmath168 @xcite , and the fitted values are given by @xmath169 and @xmath170 , respectively@xcite . we adopt the formula in @xcite to the 2hdm in our calculation . * the constraints from @xmath171 leptonic decay . we require the new physics correction to the branching ratio @xmath172 to be in the range of @xmath173 @xcite . we use the formula in @xcite in our calculation . + about the constraints ( 5 ) and ( 6 ) , two points should be noted . one is all higgs bosons are involved in the constraints by entering the self energy of @xmath171 lepton , the @xmath174 vertex correction or the @xmath175 vertex correction , and also the box diagrams for @xmath176@xcite . since the yukawa couplings of the higgs bosons to @xmath171 lepton get enhanced by @xmath54 and so do the corrections , @xmath54 must be upper bounded for given spectrum of the higgs sector . generally speaking , the lighter @xmath0 is , the more tightly @xmath54 is limited@xcite . the other point is in the type - ii 2hdm , @xmath177 , b - physics observables as well as @xmath178 decays discussed above can constraint the model in a tighter way than the constraints ( 5 ) and ( 6 ) since the yukawa couplings of @xmath171 lepton and @xmath179 quark are simultaneously enhanced by @xmath54 . but for the l2hdm , because only the yukawa couplings of @xmath171 lepton get enhanced ( see eq.[yukawa ] ) , the constraints ( 5 ) and ( 6 ) are more important in limiting @xmath54 . * indirect constraints from the precision electroweak observables such as @xmath180 , @xmath181 and @xmath182 , or their combinations @xmath183 @xcite . we require @xmath184 to be compatible with the lep / sld data at @xmath185 confidence level@xcite . we also require new physics prediction of @xmath186 is within the @xmath187 range of its experimental value . the latest results for @xmath188 are @xmath189 ( measured value ) and @xmath190 ( sm prediction ) for @xmath191 gev @xcite . in our code , we adopt the formula for these observables presented in @xcite to the type - ii 2hdm and the l2hdm respectively . + in calculating @xmath180 , @xmath181 and @xmath182 , we note that these observables get dominant contributions from the self energies of the gauge bosons @xmath1 , @xmath192 and @xmath193 . since there is no @xmath194 coupling or @xmath195 coupling , @xmath0 must be associated with the other higgs bosons to contribute to the self energies . so by the uv convergence of these quantities , one can infer that , for the case of a light @xmath0 and @xmath196 , these quantities depend on the spectrum of the higgs sector in a way like @xmath197 at leading order , which implies that a light @xmath0 can still survive the constraints from the precision electroweak observables given the splitting between @xmath150 and @xmath198 is moderate@xcite . * the constraints from b physics observables such as the branching ratios for @xmath199 , @xmath200 and @xmath201 , and the mass differences @xmath202 and @xmath203 . we require their theoretical predications to agree with the corresponding experimental values at @xmath187 level . + in the type - ii 2hdm and the l2hdm , only the charged higgs boson contributes to these observables by loops , so one can expect that @xmath198 versus @xmath54 is to be limited . combined analysis of the limits in the type - ii 2hdm has been done by the ckmfitter group , and the lower bound of @xmath204 as a function of @xmath87 was given in fig.11 of @xcite . this analysis indicates that @xmath198 must be heavier than @xmath205 at @xmath185 c.l . regardless the value of @xmath54 . in this work , we use the results of fig.11 in @xcite to exclude the unfavored points . as for the l2hdm , b physics actually can not put any constraints@xcite because in this model the couplings of the charged higgs boson to quarks are proportional to @xmath206 and in the case of large @xmath54 which we are interested in , they are suppressed . in our analysis of the l2hdm , we impose the lep bound on @xmath198 , i.e. @xmath207@xcite . * the constraints from the muon anomalous magnetic moment @xmath208 . now both the theoretical prediction and the experimental measured value of @xmath208 have reached a remarkable precision , but a significant deviation still exists : @xmath209 @xcite . in the 2hdm , @xmath208 gets additional contributions from the one - loop diagrams induced by the higgs bosons and also from the two - loop barr - zee diagrams mediated by @xmath0 and @xmath155@xcite . if the higgs bosons are much heavier than @xmath25 lepton mass , the contributions from the barr - zee diagrams are more important , and to efficiently alleviate the discrepancy of @xmath208 , one needs a light @xmath0 along with its enhanced couplings to @xmath25 lepton and also to heavy fermions such as bottom quark and @xmath171 lepton to push up the effects of the barr - zee diagram@xcite . the cp - even higgs bosons are usually preferred to be heavy since their contributions to @xmath208 are negative . + in the type - ii 2hdm , because @xmath54 is tightly constrained by the process @xmath210 at the lep@xcite and the @xmath178 decay@xcite , the barr - zee diagram contribution is insufficient to enhance @xmath208 to @xmath187 range around its measured value@xcite . so in our analysis , we require the type - ii 2hdm to explain @xmath208 at @xmath211 level . while for the l2hdm , @xmath54 is less constrained compared with the type - ii 2hdm , and the barr - zee diagram involving the @xmath171-loop is capable to push up greatly the theoretical prediction of @xmath208@xcite . therefore , we require the l2hdm to explain the discrepancy at @xmath187 level . + unlike the other constraints discussed above , the @xmath208 constraint will put a two - sided bound on @xmath54 since on the one hand , it needs a large @xmath54 to enhance the barr - zee contribution , but on the other hand , too large @xmath54 will result in an unacceptable large @xmath208 . * since this paper concentrates on a light @xmath0 , the decay @xmath212 is open up with a possible large decay width . we require the width of any higgs boson to be smaller than its mass to avoid a too fat higgs boson@xcite . we checked that for the scenario characterized by @xmath213 , the coefficient of @xmath214 interaction is usually larger than the electroweak scale @xmath125 , and consequently a large decay width is resulted . for the nmssm and nmssm , the above constraints become more complicated because in these models , not only more higgs bosons are involved in , but also sparticles enter the constraints . so it is not easy to understand some of the constraints intuitively . take the process @xmath199 as an example . in the supersymmetric models , besides the charged higgs contribution , chargino loops , gluino loops as well as neutralino loops also contribute to the process@xcite , and depending on the susy parameters , any of these contributions may become dominated over or be canceled by other contributions . as a result , although the charged higgs affects the process in the same way as that in the type - ii 2hdm , charged higgs as light as @xmath215 is still allowed even for @xmath216@xcite .    since among the constraints , @xmath208 is rather peculiar in that it needs new physics to explain the discrepancy between @xmath217 and @xmath218 , we discuss more about its dependence on susy parameters . in the nmssm and the nmssm , @xmath208 receives contributions from higgs loops and neutralino / chargino loops . for the higgs contribution , it is quite similar to that of the type - ii 2hdm except that more higgs bosons are involved in@xcite . for the neutralino / chargino contribution , in the light bino limit ( i.e. @xmath219 ) , it can be approximated by@xcite @xmath220 for @xmath221 with @xmath222 being smuon mass . so combining the two contributions together , one can learn that a light @xmath0 along with large @xmath54 and/or light smuon with moderate @xmath87 are favored to dilute the discrepancy .    because more parameters are involved in the constraints on the supersymmetric models , we consider following additional constraints to further limit their parameters :    * direct bounds on sparticle masses from the lep1 , the lep2 and the tevatron experiments @xcite . * the lep1 bound on invisible z decay @xmath223 ; the lep2 bound on neutralino production @xmath224 and @xmath225@xcite . * dark matter constraints from the wmap relic density 0.0975 @xmath226 0.1213 @xcite . note that among the above constraints , the constraint ( 2 ) on higgs sector and the constraint ( c ) on neutralino sector are very important . this is because in the supersymmetric models , the sm - like higgs is upper bounded by about @xmath227 at tree level and by about @xmath228 at loop level , and that the relic density restricts the lsp annihilation cross section in a certain narrow range .    in our analysis of the nmssm , we calculate the constraints ( 3 ) and ( 5 - 7 ) by ourselves and utilize the code nmssmtools @xcite to implement the rest constraints . we also extend nmssmtools to the nmssm to implement the constraints . for the extension , the most difficult thing we faced is how to adapt the code micromegas@xcite to the nmssm case . we solve this problem by noting the following facts :    * as we mentioned before , the nmssm is actually same as the nmssm with the trilinear singlet term setting to zero . so we can utilize the model file of the nmssm as the input of the micromegas and set @xmath229 . * since in the nmssm , the lsp is too light to annihilate into higgs pairs , there is no need to reconstruct the effective higgs potential to calculate precisely the annihilation channel @xmath230 with @xmath61 denoting any of higgs bosons@xcite . we thank the authors of the nmssmtools for helpful discussion on this issue when we finish such extension@xcite . with the above constraints , we perform four independent random scans over the parameter space of the type - ii 2hdm , the l2hdm , the nmssm and the nmssm respectively . we vary the parameters in following ranges : @xmath231 for the type - ii 2hdm , @xmath232 for the l2hdm , @xmath233 for the nmssm , and @xmath234 for the nmssm .    in performing the scans , we note that for the nmssm and the nmssm , some constraints also rely on the gaugino masses and the soft breaking parameters in the squark sector and the slepton sector . since these parameters affect little on the properties of @xmath0 , we fix them to reduce the number of free parameters in our scan . for the squark sector , we adopt the @xmath235 scenario which assumes that the soft mass parameters for the third generation squarks are degenerate : @xmath236 800 gev , and that the trilinear couplings of the third generation squarks are also degenerate , @xmath237 with @xmath238 . for the slepton sector , we assume all the soft - breaking masses and trilinear parameters to be 100 gev . this setting is necessary for the nmssm since this model is difficult to explain the muon anomalous moment at @xmath239 level for heavy sleptons@xcite . finally , we assume the grand unification relation @xmath240 for the gaugino masses with @xmath241 being fine structure constants of the different gauge group .    with large number of random points in the scans , we finally get about @xmath242 , @xmath243 , @xmath244 and @xmath242 samples for the type - ii 2hdm , the l2hdm , the nmssm and the nmssm respectively which survive the constraints and satisfy @xmath245 . analyzing the properties of the @xmath0 indicates that for most of the surviving points in the nmssm and the nmssm , its dominant component is the singlet field ( numerically speaking , @xmath246 ) so that its couplings to the sm fermions are suppressed@xcite . our analysis also indicates that the main decay products of @xmath0 are @xmath247 for the l2hdm@xcite , @xmath248 ( dominant ) and @xmath247 ( subdominant ) for the type - ii 2hdm , the nmssm and the nmssm , and in some rare cases , neutralino pairs in the nmssm@xcite .    in fig.[fig4 ] , we project the surviving samples on the @xmath249 plane . this figure shows that the allowed range of @xmath54 is from @xmath250 to @xmath251 in the type - ii 2hdm , and from @xmath252 to @xmath253 in the l2hdm . just as we introduced before , the lower bounds of @xmath254 come from the fact that we require the models to explain the muon anomalous moment , while the upper bound is due to we have imposed the constraint from the lep process @xmath255 , which have limited the upper reach of the @xmath256 coupling for light @xmath61 @xcite(for the dependence of @xmath256 coupling on @xmath54 , see sec . this figure also indicates that for the nmssm and the nmssm , @xmath54 is upper bounded by @xmath257 . for the nmssm , this is because large @xmath87 can suppress the dark matter mass to make its annihilation difficult ( see @xcite and also sec . ii ) , but for the nmssm , this is because we choose a light slepton mass so that large @xmath54 can enhance @xmath208 too significantly to be experimentally unacceptable . we checked that for the slepton mass as heavy as @xmath258 , @xmath259 is still allowed for the nmssm .    in fig.[fig5 ] and fig.[fig6 ] , we show the branching ratios of @xmath260 and @xmath261 respectively . fig.[fig5 ] indicates , among the four models , the type - ii 2hdm predicts the largest ratio for @xmath260 with its value varying from @xmath262 to @xmath263 . the underlying reason is in the type - ii 2hdm , the @xmath264 coupling is enhanced by @xmath54 ( see fig.[fig4 ] ) , while in the other three model , the coupling is suppressed either by @xmath265 or by the singlet component of the @xmath0 . fig.[fig6 ] shows that the l2hdm predicts the largest rate for @xmath266 with its value reaching @xmath5 in optimum case , and for the other three models , the ratio of @xmath261 is at least about one order smaller than that of @xmath267 . this feature can be easily understood from the @xmath268 coupling introduced in sect . we emphasize that , if the nature prefers a light @xmath0 , @xmath260 and/or @xmath269 in the type - ii 2hdm and the l2hdm will be observable at the gigaz . then by the rates of the two decays , one can determine whether the type - ii 2hdm or the l2hdm is the right theory . on the other hand , if both decays are observed with small rates or fail to be observed , the singlet extensions of the mssm are favored .    in fig.[fig7 ] , we show the rate of @xmath3 as the function of @xmath270 . this figure indicates that the branching ratio of @xmath121 can reach @xmath271 , @xmath272 , @xmath273 and @xmath274 for the optimal cases of the type - ii 2hdm , the l2hdm , the nmssm and the nmssm respectively , which implies that the decay @xmath121 will never be observable at the gigaz if the studied model is chosen by nature . the reason for the smallness is , as we pointed out before , that the decay @xmath121 proceeds only at loop level . comparing the optimum cases of the type - ii 2hdm , the nmssm and the nmssm shown in fig.5 - 7 , one may find that the relation @xmath275 holds for any of the decays . this is because the decays are all induced by the yukawa couplings with similar structure for the models . in the supersymmetric models , the large singlet component of the light @xmath0 is to suppress the yukawa couplings , and the @xmath0 in the nmssm has more singlet component than that in the nmssm . next we consider the decay @xmath11 , which , unlike the above decays , depends on the higgs self interactions . in fig.[fig8 ] we plot its rate as a function of @xmath270 and this figure indicates that the @xmath276 may be the largest among the ratios of the exotic @xmath1 decays , reaching @xmath277 in the optimum cases of the type - ii 2hdm , the l2hdm and the nmssm . the underlying reason is , in some cases , the intermediate state @xmath119 in fig.[fig3 ] ( a ) may be on - shell . in fact , we find this is one of the main differences between the nmssm and the nmssm , that is , in the nmssm , @xmath119 in fig.[fig3 ] ( a ) may be on - shell ( corresponds to the points with large @xmath278 ) while in the nmssm , this seems impossible . so we conclude that the decay @xmath11 may serve as an alternative channel to test new physics models , especially it may be used to distinguish the nmssm from the nmssm if the supersymmetry is found at the lhc and the @xmath11 is observed at the gigaz with large rate . before we end our discussion , we note that in the nmssm , the higgs boson @xmath0 may be lighter than @xmath279 without conflicting with low energy data from @xmath178 decays and the other observables ( see fig.[fig4]-[fig8 ] ) . in this case , @xmath0 is axion - like as pointed out in @xcite . we checked that , among the rare @xmath1 decays discussed in this paper , the largest branching ratio comes from @xmath280 which can reach @xmath281 . since in this case , the decay product of @xmath0 is highly collinear muon pair , detecting the decay @xmath280 may need some knowledge about detectors , which is beyond our discussion . in this paper , we studied the rare @xmath1-decays @xmath2 ( @xmath7 ) , @xmath282 and @xmath4 in the type - ii 2hdm , lepton - specific 2hdm , nmssm and nmssm , which predict a light cp - odd higgs boson @xmath0 . in the parameter space allowed by current experiments , the branching ratio can be as large as @xmath5 for @xmath118 , @xmath8 for @xmath3 and @xmath9 for @xmath4 , which implies that the decays @xmath2 and @xmath283 may be accessible at the gigaz option . since different models predict different size of branching ratios , these decays can be used to distinguish different model through the measurement of these rare decays . this work was supported in part by hastit under grant no . 2009hastit004 , by the national natural science foundation of china ( nnsfc ) under grant nos . 10821504 , 10725526 , 10635030 , 10775039 , 11075045 and by the project of knowledge innovation program ( pkip ) of chinese academy of sciences under grant no . .        for some reviews , see , e.g. , m.  a.  perez , g.  tavares - velasco and j.  j.  toscano , int . j.  mod . a * 19 * , 159 ( 2004 ) ; j. m. yang , arxiv:1006.2594 . j.  i.  illana , m.  masip , 67 , 035004 ( 2003 ) ; j. cao , z. xiong , j. m. yang , 32 , 245 ( 2004 ) . d. atwood _ et al_. , 66 , 093005 ( 2002 ) . j. kalinowski , and s. pokorski , 219 , 116 ( 1989 ) ; a. djouadi , p. m. zerwas and j. zunft , 259 , 175 ( 1991 ) ; a. djouadi , j. kalinowski , and p. m. zerwas , z. phys . c * 54 * , 255 ( 1992 ) . m. krawczyk , _ et al . _ , 19 , 463 ( 2001 ) ; 8 , 495 ( 1999 ) . j. f. gunion , g. gamberini and s. f. novaes , 38 , 3481 ( 1988 ) ; thomas j. weiler and tzu - chiang yuan , 318 , 337 ( 1989 ) ; a. djouadi , _ et al . _ , 1 , 163 ( 1998)[hep - ph/9701342 ] . d.  chang and w.  y.  keung , phys . lett .  * 77 * , 3732 ( 1996 ) . e.  keith and e.  ma , 57 , 2017 ( 1998 ) ; m.  a.  perez , g.  tavares - velasco and j.  j. toscano , int . j.  mod.phys . a * 19 * , 159 ( 2004 ) . f.  larios , g.  tavares - velasco and c. p.  yuan , 64 , 055004 ( 2001 ) ; 66 , 075006 ( 2002 ) . a. djouadi , _ et al . _ , 10 , 27 ( 1999 ) [ hep - ph/9903229 ] . for a detailed introduction of the nmssm , see f.  franke and h. fraas , int . j.  mod . a * 12 * ( 1997 ) 479 ; for a recent review of the nmssm , see for example , u. ellwanger , c. hugonie , and a. m. teixeira , arxiv : 0910.1785 . see , e.g. , j.  r.  ellis , j.  f.  gunion , h.  e.  haber , l.  roszkowski and f.  zwirner , phys .  rev . d * 39 * ( 1989 ) 844 ; m.  drees , int . j.  mod . phys .  a * 4 * ( 1989 ) 3635 ; u.  ellwanger , m.  rausch de traubenberg and c.  a.  savoy , phys . b * 315 * ( 1993 ) 331 ; nucl . b * 492 * ( 1997 ) 21 ; d.j . miller , r. nevzorov , p.m. zerwas , 681 , 3 ( 2004 ) .    c.  panagiotakopoulos , k.  tamvakis , 446 , 224 ( 1999 ) ; 469 , 145 ( 1999 ) ; c. panagiotakopoulos , a. pilaftsis , 63 , 055003 ( 2001 ) ; a.  dedes , _ et al . _ , 63 , 055009 ( 2001 ) ; a.  menon , _ et al . _ , 70 , 035005 ( 2004 ) ; v.  barger , _ et al . _ , 630 , 85 ( 2005 ) . c.  balazs , _ et al . _ , 0706 , 066 ( 2007 ) . b. a. dobrescu , k. t. matchev , 0009 , 031 ( 2000 ) ; a. arhrib , k. cheung , t. j. hou , k. w. song , hep - ph/0611211 ; 0703 , 073 ( 2007 ) ; x. g. he , j. tandean , and g. valencia , 98 , 081802 ( 2007 ) ; 0806 , 002 ( 2008 ) ; f. domingo _ et al_. , 0901 , 061 ( 2009 ) ; gudrun hiller , 70 , 034018 ( 2004 ) ; r. dermisek , and john f. gunion , 75 , 075019 ( 2007 ) ; 79 , 055014 ( 2009 ) ; 81 , 055001 ( 2010 ) ; r. dermisek , john f. gunion , and b. mcelrath , 76 , 051105 ( 2007 ) ; z. heng , _ et al_. , 77 , 095012 ( 2008 ) ; a. belyaev _ et al_. , 81 , 075021 ( 2010 ) ; d. das and u.  ellwanger , arxiv:1007.1151 [ hep - ph ] . s.  andreas , o.  lebedev , s.  ramos - sanchez and a.  ringwald , arxiv:1005.3978 [ hep - ph ] . j.  f.  gunion , jhep * 0908 * , 032 ( 2009 ) ; r. dermisek and j.  f.  gunion , phys .  rev . d * 81 * , 075003 ( 2010 ) . r.  dermisek and j.  f. gunion , phys . lett .   * 95 * , 041801 ( 2005 ) ; phys . d * 73 * , 111701 ( 2006 ) . j. cao , h. e. logan , j. m. yang , 79 , 091701 ( 2009 ) . j. cao , p. wan , l. wu , j. m. yang , 80 , 071701 ( 2009 ) . j. f. gunion and h. e. haber , 67 , 075019 ( 2003 ) . r.  m.  barnett , _ et al . _ , phys . b * 136 * , 191 ( 1984 ) ; r.  m.  barnett , g.  senjanovic and d.  wyler , phys . d * 30 * , 1529 ( 1984 ) ; y.  grossman , nucl . b * 426 * , 355 ( 1994 ) . h.  s.  goh , l.  j.  hall and p. kumar , jhep * 0905 * , 097 ( 2009 ) ; a.  g. akeroyd and w.  j.  stirling , nucl . b * 447 * , 3 ( 1995 ) ; a.  g.  akeroyd , phys . b * 377 * , 95 ( 1996 ) ; h.  e.  logan and d.  maclennan , phys .  rev . d * 79 * , 115022 ( 2009 ) ; m. aoki , _ et al . _ , arxiv:0902.4665 [ hep - ph ] . v.  barger , p.  langacker , h.  s.  lee and g. shaughnessy , phys . d * 73 * , 115010 ( 2006 ) . s. hesselbach , _ et . _ , arxiv:0810.0511v2 [ hep - ph ] . de vivie and p.  janot [ aleph collaboration ] , pa13 - 027 contribution to the international conference on high energy physics , warsaw , poland , 2531 july 1996 ; j. kurowska , o.  grajek and p.  zalewski [ delphi collaboration ] , cern - open-99 - 385 . [ aleph collaboration and delphi collaboration and l3 collaboration ] , phys . rept .   * 427 * , 257 ( 2006 ) . j.  cao and j.  m.  yang , jhep * 0812 * , 006 ( 2008 ) . m.  krawczyk and d.  temes , eur . j.   c * 44 * , 435 ( 2005 ) . g.  altarelli and r.  barbieri , 253 , 161 ( 1991 ) ; m. e. peskin , t. takeuchi , 46 , 381 ( 1992 ) . c. amsler , _ et al . _ , ( particle data group ) , 667 , 1 ( 2008 ) . o. deschamps , s.  descotes - genon , s.  monteil , v.  niess , s.  tjampens and v.  tisserand , arxiv:0907.5135 [ hep - ph ] . s.  su and b. thomas , phys . d * 79 * , 095014 ( 2009 ) . g. abbiendi , _ et al . _ , eur .  phys . j.   c * 32 * , 453 ( 2004 ) . m.  davier , _ et al . _ , 66 , 1 ( 2010 ) . k.  cheung , _ et al . _ , phys . d * 64 * , 111301 ( 2001 ) . k.  cheung and o.  c.  w. kong , phys . d * 68 * , 053003 ( 2003 ) . t. besmer , c. greub , t.hurth , 609 , 359 ( 2001 ) ; f. borzumati , _ et al . _ , 62 , 075005(2000 ) . j.  cao , k.  i.  hikasa , w.  wang , j.  m.  yang and l.  x.  yu , phys . d * 82 * , 051701 ( 2010 ) [ arxiv:1006.4811 [ hep - ph ] ] . j.  f.  gunion , _ et . d * 73 * , 015011 ( 2006 ) . martin and j.  d.  wells , phys . d * 64 * , 035003 ( 2001 ) . j.  abdallah _ et al . _ , eur . j.   c * 31 * , 421 ( 2004 ) ; g.  abbiendi _ et al . _ , eur . j. c * 35 * , 1 ( 2004 ) . j.  dunkley _ et al . _ [ wmap collaboration ] , astrophys . j.  suppl . * 180 * , 306 ( 2009 ) [ arxiv:0803.0586 [ astro - ph ] ] . u. ellwanger _ et al . _ , 02 , 066 ( 2005 ) . g.  belanger , f.  boudjema , a.  pukhov and a.  semenov , comput . commun .   * 174 * , 577 ( 2006 ) ; comput . phys .  commun . * 176 * , 367 ( 2007 ) . g.  belanger , f.  boudjema , c. hugonie , a.  pukhov and a.  semenov , jcap * 0509 * , 001 ( 2005 ) ."""

        ARTICLE_MAGNET = r"""it is well known that the classical magnetoresistance ( mr ) in metals or semiconductors with a closed free electron fermi surface increases quadratically with increasing magnetic field @xmath2 for @xmath3 and saturates when @xmath4 . here @xmath5 is the zero - magnetic - field mobility . hence , the extraordinarily high and linear mr ( lmr ) , which breaks this familiar rule , has been gaining much attention as soon as its discovery . in the past decade , this unexpected lmr has been reported in silver chalcogenide,@xcite indium antimonide,@xcite silicon,@xcite mnas - gaas composite material,@xcite and graphene.@xcite    kapitza s linear law@xcite indicates that the metal shows a magnetoresistance linear in perpendicular magnetic field when it has an open fermi surface and a mean free path longer than the electronic larmor radius . recently , another two models , irrespective of the open fermi surface , have been constructed to provide possible mechanisms for the lmr phenomenon . abrikosov suggested a quantum - limit origin of lmr for the homogenous system with a gapless linear energy spectrum.@xcite his model requires that landau levels are well formed and the carrier concentration is small that all electrons occupy only the lowest landau band . alternatively , parish and littlewood developed a classical model without involving linear spectrum.@xcite ignoring the concrete microscopic mechanism , they attributed this unusual mr to the mobility fluctuations in a strongly inhomogenous system . topological insulators@xcite ( tis ) are novel materials with a full energy gap in bulk , while there are gapless surface states . due to its unique band structure with only one helical dirac cone and linear energy dispersion,@xcite the surface states of the ti bi@xmath0se@xmath1 become an excellent platform for the study of quantum - limit lmr . the recent experiment in this flat surface system , however , reported that a large positive mr , which becomes very linear above a characteristic field of @xmath6@xmath7@xmath8 t , was observed even in an opposite situation where the carrier sheet density is high that electrons occupy more than one landau levels.@xcite moreover , they found that raising temperature to room temperature almost has no influence on the observed lmr . it is striking that this observation is in conflict with abrikosov s model and also with the classical parish - littlewood model . so far a reliable theoretical scheme capable of explaining this novel experiment has still been lacking .    in this paper , we generalize the balance - equation approach@xcite to a system modeling the surface states of a three - dimensional ti to investigate the two - dimensional magnetotransport in it . we find that a positive , nonsaturating and dominantly linear magnetoresistance can appear within quite wide magnetic - field range in the ti surface state having a positive and finite effective g - factor . this linear magnetoresistance shows up in the system of high carrier concentration and low mobility when electrons are in extended states and spread over many smeared landau levels , and persists up to room temperature , providing a possible mechanism for the recently observed linear magnetoresistance in topological insulator bi@xmath0se@xmath1 nanoribbons.@xcite we consider the surface state of a bi@xmath0se@xmath1-type large bulk gap ti in the @xmath9-@xmath10 plane under the influence of a uniform magnetic field @xmath11 applied along the @xmath12 direction.@xcite following the experimental observation,@xcite we assume that the fermi energy locates in the gap of the bulk band and above the dirac point , i.e. the surface carriers are electrons . further , the separations of the fermi energy from the bottom of bulk band and dirac point are much larger than the highest temperature ( @xmath13 ) considered in this work . hence , the contribution from the bulk band to the magnetotransport is negligible . these electrons , scattered by randomly distributed impurities and by phonons , are driven by a uniform in - plane electric field @xmath14 in the topological surface . the hamiltonian of this many - electron and phonon system consists of an electron part @xmath15 , a phonon part @xmath16 , and electron - impurity and electron - phonon interactions @xmath17 and @xmath18 : @xmath19 here , the electron hamiltonian is taken in the form @xmath20 , \ ] ] in which @xmath21 , @xmath22 , @xmath23 and @xmath24 , stand , respectively , for the canonical momentum , coordinate , momentum and spin operators of the @xmath25th electron having charge @xmath26 , @xmath27 is the vector potential of the perpendicular magnetic field @xmath28 in the landau gauge , @xmath29 is the fermi velocity , @xmath30 is the effective g - factor of the surface electron , and @xmath31 is the bohr magneton with @xmath32 the free electron mass . the sum index @xmath25 in eq.([helectron ] ) goes over all electrons of total number @xmath33 in the surface state of unit area .    in the frame work of balance equation approach,@xcite the two - dimensional center - of - mass ( c.m . ) momentum and coordinate @xmath34 and @xmath35 , and the relative - electron momenta and coordinates @xmath36 and @xmath37 are introduced to write the hamiltonian @xmath15 into the sum of a single - particle c.m . part @xmath38 and a many - particle relative - electron part @xmath39 : @xmath40 , with @xmath41.\end{aligned}\ ] ] in this , @xmath42 is the canonical momentum of the center - of - mass and @xmath43 is the canonical momentum for the @xmath25th relative electron . here we have also introduced c.m . spin operators @xmath44 and @xmath45 . the commutation relations between the c.m . spin operators @xmath46 and @xmath47 and the spin operators @xmath48 , @xmath49 and @xmath50 of the @xmath25th electron are of order of @xmath51 : @xmath52= n^{-1}2\,{\rm i}\,\varepsi lon_{\beta_1\beta_2\beta_3}\sigma_j^{\beta_3}$ ] with @xmath53 . therefore , for a macroscopic large @xmath33 system , the c.m . part @xmath38 actually commutes with the relative - electron part @xmath54 in the hamiltonian , i.e. the c.m . motion and the relative motion of electrons are truly separated from each other . the couplings between the two emerge only through the electron impurity and electron  phonon interactions . furthermore , the electric field @xmath55 shows up only in @xmath38 . and , in view of @xmath56={\rm i}\delta_{\alpha \beta}(\delta_{ij}-1/n)\simeq { \rm i}\delta_{\alpha\beta}\delta_{ij}$ ] , i.e. the relative - electron momenta and coordinates can be treated as canonical conjugate variables , the relative - motion part @xmath54 is just the hamiltonian of @xmath33 electrons in the surface state of ti in the magnetic field without the presence of the electric field .    in terms of the c.m . coordinate @xmath57 and the relative electron density operator @xmath58 , the electron impurity and electron  phonon interactions can be written as@xcite @xmath59 here @xmath60 and @xmath61 are respectively the impurity potential ( an impurity at randomly distributed position @xmath62 ) and electron  phonon coupling matrix element in the plane - wave representation , and @xmath63 with @xmath64 and @xmath65 being the creation and annihilation operators for a phonon of wavevector @xmath66 in branch @xmath67 having frequency @xmath68 . velocity ( operator ) @xmath69 is the time variation of its coordinate : @xmath70= v_{\rm f}(\sigma_{\rm c}^y\ , \hat{i}-\sigma_{\rm c}^x\ , \hat{j})$ ] . to derive a force - balance equation for steady state transport we consider the heisenberg equation for the rate of change of the c.m . canonical momentum @xmath71 : @xmath72= - n e({\bm v}\times { \bm b})- n e{\bm e}+{\bm { f}}_{\rm i}+{\bm { f}}_{\rm p},\ ] ] in which the frictional forces @xmath73 and @xmath74 share the same expressions as given in ref ..    the statistical average of the operator equation can be determined to linear order in the electron  impurity and electron phonon interactions @xmath17 and @xmath18 with the initial density matrix @xmath75 at temperature @xmath76 when the in - plane electric field @xmath77 is not strong . for steady - transport states we have @xmath78 , leading to a force - balance equation of the form @xmath79 here @xmath80 , the statistically averaged velocity of the moving center - of - mass , is identified as the average rate of change of its position , i.e. the drift velocity of the electron system driven by the electric field @xmath77 , and @xmath81 and @xmath82 are frictional forces experienced by the center - of - mass due to impurity and phonon scatterings : @xmath83,\label{fp}\end{aligned}\ ] ] in which @xmath84 is the bose distribution function , @xmath85 , and @xmath86 stands for the imaginary part of the fourier spectrum of the relative - electron density correlation function defined by @xmath87\big\rangle_{0},\ ] ] where @xmath88 and @xmath89 denotes the statistical averaging over the initial density matrix @xmath90.@xcite    the force - balance equation describes the steady - state two - dimensional magnetotransport in the surface state of a ti . note that the frictional forces @xmath81 and @xmath82 are in the opposite direction of the drift velocity @xmath91 and their magnitudes are functions of @xmath92 only . with the drift velocity @xmath93 in the @xmath9 direction , the force - balance equation eq . yields a transverse resistivity @xmath94 , and a longitudinal resistivity @xmath95 . the linear one is in the form @xmath96 for calculating the electron density correlation function @xmath97 we proceed in the landau representation.@xcite the landau levels of the single - particle hamiltonian @xmath98 of the relative - electron system in the absence of electric field are composed of a positive `` @xmath99 '' and a negative `` @xmath100 '' branch@xcite @xmath101 with @xmath102 and @xmath103 , and a zero ( @xmath104 ) level @xmath105 the corresponding landau wave functions are @xmath106 and @xmath107 for @xmath108 ; and @xmath109 for @xmath104 . here @xmath110 is the wavevector of the system along @xmath9 direction ; @xmath111 with @xmath112 ; and @xmath113 is the harmonic oscillator eigenfunction with @xmath114 being the hermite polynomial , @xmath115 , and @xmath116 . each landau level contains @xmath117 electron states for system of unit surface area . the positive branch @xmath118 and the @xmath104 level @xmath119 of the above energy spectra are indeed quite close to those of the surface states in the bulk gap of bi@xmath0se@xmath1-family materials derived from microscopic band calculation.@xcite    the landau levels are broadened due to impurity , phonon and electron - electron scatterings . we model the imaginary part of the retarded green s function , or the density - of - states , of the broadened landau level @xmath120 ( written for `` + ' ' -branch and @xmath104 levels ) , using a gaussian - type form:@xcite @xmath121,\ ] ] with a half - width @xmath122 of the form:@xcite @xmath123^{1/2}$ ] . here @xmath124 is the single - particle lifetime and @xmath125 is the cyclotron frequency of linear - energy - dispersion system with @xmath126 being the zero - temperature fermi level . using a semi - empirical parameter @xmath127 to relate @xmath124 with the transport scattering time @xmath128 , and expressing @xmath129 with the zero - field mobility @xmath5 at finite temperature,@xcite we can write the landau - level broadening as @xmath130^{1/2}.\ ] ]    in the present study we consider the case of @xmath120-doping , i.e. the fermi level is high enough above the energy zero of the dirac cone in the range of `` + ' ' -branch levels and the states of `` @xmath100''-branch levels are completely filled , that they are irrelevant to electron transport . special attention has to be paid to the @xmath104 level , since , depending on the direction of exchange potential the effective g - factor of a ti surface state , @xmath30 , can be positive , zero or negative.@xcite the sign and magnitude of the effective g - factor determines how many states of the zero level should be included in or excluded from the available states for electron occupation in the case of @xmath120-doping at a magnetic field . ( i ) if @xmath131 , the @xmath104 level center is exactly at @xmath132 and the system is electron - hole symmetric . the total number of negative energy states ( including the states of the lower half of the @xmath104 level and states of the @xmath100"-branch levels ) and that of positive energy states ( including the states of the upper half of the @xmath104 level and states of the @xmath99"-branch levels ) do not change when changing magnetic field . therefore , the lower - half negative energy states of this level are always filled and the upper - half positive - energy states of it are available for the occupation of particles which are counted as electrons participating in transport in the case of @xmath120-doping . ( ii ) for a finite positive @xmath133 , the @xmath104 level @xmath134 moves downward to negative energy and its distance to the nearest  @xmath100"-branch level is @xmath135 closer than to the nearest  + " -branch level at finite magnetic field strength @xmath2 . this is equivalent to the opening of an increasingly enlarged ( with increasing @xmath2 ) energy gap between the  + " -branch states and the states of the zero - level and the  @xmath100"-branch levels . the opening of a sufficient energy gap implies that with increasing magnetic field the states in the  + " -branch levels would no longer shrink into the zero - level , and thus the @xmath104 level should be completely excluded from the conduction band , i.e. only particles occupying the  + " -branch states are counted as electrons participating in transport in the case of @xmath120-doping , when the magnetic field @xmath2 gets larger than a certain value ( depending on the magnitude of @xmath30 ) . ( iii ) for a finite negative @xmath136 , the @xmath104 level @xmath134 moves upward to positive energy and an increasingly enlarged energy gap will be opened between the states of the zero - level and the  + " -branch and the states of  @xmath100"-branch levels , and particles occupying the @xmath104 level and  + " -branch states are electrons participating in transport when the magnetic field @xmath2 gets larger than a certain value .    as a result , the experimentally accessible sheet density @xmath33 of electrons participating in transport is related to the fermi energy @xmath137 by the following equation valid at finite @xmath30 for the magnetic field @xmath2 larger than a certain value : @xmath138 in which @xmath139 + 1\}^{-1}$ ] is the fermi distribution function at temperature @xmath76 and the summation index @xmath120 goes over @xmath140 for @xmath133 , or @xmath141 for @xmath136 . in the case of @xmath131 , @xmath142\ ] ] valid for arbitrary magnetic field , in which @xmath143 . the imaginary part of relative - electron density correlation function in the presence of a magnetic field , @xmath86 , can be expressed in the landau representation as@xcite @xmath144 in which the transform factor @xmath145 ^ 2,\end{aligned}\ ] ] with @xmath146 , @xmath147 , @xmath148 , and @xmath149 being associated laguerre polynomials . the landau - representation correlation function @xmath150 in eq.([piqw ] ) can be constructed with the imaginary part of the retarded green s function @xmath151 , or the density - of - states , of the @xmath120th landau level as@xcite @xmath152\nonumber\\ & \hspace{1.2cm}\times{\rm im}g_n(\epsilon+\omega){\rm im}g_{n'}(\epsilon).\end{aligned}\ ] ] the summation indices @xmath120 and @xmath153 in eq.([piqw ] ) are taken over @xmath140 for @xmath133 , or @xmath154 for @xmath136 . in the case of @xmath131 , eq.([piqw ] ) still works and the summation indices @xmath120 and @xmath153 go over @xmath154 but with @xmath155 replaced by @xmath156 in eq.([p2nn ] ) . numerical calculations are performed for the magnetoresistivity @xmath157 of surface state in a uniform ti bi@xmath0se@xmath1 . at zero temperature the elastic scattering contributing to the resistivity is modeled by a coulomb potential due to charged impurities:@xcite @xmath158 with @xmath159 being the impurity density , which is determined by the zero - magnetic - field mobility @xmath5 . at temperatures higher than @xmath160,@xcite phonon scatterings play increasingly important role and the dominant inelastic contribution comes from optical phonons . for this polar material , the scattering by optical phonons via the deformation potential can be neglected . hence , we take account of inelastic scattering from optical phonons via frhlich coupling : @xmath161 . in the numerical calculation we use the following parameters:@xcite fermi velocity @xmath162 , static dielectric constant @xmath163 , optical dielectric constant @xmath164 , and phonon energy @xmath165 . the broadening parameter is taken to be @xmath166 . as a function of the magnetic field @xmath2 having different effective g - factors : @xmath167 and @xmath168 for a ti surface system with electron sheet density @xmath169 in the cases of zero - magnetic - field mobility @xmath170 ( a ) and @xmath171 ( b ) . several integer - number positions of filling factor @xmath172 are marked in ( b).,scaledwidth=40.0% ]    fig.[diffg ] shows the calculated magnetoresistivity @xmath157 versus the magnetic field strength @xmath2 for a ti surface system with electron sheet density @xmath169 but having different effective g - factors : @xmath167 and @xmath168 for two values of zero - magnetic - field mobility @xmath170 and @xmath171 , representing different degree of landau - level broadening . in the case without zeeman splitting ( @xmath131 ) the resistivity @xmath157 exhibits almost no change with changing magnetic field up to 10 t , except the shubnikov - de haas ( sdh ) oscillation showing up in the case of @xmath171 . this kind of magnetoresistance behavior was indeed seen experimentally in the electron - hole symmetrical massless system of single - layer graphene.@xcite in the case of a positive g - factor , @xmath173 , the magnetoresistivity increases linearly with increasing magnetic field ; while for a negative g - factor , @xmath174 , the magnetoresistivity decreases linearly with increasing magnetic field . is shown as a function of the magnetic field @xmath2 for different values of zero - magnetic - field mobility : ( a ) @xmath175 , ( b ) @xmath176 , ( c ) @xmath177 , ( d ) @xmath178 , ( e ) @xmath179 , and ( f ) @xmath180 . the inset of ( a ) illustrates the same for a larger magnetic - field range @xmath181 . the filling factor @xmath182 is plotted versus the magnetic field in ( f ) ; and several integer - number positions of @xmath182 are also marked in ( d ) and ( e ) . here the surface electron density @xmath169 and the lattice temperature @xmath183.,scaledwidth=47.0% ]    in the following we will give more detailed examination on the linearly increasing magnetoresistance in the positive @xmath30 case . fig.[rhob ] shows the calculated resistivity @xmath157 versus the magnetic field strength @xmath2 at lattice temperature @xmath183 for system of carrier sheet density @xmath169 and @xmath173 , having different zero - field mobility @xmath184 and @xmath180 . all resistivity curves for mobility @xmath185 exhibit clear linearity in the magnetic - field range and appear no tendency of saturation at the highest field shown in the figure . especially , for the case @xmath170 , the linear behavior extends even up to the magnetic field of @xmath186 , as illustrated in the inset of fig.[rhob](a ) . this feature contradicts the classical mr which saturates at sufficiently large magnetic field @xmath187 . note that here we only present the calculated @xmath157 for magnetic field @xmath2 larger than @xmath188 t , for which a sufficient energy gap @xmath135 is assumed to open that with further increase of the magnetic field the states in the `` + ' ' -branch levels no longer shrink into the zero level and thus it should be excluded from the conduction band . this is of course not true for very weak magnetic field . when @xmath189 the energy gap @xmath190 , the situation becomes similar to the case of @xmath131 : the whole upper half of the zero - level states are available to electron occupation and we should have a flat resistivity @xmath157 when changing magnetic field . with increasing @xmath2 the portion of the zero - level states available to conduction electrons decreases until the magnetic field reaches @xmath191 . as a result the resistivity @xmath157 should exhibit a crossover from a flat changing at small @xmath2 to positively linear increasing at @xmath192 . this is just the behavior observed in the ti bi@xmath0se@xmath1.@xcite    note that in the case of @xmath170 , the broadened landau - level widths are always larger than the neighboring level interval : @xmath193 , which requires @xmath194 ^ 2 $ ] , even for the lowest landau level @xmath195 , i.e. the whole landau - level spectrum is smeared . with increasing the zero - field mobility the magnitude of resistivity @xmath157 decreases , and when the broadened landau - level width becomes smaller than the neighboring level interval , @xmath196 , a weak sdh oscillation begin to occur around the linearly - dependent average value of @xmath157 at higher portion of the magnetic field range , as seen in fig.[rhob](c ) , ( d ) and ( e ) for @xmath197 and @xmath198 . on the other hand , in the case of large mobility , e.g. @xmath199 , where the broadened landau - level widths @xmath200 are much smaller than the neighboring level interval even for level index @xmath120 as large as @xmath201 , the magnetoresistivity shows pronounced sdh oscillation and the linear - dependent behavior disappears , before the appearance of quantum hall effect,@xcite as shown in fig.[rhob](f ) . abrikosov s model for the lmr requires the applied magnetic field large enough to reach the quantum limit at which all the carriers are within the lowest landau level,@xcite while it is obvious that more than one landau levels are occupied in the experimental samples in the field range in which the linear and non - saturating magnetoresistivity was observed.@xcite for the given electron surface density @xmath202 , the number of occupied landau levels , or the filling factor @xmath172 , at different magnetic fields is shown in fig.[rhob](f ) , as well as in the fig.[rhob](d ) and ( e ) , where the integer - number positions of @xmath203 , i.e. filling up to entire @xmath182 landau levels , coincide with the minima of the density - of - states or the dips of sdh oscillation . this is in contrast with @xmath131 case , where the integer number of @xmath203 , which implies a filling up to the center position of the @xmath182th landau levels , locates at a peak of sdh oscillation , as shown in fig.[diffg]b . the observed sdh oscillations in the bi@xmath0se@xmath1 nanoribbon exhibiting nonsaturating surface lmr in the experiment@xcite favor the former case : a finite positive effective @xmath133 .     is plotted as a function of the surface electron density @xmath33 at magnetic field @xmath204 : ( a ) at different values of zero - field mobility @xmath5 , and ( b ) at different values of zero - field conductivity @xmath205.,scaledwidth=40.0% ]     at various lattice temperatures . here the zero - magnetic - field mobility at zero temperature is @xmath206.,scaledwidth=35.0% ]    next , we examine the density - dependence of the linear magnetoresistivity . to compare with abrikosov s quantum magnetoresistance which suggests a @xmath207 behavior,@xcite we show the calculated @xmath208 for above lmr versus the carrier sheet density @xmath33 in fig.[rhon ] at fixed magnetic field @xmath209 t . the mobility is taken respectively to be @xmath210 and @xmath211m@xmath212/vs to make the resistivity in the lmr regime . a clearly linear dependence of @xmath213 on the surface density @xmath33 is seen in all cases , indicating that this non - saturating linear resistivity is almost inversely proportional to the carrier density . in the figure we also show @xmath208 versus @xmath33 under the condition of different given conductivity @xmath214 and @xmath215 . in this case the half - width @xmath216 is independent of surface density . the linear dependence still holds , indicating that this linear behavior is not sensitive to the modest @xmath33-dependence of landau level broadening @xmath216 as long as the system is in the overlapped landau level regime . from the above discussion , it is obvious that lmr shows up in the system having overlapped landau levels and the separation of landau levels makes the mr departure from the linear increase . at high temperature , the thermal energy would smear the level separation and phonon scatterings further broaden landau levels . hence , it is believed that this lmr will be robust against raising temperature . this is indeed the case as seen in fig.[rhot ] , where we plot the calculated magnetoresistivity @xmath157 for the above system with zero - temperature linear mobility @xmath217m@xmath212/vs versus the magnetic field at different lattice temperatures . we can see that raising temperature to room temperature has little effect on the linearity of mr . due to the decreased mobility at higher temperature from phonon scattering , the weak sdh oscillation on the linear background tends to vanish . these features are in good agreement with the experimental report.@xcite in summary , we have studied the two - dimensional magnetotransport in the flat surface of a three - dimensional ti , which arises from the surface states with a wavevector - linear energy dispersion and a finite , positive zeeman splitting within the bulk energy gap . when the level broadening is comparable to or larger than the landau - level separation and the conduction electrons spread over many landau levels , a positive , dominantly linear and non - saturating magnetoresistance appears within a quite wide range of magnetic field and persists up to room temperature . this remarkable lmr provides a possible mechanism for the recently observed linear magnetoresistance in topological insulator bi@xmath0se@xmath1 nanoribbons.@xcite    in contrast to quantum hall effect which appears in the case of well formed landau levels and to abrikosov s quantum magnetotransport,@xcite which is limited to the extreme quantum limit that all electrons coalesce into the lowest landau level , the discussed lmr is a phenomena of pure classical two - dimensional magnetotransport in a system having linear - energy - dispersion , appearing in the regime of overlapped landau levels , irrespective of its showing up in relatively high magnetic field range . furthermore , the present scheme deals with spatially uniform case without invoking the mobility fluctuation in a strongly inhomogeneous system , which is required in the classical parish and littlewood model to produce a lmr.@xcite    the appearance of this significant positive - increasing linear magnetoresistance depends on the existence of a positive and sizable effective g - factor . if the zeeman energy splitting is quite small the resistivity @xmath157 would exhibit little change with changing magnetic field . in the case of a negative and sizable effective g - factor the magnetoresistivity would decrease linearly with increasing magnetic field . therefore , the behavior of the longitudinal resistivity versus magnetic field may provide a useful way for judging the direction and the size of the effective zeeman energy splitting in ti surface states . this work was supported by the national science foundation of china ( grant no . 11104002 ) , the national basic research program of china ( grant no . 2012cb927403 ) and by the program for science&technology innovation talents in universities of henan province ( grant no . 2012hastit029 ) ."""

        dct = tok.batch_encode_plus(
            [ARTICLE_LEP, ARTICLE_MAGNET],
            max_length=6144,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        hypotheses_batch = hf.generate(
            input_ids=dct["input_ids"].to(torch_device),
            attention_mask=dct["attention_mask"].to(torch_device),
            num_beams=4,
            max_length=512,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

        EXPECTED_LEP = (
            " the physics of @xmath0-boson will again play the central role in the frontier of particle physics if the"
            " gigaz option of the international linear collider ( ilc ) can be realized in its first phase. \n the"
            " expected sensitivity to the branching ratio of rare decays, especially its exotic or rare processes,"
            " should be investigated comprehensively to evaluate their potential in probing new physics. in this work"
            " \n, we study the rare decay into light higgs boson(s ) in the framework of the minimal supersymmetric"
            " standard model ( mssm ), where a light cp - odd higgs - boson with singlet - dominant component may"
            " naturally arise from the spontaneous breaking of some approximate global symmetry. "
        )

        EXPECTED_MAGNET = (
            " the recent experiment in the surface states of the topological insulator bi@xmath0se @xmath1, however,"
            " reported that a large positive magnetoresistance becomes very linear in perpendicular magnetic field"
            " even in an opposite situation where the carrier sheet density is high that all electrons occupy more"
            " than one landau levels. \n it is striking that this observation is in conflict with abrikosov s model"
            " and also with the classical parish - littlewood model. "
        )

        generated = tok.batch_decode(
            hypotheses_batch.tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True
        )
        assert generated == [EXPECTED_LEP, EXPECTED_MAGNET]
