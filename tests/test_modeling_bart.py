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
        model.to(torch_device)
        outputs = model.forward(input_ids=input_ids, decoder_input_ids=input_ids)
        logits = outputs[0]
        expected_shape = torch.Size((batch_size, config.num_labels))
        self.assertEqual(logits.shape, expected_shape)

        lm_model = BartForMaskedLM(config)
        lm_model.to(torch_device)
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

    def test_generate_beam_search(self):
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

        new_input_ids = lm_model.generate(
            input_ids.clone(), num_return_sequences=1, num_beams=2, no_repeat_ngram_size=3
        )
        self.assertEqual(new_input_ids.shape, (input_ids.shape[0] * 2, 20))

        # No Beam Search
        new_input_ids = lm_model.generate(input_ids)
        #self.assertEqual(new_input_ids.shape, (input_ids.shape[0], 20))

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
class BartModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head(self):
        model = BartModel.from_pretrained("bart-large").to(torch_device)
        input_ids = _long_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        inputs_dict = prepare_bart_inputs_dict(model.config, input_ids)
        with torch.no_grad():
            output = model.forward(**inputs_dict)[0]
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
            batched_logits, features = model.forward(**inputs_dict)
        expected_shape = torch.Size((2, 3))
        self.assertEqual(batched_logits.shape, expected_shape)
        expected_slice = torch.Tensor([[0.1907, 1.4342, -1.0289]]).to(torch_device)
        logits_arr = batched_logits[0].detach()

        # Test that padding does not change results
        input_ids_no_pad = _long_tensor([example_b[:-1]])

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

    @slow
    def test_cnn_summarization(self):
        hf = BartForMaskedLM.from_pretrained("bart-large-cnn", output_past=True,)
        # hf.model.decoder.generation_mode = True

        text = " (CNN)The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian"
        tok = BartTokenizer.from_pretrained("bart-large")
        tokens = tok.encode(text, return_tensors="pt")
        extra_len = 20
        gen_tokens = hf.generate(
            tokens,
            num_return_sequences=1,
            num_beams=4,
            max_length=tokens.shape[1] + extra_len,  # repetition_penalty=10.,
            do_sample=False,
        )
        expected_result = "<s>The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday."
        generated = [tok.decode(g,) for g in gen_tokens]
        self.assertEqual(expected_result, generated[0])

        FRANCE_ARTICLE =  ' Marseille, France (CNN)The French prosecutor leading an investigation into the crash of Germanwings Flight 9525 insisted Wednesday that he was not aware of any video footage from on board the plane. Marseille prosecutor Brice Robin told CNN that "so far no videos were used in the crash investigation." He added, "A person who has such a video needs to immediately give it to the investigators." Robin\'s comments follow claims by two magazines, German daily Bild and French Paris Match, of a cell phone video showing the harrowing final seconds from on board Germanwings Flight 9525 as it crashed into the French Alps. All 150 on board were killed. Paris Match and Bild reported that the video was recovered from a phone at the wreckage site. The two publications described the supposed video, but did not post it on their websites. The publications said that they watched the video, which was found by a source close to the investigation. "One can hear cries of \'My God\' in several languages," Paris Match reported. "Metallic banging can also be heard more than three times, perhaps of the pilot trying to open the cockpit door with a heavy object.  Towards the end, after a heavy shake, stronger than the others, the screaming intensifies. Then nothing." "It is a very disturbing scene," said Julian Reichelt, editor-in-chief of Bild online. An official with France\'s accident investigation agency, the BEA, said the agency is not aware of any such video. Lt. Col. Jean-Marc Menichini, a French Gendarmerie spokesman in charge of communications on rescue efforts around the Germanwings crash site, told CNN that the reports were "completely wrong" and "unwarranted." Cell phones have been collected at the site, he said, but that they "hadn\'t been exploited yet." Menichini said he believed the cell phones would need to be sent to the Criminal Research Institute in Rosny sous-Bois, near Paris, in order to be analyzed by specialized technicians working hand-in-hand with investigators. But none of the cell phones found so far have been sent to the institute, Menichini said. Asked whether staff involved in the search could have leaked a memory card to the media, Menichini answered with a categorical "no." Reichelt told "Erin Burnett: Outfront" that he had watched the video and stood by the report, saying Bild and Paris Match are "very confident" that the clip is real. He noted that investigators only revealed they\'d recovered cell phones from the crash site after Bild and Paris Match published their reports. "That is something we did not know before. ... Overall we can say many things of the investigation weren\'t revealed by the investigation at the beginning," he said. What was mental state of Germanwings co-pilot? German airline Lufthansa confirmed Tuesday that co-pilot Andreas Lubitz had battled depression years before he took the controls of Germanwings Flight 9525, which he\'s accused of deliberately crashing last week in the French Alps. Lubitz told his Lufthansa flight training school in 2009 that he had a "previous episode of severe depression," the airline said Tuesday. Email correspondence between Lubitz and the school discovered in an internal investigation, Lufthansa said, included medical documents he submitted in connection with resuming his flight training. The announcement indicates that Lufthansa, the parent company of Germanwings, knew of Lubitz\'s battle with depression, allowed him to continue training and ultimately put him in the cockpit. Lufthansa, whose CEO Carsten Spohr previously said Lubitz was 100% fit to fly, described its statement Tuesday as a "swift and seamless clarification" and said it was sharing the information and documents -- including training and medical records -- with public prosecutors. Spohr traveled to the crash site Wednesday, where recovery teams have been working for the past week to recover human remains and plane debris scattered across a steep mountainside. He saw the crisis center set up in Seyne-les-Alpes, laid a wreath in the village of Le Vernet, closer to the crash site, where grieving families have left flowers at a simple stone memorial. Menichini told CNN late Tuesday that no visible human remains were left at the site but recovery teams would keep searching. French President Francois Hollande, speaking Tuesday, said that it should be possible to identify all the victims using DNA analysis by the end of the week, sooner than authorities had previously suggested. In the meantime, the recovery of the victims\' personal belongings will start Wednesday, Menichini said. Among those personal belongings could be more cell phones belonging to the 144 passengers and six crew on board. Check out the latest from our correspondents . The details about Lubitz\'s correspondence with the flight school during his training were among several developments as investigators continued to delve into what caused the crash and Lubitz\'s possible motive for downing the jet. A Lufthansa spokesperson told CNN on Tuesday that Lubitz had a valid medical certificate, had passed all his examinations and "held all the licenses required." Earlier, a spokesman for the prosecutor\'s office in Dusseldorf, Christoph Kumpa, said medical records reveal Lubitz suffered from suicidal tendencies at some point before his aviation career and underwent psychotherapy before he got his pilot\'s license. Kumpa emphasized there\'s no evidence suggesting Lubitz was suicidal or acting aggressively before the crash. Investigators are looking into whether Lubitz feared his medical condition would cause him to lose his pilot\'s license, a European government official briefed on the investigation told CNN on Tuesday. While flying was "a big part of his life," the source said, it\'s only one theory being considered. Another source, a law enforcement official briefed on the investigation, also told CNN that authorities believe the primary motive for Lubitz to bring down the plane was that he feared he would not be allowed to fly because of his medical problems. Lubitz\'s girlfriend told investigators he had seen an eye doctor and a neuropsychologist, both of whom deemed him unfit to work recently and concluded he had psychological issues, the European government official said. But no matter what details emerge about his previous mental health struggles, there\'s more to the story, said Brian Russell, a forensic psychologist. "Psychology can explain why somebody would turn rage inward on themselves about the fact that maybe they weren\'t going to keep doing their job and they\'re upset about that and so they\'re suicidal," he said. "But there is no mental illness that explains why somebody then feels entitled to also take that rage and turn it outward on 149 other people who had nothing to do with the person\'s problems." Germanwings crash compensation: What we know . Who was the captain of Germanwings Flight 9525? CNN\'s Margot Haddad reported from Marseille and Pamela Brown from Dusseldorf, while Laura Smith-Spark wrote from London. CNN\'s Frederik Pleitgen, Pamela Boykoff, Antonia Mortensen, Sandrine Amiel and Anna-Maja Rappard contributed to this report.' # @noqa
        SHORTER_ARTICLE = ' (CNN)The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based. The Palestinians signed the ICC\'s founding Rome Statute in January, when they also accepted its jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including East Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination into the situation in Palestinian territories, paving the way for possible war crimes investigations against Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and the United States, neither of which is an ICC member, opposed the Palestinians\' efforts to join the body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday\'s ceremony, said it was a move toward greater justice. "As Palestine formally becomes a State Party to the Rome Statute today, the world is also a step closer to ending a long era of impunity and injustice," he said, according to an ICC news release. "Indeed, today brings us closer to our shared goals of justice and peace." Judge Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the Palestinians. "As the Rome Statute today enters into force for the State of Palestine, Palestine acquires all the rights as well as responsibilities that come with being a State Party to the Statute. These are substantive commitments, which cannot be taken lightly," she said. Rights group Human Rights Watch welcomed the development. "Governments seeking to penalize Palestine for joining the ICC should immediately end their pressure, and countries that support universal acceptance of the court\'s treaty should speak out to welcome its membership," said Balkees Jarrah, international justice counsel for the group. "What\'s objectionable is the attempts to undermine international justice, not Palestine\'s decision to join a treaty to which over 100 countries around the world are members." In January, when the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described it as an outrage, saying the court was overstepping its boundaries. The United States also said it "strongly" disagreed with the court\'s decision. "As we have said repeatedly, we do not believe that Palestine is a state and therefore we do not believe that it is eligible to join the ICC," the State Department said in a statement. It urged the warring sides to resolve their differences through direct negotiations. "We will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace," it said. But the ICC begs to differ with the definition of a state for its purposes and refers to the territories as "Palestine." While a preliminary examination is not a formal investigation, it allows the court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou Bensouda said her office would "conduct its analysis in full independence and impartiality." The war between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The inquiry will include alleged war crimes committed since June. The International Criminal Court was set up in 2002 to prosecute genocide, crimes against humanity and war crimes. CNN\'s Vasco Cotovio, Kareem Khadder and Faith Karimi contributed to this report.'
        EXPECTED_SUMMARY_FRANCE = 'French prosecutor says he\'s not aware of any video footage from on board the plane. German daily Bild and French Paris Match claim to have found a cell phone video of the crash. A French Gendarmerie spokesman calls the reports "completely wrong" and "unwarranted" German airline Lufthansa confirms co-pilot Andreas Lubitz had battled depression.'
        DESIRED_RESLT_SHORTER =  "The Palestinian Authority becomes the 123rd member of the International Criminal Court. The move gives the court jurisdiction over alleged crimes in Palestinian territories. Israel and the United States opposed the Palestinians' efforts to join the body. But Palestinian Foreign Minister Riad al-Malki said it was a move toward greater justice."



