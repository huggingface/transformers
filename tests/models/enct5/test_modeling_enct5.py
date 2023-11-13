# coding=utf-8
# Copyright 2018 Google T5 Authors and HuggingFace Inc. team.
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


import copy
import os
import pickle
import tempfile
import unittest

from transformers import EncT5Config, is_torch_available
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
from transformers.testing_utils import (
    require_accelerate,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    slow,
    torch_device,
)
from transformers.utils import cached_property, is_torch_fx_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_fx_available():
    from transformers.utils.fx import symbolic_trace

if is_torch_available():
    import torch

    from transformers import (
        EncT5ForSequenceClassification,
    )


class EncT5ForSequenceClassificationModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        encoder_seq_length=7,
        seq_length=1,
        # For common tests
        use_attention_mask=True,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        d_ff=37,
        relative_attention_num_buckets=8,
        is_training=False,
        dropout_rate=0.1,
        initializer_factor=0.002,
        is_encoder_decoder=True,
        eos_token_id=1,
        pad_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        # For common tests
        self.seq_length = seq_length
        self.use_attention_mask = use_attention_mask
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.d_ff = d_ff
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.initializer_factor = initializer_factor
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.scope = None
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)

        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], vocab_size=1)

        config = EncT5Config(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            is_encoder_decoder=self.is_encoder_decoder,
        )

        return (
            config,
            input_ids,
            attention_mask,
            decoder_input_ids,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        attention_mask,
        decoder_input_ids,
    ):
        model = EncT5ForSequenceClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        logits = result.logits

        self.parent.assertEqual(logits.size(), (self.batch_size, self.seq_length, config.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
            decoder_input_ids,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
        }
        return config, inputs_dict


class EncT5ForSequenceClassificationModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (EncT5ForSequenceClassification,) if is_torch_available() else ()
    test_pruning = False
    test_resize_embeddings = False
    test_model_parallel = False
    is_encoder_decoder = True
    all_parallelizable_model_classes = ()

    def setUp(self):
        self.model_tester = EncT5ForSequenceClassificationModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EncT5Config, d_model=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

#
# @require_torch
# @require_sentencepiece
# @require_tokenizers
# class T5ModelIntegrationTests(unittest.TestCase):
#     @cached_property
#     def model(self):
#         return T5ForConditionalGeneration.from_pretrained("t5-base").to(torch_device)
#
#     @cached_property
#     def tokenizer(self):
#         return T5Tokenizer.from_pretrained("t5-base")
#
#     @slow
#     def test_torch_quant(self):
#         r"""
#         Test that a simple `torch.quantization.quantize_dynamic` call works on a T5 model.
#         """
#         model_name = "google/flan-t5-small"
#         tokenizer = T5Tokenizer.from_pretrained(model_name)
#         model = T5ForConditionalGeneration.from_pretrained(model_name)
#         model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
#         input_text = "Answer the following yes/no question by reasoning step-by-step. Can you write a whole Haiku in a single tweet?"
#         input_ids = tokenizer(input_text, return_tensors="pt").input_ids
#         _ = model.generate(input_ids)
#
#     @slow
#     def test_small_generation(self):
#         model = T5ForConditionalGeneration.from_pretrained("t5-small").to(torch_device)
#         model.config.max_length = 8
#         model.config.num_beams = 1
#         model.config.do_sample = False
#         tokenizer = T5Tokenizer.from_pretrained("t5-small")
#
#         input_ids = tokenizer("summarize: Hello there", return_tensors="pt").input_ids.to(torch_device)
#
#         sequences = model.generate(input_ids)
#
#         output_str = tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]
#         self.assertTrue(output_str == "Hello there!")
#
#     @slow
#     def test_small_integration_test(self):
#         """
#         For comparision run:
#         >>> import t5  # pip install t5==0.7.1
#         >>> from t5.data.sentencepiece_vocabulary import SentencePieceVocabulary
#
#         >>> path_to_mtf_small_t5_checkpoint = '<fill_in>'
#         >>> path_to_mtf_small_spm_model_path = '<fill_in>'
#         >>> t5_model = t5.models.MtfModel(model_dir=path_to_mtf_small_t5_checkpoint, batch_size=1, tpu=None)
#         >>> vocab = SentencePieceVocabulary(path_to_mtf_small_spm_model_path, extra_ids=100)
#         >>> score = t5_model.score(inputs=["Hello there"], targets=["Hi I am"], vocabulary=vocab)
#         """
#
#         model = T5ForConditionalGeneration.from_pretrained("t5-small").to(torch_device)
#         tokenizer = T5Tokenizer.from_pretrained("t5-small")
#
#         input_ids = tokenizer("Hello there", return_tensors="pt").input_ids
#         labels = tokenizer("Hi I am", return_tensors="pt").input_ids
#
#         loss = model(input_ids.to(torch_device), labels=labels.to(torch_device)).loss
#         mtf_score = -(labels.shape[-1] * loss.item())
#
#         EXPECTED_SCORE = -19.0845
#         self.assertTrue(abs(mtf_score - EXPECTED_SCORE) < 1e-4)
#
#     @slow
#     def test_small_v1_1_integration_test(self):
#         """
#         For comparision run:
#         >>> import t5  # pip install t5==0.7.1
#         >>> from t5.data.sentencepiece_vocabulary import SentencePieceVocabulary
#
#         >>> path_to_mtf_small_t5_v1_1_checkpoint = '<fill_in>'
#         >>> path_to_mtf_small_spm_model_path = '<fill_in>'
#         >>> t5_model = t5.models.MtfModel(model_dir=path_to_mtf_small_t5_v1_1_checkpoint, batch_size=1, tpu=None)
#         >>> vocab = SentencePieceVocabulary(path_to_mtf_small_spm_model_path, extra_ids=100)
#         >>> score = t5_model.score(inputs=["Hello there"], targets=["Hi I am"], vocabulary=vocab)
#         """
#
#         model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-small").to(torch_device)
#         tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-small")
#
#         input_ids = tokenizer("Hello there", return_tensors="pt").input_ids
#         labels = tokenizer("Hi I am", return_tensors="pt").input_ids
#
#         loss = model(input_ids.to(torch_device), labels=labels.to(torch_device)).loss
#         mtf_score = -(labels.shape[-1] * loss.item())
#
#         EXPECTED_SCORE = -59.0293
#         self.assertTrue(abs(mtf_score - EXPECTED_SCORE) < 1e-4)
#
#     @slow
#     def test_small_byt5_integration_test(self):
#         """
#         For comparision run:
#         >>> import t5  # pip install t5==0.9.1
#
#         >>> path_to_byt5_small_checkpoint = '<fill_in>'
#         >>> t5_model = t5.models.MtfModel(model_dir=path_to_tf_checkpoint, batch_size=1, tpu=None)
#         >>> vocab = t5.data.ByteVocabulary()
#         >>> score = t5_model.score(inputs=["Hello there"], targets=["Hi I am"], vocabulary=vocab)
#         """
#
#         model = T5ForConditionalGeneration.from_pretrained("google/byt5-small").to(torch_device)
#         tokenizer = ByT5Tokenizer.from_pretrained("google/byt5-small")
#
#         input_ids = tokenizer("Hello there", return_tensors="pt").input_ids
#         labels = tokenizer("Hi I am", return_tensors="pt").input_ids
#
#         loss = model(input_ids.to(torch_device), labels=labels.to(torch_device)).loss
#         mtf_score = -(labels.shape[-1] * loss.item())
#
#         EXPECTED_SCORE = -60.7397
#         self.assertTrue(abs(mtf_score - EXPECTED_SCORE) < 1e-4)
#
#     @slow
#     def test_summarization(self):
#         model = self.model
#         tok = self.tokenizer
#
#         FRANCE_ARTICLE = (  # @noqa
#             "Marseille, France (CNN)The French prosecutor leading an investigation into the crash of Germanwings"
#             " Flight 9525 insisted Wednesday that he was not aware of any video footage from on board the plane."
#             ' Marseille prosecutor Brice Robin told CNN that "so far no videos were used in the crash investigation."'
#             ' He added, "A person who has such a video needs to immediately give it to the investigators." Robin\'s'
#             " comments follow claims by two magazines, German daily Bild and French Paris Match, of a cell phone video"
#             " showing the harrowing final seconds from on board Germanwings Flight 9525 as it crashed into the French"
#             " Alps. All 150 on board were killed. Paris Match and Bild reported that the video was recovered from a"
#             " phone at the wreckage site. The two publications described the supposed video, but did not post it on"
#             " their websites. The publications said that they watched the video, which was found by a source close to"
#             " the investigation. \"One can hear cries of 'My God' in several languages,\" Paris Match reported."
#             ' "Metallic banging can also be heard more than three times, perhaps of the pilot trying to open the'
#             " cockpit door with a heavy object.  Towards the end, after a heavy shake, stronger than the others, the"
#             ' screaming intensifies. Then nothing." "It is a very disturbing scene," said Julian Reichelt,'
#             " editor-in-chief of Bild online. An official with France's accident investigation agency, the BEA, said"
#             " the agency is not aware of any such video. Lt. Col. Jean-Marc Menichini, a French Gendarmerie spokesman"
#             " in charge of communications on rescue efforts around the Germanwings crash site, told CNN that the"
#             ' reports were "completely wrong" and "unwarranted." Cell phones have been collected at the site, he said,'
#             ' but that they "hadn\'t been exploited yet." Menichini said he believed the cell phones would need to be'
#             " sent to the Criminal Research Institute in Rosny sous-Bois, near Paris, in order to be analyzed by"
#             " specialized technicians working hand-in-hand with investigators. But none of the cell phones found so"
#             " far have been sent to the institute, Menichini said. Asked whether staff involved in the search could"
#             ' have leaked a memory card to the media, Menichini answered with a categorical "no." Reichelt told "Erin'
#             ' Burnett: Outfront" that he had watched the video and stood by the report, saying Bild and Paris Match'
#             ' are "very confident" that the clip is real. He noted that investigators only revealed they\'d recovered'
#             ' cell phones from the crash site after Bild and Paris Match published their reports. "That is something'
#             " we did not know before. ... Overall we can say many things of the investigation weren't revealed by the"
#             ' investigation at the beginning," he said. What was mental state of Germanwings co-pilot? German airline'
#             " Lufthansa confirmed Tuesday that co-pilot Andreas Lubitz had battled depression years before he took the"
#             " controls of Germanwings Flight 9525, which he's accused of deliberately crashing last week in the"
#             ' French Alps. Lubitz told his Lufthansa flight training school in 2009 that he had a "previous episode of'
#             ' severe depression," the airline said Tuesday. Email correspondence between Lubitz and the school'
#             " discovered in an internal investigation, Lufthansa said, included medical documents he submitted in"
#             " connection with resuming his flight training. The announcement indicates that Lufthansa, the parent"
#             " company of Germanwings, knew of Lubitz's battle with depression, allowed him to continue training and"
#             " ultimately put him in the cockpit. Lufthansa, whose CEO Carsten Spohr previously said Lubitz was 100%"
#             ' fit to fly, described its statement Tuesday as a "swift and seamless clarification" and said it was'
#             " sharing the information and documents -- including training and medical records -- with public"
#             " prosecutors. Spohr traveled to the crash site Wednesday, where recovery teams have been working for the"
#             " past week to recover human remains and plane debris scattered across a steep mountainside. He saw the"
#             " crisis center set up in Seyne-les-Alpes, laid a wreath in the village of Le Vernet, closer to the crash"
#             " site, where grieving families have left flowers at a simple stone memorial. Menichini told CNN late"
#             " Tuesday that no visible human remains were left at the site but recovery teams would keep searching."
#             " French President Francois Hollande, speaking Tuesday, said that it should be possible to identify all"
#             " the victims using DNA analysis by the end of the week, sooner than authorities had previously suggested."
#             " In the meantime, the recovery of the victims' personal belongings will start Wednesday, Menichini said."
#             " Among those personal belongings could be more cell phones belonging to the 144 passengers and six crew"
#             " on board. Check out the latest from our correspondents . The details about Lubitz's correspondence with"
#             " the flight school during his training were among several developments as investigators continued to"
#             " delve into what caused the crash and Lubitz's possible motive for downing the jet. A Lufthansa"
#             " spokesperson told CNN on Tuesday that Lubitz had a valid medical certificate, had passed all his"
#             ' examinations and "held all the licenses required." Earlier, a spokesman for the prosecutor\'s office in'
#             " Dusseldorf, Christoph Kumpa, said medical records reveal Lubitz suffered from suicidal tendencies at"
#             " some point before his aviation career and underwent psychotherapy before he got his pilot's license."
#             " Kumpa emphasized there's no evidence suggesting Lubitz was suicidal or acting aggressively before the"
#             " crash. Investigators are looking into whether Lubitz feared his medical condition would cause him to"
#             " lose his pilot's license, a European government official briefed on the investigation told CNN on"
#             ' Tuesday. While flying was "a big part of his life," the source said, it\'s only one theory being'
#             " considered. Another source, a law enforcement official briefed on the investigation, also told CNN that"
#             " authorities believe the primary motive for Lubitz to bring down the plane was that he feared he would"
#             " not be allowed to fly because of his medical problems. Lubitz's girlfriend told investigators he had"
#             " seen an eye doctor and a neuropsychologist, both of whom deemed him unfit to work recently and concluded"
#             " he had psychological issues, the European government official said. But no matter what details emerge"
#             " about his previous mental health struggles, there's more to the story, said Brian Russell, a forensic"
#             ' psychologist. "Psychology can explain why somebody would turn rage inward on themselves about the fact'
#             " that maybe they weren't going to keep doing their job and they're upset about that and so they're"
#             ' suicidal," he said. "But there is no mental illness that explains why somebody then feels entitled to'
#             " also take that rage and turn it outward on 149 other people who had nothing to do with the person's"
#             ' problems." Germanwings crash compensation: What we know . Who was the captain of Germanwings Flight'
#             " 9525? CNN's Margot Haddad reported from Marseille and Pamela Brown from Dusseldorf, while Laura"
#             " Smith-Spark wrote from London. CNN's Frederik Pleitgen, Pamela Boykoff, Antonia Mortensen, Sandrine"
#             " Amiel and Anna-Maja Rappard contributed to this report."
#         )
#         SHORTER_ARTICLE = (
#             "(CNN)The Palestinian Authority officially became the 123rd member of the International Criminal Court on"
#             " Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The"
#             " formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based."
#             " The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted its"
#             ' jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including East'
#             ' Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination into the'
#             " situation in Palestinian territories, paving the way for possible war crimes investigations against"
#             " Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and"
#             " the United States, neither of which is an ICC member, opposed the Palestinians' efforts to join the"
#             " body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday's ceremony, said it was a"
#             ' move toward greater justice. "As Palestine formally becomes a State Party to the Rome Statute today, the'
#             ' world is also a step closer to ending a long era of impunity and injustice," he said, according to an'
#             ' ICC news release. "Indeed, today brings us closer to our shared goals of justice and peace." Judge'
#             " Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the"
#             ' Palestinians. "As the Rome Statute today enters into force for the State of Palestine, Palestine'
#             " acquires all the rights as well as responsibilities that come with being a State Party to the Statute."
#             ' These are substantive commitments, which cannot be taken lightly," she said. Rights group Human Rights'
#             ' Watch welcomed the development. "Governments seeking to penalize Palestine for joining the ICC should'
#             " immediately end their pressure, and countries that support universal acceptance of the court's treaty"
#             ' should speak out to welcome its membership," said Balkees Jarrah, international justice counsel for the'
#             " group. \"What's objectionable is the attempts to undermine international justice, not Palestine's"
#             ' decision to join a treaty to which over 100 countries around the world are members." In January, when'
#             " the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described it as an"
#             ' outrage, saying the court was overstepping its boundaries. The United States also said it "strongly"'
#             " disagreed with the court's decision. \"As we have said repeatedly, we do not believe that Palestine is a"
#             ' state and therefore we do not believe that it is eligible to join the ICC," the State Department said in'
#             ' a statement. It urged the warring sides to resolve their differences through direct negotiations. "We'
#             ' will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace,"'
#             " it said. But the ICC begs to differ with the definition of a state for its purposes and refers to the"
#             ' territories as "Palestine." While a preliminary examination is not a formal investigation, it allows the'
#             " court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou"
#             ' Bensouda said her office would "conduct its analysis in full independence and impartiality." The war'
#             " between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The inquiry"
#             " will include alleged war crimes committed since June. The International Criminal Court was set up in"
#             " 2002 to prosecute genocide, crimes against humanity and war crimes. CNN's Vasco Cotovio, Kareem Khadder"
#             " and Faith Karimi contributed to this report."
#         )
#         IRAN_ARTICLE = (
#             "(CNN)The United States and its negotiating partners reached a very strong framework agreement with Iran"
#             " in Lausanne, Switzerland, on Thursday that limits Iran's nuclear program in such a way as to effectively"
#             " block it from building a nuclear weapon. Expect pushback anyway, if the recent past is any harbinger."
#             " Just last month, in an attempt to head off such an agreement, House Speaker John Boehner invited Israeli"
#             " Prime Minister Benjamin Netanyahu to preemptively blast it before Congress, and 47 senators sent a"
#             " letter to the Iranian leadership warning them away from a deal. The debate that has already begun since"
#             " the announcement of the new framework will likely result in more heat than light. It will not be helped"
#             " by the gathering swirl of dubious assumptions and doubtful assertions. Let us address some of these: ."
#             " The most misleading assertion, despite universal rejection by experts, is that the negotiations'"
#             " objective at the outset was the total elimination of any nuclear program in Iran. That is the position"
#             " of Netanyahu and his acolytes in the U.S. Congress. But that is not and never was the objective. If it"
#             " had been, there would have been no Iranian team at the negotiating table. Rather, the objective has"
#             " always been to structure an agreement or series of agreements so that Iran could not covertly develop a"
#             " nuclear arsenal before the United States and its allies could respond. The new framework has exceeded"
#             " expectations in achieving that goal. It would reduce Iran's low-enriched uranium stockpile, cut by"
#             " two-thirds its number of installed centrifuges and implement a rigorous inspection regime. Another"
#             " dubious assumption of opponents is that the Iranian nuclear program is a covert weapons program. Despite"
#             " sharp accusations by some in the United States and its allies, Iran denies having such a program, and"
#             " U.S. intelligence contends that Iran has not yet made the decision to build a nuclear weapon. Iran's"
#             " continued cooperation with International Atomic Energy Agency inspections is further evidence on this"
#             " point, and we'll know even more about Iran's program in the coming months and years because of the deal."
#             " In fact, the inspections provisions that are part of this agreement are designed to protect against any"
#             " covert action by the Iranians. What's more, the rhetoric of some members of Congress has implied that"
#             " the negotiations have been between only the United States and Iran (i.e., the 47 senators' letter"
#             " warning that a deal might be killed by Congress or a future president). This of course is not the case."
#             " The talks were between Iran and the five permanent members of the U.N. Security Council (United States,"
#             " United Kingdom, France, China and Russia) plus Germany, dubbed the P5+1. While the United States has"
#             " played a leading role in the effort, it negotiated the terms alongside its partners. If the agreement"
#             " reached by the P5+1 is rejected by Congress, it could result in an unraveling of the sanctions on Iran"
#             " and threaten NATO cohesion in other areas. Another questionable assertion is that this agreement"
#             " contains a sunset clause, after which Iran will be free to do as it pleases. Again, this is not the"
#             " case. Some of the restrictions on Iran's nuclear activities, such as uranium enrichment, will be eased"
#             " or eliminated over time, as long as 15 years. But most importantly, the framework agreement includes"
#             " Iran's ratification of the Additional Protocol, which allows IAEA inspectors expanded access to nuclear"
#             " sites both declared and nondeclared. This provision will be permanent. It does not sunset. Thus, going"
#             " forward, if Iran decides to enrich uranium to weapons-grade levels, monitors will be able to detect such"
#             " a move in a matter of days and alert the U.N. Security Council. Many in Congress have said that the"
#             ' agreement should be a formal treaty requiring the Senate to "advise and consent." But the issue is not'
#             " suited for a treaty. Treaties impose equivalent obligations on all signatories. For example, the New"
#             " START treaty limits Russia and the United States to 1,550 deployed strategic warheads. But any agreement"
#             " with Iran will not be so balanced.  The restrictions and obligations in the final framework agreement"
#             " will be imposed almost exclusively on Iran. The P5+1 are obligated only to ease and eventually remove"
#             " most but not all economic sanctions, which were imposed as leverage to gain this final deal. Finally"
#             " some insist that any agreement must address Iranian missile programs, human rights violations or support"
#             " for Hamas or Hezbollah.  As important as these issues are, and they must indeed be addressed, they are"
#             " unrelated to the most important aim of a nuclear deal: preventing a nuclear Iran.  To include them in"
#             " the negotiations would be a poison pill. This agreement should be judged on its merits and on how it"
#             " affects the security of our negotiating partners and allies, including Israel. Those judgments should be"
#             " fact-based, not based on questionable assertions or dubious assumptions."
#         )
#         ARTICLE_SUBWAY = (
#             "New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. A"
#             " year later, she got married again in Westchester County, but to a different man and without divorcing"
#             " her first husband.  Only 18 days after that marriage, she got hitched yet again. Then, Barrientos"
#             ' declared "I do" five more times, sometimes only within two weeks of each other. In 2010, she married'
#             " once more, this time in the Bronx. In an application for a marriage license, she stated it was her"
#             ' "first and only" marriage. Barrientos, now 39, is facing two criminal counts of "offering a false'
#             ' instrument for filing in the first degree," referring to her false statements on the 2010 marriage'
#             " license application, according to court documents. Prosecutors said the marriages were part of an"
#             " immigration scam. On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to"
#             " her attorney, Christopher Wright, who declined to comment further. After leaving court, Barrientos was"
#             " arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New"
#             " York subway through an emergency exit, said Detective Annette Markowski, a police spokeswoman. In total,"
#             " Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.  All"
#             " occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be"
#             " married to four men, and at one time, she was married to eight men at once, prosecutors say. Prosecutors"
#             " said the immigration scam involved some of her husbands, who filed for permanent residence status"
#             " shortly after the marriages.  Any divorces happened only after such filings were approved. It was"
#             " unclear whether any of the men will be prosecuted. The case was referred to the Bronx District"
#             " Attorney's Office by Immigration and Customs Enforcement and the Department of Homeland Security's"
#             ' Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt,'
#             " Turkey, Georgia, Pakistan and Mali. Her eighth husband, Rashid Rajput, was deported in 2006 to his"
#             " native Pakistan after an investigation by the Joint Terrorism Task Force. If convicted, Barrientos faces"
#             " up to four years in prison.  Her next court appearance is scheduled for May 18."
#         )
#
#         expected_summaries = [
#             'prosecutor: "so far no videos were used in the crash investigation" two magazines claim to have found a'
#             " cell phone video of the final seconds . \"one can hear cries of 'My God' in several languages,\" one"
#             " magazine says .",
#             "the formal accession was marked by a ceremony at The Hague, in the Netherlands . the ICC opened a"
#             " preliminary examination into the situation in the occupied Palestinian territory . as members of the"
#             " court, Palestinians may be subject to counter-charges as well .",
#             "the u.s. and its negotiating partners reached a very strong framework agreement with Iran . aaron miller:"
#             " the debate that has already begun since the announcement of the new framework will likely result in more"
#             " heat than light . the deal would reduce Iran's low-enriched uranium stockpile, cut centrifuges and"
#             " implement a rigorous inspection regime .",
#             "prosecutors say the marriages were part of an immigration scam . if convicted, barrientos faces two"
#             ' criminal counts of "offering a false instrument for filing in the first degree" she has been married 10'
#             " times, with nine of her marriages occurring between 1999 and 2002 .",
#         ]
#
#         use_task_specific_params(model, "summarization")
#
#         dct = tok(
#             [model.config.prefix + x for x in [FRANCE_ARTICLE, SHORTER_ARTICLE, IRAN_ARTICLE, ARTICLE_SUBWAY]],
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt",
#         ).to(torch_device)
#         self.assertEqual(512, dct["input_ids"].shape[1])
#
#         hypotheses_batch = model.generate(
#             **dct,
#             num_beams=4,
#             length_penalty=2.0,
#             max_length=142,
#             min_length=56,
#             no_repeat_ngram_size=3,
#             do_sample=False,
#             early_stopping=True,
#         )
#
#         decoded = tok.batch_decode(hypotheses_batch, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#         self.assertListEqual(
#             expected_summaries,
#             decoded,
#         )
#
#     @slow
#     def test_translation_en_to_de(self):
#         model = self.model
#         tok = self.tokenizer
#         use_task_specific_params(model, "translation_en_to_de")
#
#         en_text = '"Luigi often said to me that he never wanted the brothers to end up in court", she wrote.'
#         expected_translation = (
#             '"Luigi sagte mir oft, dass er nie wollte, dass die Brüder am Gericht sitzen", schrieb sie.'
#         )
#
#         input_ids = tok.encode(model.config.prefix + en_text, return_tensors="pt")
#         input_ids = input_ids.to(torch_device)
#         output = model.generate(input_ids)
#         translation = tok.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
#         self.assertEqual(translation, expected_translation)
#
#     @slow
#     def test_translation_en_to_fr(self):
#         model = self.model  # t5-base
#         tok = self.tokenizer
#         use_task_specific_params(model, "translation_en_to_fr")
#
#         en_text = (
#             ' This image section from an infrared recording by the Spitzer telescope shows a "family portrait" of'
#             " countless generations of stars: the oldest stars are seen as blue dots. "
#         )
#
#         input_ids = tok.encode(model.config.prefix + en_text, return_tensors="pt")
#         input_ids = input_ids.to(torch_device)
#
#         output = model.generate(
#             input_ids=input_ids,
#             num_beams=4,
#             length_penalty=2.0,
#             max_length=100,
#             no_repeat_ngram_size=3,
#             do_sample=False,
#             early_stopping=True,
#         )
#         translation = tok.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
#         new_truncated_translation = (
#             "Cette section d'images provenant de l'enregistrement infrarouge effectué par le télescope Spitzer montre "
#             "un "
#             "« portrait familial » de générations innombrables d’étoiles : les plus anciennes sont observées "
#             "sous forme "
#             "de points bleus."
#         )
#
#         self.assertEqual(translation, new_truncated_translation)
#
#     @slow
#     def test_translation_en_to_ro(self):
#         model = self.model
#         tok = self.tokenizer
#         use_task_specific_params(model, "translation_en_to_ro")
#         en_text = "Taco Bell said it plans to add 2,000 locations in the US by 2022."
#         expected_translation = "Taco Bell a declarat că intenţionează să adauge 2 000 de locaţii în SUA până în 2022."
#
#         inputs = tok(model.config.prefix + en_text, return_tensors="pt").to(torch_device)
#         output = model.generate(**inputs)
#         translation = tok.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
#         self.assertEqual(translation, expected_translation)
#
#     @slow
#     def test_contrastive_search_t5(self):
#         article = (
#             " New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. A"
#             " year later, she got married again in Westchester County, but to a different man and without divorcing"
#             " her first husband.  Only 18 days after that marriage, she got hitched yet again. Then, Barrientos"
#             ' declared "I do" five more times, sometimes only within two weeks of each other. In 2010, she married'
#             " once more, this time in the Bronx. In an application for a marriage license, she stated it was her"
#             ' "first and only" marriage. Barrientos, now 39, is facing two criminal counts of "offering a false'
#             ' instrument for filing in the first degree," referring to her false statements on the 2010 marriage'
#             " license application, according to court documents. Prosecutors said the marriages were part of an"
#             " immigration scam. On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to"
#             " her attorney, Christopher Wright, who declined to comment further. After leaving court, Barrientos was"
#             " arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New"
#             " York subway through an emergency exit, said Detective Annette Markowski, a police spokeswoman. In total,"
#             " Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.  All"
#             " occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be"
#             " married to four men, and at one time, she was married to eight men at once, prosecutors say. Prosecutors"
#             " said the immigration scam involved some of her husbands, who filed for permanent residence status"
#             " shortly after the marriages.  Any divorces happened only after such filings were approved. It was"
#             " unclear whether any of the men will be prosecuted. The case was referred to the Bronx District"
#             " Attorney's Office by Immigration and Customs Enforcement and the Department of Homeland Security's"
#             ' Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt,'
#             " Turkey, Georgia, Pakistan and Mali. Her eighth husband, Rashid Rajput, was deported in 2006 to his"
#             " native Pakistan after an investigation by the Joint Terrorism Task Force. If convicted, Barrientos faces"
#             " up to four years in prison.  Her next court appearance is scheduled for May 18."
#         )
#         article = "summarize: " + article.strip()
#         t5_tokenizer = AutoTokenizer.from_pretrained("flax-community/t5-base-cnn-dm")
#         t5_model = T5ForConditionalGeneration.from_pretrained("flax-community/t5-base-cnn-dm").to(torch_device)
#         input_ids = t5_tokenizer(
#             article, add_special_tokens=False, truncation=True, max_length=512, return_tensors="pt"
#         ).input_ids.to(torch_device)
#
#         outputs = t5_model.generate(input_ids, penalty_alpha=0.5, top_k=5, max_length=64)
#         generated_text = t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
#
#         self.assertListEqual(
#             generated_text,
#             [
#                 "Liana Barrientos has been married 10 times, nine of them in the Bronx. Her husbands filed for "
#                 "permanent residence after the marriages, prosecutors say."
#             ],
#         )
#
#
# @require_torch
# class TestAsymmetricT5(unittest.TestCase):
#     def build_model_and_check_forward_pass(self, **kwargs):
#         tester = T5ModelTester(self, **kwargs)
#         config, *inputs = tester.prepare_config_and_inputs()
#         (
#             input_ids,
#             decoder_input_ids,
#             attention_mask,
#             decoder_attention_mask,
#             lm_labels,
#         ) = inputs
#         model = T5ForConditionalGeneration(config=config).to(torch_device).eval()
#         outputs = model(
#             input_ids=input_ids,
#             decoder_input_ids=decoder_input_ids,
#             decoder_attention_mask=decoder_attention_mask,
#             labels=lm_labels,
#         )
#         # outputs = model(*inputs)
#         assert len(outputs) == 4
#         assert outputs["logits"].size() == (tester.batch_size, tester.decoder_seq_length, tester.vocab_size)
#         assert outputs["loss"].size() == ()
#         return model
#
#     def test_small_decoder(self):
#         # num_hidden_layers is passed to T5Config as num_layers
#         model = self.build_model_and_check_forward_pass(decoder_layers=1, num_hidden_layers=2)
#         assert len(model.encoder.block) == 2
#         assert len(model.decoder.block) == 1
#
#     def test_defaulting_to_symmetry(self):
#         # num_hidden_layers is passed to T5Config as num_layers
#         model = self.build_model_and_check_forward_pass(num_hidden_layers=2)
#         assert len(model.decoder.block) == len(model.encoder.block) == 2
