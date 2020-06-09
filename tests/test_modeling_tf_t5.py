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


import unittest

from transformers import T5Config, is_tf_available

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor
from .utils import require_tf, slow


if is_tf_available():
    import tensorflow as tf
    from transformers import TFT5Model, TFT5ForConditionalGeneration, T5Tokenizer


@require_tf
class TFT5ModelTest(TFModelTesterMixin, unittest.TestCase):

    is_encoder_decoder = True
    all_model_classes = (TFT5Model, TFT5ForConditionalGeneration) if is_tf_available() else ()
    all_generative_model_classes = (TFT5ForConditionalGeneration,) if is_tf_available() else ()

    class TFT5ModelTester(object):
        def __init__(
            self,
            parent,
            batch_size=13,
            seq_length=7,
            is_training=True,
            use_input_mask=True,
            use_labels=True,
            vocab_size=99,
            n_positions=14,
            hidden_size=32,
            num_hidden_layers=5,
            num_attention_heads=4,
            d_ff=37,
            relative_attention_num_buckets=8,
            dropout_rate=0.1,
            initializer_factor=0.002,
            eos_token_id=1,
            pad_token_id=0,
            scope=None,
        ):
            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.is_training = is_training
            self.use_input_mask = use_input_mask
            self.use_labels = use_labels
            self.vocab_size = vocab_size
            self.n_positions = n_positions
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.d_ff = d_ff
            self.relative_attention_num_buckets = relative_attention_num_buckets
            self.dropout_rate = dropout_rate
            self.initializer_factor = initializer_factor
            self.eos_token_id = eos_token_id
            self.pad_token_id = pad_token_id
            self.scope = scope

        def prepare_config_and_inputs(self):
            input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

            input_mask = None
            if self.use_input_mask:
                input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

            token_labels = None
            if self.use_labels:
                token_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

            config = T5Config(
                vocab_size=self.vocab_size,
                n_positions=self.n_positions,
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
            )

            return (config, input_ids, input_mask, token_labels)

        def create_and_check_t5_model(self, config, input_ids, input_mask, token_labels):
            model = TFT5Model(config=config)
            inputs = {
                "inputs": input_ids,
                "decoder_input_ids": input_ids,
                "decoder_attention_mask": input_mask,
            }
            decoder_output, decoder_past, encoder_output = model(inputs)

            decoder_output, decoder_past, encoder_output = model(
                input_ids, decoder_attention_mask=input_mask, decoder_input_ids=input_ids
            )
            result = {
                "encoder_output": encoder_output.numpy(),
                "decoder_past": decoder_past,
                "decoder_output": decoder_output.numpy(),
            }
            self.parent.assertListEqual(
                list(result["encoder_output"].shape), [self.batch_size, self.seq_length, self.hidden_size]
            )
            self.parent.assertListEqual(
                list(result["decoder_output"].shape), [self.batch_size, self.seq_length, self.hidden_size]
            )
            self.parent.assertEqual(len(decoder_past), 2)
            # decoder_past[0] should correspond to encoder output
            self.parent.assertTrue(tf.reduce_all(tf.math.equal(decoder_past[0][0], encoder_output)))
            # There should be `num_layers` key value embeddings stored in decoder_past[1]
            self.parent.assertEqual(len(decoder_past[1]), config.num_layers)
            # There should be a self attn key, a self attn value, a cross attn key and a cross attn value stored in each decoder_past[1] tuple
            self.parent.assertEqual(len(decoder_past[1][0]), 4)

        def create_and_check_t5_with_lm_head(self, config, input_ids, input_mask, token_labels):
            model = TFT5ForConditionalGeneration(config=config)
            inputs_dict = {
                "inputs": input_ids,
                "decoder_input_ids": input_ids,
                "decoder_attention_mask": input_mask,
            }

            prediction_scores, _, _ = model(inputs_dict)

            result = {
                "prediction_scores": prediction_scores.numpy(),
            }
            self.parent.assertListEqual(
                list(result["prediction_scores"].shape), [self.batch_size, self.seq_length, self.vocab_size]
            )

        def create_and_check_t5_decoder_model_past(self, config, input_ids, decoder_input_ids, attention_mask):
            model = TFT5Model(config=config).get_decoder()

            input_ids = input_ids[:1, :]
            self.batch_size = 1

            # first forward pass
            _, past_key_value_states = model(input_ids, use_cache=True)

            # create hypothetical next token and extent to next_input_ids
            next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

            # append to next input_ids and
            next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)

            output_from_no_past = model(next_input_ids)[0]
            output_from_past = model(next_tokens, past_key_value_states=past_key_value_states)[0]

            # select random slice
            random_slice_idx = int(ids_tensor((1,), output_from_past.shape[-1]))
            output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx]
            output_from_past_slice = output_from_past[:, 0, random_slice_idx]

            # test that outputs are equal for slice
            tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-3)

        def create_and_check_t5_decoder_model_attention_mask_past(
            self, config, input_ids, decoder_input_ids, attention_mask
        ):
            model = TFT5Model(config=config).get_decoder()

            # create attention mask
            half_seq_length = self.seq_length // 2
            attn_mask_begin = tf.ones((self.batch_size, half_seq_length), dtype=tf.int32)
            attn_mask_end = tf.zeros((self.batch_size, self.seq_length - half_seq_length), dtype=tf.int32)
            attn_mask = tf.concat([attn_mask_begin, attn_mask_end], axis=1)

            # first forward pass
            _, past_key_value_states = model(input_ids, attention_mask=attn_mask, use_cache=True)

            # create hypothetical next token and extent to next_input_ids
            next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

            # change a random masked slice from input_ids
            random_seq_idx_to_change = ids_tensor((1,), half_seq_length).numpy() + 1
            random_other_next_tokens = ids_tensor((self.batch_size, self.seq_length), config.vocab_size)
            vector_condition = tf.range(self.seq_length) == (self.seq_length - random_seq_idx_to_change)
            condition = tf.transpose(
                tf.broadcast_to(tf.expand_dims(vector_condition, -1), (self.seq_length, self.batch_size))
            )
            input_ids = tf.where(condition, random_other_next_tokens, input_ids)

            # append to next input_ids and attn_mask
            next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
            attn_mask = tf.concat([attn_mask, tf.ones((attn_mask.shape[0], 1), dtype=tf.int32)], axis=1,)

            # get two different outputs
            output_from_no_past = model(next_input_ids, attention_mask=attn_mask)[0]
            output_from_past = model(
                next_tokens, past_key_value_states=past_key_value_states, attention_mask=attn_mask
            )[0]

            # select random slice
            random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).numpy().item()
            output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx]
            output_from_past_slice = output_from_past[:, 0, random_slice_idx]

            # test that outputs are equal for slice
            tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-3)

        def prepare_config_and_inputs_for_common(self):
            config_and_inputs = self.prepare_config_and_inputs()
            (config, input_ids, input_mask, token_labels) = config_and_inputs
            inputs_dict = {
                "inputs": input_ids,
                "decoder_input_ids": input_ids,
                "decoder_attention_mask": input_mask,
                "use_cache": tf.convert_to_tensor([False]),
            }
            return config, inputs_dict

    def setUp(self):
        self.model_tester = TFT5ModelTest.TFT5ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=T5Config, d_model=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_t5_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_t5_model(*config_and_inputs)

    def test_with_lm_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_t5_with_lm_head(*config_and_inputs)

    def test_t5_decoder_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_t5_decoder_model_past(*config_and_inputs)

    def test_t5_decoder_model_past_with_attn_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_t5_decoder_model_attention_mask_past(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in ["t5-small"]:
            model = TFT5Model.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_tf
class TFT5ModelIntegrationTests(unittest.TestCase):
    @slow
    def test_summarization(self):
        model = TFT5ForConditionalGeneration.from_pretrained("t5-base")
        tok = T5Tokenizer.from_pretrained("t5-base")

        FRANCE_ARTICLE = 'Marseille, France (CNN)The French prosecutor leading an investigation into the crash of Germanwings Flight 9525 insisted Wednesday that he was not aware of any video footage from on board the plane. Marseille prosecutor Brice Robin told CNN that "so far no videos were used in the crash investigation." He added, "A person who has such a video needs to immediately give it to the investigators." Robin\'s comments follow claims by two magazines, German daily Bild and French Paris Match, of a cell phone video showing the harrowing final seconds from on board Germanwings Flight 9525 as it crashed into the French Alps. All 150 on board were killed. Paris Match and Bild reported that the video was recovered from a phone at the wreckage site. The two publications described the supposed video, but did not post it on their websites. The publications said that they watched the video, which was found by a source close to the investigation. "One can hear cries of \'My God\' in several languages," Paris Match reported. "Metallic banging can also be heard more than three times, perhaps of the pilot trying to open the cockpit door with a heavy object.  Towards the end, after a heavy shake, stronger than the others, the screaming intensifies. Then nothing." "It is a very disturbing scene," said Julian Reichelt, editor-in-chief of Bild online. An official with France\'s accident investigation agency, the BEA, said the agency is not aware of any such video. Lt. Col. Jean-Marc Menichini, a French Gendarmerie spokesman in charge of communications on rescue efforts around the Germanwings crash site, told CNN that the reports were "completely wrong" and "unwarranted." Cell phones have been collected at the site, he said, but that they "hadn\'t been exploited yet." Menichini said he believed the cell phones would need to be sent to the Criminal Research Institute in Rosny sous-Bois, near Paris, in order to be analyzed by specialized technicians working hand-in-hand with investigators. But none of the cell phones found so far have been sent to the institute, Menichini said. Asked whether staff involved in the search could have leaked a memory card to the media, Menichini answered with a categorical "no." Reichelt told "Erin Burnett: Outfront" that he had watched the video and stood by the report, saying Bild and Paris Match are "very confident" that the clip is real. He noted that investigators only revealed they\'d recovered cell phones from the crash site after Bild and Paris Match published their reports. "That is something we did not know before. ... Overall we can say many things of the investigation weren\'t revealed by the investigation at the beginning," he said. What was mental state of Germanwings co-pilot? German airline Lufthansa confirmed Tuesday that co-pilot Andreas Lubitz had battled depression years before he took the controls of Germanwings Flight 9525, which he\'s accused of deliberately crashing last week in the French Alps. Lubitz told his Lufthansa flight training school in 2009 that he had a "previous episode of severe depression," the airline said Tuesday. Email correspondence between Lubitz and the school discovered in an internal investigation, Lufthansa said, included medical documents he submitted in connection with resuming his flight training. The announcement indicates that Lufthansa, the parent company of Germanwings, knew of Lubitz\'s battle with depression, allowed him to continue training and ultimately put him in the cockpit. Lufthansa, whose CEO Carsten Spohr previously said Lubitz was 100% fit to fly, described its statement Tuesday as a "swift and seamless clarification" and said it was sharing the information and documents -- including training and medical records -- with public prosecutors. Spohr traveled to the crash site Wednesday, where recovery teams have been working for the past week to recover human remains and plane debris scattered across a steep mountainside. He saw the crisis center set up in Seyne-les-Alpes, laid a wreath in the village of Le Vernet, closer to the crash site, where grieving families have left flowers at a simple stone memorial. Menichini told CNN late Tuesday that no visible human remains were left at the site but recovery teams would keep searching. French President Francois Hollande, speaking Tuesday, said that it should be possible to identify all the victims using DNA analysis by the end of the week, sooner than authorities had previously suggested. In the meantime, the recovery of the victims\' personal belongings will start Wednesday, Menichini said. Among those personal belongings could be more cell phones belonging to the 144 passengers and six crew on board. Check out the latest from our correspondents . The details about Lubitz\'s correspondence with the flight school during his training were among several developments as investigators continued to delve into what caused the crash and Lubitz\'s possible motive for downing the jet. A Lufthansa spokesperson told CNN on Tuesday that Lubitz had a valid medical certificate, had passed all his examinations and "held all the licenses required." Earlier, a spokesman for the prosecutor\'s office in Dusseldorf, Christoph Kumpa, said medical records reveal Lubitz suffered from suicidal tendencies at some point before his aviation career and underwent psychotherapy before he got his pilot\'s license. Kumpa emphasized there\'s no evidence suggesting Lubitz was suicidal or acting aggressively before the crash. Investigators are looking into whether Lubitz feared his medical condition would cause him to lose his pilot\'s license, a European government official briefed on the investigation told CNN on Tuesday. While flying was "a big part of his life," the source said, it\'s only one theory being considered. Another source, a law enforcement official briefed on the investigation, also told CNN that authorities believe the primary motive for Lubitz to bring down the plane was that he feared he would not be allowed to fly because of his medical problems. Lubitz\'s girlfriend told investigators he had seen an eye doctor and a neuropsychologist, both of whom deemed him unfit to work recently and concluded he had psychological issues, the European government official said. But no matter what details emerge about his previous mental health struggles, there\'s more to the story, said Brian Russell, a forensic psychologist. "Psychology can explain why somebody would turn rage inward on themselves about the fact that maybe they weren\'t going to keep doing their job and they\'re upset about that and so they\'re suicidal," he said. "But there is no mental illness that explains why somebody then feels entitled to also take that rage and turn it outward on 149 other people who had nothing to do with the person\'s problems." Germanwings crash compensation: What we know . Who was the captain of Germanwings Flight 9525? CNN\'s Margot Haddad reported from Marseille and Pamela Brown from Dusseldorf, while Laura Smith-Spark wrote from London. CNN\'s Frederik Pleitgen, Pamela Boykoff, Antonia Mortensen, Sandrine Amiel and Anna-Maja Rappard contributed to this report.'  # @noqa
        EXPECTED_SUMMARY_FRANCE = 'french prosecutor says he is not aware of any video footage from on board the plane . prosecutor: "so far no videos were used in the crash investigation" two magazines claim to have found a cell phone video of the final seconds of flight 9525 . all 150 on board were killed when the plane crashed into the french Alps .'

        SHORTER_ARTICLE = '(CNN)The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based. The Palestinians signed the ICC\'s founding Rome Statute in January, when they also accepted its jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including East Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination into the situation in Palestinian territories, paving the way for possible war crimes investigations against Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and the United States, neither of which is an ICC member, opposed the Palestinians\' efforts to join the body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday\'s ceremony, said it was a move toward greater justice. "As Palestine formally becomes a State Party to the Rome Statute today, the world is also a step closer to ending a long era of impunity and injustice," he said, according to an ICC news release. "Indeed, today brings us closer to our shared goals of justice and peace." Judge Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the Palestinians. "As the Rome Statute today enters into force for the State of Palestine, Palestine acquires all the rights as well as responsibilities that come with being a State Party to the Statute. These are substantive commitments, which cannot be taken lightly," she said. Rights group Human Rights Watch welcomed the development. "Governments seeking to penalize Palestine for joining the ICC should immediately end their pressure, and countries that support universal acceptance of the court\'s treaty should speak out to welcome its membership," said Balkees Jarrah, international justice counsel for the group. "What\'s objectionable is the attempts to undermine international justice, not Palestine\'s decision to join a treaty to which over 100 countries around the world are members." In January, when the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described it as an outrage, saying the court was overstepping its boundaries. The United States also said it "strongly" disagreed with the court\'s decision. "As we have said repeatedly, we do not believe that Palestine is a state and therefore we do not believe that it is eligible to join the ICC," the State Department said in a statement. It urged the warring sides to resolve their differences through direct negotiations. "We will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace," it said. But the ICC begs to differ with the definition of a state for its purposes and refers to the territories as "Palestine." While a preliminary examination is not a formal investigation, it allows the court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou Bensouda said her office would "conduct its analysis in full independence and impartiality." The war between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The inquiry will include alleged war crimes committed since June. The International Criminal Court was set up in 2002 to prosecute genocide, crimes against humanity and war crimes. CNN\'s Vasco Cotovio, Kareem Khadder and Faith Karimi contributed to this report.'
        EXPECTED_SUMMARY_SHORTER = "the formal accession was marked with a ceremony at The Hague, in the Netherlands . the Palestinians signed the ICC's founding Rome Statute in January . they also accepted its jurisdiction over alleged crimes committed in occupied Palestinian territory . as members, Palestinians may be subject to counter-charges as well ."

        IRAN_ARTICLE = "(CNN)The United States and its negotiating partners reached a very strong framework agreement with Iran in Lausanne, Switzerland, on Thursday that limits Iran's nuclear program in such a way as to effectively block it from building a nuclear weapon. Expect pushback anyway, if the recent past is any harbinger. Just last month, in an attempt to head off such an agreement, House Speaker John Boehner invited Israeli Prime Minister Benjamin Netanyahu to preemptively blast it before Congress, and 47 senators sent a letter to the Iranian leadership warning them away from a deal. The debate that has already begun since the announcement of the new framework will likely result in more heat than light. It will not be helped by the gathering swirl of dubious assumptions and doubtful assertions. Let us address some of these: . The most misleading assertion, despite universal rejection by experts, is that the negotiations' objective at the outset was the total elimination of any nuclear program in Iran. That is the position of Netanyahu and his acolytes in the U.S. Congress. But that is not and never was the objective. If it had been, there would have been no Iranian team at the negotiating table. Rather, the objective has always been to structure an agreement or series of agreements so that Iran could not covertly develop a nuclear arsenal before the United States and its allies could respond. The new framework has exceeded expectations in achieving that goal. It would reduce Iran's low-enriched uranium stockpile, cut by two-thirds its number of installed centrifuges and implement a rigorous inspection regime. Another dubious assumption of opponents is that the Iranian nuclear program is a covert weapons program. Despite sharp accusations by some in the United States and its allies, Iran denies having such a program, and U.S. intelligence contends that Iran has not yet made the decision to build a nuclear weapon. Iran's continued cooperation with International Atomic Energy Agency inspections is further evidence on this point, and we'll know even more about Iran's program in the coming months and years because of the deal. In fact, the inspections provisions that are part of this agreement are designed to protect against any covert action by the Iranians. What's more, the rhetoric of some members of Congress has implied that the negotiations have been between only the United States and Iran (i.e., the 47 senators' letter warning that a deal might be killed by Congress or a future president). This of course is not the case. The talks were between Iran and the five permanent members of the U.N. Security Council (United States, United Kingdom, France, China and Russia) plus Germany, dubbed the P5+1. While the United States has played a leading role in the effort, it negotiated the terms alongside its partners. If the agreement reached by the P5+1 is rejected by Congress, it could result in an unraveling of the sanctions on Iran and threaten NATO cohesion in other areas. Another questionable assertion is that this agreement contains a sunset clause, after which Iran will be free to do as it pleases. Again, this is not the case. Some of the restrictions on Iran's nuclear activities, such as uranium enrichment, will be eased or eliminated over time, as long as 15 years. But most importantly, the framework agreement includes Iran's ratification of the Additional Protocol, which allows IAEA inspectors expanded access to nuclear sites both declared and nondeclared. This provision will be permanent. It does not sunset. Thus, going forward, if Iran decides to enrich uranium to weapons-grade levels, monitors will be able to detect such a move in a matter of days and alert the U.N. Security Council. Many in Congress have said that the agreement should be a formal treaty requiring the Senate to \"advise and consent.\" But the issue is not suited for a treaty. Treaties impose equivalent obligations on all signatories. For example, the New START treaty limits Russia and the United States to 1,550 deployed strategic warheads. But any agreement with Iran will not be so balanced.  The restrictions and obligations in the final framework agreement will be imposed almost exclusively on Iran. The P5+1 are obligated only to ease and eventually remove most but not all economic sanctions, which were imposed as leverage to gain this final deal. Finally some insist that any agreement must address Iranian missile programs, human rights violations or support for Hamas or Hezbollah.  As important as these issues are, and they must indeed be addressed, they are unrelated to the most important aim of a nuclear deal: preventing a nuclear Iran.  To include them in the negotiations would be a poison pill. This agreement should be judged on its merits and on how it affects the security of our negotiating partners and allies, including Israel. Those judgments should be fact-based, not based on questionable assertions or dubious assumptions."
        EXPECTED_SUMMARY_IRAN = "the united states and its negotiating partners reached a very strong framework agreement with Iran . the agreement limits Iran's nuclear program in such a way as to effectively block it from building a nuclear weapon . expect pushback anyway, if the recent past is any harbinger ."

        ARTICLE_SUBWAY = 'New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.  Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other. In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage. Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the 2010 marriage license application, according to court documents. Prosecutors said the marriages were part of an immigration scam. On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further. After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.  All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say. Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.  Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted. The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali. Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force. If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.'
        EXPECTED_SUMMARY_SUBWAY = "in total, barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002 . she is believed to still be married to four men, and at one time, she was married to eight men at once . prosecutors say the marriages were part of an immigration scam ."

        task_specific_config = getattr(model.config, "task_specific_params", {})
        summarization_config = task_specific_config.get("summarization", {})
        model.config.update(summarization_config)

        dct = tok.batch_encode_plus(
            [model.config.prefix + x for x in [FRANCE_ARTICLE, SHORTER_ARTICLE, IRAN_ARTICLE, ARTICLE_SUBWAY]],
            max_length=512,
            pad_to_max_length=True,
            return_tensors="tf",
        )
        self.assertEqual(512, dct["input_ids"].shape[1])

        hypotheses_batch = model.generate(
            input_ids=dct["input_ids"],
            attention_mask=dct["attention_mask"],
            num_beams=4,
            length_penalty=2.0,
            max_length=142,
            min_length=56,
            no_repeat_ngram_size=3,
            do_sample=False,
            early_stopping=True,
        )

        decoded = [
            tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in hypotheses_batch
        ]

        self.assertListEqual(
            [EXPECTED_SUMMARY_FRANCE, EXPECTED_SUMMARY_SHORTER, EXPECTED_SUMMARY_IRAN, EXPECTED_SUMMARY_SUBWAY],
            decoded,
        )

    @slow
    def test_translation_en_to_de(self):
        model = TFT5ForConditionalGeneration.from_pretrained("t5-base")
        tok = T5Tokenizer.from_pretrained("t5-base")

        task_specific_config = getattr(model.config, "task_specific_params", {})
        translation_config = task_specific_config.get("translation_en_to_de", {})
        model.config.update(translation_config)

        original_input = '"Luigi often said to me that he never wanted the brothers to end up in court", she wrote.'
        expected_translation = (
            '"Luigi sagte mir oft, dass er nie wollte, dass die Brüder am Gericht sitzen", schrieb sie.'
        )

        input_ids = tok.encode(model.config.prefix + original_input, return_tensors="tf")

        output = model.generate(
            input_ids=input_ids,
            num_beams=4,
            length_penalty=2.0,
            max_length=50,
            no_repeat_ngram_size=3,
            do_sample=False,
            early_stopping=True,
        )
        translation = tok.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        self.assertEqual(translation, expected_translation)

    @slow
    def test_translation_en_to_fr(self):
        model = TFT5ForConditionalGeneration.from_pretrained("t5-base")
        tok = T5Tokenizer.from_pretrained("t5-base")

        task_specific_config = getattr(model.config, "task_specific_params", {})
        translation_config = task_specific_config.get("translation_en_to_fr", {})
        model.config.update(translation_config)

        original_input = 'This image section from an infrared recording by the Spitzer telescope shows a "family portrait" of countless generations of stars: the oldest stars are seen as blue dots, while more difficult to identify are the pink-coloured "new-borns" in the star delivery room.'
        expected_translation = "Cette section d'images provenant de l'enregistrement infrarouge effectué par le télescope Spitzer montre un « portrait familial » de générations innombrables de étoiles : les plus anciennes sont observées sous forme de pointes bleues, alors que les « nouveau-nés » de couleur rose dans la salle des accouchements doivent être plus difficiles "

        input_ids = tok.encode(model.config.prefix + original_input, return_tensors="tf")

        output = model.generate(
            input_ids=input_ids,
            num_beams=4,
            length_penalty=2.0,
            max_length=100,
            no_repeat_ngram_size=3,
            do_sample=False,
            early_stopping=True,
        )
        translation = tok.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        self.assertEqual(translation, expected_translation)

    @slow
    def test_translation_en_to_ro(self):
        model = TFT5ForConditionalGeneration.from_pretrained("t5-base")
        tok = T5Tokenizer.from_pretrained("t5-base")

        task_specific_config = getattr(model.config, "task_specific_params", {})
        translation_config = task_specific_config.get("translation_en_to_ro", {})
        model.config.update(translation_config)

        original_input = "Taco Bell said it plans to add 2,000 locations in the US by 2022."
        expected_translation = "Taco Bell a declarat că intenţionează să adauge 2 000 de locaţii în SUA până în 2022."

        input_ids = tok.encode(model.config.prefix + original_input, return_tensors="tf")

        output = model.generate(
            input_ids=input_ids,
            num_beams=4,
            length_penalty=2.0,
            max_length=50,
            no_repeat_ngram_size=3,
            do_sample=False,
            early_stopping=True,
        )
        translation = tok.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        self.assertEqual(translation, expected_translation)
