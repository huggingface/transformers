import unittest
from typing import Iterable, List, Optional

from transformers import pipeline
from transformers.pipelines import SUPPORTED_TASKS, Conversation, DefaultArgumentHandler, Pipeline
from transformers.testing_utils import require_tf, require_tokenizers, require_torch, slow, torch_device


DEFAULT_DEVICE_NUM = -1 if torch_device == "cpu" else 0
VALID_INPUTS = ["A simple string", ["list of strings"]]

NER_FINETUNED_MODELS = ["sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english"]
TF_NER_FINETUNED_MODELS = ["Narsil/small"]

# xlnet-base-cased disabled for now, since it crashes TF2
FEATURE_EXTRACT_FINETUNED_MODELS = ["sshleifer/tiny-distilbert-base-cased"]
TEXT_CLASSIF_FINETUNED_MODELS = ["sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english"]
TEXT_GENERATION_FINETUNED_MODELS = ["sshleifer/tiny-ctrl"]

FILL_MASK_FINETUNED_MODELS = ["sshleifer/tiny-distilroberta-base"]
LARGE_FILL_MASK_FINETUNED_MODELS = ["distilroberta-base"]  # @slow

SUMMARIZATION_FINETUNED_MODELS = ["sshleifer/bart-tiny-random", "patrickvonplaten/t5-tiny-random"]
TF_SUMMARIZATION_FINETUNED_MODELS = ["patrickvonplaten/t5-tiny-random"]

TRANSLATION_FINETUNED_MODELS = [
    ("patrickvonplaten/t5-tiny-random", "translation_en_to_de"),
    ("patrickvonplaten/t5-tiny-random", "translation_en_to_ro"),
]
TF_TRANSLATION_FINETUNED_MODELS = [("patrickvonplaten/t5-tiny-random", "translation_en_to_fr")]

TEXT2TEXT_FINETUNED_MODELS = ["patrickvonplaten/t5-tiny-random"]
TF_TEXT2TEXT_FINETUNED_MODELS = ["patrickvonplaten/t5-tiny-random"]

DIALOGUE_FINETUNED_MODELS = ["microsoft/DialoGPT-medium"]  # @slow

expected_fill_mask_result = [
    [
        {"sequence": "<s>My name is John</s>", "score": 0.00782308354973793, "token": 610, "token_str": "ĠJohn"},
        {"sequence": "<s>My name is Chris</s>", "score": 0.007475061342120171, "token": 1573, "token_str": "ĠChris"},
    ],
    [
        {"sequence": "<s>The largest city in France is Paris</s>", "score": 0.3185044229030609, "token": 2201},
        {"sequence": "<s>The largest city in France is Lyon</s>", "score": 0.21112334728240967, "token": 12790},
    ],
]

expected_fill_mask_target_result = [
    [
        {
            "sequence": "<s>My name is Patrick</s>",
            "score": 0.004992353264242411,
            "token": 3499,
            "token_str": "ĠPatrick",
        },
        {
            "sequence": "<s>My name is Clara</s>",
            "score": 0.00019297805556561798,
            "token": 13606,
            "token_str": "ĠClara",
        },
    ]
]

SUMMARIZATION_KWARGS = dict(num_beams=2, min_length=2, max_length=5)


class DefaultArgumentHandlerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.handler = DefaultArgumentHandler()

    def test_kwargs_x(self):
        mono_data = {"X": "This is a sample input"}
        mono_args = self.handler(**mono_data)

        self.assertTrue(isinstance(mono_args, list))
        self.assertEqual(len(mono_args), 1)

        multi_data = {"x": ["This is a sample input", "This is a second sample input"]}
        multi_args = self.handler(**multi_data)

        self.assertTrue(isinstance(multi_args, list))
        self.assertEqual(len(multi_args), 2)

    def test_kwargs_data(self):
        mono_data = {"data": "This is a sample input"}
        mono_args = self.handler(**mono_data)

        self.assertTrue(isinstance(mono_args, list))
        self.assertEqual(len(mono_args), 1)

        multi_data = {"data": ["This is a sample input", "This is a second sample input"]}
        multi_args = self.handler(**multi_data)

        self.assertTrue(isinstance(multi_args, list))
        self.assertEqual(len(multi_args), 2)

    def test_multi_kwargs(self):
        mono_data = {"data": "This is a sample input", "X": "This is a sample input 2"}
        mono_args = self.handler(**mono_data)

        self.assertTrue(isinstance(mono_args, list))
        self.assertEqual(len(mono_args), 2)

        multi_data = {
            "data": ["This is a sample input", "This is a second sample input"],
            "test": ["This is a sample input 2", "This is a second sample input 2"],
        }
        multi_args = self.handler(**multi_data)

        self.assertTrue(isinstance(multi_args, list))
        self.assertEqual(len(multi_args), 4)

    def test_args(self):
        mono_data = "This is a sample input"
        mono_args = self.handler(mono_data)

        self.assertTrue(isinstance(mono_args, list))
        self.assertEqual(len(mono_args), 1)

        mono_data = ["This is a sample input"]
        mono_args = self.handler(mono_data)

        self.assertTrue(isinstance(mono_args, list))
        self.assertEqual(len(mono_args), 1)

        multi_data = ["This is a sample input", "This is a second sample input"]
        multi_args = self.handler(multi_data)

        self.assertTrue(isinstance(multi_args, list))
        self.assertEqual(len(multi_args), 2)

        multi_data = ["This is a sample input", "This is a second sample input"]
        multi_args = self.handler(*multi_data)

        self.assertTrue(isinstance(multi_args, list))
        self.assertEqual(len(multi_args), 2)


class MonoColumnInputTestCase(unittest.TestCase):
    def _test_mono_column_pipeline(
        self,
        nlp: Pipeline,
        valid_inputs: List,
        output_keys: Iterable[str],
        invalid_inputs: List = [None],
        expected_multi_result: Optional[List] = None,
        expected_check_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        self.assertIsNotNone(nlp)

        mono_result = nlp(valid_inputs[0], **kwargs)
        self.assertIsInstance(mono_result, list)
        self.assertIsInstance(mono_result[0], (dict, list))

        if isinstance(mono_result[0], list):
            mono_result = mono_result[0]

        for key in output_keys:
            self.assertIn(key, mono_result[0])

        multi_result = [nlp(input, **kwargs) for input in valid_inputs]
        self.assertIsInstance(multi_result, list)
        self.assertIsInstance(multi_result[0], (dict, list))

        if expected_multi_result is not None:
            for result, expect in zip(multi_result, expected_multi_result):
                for key in expected_check_keys or []:
                    self.assertEqual(
                        set([o[key] for o in result]),
                        set([o[key] for o in expect]),
                    )

        if isinstance(multi_result[0], list):
            multi_result = multi_result[0]

        for result in multi_result:
            for key in output_keys:
                self.assertIn(key, result)

        self.assertRaises(Exception, nlp, invalid_inputs)

    @require_torch
    def test_torch_sentiment_analysis(self):
        mandatory_keys = {"label", "score"}
        for model_name in TEXT_CLASSIF_FINETUNED_MODELS:
            nlp = pipeline(task="sentiment-analysis", model=model_name, tokenizer=model_name)
            self._test_mono_column_pipeline(nlp, VALID_INPUTS, mandatory_keys)

    @require_tf
    def test_tf_sentiment_analysis(self):
        mandatory_keys = {"label", "score"}
        for model_name in TEXT_CLASSIF_FINETUNED_MODELS:
            nlp = pipeline(task="sentiment-analysis", model=model_name, tokenizer=model_name, framework="tf")
            self._test_mono_column_pipeline(nlp, VALID_INPUTS, mandatory_keys)

    @require_torch
    def test_torch_feature_extraction(self):
        for model_name in FEATURE_EXTRACT_FINETUNED_MODELS:
            nlp = pipeline(task="feature-extraction", model=model_name, tokenizer=model_name)
            self._test_mono_column_pipeline(nlp, VALID_INPUTS, {})

    @require_tf
    def test_tf_feature_extraction(self):
        for model_name in FEATURE_EXTRACT_FINETUNED_MODELS:
            nlp = pipeline(task="feature-extraction", model=model_name, tokenizer=model_name, framework="tf")
            self._test_mono_column_pipeline(nlp, VALID_INPUTS, {})

    @require_torch
    def test_torch_fill_mask(self):
        mandatory_keys = {"sequence", "score", "token"}
        valid_inputs = [
            "My name is <mask>",
            "The largest city in France is <mask>",
        ]
        invalid_inputs = [
            "This is <mask> <mask>"  # More than 1 mask_token in the input is not supported
            "This is"  # No mask_token is not supported
        ]
        for model_name in FILL_MASK_FINETUNED_MODELS:
            nlp = pipeline(
                task="fill-mask",
                model=model_name,
                tokenizer=model_name,
                framework="pt",
                topk=2,
            )
            self._test_mono_column_pipeline(
                nlp, valid_inputs, mandatory_keys, invalid_inputs, expected_check_keys=["sequence"]
            )

    @require_tf
    def test_tf_fill_mask(self):
        mandatory_keys = {"sequence", "score", "token"}
        valid_inputs = [
            "My name is <mask>",
            "The largest city in France is <mask>",
        ]
        invalid_inputs = [
            "This is <mask> <mask>"  # More than 1 mask_token in the input is not supported
            "This is"  # No mask_token is not supported
        ]
        for model_name in FILL_MASK_FINETUNED_MODELS:
            nlp = pipeline(
                task="fill-mask",
                model=model_name,
                tokenizer=model_name,
                framework="tf",
                topk=2,
            )
            self._test_mono_column_pipeline(
                nlp, valid_inputs, mandatory_keys, invalid_inputs, expected_check_keys=["sequence"]
            )

    @require_torch
    def test_torch_fill_mask_with_targets(self):
        valid_inputs = ["My name is <mask>"]
        valid_targets = [[" Teven", " Patrick", " Clara"], [" Sam"]]
        invalid_targets = [[], [""], ""]
        for model_name in FILL_MASK_FINETUNED_MODELS:
            nlp = pipeline(task="fill-mask", model=model_name, tokenizer=model_name, framework="pt")
            for targets in valid_targets:
                outputs = nlp(valid_inputs, targets=targets)
                self.assertIsInstance(outputs, list)
                self.assertEqual(len(outputs), len(targets))
            for targets in invalid_targets:
                self.assertRaises(ValueError, nlp, valid_inputs, targets=targets)

    @require_tf
    def test_tf_fill_mask_with_targets(self):
        valid_inputs = ["My name is <mask>"]
        valid_targets = [[" Teven", " Patrick", " Clara"], [" Sam"]]
        invalid_targets = [[], [""], ""]
        for model_name in FILL_MASK_FINETUNED_MODELS:
            nlp = pipeline(task="fill-mask", model=model_name, tokenizer=model_name, framework="tf")
            for targets in valid_targets:
                outputs = nlp(valid_inputs, targets=targets)
                self.assertIsInstance(outputs, list)
                self.assertEqual(len(outputs), len(targets))
            for targets in invalid_targets:
                self.assertRaises(ValueError, nlp, valid_inputs, targets=targets)

    @require_torch
    @slow
    def test_torch_fill_mask_results(self):
        mandatory_keys = {"sequence", "score", "token"}
        valid_inputs = [
            "My name is <mask>",
            "The largest city in France is <mask>",
        ]
        valid_targets = [" Patrick", " Clara"]
        for model_name in LARGE_FILL_MASK_FINETUNED_MODELS:
            nlp = pipeline(
                task="fill-mask",
                model=model_name,
                tokenizer=model_name,
                framework="pt",
                topk=2,
            )
            self._test_mono_column_pipeline(
                nlp,
                valid_inputs,
                mandatory_keys,
                expected_multi_result=expected_fill_mask_result,
                expected_check_keys=["sequence"],
            )
            self._test_mono_column_pipeline(
                nlp,
                valid_inputs[:1],
                mandatory_keys,
                expected_multi_result=expected_fill_mask_target_result,
                expected_check_keys=["sequence"],
                targets=valid_targets,
            )

    @require_tf
    @slow
    def test_tf_fill_mask_results(self):
        mandatory_keys = {"sequence", "score", "token"}
        valid_inputs = [
            "My name is <mask>",
            "The largest city in France is <mask>",
        ]
        valid_targets = [" Patrick", " Clara"]
        for model_name in LARGE_FILL_MASK_FINETUNED_MODELS:
            nlp = pipeline(task="fill-mask", model=model_name, tokenizer=model_name, framework="tf", topk=2)
            self._test_mono_column_pipeline(
                nlp,
                valid_inputs,
                mandatory_keys,
                expected_multi_result=expected_fill_mask_result,
                expected_check_keys=["sequence"],
            )
            self._test_mono_column_pipeline(
                nlp,
                valid_inputs[:1],
                mandatory_keys,
                expected_multi_result=expected_fill_mask_target_result,
                expected_check_keys=["sequence"],
                targets=valid_targets,
            )

    @require_torch
    @require_tokenizers
    def test_torch_summarization(self):
        invalid_inputs = [4, "<mask>"]
        mandatory_keys = ["summary_text"]
        for model in SUMMARIZATION_FINETUNED_MODELS:
            nlp = pipeline(task="summarization", model=model, tokenizer=model)
            self._test_mono_column_pipeline(
                nlp, VALID_INPUTS, mandatory_keys, invalid_inputs=invalid_inputs, **SUMMARIZATION_KWARGS
            )

    @require_torch
    @slow
    def test_integration_torch_summarization(self):
        nlp = pipeline(task="summarization", device=DEFAULT_DEVICE_NUM)
        cnn_article = ' (CNN)The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based. The Palestinians signed the ICC\'s founding Rome Statute in January, when they also accepted its jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including East Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination into the situation in Palestinian territories, paving the way for possible war crimes investigations against Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and the United States, neither of which is an ICC member, opposed the Palestinians\' efforts to join the body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday\'s ceremony, said it was a move toward greater justice. "As Palestine formally becomes a State Party to the Rome Statute today, the world is also a step closer to ending a long era of impunity and injustice," he said, according to an ICC news release. "Indeed, today brings us closer to our shared goals of justice and peace." Judge Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the Palestinians. "As the Rome Statute today enters into force for the State of Palestine, Palestine acquires all the rights as well as responsibilities that come with being a State Party to the Statute. These are substantive commitments, which cannot be taken lightly," she said. Rights group Human Rights Watch welcomed the development. "Governments seeking to penalize Palestine for joining the ICC should immediately end their pressure, and countries that support universal acceptance of the court\'s treaty should speak out to welcome its membership," said Balkees Jarrah, international justice counsel for the group. "What\'s objectionable is the attempts to undermine international justice, not Palestine\'s decision to join a treaty to which over 100 countries around the world are members." In January, when the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described it as an outrage, saying the court was overstepping its boundaries. The United States also said it "strongly" disagreed with the court\'s decision. "As we have said repeatedly, we do not believe that Palestine is a state and therefore we do not believe that it is eligible to join the ICC," the State Department said in a statement. It urged the warring sides to resolve their differences through direct negotiations. "We will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace," it said. But the ICC begs to differ with the definition of a state for its purposes and refers to the territories as "Palestine." While a preliminary examination is not a formal investigation, it allows the court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou Bensouda said her office would "conduct its analysis in full independence and impartiality." The war between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The inquiry will include alleged war crimes committed since June. The International Criminal Court was set up in 2002 to prosecute genocide, crimes against humanity and war crimes. CNN\'s Vasco Cotovio, Kareem Khadder and Faith Karimi contributed to this report.'
        expected_cnn_summary = " The Palestinian Authority becomes the 123rd member of the International Criminal Court . The move gives the court jurisdiction over alleged crimes in Palestinian territories . Israel and the United States opposed the Palestinians' efforts to join the court . Rights group Human Rights Watch welcomes the move, says governments seeking to penalize Palestine should end pressure ."
        result = nlp(cnn_article)
        self.assertEqual(result[0]["summary_text"], expected_cnn_summary)

    @require_tf
    @slow
    def test_tf_summarization(self):
        invalid_inputs = [4, "<mask>"]
        mandatory_keys = ["summary_text"]
        for model_name in TF_SUMMARIZATION_FINETUNED_MODELS:
            nlp = pipeline(
                task="summarization",
                model=model_name,
                tokenizer=model_name,
                framework="tf",
            )
            self._test_mono_column_pipeline(
                nlp, VALID_INPUTS, mandatory_keys, invalid_inputs=invalid_inputs, **SUMMARIZATION_KWARGS
            )

    @require_torch
    @require_tokenizers
    def test_torch_translation(self):
        invalid_inputs = [4, "<mask>"]
        mandatory_keys = ["translation_text"]
        for model_name, task in TRANSLATION_FINETUNED_MODELS:
            nlp = pipeline(task=task, model=model_name, tokenizer=model_name)
            self._test_mono_column_pipeline(
                nlp,
                VALID_INPUTS,
                mandatory_keys,
                invalid_inputs,
            )

    @require_tf
    @slow
    def test_tf_translation(self):
        invalid_inputs = [4, "<mask>"]
        mandatory_keys = ["translation_text"]
        for model, task in TF_TRANSLATION_FINETUNED_MODELS:
            nlp = pipeline(task=task, model=model, tokenizer=model, framework="tf")
            self._test_mono_column_pipeline(nlp, VALID_INPUTS, mandatory_keys, invalid_inputs=invalid_inputs)

    @require_torch
    @require_tokenizers
    def test_torch_text2text(self):
        invalid_inputs = [4, "<mask>"]
        mandatory_keys = ["generated_text"]
        for model_name in TEXT2TEXT_FINETUNED_MODELS:
            nlp = pipeline(task="text2text-generation", model=model_name, tokenizer=model_name)
            self._test_mono_column_pipeline(
                nlp,
                VALID_INPUTS,
                mandatory_keys,
                invalid_inputs,
            )

    @require_tf
    @slow
    def test_tf_text2text(self):
        invalid_inputs = [4, "<mask>"]
        mandatory_keys = ["generated_text"]
        for model in TEXT2TEXT_FINETUNED_MODELS:
            nlp = pipeline(task="text2text-generation", model=model, tokenizer=model, framework="tf")
            self._test_mono_column_pipeline(nlp, VALID_INPUTS, mandatory_keys, invalid_inputs=invalid_inputs)

    @require_torch
    def test_torch_text_generation(self):
        for model_name in TEXT_GENERATION_FINETUNED_MODELS:
            nlp = pipeline(task="text-generation", model=model_name, tokenizer=model_name, framework="pt")
            self._test_mono_column_pipeline(nlp, VALID_INPUTS, {})
        self._test_mono_column_pipeline(nlp, VALID_INPUTS, {}, prefix="This is ")

    @require_tf
    def test_tf_text_generation(self):
        for model_name in TEXT_GENERATION_FINETUNED_MODELS:
            nlp = pipeline(task="text-generation", model=model_name, tokenizer=model_name, framework="tf")
            self._test_mono_column_pipeline(nlp, VALID_INPUTS, {})
        self._test_mono_column_pipeline(nlp, VALID_INPUTS, {}, prefix="This is ")

    @require_torch
    @slow
    def test_integration_torch_conversation(self):
        # When
        nlp = pipeline(task="conversational", device=DEFAULT_DEVICE_NUM)
        conversation_1 = Conversation("Going to the movies tonight - any suggestions?")
        conversation_2 = Conversation("What's the last book you have read?")
        # Then
        self.assertEqual(len(conversation_1.past_user_inputs), 0)
        self.assertEqual(len(conversation_2.past_user_inputs), 0)
        # When
        result = nlp([conversation_1, conversation_2], do_sample=False, max_length=1000)
        # Then
        self.assertEqual(result, [conversation_1, conversation_2])
        self.assertEqual(len(result[0].past_user_inputs), 1)
        self.assertEqual(len(result[1].past_user_inputs), 1)
        self.assertEqual(len(result[0].generated_responses), 1)
        self.assertEqual(len(result[1].generated_responses), 1)
        self.assertEqual(result[0].past_user_inputs[0], "Going to the movies tonight - any suggestions?")
        self.assertEqual(result[0].generated_responses[0], "The Big Lebowski")
        self.assertEqual(result[1].past_user_inputs[0], "What's the last book you have read?")
        self.assertEqual(result[1].generated_responses[0], "The Last Question")
        # When
        conversation_2.add_user_input("Why do you recommend it?")
        result = nlp(conversation_2, do_sample=False, max_length=1000)
        # Then
        self.assertEqual(result, conversation_2)
        self.assertEqual(len(result.past_user_inputs), 2)
        self.assertEqual(len(result.generated_responses), 2)
        self.assertEqual(result.past_user_inputs[1], "Why do you recommend it?")
        self.assertEqual(result.generated_responses[1], "It's a good book.")

    @require_torch
    @slow
    def test_integration_torch_conversation_truncated_history(self):
        # When
        nlp = pipeline(task="conversational", min_length_for_response=24, device=DEFAULT_DEVICE_NUM)
        conversation_1 = Conversation("Going to the movies tonight - any suggestions?")
        # Then
        self.assertEqual(len(conversation_1.past_user_inputs), 0)
        # When
        result = nlp(conversation_1, do_sample=False, max_length=36)
        # Then
        self.assertEqual(result, conversation_1)
        self.assertEqual(len(result.past_user_inputs), 1)
        self.assertEqual(len(result.generated_responses), 1)
        self.assertEqual(result.past_user_inputs[0], "Going to the movies tonight - any suggestions?")
        self.assertEqual(result.generated_responses[0], "The Big Lebowski")
        # When
        conversation_1.add_user_input("Is it an action movie?")
        result = nlp(conversation_1, do_sample=False, max_length=36)
        # Then
        self.assertEqual(result, conversation_1)
        self.assertEqual(len(result.past_user_inputs), 2)
        self.assertEqual(len(result.generated_responses), 2)
        self.assertEqual(result.past_user_inputs[1], "Is it an action movie?")
        self.assertEqual(result.generated_responses[1], "It's a comedy.")


QA_FINETUNED_MODELS = ["sshleifer/tiny-distilbert-base-cased-distilled-squad"]


class ZeroShotClassificationPipelineTests(unittest.TestCase):
    def _test_scores_sum_to_one(self, result):
        sum = 0.0
        for score in result["scores"]:
            sum += score
        self.assertAlmostEqual(sum, 1.0)

    def _test_zero_shot_pipeline(self, nlp):
        output_keys = {"sequence", "labels", "scores"}
        valid_mono_inputs = [
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": "politics"},
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": ["politics"]},
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": "politics, public health"},
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": ["politics", "public health"]},
            {"sequences": ["Who are you voting for in 2020?"], "candidate_labels": "politics"},
            {
                "sequences": "Who are you voting for in 2020?",
                "candidate_labels": "politics",
                "hypothesis_template": "This text is about {}",
            },
        ]
        valid_multi_input = {
            "sequences": ["Who are you voting for in 2020?", "What is the capital of Spain?"],
            "candidate_labels": "politics",
        }
        invalid_inputs = [
            {"sequences": None, "candidate_labels": "politics"},
            {"sequences": "", "candidate_labels": "politics"},
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": None},
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": ""},
            {
                "sequences": "Who are you voting for in 2020?",
                "candidate_labels": "politics",
                "hypothesis_template": None,
            },
            {
                "sequences": "Who are you voting for in 2020?",
                "candidate_labels": "politics",
                "hypothesis_template": "",
            },
            {
                "sequences": "Who are you voting for in 2020?",
                "candidate_labels": "politics",
                "hypothesis_template": "Template without formatting syntax.",
            },
        ]
        self.assertIsNotNone(nlp)

        for mono_input in valid_mono_inputs:
            mono_result = nlp(**mono_input)
            self.assertIsInstance(mono_result, dict)
            if len(mono_result["labels"]) > 1:
                self._test_scores_sum_to_one(mono_result)

            for key in output_keys:
                self.assertIn(key, mono_result)

        multi_result = nlp(**valid_multi_input)
        self.assertIsInstance(multi_result, list)
        self.assertIsInstance(multi_result[0], dict)
        self.assertEqual(len(multi_result), len(valid_multi_input["sequences"]))

        for result in multi_result:
            for key in output_keys:
                self.assertIn(key, result)

            if len(result["labels"]) > 1:
                self._test_scores_sum_to_one(result)

        for bad_input in invalid_inputs:
            self.assertRaises(Exception, nlp, **bad_input)

    def _test_zero_shot_pipeline_outputs(self, nlp):
        inputs = [
            {
                "sequences": "Who are you voting for in 2020?",
                "candidate_labels": ["politics", "public health", "science"],
            },
            {
                "sequences": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.",
                "candidate_labels": ["machine learning", "statistics", "translation", "vision"],
                "multi_class": True,
            },
        ]

        expected_outputs = [
            {
                "sequence": "Who are you voting for in 2020?",
                "labels": ["politics", "public health", "science"],
                "scores": [0.975, 0.015, 0.008],
            },
            {
                "sequence": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.",
                "labels": ["translation", "machine learning", "vision", "statistics"],
                "scores": [0.817, 0.712, 0.018, 0.017],
            },
        ]

        for input, expected_output in zip(inputs, expected_outputs):
            output = nlp(**input)
            for key in output:
                if key == "scores":
                    for output_score, expected_score in zip(output[key], expected_output[key]):
                        self.assertAlmostEqual(output_score, expected_score, places=2)
                else:
                    self.assertEqual(output[key], expected_output[key])

    @require_torch
    def test_torch_zero_shot_classification(self):
        for model_name in TEXT_CLASSIF_FINETUNED_MODELS:
            nlp = pipeline(task="zero-shot-classification", model=model_name, tokenizer=model_name)
            self._test_zero_shot_pipeline(nlp)

    @require_tf
    def test_tf_zero_shot_classification(self):
        for model_name in TEXT_CLASSIF_FINETUNED_MODELS:
            nlp = pipeline(task="zero-shot-classification", model=model_name, tokenizer=model_name, framework="tf")
            self._test_zero_shot_pipeline(nlp)

    @require_torch
    @slow
    def test_torch_zero_shot_outputs(self):
        nlp = pipeline(task="zero-shot-classification", model="roberta-large-mnli")
        self._test_zero_shot_pipeline_outputs(nlp)

    @require_tf
    @slow
    def test_tf_zero_shot_outputs(self):
        nlp = pipeline(task="zero-shot-classification", model="roberta-large-mnli", framework="tf")
        self._test_zero_shot_pipeline_outputs(nlp)


class DialoguePipelineTests(unittest.TestCase):
    def _test_conversation_pipeline(self, nlp):
        valid_inputs = [Conversation("Hi there!"), [Conversation("Hi there!"), Conversation("How are you?")]]
        invalid_inputs = ["Hi there!", Conversation()]
        self.assertIsNotNone(nlp)

        mono_result = nlp(valid_inputs[0])
        self.assertIsInstance(mono_result, Conversation)

        multi_result = nlp(valid_inputs[1])
        self.assertIsInstance(multi_result, list)
        self.assertIsInstance(multi_result[0], Conversation)
        # Inactive conversations passed to the pipeline raise a ValueError
        self.assertRaises(ValueError, nlp, valid_inputs[1])

        for bad_input in invalid_inputs:
            self.assertRaises(Exception, nlp, bad_input)
        self.assertRaises(Exception, nlp, invalid_inputs)

    @require_torch
    @slow
    def test_torch_conversation(self):
        for model_name in DIALOGUE_FINETUNED_MODELS:
            nlp = pipeline(task="conversational", model=model_name, tokenizer=model_name)
            self._test_conversation_pipeline(nlp)

    @require_tf
    @slow
    def test_tf_conversation(self):
        for model_name in DIALOGUE_FINETUNED_MODELS:
            nlp = pipeline(task="conversational", model=model_name, tokenizer=model_name, framework="tf")
            self._test_conversation_pipeline(nlp)


class QAPipelineTests(unittest.TestCase):
    def _test_qa_pipeline(self, nlp):
        output_keys = {"score", "answer", "start", "end"}
        valid_inputs = [
            {"question": "Where was HuggingFace founded ?", "context": "HuggingFace was founded in Paris."},
            {
                "question": "In what field is HuggingFace working ?",
                "context": "HuggingFace is a startup based in New-York founded in Paris which is trying to solve NLP.",
            },
        ]
        invalid_inputs = [
            {"question": "", "context": "This is a test to try empty question edge case"},
            {"question": None, "context": "This is a test to try empty question edge case"},
            {"question": "What is does with empty context ?", "context": ""},
            {"question": "What is does with empty context ?", "context": None},
        ]
        self.assertIsNotNone(nlp)

        mono_result = nlp(valid_inputs[0])
        self.assertIsInstance(mono_result, dict)

        for key in output_keys:
            self.assertIn(key, mono_result)

        multi_result = nlp(valid_inputs)
        self.assertIsInstance(multi_result, list)
        self.assertIsInstance(multi_result[0], dict)

        for result in multi_result:
            for key in output_keys:
                self.assertIn(key, result)
        for bad_input in invalid_inputs:
            self.assertRaises(Exception, nlp, bad_input)
        self.assertRaises(Exception, nlp, invalid_inputs)

    @require_torch
    def test_torch_question_answering(self):
        for model_name in QA_FINETUNED_MODELS:
            nlp = pipeline(task="question-answering", model=model_name, tokenizer=model_name)
            self._test_qa_pipeline(nlp)

    @require_tf
    def test_tf_question_answering(self):
        for model_name in QA_FINETUNED_MODELS:
            nlp = pipeline(task="question-answering", model=model_name, tokenizer=model_name, framework="tf")
            self._test_qa_pipeline(nlp)


class NerPipelineTests(unittest.TestCase):
    def _test_ner_pipeline(
        self,
        nlp: Pipeline,
        output_keys: Iterable[str],
    ):

        ungrouped_ner_inputs = [
            [
                {"entity": "B-PER", "index": 1, "score": 0.9994944930076599, "word": "Cons"},
                {"entity": "B-PER", "index": 2, "score": 0.8025449514389038, "word": "##uelo"},
                {"entity": "I-PER", "index": 3, "score": 0.9993102550506592, "word": "Ara"},
                {"entity": "I-PER", "index": 4, "score": 0.9993743896484375, "word": "##új"},
                {"entity": "I-PER", "index": 5, "score": 0.9992871880531311, "word": "##o"},
                {"entity": "I-PER", "index": 6, "score": 0.9993029236793518, "word": "No"},
                {"entity": "I-PER", "index": 7, "score": 0.9981776475906372, "word": "##guera"},
                {"entity": "B-PER", "index": 15, "score": 0.9998136162757874, "word": "Andrés"},
                {"entity": "I-PER", "index": 16, "score": 0.999740719795227, "word": "Pas"},
                {"entity": "I-PER", "index": 17, "score": 0.9997414350509644, "word": "##tran"},
                {"entity": "I-PER", "index": 18, "score": 0.9996136426925659, "word": "##a"},
                {"entity": "B-ORG", "index": 28, "score": 0.9989739060401917, "word": "Far"},
                {"entity": "I-ORG", "index": 29, "score": 0.7188422083854675, "word": "##c"},
            ],
            [
                {"entity": "I-PER", "index": 1, "score": 0.9968166351318359, "word": "En"},
                {"entity": "I-PER", "index": 2, "score": 0.9957635998725891, "word": "##zo"},
                {"entity": "I-ORG", "index": 7, "score": 0.9986497163772583, "word": "UN"},
            ],
        ]
        expected_grouped_ner_results = [
            [
                {"entity_group": "B-PER", "score": 0.9710702640669686, "word": "Consuelo Araújo Noguera"},
                {"entity_group": "B-PER", "score": 0.9997273534536362, "word": "Andrés Pastrana"},
                {"entity_group": "B-ORG", "score": 0.8589080572128296, "word": "Farc"},
            ],
            [
                {"entity_group": "I-PER", "score": 0.9962901175022125, "word": "Enzo"},
                {"entity_group": "I-ORG", "score": 0.9986497163772583, "word": "UN"},
            ],
        ]

        self.assertIsNotNone(nlp)

        mono_result = nlp(VALID_INPUTS[0])
        self.assertIsInstance(mono_result, list)
        self.assertIsInstance(mono_result[0], (dict, list))

        if isinstance(mono_result[0], list):
            mono_result = mono_result[0]

        for key in output_keys:
            self.assertIn(key, mono_result[0])

        multi_result = [nlp(input) for input in VALID_INPUTS]
        self.assertIsInstance(multi_result, list)
        self.assertIsInstance(multi_result[0], (dict, list))

        if isinstance(multi_result[0], list):
            multi_result = multi_result[0]

        for result in multi_result:
            for key in output_keys:
                self.assertIn(key, result)

        for ungrouped_input, grouped_result in zip(ungrouped_ner_inputs, expected_grouped_ner_results):
            self.assertEqual(nlp.group_entities(ungrouped_input), grouped_result)

    @require_torch
    def test_torch_ner(self):
        mandatory_keys = {"entity", "word", "score"}
        for model_name in NER_FINETUNED_MODELS:
            nlp = pipeline(task="ner", model=model_name, tokenizer=model_name)
            self._test_ner_pipeline(nlp, mandatory_keys)

    @require_torch
    def test_ner_grouped(self):
        mandatory_keys = {"entity_group", "word", "score"}
        for model_name in NER_FINETUNED_MODELS:
            nlp = pipeline(task="ner", model=model_name, tokenizer=model_name, grouped_entities=True)
            self._test_ner_pipeline(nlp, mandatory_keys)

    @require_tf
    def test_tf_ner(self):
        mandatory_keys = {"entity", "word", "score"}
        for model_name in NER_FINETUNED_MODELS:
            nlp = pipeline(task="ner", model=model_name, tokenizer=model_name, framework="tf")
            self._test_ner_pipeline(nlp, mandatory_keys)

    @require_tf
    def test_tf_ner_grouped(self):
        mandatory_keys = {"entity_group", "word", "score"}
        for model_name in NER_FINETUNED_MODELS:
            nlp = pipeline(task="ner", model=model_name, tokenizer=model_name, framework="tf", grouped_entities=True)
            self._test_ner_pipeline(nlp, mandatory_keys)

    @require_tf
    def test_tf_only_ner(self):
        mandatory_keys = {"entity", "word", "score"}
        for model_name in TF_NER_FINETUNED_MODELS:
            # We don't specificy framework='tf' but it gets detected automatically
            nlp = pipeline(task="ner", model=model_name, tokenizer=model_name)
            self._test_ner_pipeline(nlp, mandatory_keys)


class PipelineCommonTests(unittest.TestCase):
    pipelines = SUPPORTED_TASKS.keys()

    @require_tf
    @slow
    def test_tf_defaults(self):
        # Test that pipelines can be correctly loaded without any argument
        for task in self.pipelines:
            with self.subTest(msg="Testing TF defaults with TF and {}".format(task)):
                pipeline(task, framework="tf")
                pipeline(task)

    @require_torch
    @slow
    def test_pt_defaults(self):
        # Test that pipelines can be correctly loaded without any argument
        for task in self.pipelines:
            with self.subTest(msg="Testing Torch defaults with PyTorch and {}".format(task)):
                pipeline(task, framework="pt")
                pipeline(task)
