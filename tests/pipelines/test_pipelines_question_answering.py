# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    LxmertConfig,
    QuestionAnsweringPipeline,
)
from transformers.data.processors.squad import SquadExample
from transformers.pipelines import QuestionAnsweringArgumentHandler, pipeline
from transformers.testing_utils import (
    is_pipeline_test,
    nested_simplify,
    require_tf,
    require_torch,
    require_torch_or_tf,
    slow,
)

from .test_pipelines_common import ANY


# These 2 model types require different inputs than those of the usual text models.
_TO_SKIP = {"LayoutLMv2Config", "LayoutLMv3Config"}


@is_pipeline_test
class QAPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_QUESTION_ANSWERING_MAPPING
    tf_model_mapping = TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING

    if model_mapping is not None:
        model_mapping = {config: model for config, model in model_mapping.items() if config.__name__ not in _TO_SKIP}
    if tf_model_mapping is not None:
        tf_model_mapping = {
            config: model for config, model in tf_model_mapping.items() if config.__name__ not in _TO_SKIP
        }

    def get_test_pipeline(self, model, tokenizer, processor):
        if isinstance(model.config, LxmertConfig):
            # This is an bimodal model, we need to find a more consistent way
            # to switch on those models.
            return None, None
        question_answerer = QuestionAnsweringPipeline(model, tokenizer)

        examples = [
            {"question": "Where was HuggingFace founded ?", "context": "HuggingFace was founded in Paris."},
            {"question": "In what field is HuggingFace ?", "context": "HuggingFace is  an AI startup."},
        ]
        return question_answerer, examples

    def run_pipeline_test(self, question_answerer, _):
        outputs = question_answerer(
            question="Where was HuggingFace founded ?", context="HuggingFace was founded in Paris."
        )
        self.assertEqual(outputs, {"answer": ANY(str), "start": ANY(int), "end": ANY(int), "score": ANY(float)})
        outputs = question_answerer(
            question="Where was HuggingFace founded ?",
            context="HuggingFace was founded in Paris.",
            handle_impossible_answer=True,
        )
        self.assertEqual(outputs, {"answer": ANY(str), "start": ANY(int), "end": ANY(int), "score": ANY(float)})

        outputs = question_answerer(
            question=["In what field is HuggingFace working ?", "In what field is HuggingFace working ?"],
            context="HuggingFace was founded in Paris.",
        )
        self.assertEqual(
            outputs,
            [
                {"answer": ANY(str), "start": ANY(int), "end": ANY(int), "score": ANY(float)},
                {"answer": ANY(str), "start": ANY(int), "end": ANY(int), "score": ANY(float)},
            ],
        )

        outputs = question_answerer(
            question=["What field is HuggingFace working ?", "In what field is HuggingFace ?"],
            context=[
                "HuggingFace is a startup based in New-York",
                "HuggingFace is a startup founded in Paris",
            ],
        )
        self.assertEqual(
            outputs,
            [
                {"answer": ANY(str), "start": ANY(int), "end": ANY(int), "score": ANY(float)},
                {"answer": ANY(str), "start": ANY(int), "end": ANY(int), "score": ANY(float)},
            ],
        )

        with self.assertRaises(ValueError):
            question_answerer(question="", context="HuggingFace was founded in Paris.")
        with self.assertRaises(ValueError):
            question_answerer(question=None, context="HuggingFace was founded in Paris.")
        with self.assertRaises(ValueError):
            question_answerer(question="In what field is HuggingFace working ?", context="")
        with self.assertRaises(ValueError):
            question_answerer(question="In what field is HuggingFace working ?", context=None)

        outputs = question_answerer(
            question="Where was HuggingFace founded ?", context="HuggingFace was founded in Paris.", top_k=20
        )
        self.assertEqual(
            outputs, [{"answer": ANY(str), "start": ANY(int), "end": ANY(int), "score": ANY(float)} for i in range(20)]
        )

        # Very long context require multiple features
        outputs = question_answerer(
            question="Where was HuggingFace founded ?", context="HuggingFace was founded in Paris." * 20
        )
        self.assertEqual(outputs, {"answer": ANY(str), "start": ANY(int), "end": ANY(int), "score": ANY(float)})

        # Using batch is OK
        if question_answerer.tokenizer.pad_token_id is None:
            question_answerer.tokenizer.pad_token_id = question_answerer.model.config.eos_token_id
        new_outputs = question_answerer(
            question="Where was HuggingFace founded ?", context="HuggingFace was founded in Paris." * 20, batch_size=2
        )
        self.assertEqual(new_outputs, {"answer": ANY(str), "start": ANY(int), "end": ANY(int), "score": ANY(float)})
        self.assertEqual(nested_simplify(outputs), nested_simplify(new_outputs))

    @require_torch
    def test_small_model_pt(self):
        question_answerer = pipeline(
            "question-answering", model="sshleifer/tiny-distilbert-base-cased-distilled-squad"
        )

        outputs = question_answerer(
            question="Where was HuggingFace founded ?", context="HuggingFace was founded in Paris."
        )

        self.assertEqual(nested_simplify(outputs), {"score": 0.01, "start": 0, "end": 11, "answer": "HuggingFace"})

    @require_torch
    def test_small_model_pt_iterator(self):
        # https://github.com/huggingface/transformers/issues/18510
        pipe = pipeline(model="sshleifer/tiny-distilbert-base-cased-distilled-squad", batch_size=16, framework="pt")

        def data():
            for i in range(10):
                yield {"question": "Where was HuggingFace founded ?", "context": "HuggingFace was founded in Paris."}

        for outputs in pipe(data()):
            self.assertEqual(nested_simplify(outputs), {"score": 0.01, "start": 0, "end": 11, "answer": "HuggingFace"})

    @require_torch
    def test_small_model_pt_softmax_trick(self):
        question_answerer = pipeline(
            "question-answering", model="sshleifer/tiny-distilbert-base-cased-distilled-squad"
        )

        real_postprocess = question_answerer.postprocess

        # Tweak start and stop to make sure we encounter the softmax logits
        # bug.
        def ensure_large_logits_postprocess(
            model_outputs,
            top_k=1,
            handle_impossible_answer=False,
            max_answer_len=15,
        ):
            for output in model_outputs:
                output["start"] = output["start"] * 1e6
                output["end"] = output["end"] * 1e6
            return real_postprocess(
                model_outputs,
                top_k=top_k,
                handle_impossible_answer=handle_impossible_answer,
                max_answer_len=max_answer_len,
            )

        question_answerer.postprocess = ensure_large_logits_postprocess

        outputs = question_answerer(
            question="Where was HuggingFace founded ?", context="HuggingFace was founded in Paris."
        )

        self.assertEqual(nested_simplify(outputs), {"score": 0.028, "start": 0, "end": 11, "answer": "HuggingFace"})

    @slow
    @require_torch
    def test_small_model_japanese(self):
        question_answerer = pipeline(
            "question-answering",
            model="KoichiYasuoka/deberta-base-japanese-aozora-ud-head",
        )
        output = question_answerer(question="国語", context="全学年にわたって小学校の国語の教科書に挿し絵が用いられている")  # fmt: skip

        # Wrong answer, the whole text is identified as one "word" since the tokenizer does not include
        # a pretokenizer
        self.assertEqual(nested_simplify(output),{"score": 1.0, "start": 0, "end": 30, "answer": "全学年にわたって小学校の国語の教科書に挿し絵が用いられている"})  # fmt: skip

        # Disable word alignment
        output = question_answerer(question="国語", context="全学年にわたって小学校の国語の教科書に挿し絵が用いられている", align_to_words=False)  # fmt: skip
        self.assertEqual(
            nested_simplify(output),
            {"score": 1.0, "start": 15, "end": 18, "answer": "教科書"},
        )

    @slow
    @require_torch
    def test_small_model_long_context_cls_slow(self):
        question_answerer = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            handle_impossible_answer=True,
            max_seq_length=512,
        )
        outputs = question_answerer(
            question="What country is Paris the capital of?",
            context="""London is the capital and largest city of England and the United Kingdom. It stands on the River Thames in south-east England at the head of a 50-mile (80 km) estuary down to the North Sea, and has been a major settlement for two millennia. The City of London, its ancient core and financial centre, was founded by the Romans as Londinium and retains boundaries close to its medieval ones. Since the 19th century, \"London\" has also referred to the metropolis around this core, historically split between the counties of Middlesex, Essex, Surrey, Kent, and Hertfordshire, which largely comprises Greater London, governed by the Greater London Authority. The City of Westminster, to the west of the City of London, has for centuries held the national government and parliament. As one of the world's global cities, London exerts strong influence on its arts, commerce, education, entertainment, fashion, finance, health care, media, tourism, and communications, and has sometimes been called the capital of the world. Its GDP (€801.66 billion in 2017) makes it the biggest urban economy in Europe, and it is one of the major financial centres in the world. In 2019 it had the second-highest number of ultra high-net-worth individuals in Europe after Paris and the second-highest number of billionaires in Europe after Moscow. As of 2021, London has the most millionaires of any city. With Europe's largest concentration of higher education institutions, it includes Imperial College London in natural and applied sciences, the London School of Economics in social sciences, and the comprehensive University College London. The city is home to the most 5-star hotels of any city in the world. In 2012, London became the first city to host three Summer Olympic Games. London is the capital and largest city of England and the United Kingdom. It stands on the River Thames in south-east England at the head of a 50-mile (80 km) estuary down to the North Sea, and has been a major settlement for two millennia. The City of London, its ancient core and financial centre, was founded by the Romans as Londinium and retains boundaries close to its medieval ones. Since the 19th century, \"London\" has also referred to the metropolis around this core, historically split between the counties of Middlesex, Essex, Surrey, Kent, and Hertfordshire, which largely comprises Greater London, governed by the Greater London Authority. The City of Westminster, to the west of the City of London, has for centuries held the national government and parliament. As one of the world's global cities, London exerts strong influence on its arts, commerce, education, entertainment, fashion, finance, health care, media, tourism, and communications, and has sometimes been called the capital of the world. Its GDP (€801.66 billion in 2017) makes it the biggest urban economy in Europe, and it is one of the major financial centres in the world. In 2019 it had the second-highest number of ultra high-net-worth individuals in Europe after Paris and the second-highest number of billionaires in Europe after Moscow. As of 2021, London has the most millionaires of any city. With Europe's largest concentration of higher education institutions, it includes Imperial College London in natural and applied sciences, the London School of Economics in social sciences, and the comprehensive University College London. The city is home to the most 5-star hotels of any city in the world. In 2012, London became the first city to host three Summer Olympic Games.""",
        )
        self.assertEqual(nested_simplify(outputs), {"score": 0.988, "start": 0, "end": 0, "answer": ""})

    @require_tf
    def test_small_model_tf(self):
        question_answerer = pipeline(
            "question-answering", model="sshleifer/tiny-distilbert-base-cased-distilled-squad", framework="tf"
        )
        outputs = question_answerer(
            question="Where was HuggingFace founded ?", context="HuggingFace was founded in Paris."
        )

        self.assertEqual(nested_simplify(outputs), {"score": 0.011, "start": 0, "end": 11, "answer": "HuggingFace"})

    @slow
    @require_torch
    def test_large_model_pt(self):
        question_answerer = pipeline(
            "question-answering",
        )
        outputs = question_answerer(
            question="Where was HuggingFace founded ?", context="HuggingFace was founded in Paris."
        )

        self.assertEqual(nested_simplify(outputs), {"score": 0.979, "start": 27, "end": 32, "answer": "Paris"})

    @slow
    @require_torch
    def test_large_model_issue(self):
        qa_pipeline = pipeline(
            "question-answering",
            model="mrm8488/bert-multi-cased-finetuned-xquadv1",
        )
        outputs = qa_pipeline(
            {
                "context": (
                    "Yes Bank founder Rana Kapoor has approached the Bombay High Court, challenging a special court's"
                    " order from August this year that had remanded him in police custody for a week in a multi-crore"
                    " loan fraud case. Kapoor, who is currently lodged in Taloja Jail, is an accused in the loan fraud"
                    " case and some related matters being probed by the CBI and Enforcement Directorate. A single"
                    " bench presided over by Justice S K Shinde on Tuesday posted the plea for further hearing on"
                    " October 14. In his plea filed through advocate Vijay Agarwal, Kapoor claimed that the special"
                    " court's order permitting the CBI's request for police custody on August 14 was illegal and in"
                    " breach of the due process of law. Therefore, his police custody and subsequent judicial custody"
                    " in the case were all illegal. Kapoor has urged the High Court to quash and set aside the special"
                    " court's order dated August 14. As per his plea, in August this year, the CBI had moved two"
                    " applications before the special court, one seeking permission to arrest Kapoor, who was already"
                    " in judicial custody at the time in another case, and the other, seeking his police custody."
                    " While the special court refused to grant permission to the CBI to arrest Kapoor, it granted the"
                    " central agency's plea for his custody. Kapoor, however, said in his plea that before filing an"
                    " application for his arrest, the CBI had not followed the process of issuing him a notice under"
                    " Section 41 of the CrPC for appearance before it. He further said that the CBI had not taken"
                    " prior sanction as mandated under section 17 A of the Prevention of Corruption Act for"
                    " prosecuting him. The special court, however, had said in its order at the time that as Kapoor"
                    " was already in judicial custody in another case and was not a free man the procedure mandated"
                    " under Section 41 of the CrPC need not have been adhered to as far as issuing a prior notice of"
                    " appearance was concerned. ADVERTISING It had also said that case records showed that the"
                    " investigating officer had taken an approval from a managing director of Yes Bank before"
                    " beginning the proceedings against Kapoor and such a permission was a valid sanction. However,"
                    " Kapoor in his plea said that the above order was bad in law and sought that it be quashed and"
                    " set aside. The law mandated that if initial action was not in consonance with legal procedures,"
                    " then all subsequent actions must be held as illegal, he said, urging the High Court to declare"
                    " the CBI remand and custody and all subsequent proceedings including the further custody as"
                    " illegal and void ab-initio. In a separate plea before the High Court, Kapoor's daughter Rakhee"
                    " Kapoor-Tandon has sought exemption from in-person appearance before a special PMLA court. Rakhee"
                    " has stated that she is a resident of the United Kingdom and is unable to travel to India owing"
                    " to restrictions imposed due to the COVID-19 pandemic. According to the CBI, in the present case,"
                    " Kapoor had obtained a gratification or pecuniary advantage of ₹ 307 crore, and thereby caused"
                    " Yes Bank a loss of ₹ 1,800 crore by extending credit facilities to Avantha Group, when it was"
                    " not eligible for the same"
                ),
                "question": "Is this person invovled in fraud?",
            }
        )
        self.assertEqual(
            nested_simplify(outputs),
            {"answer": "an accused in the loan fraud case", "end": 294, "score": 0.001, "start": 261},
        )

    @slow
    @require_torch
    def test_large_model_course(self):
        question_answerer = pipeline("question-answering")
        long_context = """
🤗 Transformers: State of the Art NLP

🤗 Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction,
question answering, summarization, translation, text generation and more in over 100 languages.
Its aim is to make cutting-edge NLP easier to use for everyone.

🤗 Transformers provides APIs to quickly download and use those pretrained models on a given text, fine-tune them on your own datasets and
then share them with the community on our model hub. At the same time, each python module defining an architecture is fully standalone and
can be modified to enable quick research experiments.

Why should I use transformers?

1. Easy-to-use state-of-the-art models:
  - High performance on NLU and NLG tasks.
  - Low barrier to entry for educators and practitioners.
  - Few user-facing abstractions with just three classes to learn.
  - A unified API for using all our pretrained models.
  - Lower compute costs, smaller carbon footprint:

2. Researchers can share trained models instead of always retraining.
  - Practitioners can reduce compute time and production costs.
  - Dozens of architectures with over 10,000 pretrained models, some in more than 100 languages.

3. Choose the right framework for every part of a model's lifetime:
  - Train state-of-the-art models in 3 lines of code.
  - Move a single model between TF2.0/PyTorch frameworks at will.
  - Seamlessly pick the right framework for training, evaluation and production.

4. Easily customize a model or an example to your needs:
  - We provide examples for each architecture to reproduce the results published by its original authors.
  - Model internals are exposed as consistently as possible.
  - Model files can be used independently of the library for quick experiments.

🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
        question = "Which deep learning libraries back 🤗 Transformers?"
        outputs = question_answerer(question=question, context=long_context)

        self.assertEqual(
            nested_simplify(outputs),
            {"answer": "Jax, PyTorch and TensorFlow", "end": 1919, "score": 0.971, "start": 1892},
        )

    @slow
    @require_tf
    def test_large_model_tf(self):
        question_answerer = pipeline("question-answering", framework="tf")
        outputs = question_answerer(
            question="Where was HuggingFace founded ?", context="HuggingFace was founded in Paris."
        )

        self.assertEqual(nested_simplify(outputs), {"score": 0.979, "start": 27, "end": 32, "answer": "Paris"})


@require_torch_or_tf
class QuestionAnsweringArgumentHandlerTests(unittest.TestCase):
    def test_argument_handler(self):
        qa = QuestionAnsweringArgumentHandler()

        Q = "Where was HuggingFace founded ?"
        C = "HuggingFace was founded in Paris"

        normalized = qa(Q, C)
        self.assertEqual(type(normalized), list)
        self.assertEqual(len(normalized), 1)
        self.assertEqual({type(el) for el in normalized}, {SquadExample})

        normalized = qa(question=Q, context=C)
        self.assertEqual(type(normalized), list)
        self.assertEqual(len(normalized), 1)
        self.assertEqual({type(el) for el in normalized}, {SquadExample})

        normalized = qa(question=Q, context=C)
        self.assertEqual(type(normalized), list)
        self.assertEqual(len(normalized), 1)
        self.assertEqual({type(el) for el in normalized}, {SquadExample})

        normalized = qa(question=[Q, Q], context=C)
        self.assertEqual(type(normalized), list)
        self.assertEqual(len(normalized), 2)
        self.assertEqual({type(el) for el in normalized}, {SquadExample})

        normalized = qa({"question": Q, "context": C})
        self.assertEqual(type(normalized), list)
        self.assertEqual(len(normalized), 1)
        self.assertEqual({type(el) for el in normalized}, {SquadExample})

        normalized = qa([{"question": Q, "context": C}])
        self.assertEqual(type(normalized), list)
        self.assertEqual(len(normalized), 1)
        self.assertEqual({type(el) for el in normalized}, {SquadExample})

        normalized = qa([{"question": Q, "context": C}, {"question": Q, "context": C}])
        self.assertEqual(type(normalized), list)
        self.assertEqual(len(normalized), 2)
        self.assertEqual({type(el) for el in normalized}, {SquadExample})

        normalized = qa(X={"question": Q, "context": C})
        self.assertEqual(type(normalized), list)
        self.assertEqual(len(normalized), 1)
        self.assertEqual({type(el) for el in normalized}, {SquadExample})

        normalized = qa(X=[{"question": Q, "context": C}])
        self.assertEqual(type(normalized), list)
        self.assertEqual(len(normalized), 1)
        self.assertEqual({type(el) for el in normalized}, {SquadExample})

        normalized = qa(data={"question": Q, "context": C})
        self.assertEqual(type(normalized), list)
        self.assertEqual(len(normalized), 1)
        self.assertEqual({type(el) for el in normalized}, {SquadExample})

    def test_argument_handler_error_handling(self):
        qa = QuestionAnsweringArgumentHandler()

        Q = "Where was HuggingFace founded ?"
        C = "HuggingFace was founded in Paris"

        with self.assertRaises(KeyError):
            qa({"context": C})
        with self.assertRaises(KeyError):
            qa({"question": Q})
        with self.assertRaises(KeyError):
            qa([{"context": C}])
        with self.assertRaises(ValueError):
            qa(None, C)
        with self.assertRaises(ValueError):
            qa("", C)
        with self.assertRaises(ValueError):
            qa(Q, None)
        with self.assertRaises(ValueError):
            qa(Q, "")

        with self.assertRaises(ValueError):
            qa(question=None, context=C)
        with self.assertRaises(ValueError):
            qa(question="", context=C)
        with self.assertRaises(ValueError):
            qa(question=Q, context=None)
        with self.assertRaises(ValueError):
            qa(question=Q, context="")

        with self.assertRaises(ValueError):
            qa({"question": None, "context": C})
        with self.assertRaises(ValueError):
            qa({"question": "", "context": C})
        with self.assertRaises(ValueError):
            qa({"question": Q, "context": None})
        with self.assertRaises(ValueError):
            qa({"question": Q, "context": ""})

        with self.assertRaises(ValueError):
            qa([{"question": Q, "context": C}, {"question": None, "context": C}])
        with self.assertRaises(ValueError):
            qa([{"question": Q, "context": C}, {"question": "", "context": C}])

        with self.assertRaises(ValueError):
            qa([{"question": Q, "context": C}, {"question": Q, "context": None}])
        with self.assertRaises(ValueError):
            qa([{"question": Q, "context": C}, {"question": Q, "context": ""}])

        with self.assertRaises(ValueError):
            qa(question={"This": "Is weird"}, context="This is a context")

        with self.assertRaises(ValueError):
            qa(question=[Q, Q], context=[C, C, C])

        with self.assertRaises(ValueError):
            qa(question=[Q, Q, Q], context=[C, C])

    def test_argument_handler_old_format(self):
        qa = QuestionAnsweringArgumentHandler()

        Q = "Where was HuggingFace founded ?"
        C = "HuggingFace was founded in Paris"
        # Backward compatibility for this
        normalized = qa(question=[Q, Q], context=[C, C])
        self.assertEqual(type(normalized), list)
        self.assertEqual(len(normalized), 2)
        self.assertEqual({type(el) for el in normalized}, {SquadExample})

    def test_argument_handler_error_handling_odd(self):
        qa = QuestionAnsweringArgumentHandler()
        with self.assertRaises(ValueError):
            qa(None)

        with self.assertRaises(ValueError):
            qa(Y=None)

        with self.assertRaises(ValueError):
            qa(1)
