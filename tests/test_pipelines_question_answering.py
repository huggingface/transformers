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
from transformers.testing_utils import is_pipeline_test, nested_simplify, require_tf, require_torch, slow

from .test_pipelines_common import ANY, PipelineTestCaseMeta


@is_pipeline_test
class QAPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_FOR_QUESTION_ANSWERING_MAPPING
    tf_model_mapping = TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING

    def run_pipeline_test(self, model, tokenizer, feature_extractor):
        if isinstance(model.config, LxmertConfig):
            # This is an bimodal model, we need to find a more consistent way
            # to switch on those models.
            return
        question_answerer = QuestionAnsweringPipeline(model, tokenizer)

        outputs = question_answerer(
            question="Where was HuggingFace founded ?", context="HuggingFace was founded in Paris."
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
            question="Where was HuggingFace founded ?", context="HuggingFace was founded in Paris.", topk=20
        )
        self.assertEqual(
            outputs, [{"answer": ANY(str), "start": ANY(int), "end": ANY(int), "score": ANY(float)} for i in range(20)]
        )

    @require_torch
    def test_small_model_pt(self):
        question_answerer = pipeline(
            "question-answering", model="sshleifer/tiny-distilbert-base-cased-distilled-squad"
        )
        outputs = question_answerer(
            question="Where was HuggingFace founded ?", context="HuggingFace was founded in Paris."
        )

        self.assertEqual(nested_simplify(outputs), {"score": 0.01, "start": 0, "end": 11, "answer": "HuggingFace"})

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
    @require_tf
    def test_large_model_tf(self):
        question_answerer = pipeline("question-answering", framework="tf")
        outputs = question_answerer(
            question="Where was HuggingFace founded ?", context="HuggingFace was founded in Paris."
        )

        self.assertEqual(nested_simplify(outputs), {"score": 0.979, "start": 27, "end": 32, "answer": "Paris"})


@is_pipeline_test
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
