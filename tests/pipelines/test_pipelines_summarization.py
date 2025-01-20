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
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    SummarizationPipeline,
    TFPreTrainedModel,
    pipeline,
)
from transformers.testing_utils import is_pipeline_test, require_tf, require_torch, slow, torch_device, skipIfRocm
from transformers.tokenization_utils import TruncationStrategy

from .test_pipelines_common import ANY


@is_pipeline_test
class SummarizationPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
    tf_model_mapping = TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING

    def get_test_pipeline(
        self,
        model,
        tokenizer=None,
        image_processor=None,
        feature_extractor=None,
        processor=None,
        torch_dtype="float32",
    ):
        summarizer = SummarizationPipeline(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            processor=processor,
            torch_dtype=torch_dtype,
        )
        return summarizer, ["(CNN)The Palestinian Authority officially became", "Some other text"]

    def run_pipeline_test(self, summarizer, _):
        model = summarizer.model

        outputs = summarizer("(CNN)The Palestinian Authority officially became")
        self.assertEqual(outputs, [{"summary_text": ANY(str)}])

        outputs = summarizer(
            "(CNN)The Palestinian Authority officially became ",
            num_beams=2,
            min_length=2,
            max_length=5,
        )
        self.assertEqual(outputs, [{"summary_text": ANY(str)}])

        # Some models (Switch Transformers, LED, T5, LongT5, etc) can handle long sequences.
        model_can_handle_longer_seq = [
            "SwitchTransformersConfig",
            "T5Config",
            "LongT5Config",
            "LEDConfig",
            "PegasusXConfig",
            "FSMTConfig",
            "M2M100Config",
            "ProphetNetConfig",  # positional embeddings up to a fixed maximum size (otherwise clamping the values)
        ]
        if model.config.__class__.__name__ not in model_can_handle_longer_seq:
            # Too long and exception is expected.
            # For TF models, if the weights are initialized in GPU context, we won't get expected index error from
            # the embedding layer.
            if not (
                isinstance(model, TFPreTrainedModel)
                and len(summarizer.model.trainable_weights) > 0
                and "GPU" in summarizer.model.trainable_weights[0].device
            ):
                if str(summarizer.device) == "cpu":
                    with self.assertRaises(Exception):
                        outputs = summarizer("This " * 1000)
        outputs = summarizer("This " * 1000, truncation=TruncationStrategy.ONLY_FIRST)

    @require_torch
    @skipIfRocm(arch='gfx1201')
    def test_small_model_pt(self):
        summarizer = pipeline(task="summarization", model="sshleifer/tiny-mbart", framework="pt")
        outputs = summarizer("This is a small test")
        self.assertEqual(
            outputs,
            [
                {
                    "summary_text": "เข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไป"
                }
            ],
        )

    @require_tf
    def test_small_model_tf(self):
        summarizer = pipeline(task="summarization", model="sshleifer/tiny-mbart", framework="tf")
        outputs = summarizer("This is a small test")
        self.assertEqual(
            outputs,
            [
                {
                    "summary_text": "เข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไปเข้าไป"
                }
            ],
        )

    @require_torch
    @slow
    def test_integration_torch_summarization(self):
        summarizer = pipeline(task="summarization", device=torch_device)
        cnn_article = (
            " (CNN)The Palestinian Authority officially became the 123rd member of the International Criminal Court on"
            " Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The"
            " formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based."
            " The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted its"
            ' jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including East'
            ' Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination into the'
            " situation in Palestinian territories, paving the way for possible war crimes investigations against"
            " Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and"
            " the United States, neither of which is an ICC member, opposed the Palestinians' efforts to join the"
            " body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday's ceremony, said it was a"
            ' move toward greater justice. "As Palestine formally becomes a State Party to the Rome Statute today, the'
            ' world is also a step closer to ending a long era of impunity and injustice," he said, according to an'
            ' ICC news release. "Indeed, today brings us closer to our shared goals of justice and peace." Judge'
            " Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the"
            ' Palestinians. "As the Rome Statute today enters into force for the State of Palestine, Palestine'
            " acquires all the rights as well as responsibilities that come with being a State Party to the Statute."
            ' These are substantive commitments, which cannot be taken lightly," she said. Rights group Human Rights'
            ' Watch welcomed the development. "Governments seeking to penalize Palestine for joining the ICC should'
            " immediately end their pressure, and countries that support universal acceptance of the court's treaty"
            ' should speak out to welcome its membership," said Balkees Jarrah, international justice counsel for the'
            " group. \"What's objectionable is the attempts to undermine international justice, not Palestine's"
            ' decision to join a treaty to which over 100 countries around the world are members." In January, when'
            " the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described it as an"
            ' outrage, saying the court was overstepping its boundaries. The United States also said it "strongly"'
            " disagreed with the court's decision. \"As we have said repeatedly, we do not believe that Palestine is a"
            ' state and therefore we do not believe that it is eligible to join the ICC," the State Department said in'
            ' a statement. It urged the warring sides to resolve their differences through direct negotiations. "We'
            ' will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace,"'
            " it said. But the ICC begs to differ with the definition of a state for its purposes and refers to the"
            ' territories as "Palestine." While a preliminary examination is not a formal investigation, it allows the'
            " court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou"
            ' Bensouda said her office would "conduct its analysis in full independence and impartiality." The war'
            " between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The inquiry"
            " will include alleged war crimes committed since June. The International Criminal Court was set up in"
            " 2002 to prosecute genocide, crimes against humanity and war crimes. CNN's Vasco Cotovio, Kareem Khadder"
            " and Faith Karimi contributed to this report."
        )
        expected_cnn_summary = (
            " The Palestinian Authority becomes the 123rd member of the International Criminal Court . The move gives"
            " the court jurisdiction over alleged crimes in Palestinian territories . Israel and the United States"
            " opposed the Palestinians' efforts to join the court . Rights group Human Rights Watch welcomes the move,"
            " says governments seeking to penalize Palestine should end pressure ."
        )
        result = summarizer(cnn_article)
        self.assertEqual(result[0]["summary_text"], expected_cnn_summary)
