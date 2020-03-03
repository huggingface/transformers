# coding=utf-8
# Copyright 2018 HuggingFace Inc..
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


import argparse
import logging
import sys
import unittest
from unittest.mock import patch

import evaluate_bart_cnn
import run_generation
import run_glue
import run_squad
from transformers.tests import slow

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


def get_setup_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f")
    args = parser.parse_args()
    return args.f


class ExamplesTests(unittest.TestCase):
    def test_run_glue(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        testargs = [
            "run_glue.py",
            "--data_dir=./examples/tests_samples/MRPC/",
            "--task_name=mrpc",
            "--do_train",
            "--do_eval",
            "--output_dir=./examples/tests_samples/temp_dir",
            "--per_gpu_train_batch_size=2",
            "--per_gpu_eval_batch_size=1",
            "--learning_rate=1e-4",
            "--max_steps=10",
            "--warmup_steps=2",
            "--overwrite_output_dir",
            "--seed=42",
        ]
        model_type, model_name = ("--model_type=bert", "--model_name_or_path=bert-base-uncased")
        with patch.object(sys, "argv", testargs + [model_type, model_name]):
            result = run_glue.main()
            for value in result.values():
                self.assertGreaterEqual(value, 0.75)

    def test_run_squad(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        testargs = [
            "run_squad.py",
            "--data_dir=./examples/tests_samples/SQUAD",
            "--model_name=bert-base-uncased",
            "--output_dir=./examples/tests_samples/temp_dir",
            "--max_steps=10",
            "--warmup_steps=2",
            "--do_train",
            "--do_eval",
            "--version_2_with_negative",
            "--learning_rate=2e-4",
            "--per_gpu_train_batch_size=2",
            "--per_gpu_eval_batch_size=1",
            "--overwrite_output_dir",
            "--seed=42",
        ]
        model_type, model_name = ("--model_type=bert", "--model_name_or_path=bert-base-uncased")
        with patch.object(sys, "argv", testargs + [model_type, model_name]):
            result = run_squad.main()
            self.assertGreaterEqual(result["f1"], 30)
            self.assertGreaterEqual(result["exact"], 30)

    def test_generation(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        testargs = ["run_generation.py", "--prompt=Hello", "--length=10", "--seed=42"]
        model_type, model_name = ("--model_type=openai-gpt", "--model_name_or_path=openai-gpt")
        with patch.object(sys, "argv", testargs + [model_type, model_name]):
            result = run_generation.main()
            self.assertGreaterEqual(len(result[0]), 10)

    @slow
    def test_bart_cnn_summarization(self):
        articles = [
            ' New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.  Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other. In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage. Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the 2010 marriage license application, according to court documents. Prosecutors said the marriages were part of an immigration scam. On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further. After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.  All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say. Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.  Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted. The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali. Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force. If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.'
        ]
        evaluate_bart_cnn.generate_summaries(articles, "summaries.txt", batch_size=1)
