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

import run_generation
import run_glue
import run_squad
import evaluate_bart_cnn


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

    def test_bart_cnn_summarization(self):
        articles = [' New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.  Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other. In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage. Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the 2010 marriage license application, according to court documents. Prosecutors said the marriages were part of an immigration scam. On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further. After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.  All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say. Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.  Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted. The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali. Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force. If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.'
        " (CNN)At first police in Marana, Arizona, thought the shoplifted gun Mario Valencia held as he walked through a busy office park was locked and unable to fire. The cable through the lever and trigger couldn't be taken off, an officer was told by an employee of the Walmart where Valencia took the gun and some rounds of ammunition. But just 10 seconds after the worker told police that ... a shot. Valencia had fired into the air, and less than a minute later a police car slammed into him in a move that ended a crime spree and sparked nationwide discussion on the officer's unusual tactic. The 36-year-old Valencia was hospitalized and within a few days transferred to jail where he faces 15 charges, including shoplifting the .30-30 rifle. That February morning, police have said, Valencia committed several crimes in nearby Tucson before stealing a car and driving to the Walmart in Marana. There he went to the sporting goods department, asked to see a rifle, then told an employee he wanted the ammunition. Officer who drove into suspect justified, chief says . The woman told police she gave Valencia the rounds because he told her he would break the case with the bullets inside. He also told her not to do anything stupid. In spite of that she also said she didn't feel threatened, leading police to charge him with shoplifting and not armed robbery. Walmart told CNN's Miguel Marquez that  the store clerk acted appropriately, even using a code to alert security to call police. Valencia took the gun and ammo and fled into a nearby business park where he encountered an officer in a slow-moving patrol car. At one point he pointed the weapon at an officer and at another he pointed it at his head. The officer told him several times to put down the gun, police have said. The officers that were tailing him assumed that he likely couldn't shoot anyone because of the store's lock. Marana police on Thursday said the cable gun lock was still on the rifle when it was recovered. But the wire that goes through the trigger and the lever to reload the gun were loose enough to allow it to still be used, police said. It also should have been wrapped through the lever twice, not once, police said. A Walmart spokesman told CNN that the rifle had been properly locked and might have been affected by the hard blow caused by the police car. Valencia, who is in Pima County Jail, will appear in court again on May 18."]
        evaluate_bart_cnn.generate_summaries(articles, 'summaries.txt', batch_size=2)
