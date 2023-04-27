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

import logging
import os
import sys
from pathlib import Path
from unittest.mock import patch

from parameterized import parameterized
from run_eval import run_generate
from run_eval_search import run_search

from transformers.testing_utils import CaptureStdout, TestCasePlus, slow
from utils import ROUGE_KEYS


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


def _dump_articles(path: Path, articles: list):
    content = "\n".join(articles)
    Path(path).open("w").writelines(content)


T5_TINY = "patrickvonplaten/t5-tiny-random"
BART_TINY = "sshleifer/bart-tiny-random"
MBART_TINY = "sshleifer/tiny-mbart"

stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
logging.disable(logging.CRITICAL)  # remove noisy download output from tracebacks


class TestTheRest(TestCasePlus):
    def run_eval_tester(self, model):
        input_file_name = Path(self.get_auto_remove_tmp_dir()) / "utest_input.source"
        output_file_name = input_file_name.parent / "utest_output.txt"
        assert not output_file_name.exists()
        articles = [" New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County."]
        _dump_articles(input_file_name, articles)

        score_path = str(Path(self.get_auto_remove_tmp_dir()) / "scores.json")
        task = "translation_en_to_de" if model == T5_TINY else "summarization"
        testargs = f"""
            run_eval_search.py
            {model}
            {input_file_name}
            {output_file_name}
            --score_path {score_path}
            --task {task}
            --num_beams 2
            --length_penalty 2.0
            """.split()

        with patch.object(sys, "argv", testargs):
            run_generate()
            assert Path(output_file_name).exists()
            # os.remove(Path(output_file_name))

    # test one model to quickly (no-@slow) catch simple problems and do an
    # extensive testing of functionality with multiple models as @slow separately
    def test_run_eval(self):
        self.run_eval_tester(T5_TINY)

    # any extra models should go into the list here - can be slow
    @parameterized.expand([BART_TINY, MBART_TINY])
    @slow
    def test_run_eval_slow(self, model):
        self.run_eval_tester(model)

    # testing with 2 models to validate: 1. translation (t5) 2. summarization (mbart)
    @parameterized.expand([T5_TINY, MBART_TINY])
    @slow
    def test_run_eval_search(self, model):
        input_file_name = Path(self.get_auto_remove_tmp_dir()) / "utest_input.source"
        output_file_name = input_file_name.parent / "utest_output.txt"
        assert not output_file_name.exists()

        text = {
            "en": ["Machine learning is great, isn't it?", "I like to eat bananas", "Tomorrow is another great day!"],
            "de": [
                "Maschinelles Lernen ist großartig, oder?",
                "Ich esse gerne Bananen",
                "Morgen ist wieder ein toller Tag!",
            ],
        }

        tmp_dir = Path(self.get_auto_remove_tmp_dir())
        score_path = str(tmp_dir / "scores.json")
        reference_path = str(tmp_dir / "val.target")
        _dump_articles(input_file_name, text["en"])
        _dump_articles(reference_path, text["de"])
        task = "translation_en_to_de" if model == T5_TINY else "summarization"
        testargs = f"""
            run_eval_search.py
            {model}
            {str(input_file_name)}
            {str(output_file_name)}
            --score_path {score_path}
            --reference_path {reference_path}
            --task {task}
            """.split()
        testargs.extend(["--search", "num_beams=1:2 length_penalty=0.9:1.0"])

        with patch.object(sys, "argv", testargs):
            with CaptureStdout() as cs:
                run_search()
            expected_strings = [" num_beams | length_penalty", model, "Best score args"]
            un_expected_strings = ["Info"]
            if "translation" in task:
                expected_strings.append("bleu")
            else:
                expected_strings.extend(ROUGE_KEYS)
            for w in expected_strings:
                assert w in cs.out
            for w in un_expected_strings:
                assert w not in cs.out
            assert Path(output_file_name).exists()
            os.remove(Path(output_file_name))
