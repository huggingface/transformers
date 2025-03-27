#!/usr/bin/env python
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

import argparse
import itertools
import operator
import sys
from collections import OrderedDict

from run_eval import datetime_now, run_generate

from utils import ROUGE_KEYS


# A table of supported tasks and the list of scores in the order of importance to be sorted by.
# To add a new task, simply list the score names that `run_eval.run_generate()` returns
task_score_names = {
    "translation": ["bleu"],
    "summarization": ROUGE_KEYS,
}


def parse_search_arg(search):
    groups = search.split()
    entries = dict((g.split("=") for g in groups))
    entry_names = list(entries.keys())
    sets = [[f"--{k} {v}" for v in vs.split(":")] for k, vs in entries.items()]
    matrix = [list(x) for x in itertools.product(*sets)]
    return matrix, entry_names


def run_search():
    """
     Run parametric search over the desired hparam space with help of ``run_eval.py``.

     All the arguments except ``--search`` are passed to ``run_eval.py`` as is. The values inside of "--search" are parsed, reformatted and fed to ``run_eval.py`` as additional args.

    The format for the ``--search`` value is a simple string with hparams and colon separated values to try, e.g.:
    ```
     --search "num_beams=5:10 length_penalty=0.8:1.0:1.2 early_stopping=true:false"
    ```
    which will generate ``12`` ``(2*3*2)`` searches for a product of each hparam. For example the example that was just used will invoke ``run_eval.py`` repeatedly with:

    ```
     --num_beams 5 --length_penalty 0.8 --early_stopping true
     --num_beams 5 --length_penalty 0.8 --early_stopping false
     [...]
     --num_beams 10 --length_penalty 1.2 --early_stopping false
    ```

    On completion, this function prints a markdown table of the results sorted by the best BLEU score and the winning arguments.


    """
    prog = sys.argv[0]

    parser = argparse.ArgumentParser(
        usage=(
            "\n\nImportant: this script accepts all arguments `run_eval.py` accepts and then a few extra, therefore"
            " refer to `run_eval.py -h` for the complete list."
        )
    )
    parser.add_argument(
        "--search",
        type=str,
        required=False,
        help='param space to search, e.g. "num_beams=5:10 length_penalty=0.8:1.0:1.2"',
    )
    parser.add_argument(
        "--bs", type=int, default=8, required=False, help="initial batch size (may get reduced if it's too big)"
    )
    parser.add_argument("--task", type=str, help="used for task_specific_params + metrics")
    parser.add_argument(
        "--info",
        nargs="?",
        type=str,
        const=datetime_now(),
        help=(
            "add custom notes to be printed before the results table. If no value is passed, the current datetime"
            " string will be used."
        ),
    )
    args, args_main = parser.parse_known_args()
    # we share some of the args
    args_main.extend(["--task", args.task])
    args_normal = [prog] + args_main

    # to support variations like translation_en_to_de"
    task = "translation" if "translation" in args.task else "summarization"

    matrix, col_names = parse_search_arg(args.search)
    col_names[0:0] = task_score_names[task]  # score cols first
    col_widths = {col: len(str(col)) for col in col_names}
    results = []
    for r in matrix:
        hparams = dict((x.replace("--", "").split() for x in r))
        args_exp = " ".join(r).split()
        args_exp.extend(["--bs", str(args.bs)])  # in case we need to reduce its size due to CUDA OOM
        sys.argv = args_normal + args_exp

        # XXX: need to trap CUDA OOM and lower args.bs if that happens and retry

        scores = run_generate(verbose=False)
        # make sure scores are first in the table
        result = OrderedDict()
        for score in task_score_names[task]:
            result[score] = scores[score]
        result.update(hparams)
        results.append(result)

        # find widest entries
        for k, v in result.items():
            l = len(str(v))
            if l > col_widths[k]:
                col_widths[k] = l

    results_sorted = sorted(results, key=operator.itemgetter(*task_score_names[task]), reverse=True)
    print(" | ".join([f"{col:{col_widths[col]}}" for col in col_names]))
    print(" | ".join([f"{'-' * col_widths[col]}" for col in col_names]))
    for row in results_sorted:
        print(" | ".join([f"{row[col]:{col_widths[col]}}" for col in col_names]))

    best = results_sorted[0]
    for score in task_score_names[task]:
        del best[score]
    best_args = [f"--{k} {v}" for k, v in best.items()]
    dyn_args = ["--bs", str(args.bs)]
    if args.info:
        print(f"\nInfo: {args.info}")
    print("\nBest score args:")
    print(" ".join(args_main + best_args + dyn_args))

    return results_sorted


if __name__ == "__main__":
    # Usage:
    # [normal-run_eval_search.py cmd plus] \
    # --search="num_beams=1:5:10 length_penalty=0.8:1:1.2 early_stopping=true:false"
    #
    # Example:
    # PYTHONPATH="src:examples/seq2seq" python examples/seq2seq/run_eval_search.py $MODEL_NAME \
    # $DATA_DIR/val.source $SAVE_DIR/test_translations.txt --reference_path $DATA_DIR/val.target \
    # --score_path $SAVE_DIR/test_bleu.json --bs $BS --task translation \
    # --search="num_beams=1:5:10 length_penalty=0.8:1:1.2 early_stopping=true:false"
    run_search()
