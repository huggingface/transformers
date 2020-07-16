# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import ast
import glob
import json
import logging
import os
import re
import string
from collections import Counter

import torch
from tqdm import tqdm

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    RagConfig,
    RagDefaultSequenceModel,
    RagDefaultTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)


logger = logging.getLogger(__name__)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_scores(preds_path, gold_data_path, gold_data_mode):
    hypos = open(preds_path, "r").readlines()
    hypos = [line.strip() for line in hypos]
    references = open(gold_data_path, "r").readlines()
    references = [line.strip() for line in references]

    answers = []
    if gold_data_mode == "qa":
        for reference in references:
            _, answer_list = reference.split("\t")
            answers.append(ast.literal_eval(answer_list))
    else:
        answers = [[reference] for reference in references]

    f1 = em = total = 0
    for prediction, ground_truths in zip(hypos, answers):  # data[0], data[1]):
        total += 1
        em += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    em = 100.0 * em / total
    f1 = 100.0 * f1 / total

    print("F1: {}".format(f1))
    print("EM: {}".format(em))


def evaluate_batch(args, model, tokenizer, questions):
    with torch.no_grad():
        input_ids = tokenizer.batch_encode_plus(questions, return_tensors="pt", padding=True, truncation=True)[
            "input_ids"
        ].to(args.device)
        outputs = model.generate(
            input_ids,
            num_beams=args.n_beams,
            min_length=1,  # make sure short answers are allowed
            max_length=10,  # no need for crazy long answers in NQ
            early_stopping=False,
            num_return_sequences=1,
            bad_words_ids=[[0, 0]]
            # BART likes to repeat BOS tokens, dont allow it to generate more than one
        )
        output_strings = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_strings


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained checkpoints or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--evaluation_set", default=None, type=str, required=True, help="Path to a file containing evaluation samples",
    )
    parser.add_argument(
        "--gold_data_path",
        default=None,
        type=str,
        required=True,
        help="Path to a tab-separated file with gold samples",
    )
    parser.add_argument(
        "--gold_data_mode",
        default="qa",
        type=str,
        choices=["qa", "ans"],
        help="Format of the gold data file"
        "qa - a sinle line in the following format: question [tab] answer_list"
        "ans - a single line of the gold file contains the expected answer string",
    )
    parser.add_argument(
        "--predictions_filename",
        type=str,
        default="predictions.txt",
        help="Name of the predictions file, to be stored in the checkpoints directry",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--n_beams", default=3, type=int, help="Number of beams to be used when generating answers",
    )
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        "--recalculate", help="Recalculate predictions even if the prediction file exists", action="store_true"
    )

    args = parser.parse_args()

    logging.getLogger(__name__).setLevel(logging.INFO)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints = [args.model_name_or_path]
    if args.eval_all_checkpoints:
        checkpoints = list(
            os.path.dirname(c)
            for c in sorted(glob.glob(args.model_name_or_path + "/**/" + WEIGHTS_NAME, recursive=True))
        )

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    tokenizer = RagDefaultTokenizer.from_pretrained("facebook/bart-large", do_lower_case=args.do_lower_case)
    for checkpoint in checkpoints:

        predictions_path = os.path.join(checkpoint, args.predictions_filename)

        if os.path.exists(predictions_path) and (not args.recalculate):
            logger.info("Calculating metrics based on an existing predictions file: {}".format(predictions_path))
            get_scores(predictions_path, args.gold_data_path, args.gold_data_mode)
            continue

        logger.info("***** Running evaluation for {} *****".format(checkpoint))
        logger.info("  Batch size = %d", args.eval_batch_size)
        logger.info("  Predictions will be stored under {}".format(predictions_path))

        model = RagDefaultSequenceModel.from_pretrained(checkpoint)
        model.to(args.device)

        with open(args.evaluation_set, "r") as eval_file, open(predictions_path, "w") as preds_file:
            questions = []
            for line in tqdm(eval_file):
                questions.append(line.strip())
                if len(questions) == args.eval_batch_size:
                    answers = evaluate_batch(args, model, tokenizer, questions)
                    questions = []
                    preds_file.write("\n".join(answers))
                    preds_file.flush()
            if len(datapoints) > 0:
                answers = evaluate_batch(datapoints)
                preds_file.write("\n".join(answers))
                preds_file.flush()

            evaluate(args, model, tokenizer, predictions_path)
            get_scores(predictions_path, args.gold_data_path, args.gold_data_mode)


if __name__ == "__main__":
    main()
