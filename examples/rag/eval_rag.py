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
import logging
import os
import re
import string

import pandas as pd
import torch
from tqdm import tqdm

from transformers import WEIGHTS_NAME, BartForConditionalGeneration, BartTokenizer, RagSequenceModel, RagTokenModel
from utils import exact_match_score, f1_score


logger = logging.getLogger(__name__)


def infer_model_type(model_name_or_path):
    if "token" in model_name_or_path:
        return "rag_token"
    if "sequence" in model_name_or_path:
        return "rag_sequence"
    if "bart" in model_name_or_path:
        return "bart"
    return None


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_scores(preds_path, gold_data_path, gold_data_mode):
    hypos = [line.strip() for line in open(preds_path, "r").readlines()]
    answers = []

    if gold_data_mode == "qa":
        data = pd.read_csv(gold_data_path, sep="\t", header=None)
        for answer_list in data[1]:
            ground_truths = ast.literal_eval(answer_list)
            answers.append(ground_truths)
    else:
        references = [line.strip() for line in open(gold_data_path, "r").readlines()]
        answers = [[reference] for reference in references]

    f1 = em = total = 0
    for prediction, ground_truths in zip(hypos, answers):
        total += 1
        em += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    em = 100.0 * em / total
    f1 = 100.0 * f1 / total

    print("F1: {}".format(f1))
    print("EM: {}".format(em))


def evaluate_batch(args, rag_model, tokenizer, questions):
    with torch.no_grad():
        input_ids = tokenizer.batch_encode_plus(questions, return_tensors="pt", padding=True, truncation=True)[
            "input_ids"
        ].to(args.device)
        outputs = rag_model.generate(
            input_ids,
            num_beams=args.num_beams,
            min_length=args.min_length,  # make sure short answers are allowed
            max_length=args.max_length,  # no need for crazy long answers in NQ
            early_stopping=False,
            num_return_sequences=1,
            bad_words_ids=[[0, 0]],  # BART likes to repeat BOS tokens, dont allow it to generate more than one
            clean_up_tokenization=True,
        )
        output_strings = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_strings


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        choices=["rag_sequence", "rag_token", "bart"],
        type=str,
        help="RAG model type: rag_sequence, rag_token or bart, if none specified, the type is inferred from the model_name_or_path",
    )
    parser.add_argument(
        "--retriever_type",
        default=None,
        choices=["hf_retriever", "mpi_retriever"],
        type=str,
        help="RAG model retriever type",
    )
    parser.add_argument("--n_docs", default=5, type=int, help="Number of retrieved docs")
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
        "qa - a single line in the following format: question [tab] answer_list"
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
        "--eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--recalculate", help="Recalculate predictions even if the prediction file exists", action="store_true",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--num_beams", default=4, type=int, help="Number of beams to be used when generating answers",
    )
    parser.add_argument("--min_length", default=1, type=int, help="Min length of the generated answers")
    parser.add_argument("--max_length", default=50, type=int, help="Max length of the generated answers")

    args = parser.parse_args()

    logging.getLogger(__name__).setLevel(logging.INFO)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_kwargs = {}
    if args.model_type is None:
        args.model_type = infer_model_type(args.model_name_or_path)
        assert args.model_type is not None
    if args.model_type.startswith("rag"):
        model_class = RagTokenModel if args.model_type == "rag_token" else RagSequenceModel
        model_kwargs["n_docs"] = args.n_docs
        if args.retriever_type is not None:
            model_kwargs["retriever_type"] = args.retriever_type
    else:
        model_class = BartForConditionalGeneration

    checkpoints = (
        [f.path for f in os.scandir(args.model_name_or_path) if f.is_dir()]
        if args.eval_all_checkpoints
        else [args.model_name_or_path]
    )

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    for checkpoint in checkpoints:
        predictions_path = os.path.join(checkpoint, args.predictions_filename)
        if os.path.exists(predictions_path) and (not args.recalculate):
            logger.info("Calculating metrics based on an existing predictions file: {}".format(predictions_path))
            get_scores(predictions_path, args.gold_data_path, args.gold_data_mode)
            continue

        logger.info("***** Running evaluation for {} *****".format(checkpoint))
        logger.info("  Batch size = %d", args.eval_batch_size)
        logger.info("  Predictions will be stored under {}".format(predictions_path))

        model = model_class.from_pretrained(checkpoint, **model_kwargs)
        model.to(args.device)
        tokenizer = (
            model.model.generator_tokenizer
            if args.model_type != "bart"
            else BartTokenizer.from_pretrained("facebook/bart-large")
        )
        if args.model_type != "bart":
            model.model.retriever.init_retrieval(12345)

        with open(args.evaluation_set, "r") as eval_file, open(predictions_path, "w") as preds_file:
            questions = []
            for line in tqdm(eval_file):
                questions.append(line.strip())
                if len(questions) == args.eval_batch_size:
                    answers = evaluate_batch(args, model, tokenizer, questions)
                    preds_file.write("\n".join(answers) + "\n")
                    preds_file.flush()
                    for q, a in zip(questions, answers):
                        print("Q:", q, "- A:", a)
                    questions = []
                    print()
            if len(questions) > 0:
                answers = evaluate_batch(args, model, tokenizer, questions)
                preds_file.write("\n".join(answers))
                preds_file.flush()

            get_scores(predictions_path, args.gold_data_path, args.gold_data_mode)


if __name__ == "__main__":
    main()
