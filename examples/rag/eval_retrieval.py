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
"""
Evaluation script to evaluate quality of the Retriever used in the RAG model.
"""


import argparse
import json
import logging
import os

import torch
from tqdm import tqdm

from transformers import WEIGHTS_NAME, RagConfig, RagDefaultSequenceModel, RagDefaultTokenModel


logger = logging.getLogger(__name__)


def infer_model_type(model_name_or_path):
    if "token" in model_name_or_path:
        return "token"
    if "sequence" in model_name_or_path:
        return "sequence"
    return None


def evaluate_batch(args, model, input_strings):
    def strip_title(title):
        if title.startswith('"'):
            title = title[1:]
        if title.endswith('"'):
            title = title[:-1]
        return title

    rag_model = model.model
    retriever_inputs = rag_model.retriever_tokenizer.batch_encode_plus(
        input_strings, return_tensors="pt", padding=True, truncation=True,
    )
    retriever_input_embs = rag_model.question_encoder(retriever_inputs["input_ids"].to(args.device))[0]

    _, all_docs = rag_model.retriever.retrieve(retriever_input_embs)

    provenance_strings = []
    for docs in all_docs:
        provenance = [strip_title(title) for title in docs["title"]]
        provenance_strings.append("\t".join(provenance))
    return provenance_strings


def get_precision_at_k(preds_path, gold_data_path, k=1):
    hypos = [line.strip() for line in open(preds_path, "r").readlines()]
    references = [line.strip() for line in open(gold_data_path, "r").readlines()]

    em = total = 0
    for hypo, reference in zip(hypos, references):
        hypo_provenance = set(hypo.split("\t")[:k])
        ref_provenance = set(reference.split("\t")[1 : (k + 1)])
        total += 1
        em += len(hypo_provenance & ref_provenance) / k

    em = 100.0 * em / total
    logger.info("Precision@{}: {}".format(k, em))


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        choices=["sequence", "token"],
        type=str,
        help="RAG model type: sequence or token, if none specified, the type is inferred from the model_name_or_path",
    )
    parser.add_argument(
        "--retriever_type",
        default="hf_retriever",
        choices=["hf_retriever", "mpi_retriever"],
        type=str,
        help="RAG model retriever type",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="facebook/bart-large",
        type=str,
        help="Path to pretrained checkpoints or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="dpr_predictions.tsv",
        help="Name of the predictions file, to be stored in the checkpoints directry",
    )
    parser.add_argument(
        "--gold_data_path",
        default=None,
        type=str,
        required=True,
        help="Path to a tab-separated file with gold samples",
    )
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        "--recalculate", help="Recalculate predictions even if the prediction file exists", action="store_true"
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--k", default=1, type=int, help="k for the precision@k calculation")

    args = parser.parse_args()

    logging.getLogger(__name__).setLevel(logging.INFO)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type is None:
        args.model_type = infer_model_type(args.model_name_or_path)
        assert args.model_type is not None
        logger.info("Inferred model type:", args.model_type)
    model_class = RagDefaultSequenceModel if args.model_type == "sequence" else RagDefaultTokenModel

    checkpoints = [args.model_name_or_path]
    if args.eval_all_checkpoints:
        checkpoints = list(
            os.path.dirname(c)
            for c in sorted(glob.glob(args.model_name_or_path + "/**/" + WEIGHTS_NAME, recursive=True))
        )
        rag_config = None
    else:
        rag_config = RagConfig(
            pretrained_generator_name_or_path="facebook/bart-large", retriever_type=args.retriever_type
        )
    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    for checkpoint in checkpoints:

        if os.path.exists(args.predictions_path) and (not args.recalculate):
            logger.info("Prediction file already exists: {}".format(args.predictions_path))
            get_precision_at_k(args.predictions_path, args.gold_data_path, args.k)
            continue

        logger.info("***** Running retrieval evaluation for {} *****".format(checkpoint))
        logger.info("  Batch size = %d", args.eval_batch_size)
        logger.info("  Predictions will be stored under {}".format(args.predictions_path))

        model = model_class.from_pretrained(checkpoint) if rag_config is None else model_class(rag_config)
        model.to(args.device)

        with open(args.gold_data_path, "r") as eval_file, open(args.predictions_path, "w") as preds_file:
            questions = []
            for line in tqdm(eval_file):
                question = line.strip().split("\t")[0]
                questions.append(question)
                if len(questions) % args.eval_batch_size == 0:
                    predictions = evaluate_batch(args, model, questions)
                    preds_file.write("\n".join(predictions) + "\n")
                    preds_file.flush()
                    questions = []
            if len(questions) > 0:
                predictions = evaluate_batch(args, model, questions)
                preds_file.write("\n".join(predictions) + "\n")
                preds_file.flush()

        get_precision_at_k(args.predictions_path, args.gold_data_path, args.k)


if __name__ == "__main__":
    main()
