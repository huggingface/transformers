#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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

"""
import argparse
import logging
import io
import os
import onnxruntime
import sys
import torch

import numpy as np

from datasets import load_metric
from tqdm.auto import tqdm

import transformers
from transformers import (
    MODEL_MAPPING,
    BartForConditionalGeneration,
    BartTokenizer,
)

from bart_onnx.generation_onnx import (
    BARTBeamSearchGenerator,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s |  [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

model_dic = {'facebook/bart-base' : BartForConditionalGeneration}
tokenizer_dic = {'facebook/bart-base' : BartTokenizer}


# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=5,
        help=(
            "The maximum total input sequence length after tokenization."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Device where the model will be run",
    )
    parser.add_argument("--output_file_path", type=str, default=None, help="Where to store the final ONNX file.")

    args = parser.parse_args()

    # if args.validation_file is not None:
    #     extension = args.validation_file.split(".")[-1]
    #     assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args

def load_model_tokenizer(model_name, device='cpu'):
    huggingface_model = model_dic[model_name].from_pretrained(model_name).to(device)
    tokenizer = tokenizer_dic[model_name].from_pretrained(model_name)

    if model_name in ['facebook/bart-base']:
        huggingface_model.config.no_repeat_ngram_size = 0
        huggingface_model.config.forced_bos_token_id = None
        huggingface_model.config.min_length = 0

    return huggingface_model, tokenizer

def export_and_validate_model(model, tokenizer, output_obj, num_beams, max_length, device='cpu'):
    model.eval()
    ort_sess = None
    onnx_bart = torch.jit.script(BARTBeamSearchGenerator(model))

    with torch.no_grad():
        ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
        inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt').to(device)

        # Test export here.
        summary_ids = model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        num_beams=num_beams,
                        max_length=max_length,
                        early_stopping=True,
                        decoder_start_token_id=model.config.decoder_start_token_id)

        if not ort_sess:
            torch.onnx.export(onnx_bart,
                (inputs['input_ids'], inputs['attention_mask'], num_beams, max_length, model.config.decoder_start_token_id),
                output_obj,
                opset_version=14,
                input_names=['input_ids', 'attention_mask', 'num_beams', 'max_length', 'decoder_start_token_id'],
                output_names = ['output_ids'],
                dynamic_axes={
                    'input_ids': {0: 'batch', 1: 'seq'},
                    'output_ids': {0: 'batch', 1: 'seq_out'},
                },
                verbose=False,
                strip_doc_string=False,
                example_outputs=summary_ids)

            if isinstance(output_obj, str):
                ort_sess = onnxruntime.InferenceSession(output_obj)
            else:
                ort_sess = onnxruntime.InferenceSession(output_obj.getvalue())

            ort_out = ort_sess.run(None, {
                'input_ids': inputs['input_ids'].cpu().numpy(),
                'attention_mask': inputs['attention_mask'].cpu().numpy(),
                'num_beams': np.array(num_beams),
                'max_length': np.array(max_length),
                'decoder_start_token_id': np.array(model.config.decoder_start_token_id)
            })

            np.testing.assert_allclose(summary_ids.cpu().numpy(), ort_out[0], rtol=1e-3, atol=1e-3)

            print("Pass - Results are matched.")


def main():
    args = parse_args()
    local_device = 'cpu'
    local_max_length = 5
    local_num_beams = 4

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.setLevel(logging.ERROR)
    transformers.utils.logging.set_verbosity_error()

    if args.device:
        local_device = args.device

    if args.model_name_or_path:
        model, tokenizer = load_model_tokenizer(args.model_name_or_path, local_device)
    else:
        raise ValueError("Make sure that model name has been passed")

    # model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if args.max_length:
        local_max_length = args.max_length

    if args.num_beams:
        local_num_beams = args.num_beams

    if args.output_file_path:
        output_name = args.output_file_path
    else:
        output_name = io.BytesIO()

    export_and_validate_model(model, tokenizer, output_name, local_num_beams, local_max_length, local_device)


    logger.info("***** Running export *****")


if __name__ == "__main__":
    main()
