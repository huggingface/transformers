#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 University of Cambridge, Tencent AI Lab, DeepMind and The University of Hong Kong Authors and The HuggingFace Inc. team. All rights reserved.
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
"""The examples of running contrastive search on the auto-APIs;

Running this example:
python run_generation_contrastive_search.py --model_name_or_path=openai-community/gpt2-large --penalty_alpha=0.6 --k=4 --length=256
"""

import argparse
import logging

from accelerate import PartialState
from accelerate.utils import set_seed

from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--penalty_alpha", type=float, default=0.0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Whether or not to use cpu. If set to False, we will use gpu/npu or mps device if available",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    # Initialize the distributed state.
    distributed_state = PartialState(cpu=args.use_cpu)

    logger.warning(f"device: {distributed_state.device}, 16-bits inference: {args.fp16}")

    if args.seed is not None:
        set_seed(args.seed)

    # Initialize the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    # model = OPTForCausalLM.from_pretrained(args.model_name_or_path)
    # Set the model to the right device
    model.to(distributed_state.device)

    if args.fp16:
        model.half()

    logger.info(args)
    prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")

    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(distributed_state.device) for key, value in inputs.items()}

    output_sequences = model.generate(
        **inputs,
        max_length=args.length + len(inputs["input_ids"][0]),
        penalty_alpha=args.penalty_alpha,
        top_k=args.k,
    )

    generated_sequences = []
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True, add_special_tokens=False)

        # Remove all text after the stop token
        text = text[: text.find(args.stop_token) if args.stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text + text[len(tokenizer.decode(inputs["input_ids"][0], clean_up_tokenization_spaces=True)) :]
        )

        generated_sequences.append(total_sequence)
        print(total_sequence)

    return generated_sequences


if __name__ == "__main__":
    main()
