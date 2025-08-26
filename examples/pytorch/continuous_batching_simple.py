# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team
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
import time

import datasets
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DISPLAYED_SAMPLES = 3


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-blocks", "-n", type=int, default=None)
    parser.add_argument("--max-batch-tokens", "-b", type=int, default=None)
    parser.add_argument(
        "--attn", type=str, default="paged_attention|kernels-community/flash-attn", help="Attention implementation"
    )
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()

    # Prepare model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        attn_implementation=args.attn,
        dtype=torch.bfloat16,
    )
    model = model.cuda().eval()

    # Prepare tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    dataset = datasets.load_dataset("openai/gsm8k", "socratic", split="test")
    dataset = dataset.select(range(args.samples))
    tokenized_datasets = dataset.map(lambda x: tokenizer(x["question"]), batched=True)
    simple_batch_inputs = [item["input_ids"] for item in tokenized_datasets]

    # Prepare generation config
    generation_config = GenerationConfig(
        max_new_tokens=512,
        use_cuda_graph=False,  # Not supported for simple version
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        num_blocks=args.num_blocks,
        max_batch_tokens=args.max_batch_tokens,
    )

    # Warmup iterations
    _ = model.generate_batch(
        inputs=simple_batch_inputs[: min(5, args.samples)],
        generation_config=generation_config,
        slice_inputs=True,
    )

    # Actual batch generation
    print("--- Running CB Generation Example ---")
    start_time = time.time()
    batch_outputs = model.generate_batch(
        inputs=simple_batch_inputs,
        generation_config=generation_config,
        slice_inputs=True,
    )
    end_time = time.time()
    print("Done with batch generation.")

    # Decode outputs
    token_count = 0
    for i, request in enumerate(batch_outputs):
        input_text = tokenizer.decode(batch_outputs[request].prompt_ids, skip_special_tokens=True)
        # Try to decode the output
        try:
            output_text = tokenizer.decode(batch_outputs[request].generated_tokens, skip_special_tokens=True)
            token_count += len(batch_outputs[request].generated_tokens[1:])
        except Exception as e:
            print(f"Decoding failed for request {request}: {e}")
            continue

        # Display sample if asked
        if i < DISPLAYED_SAMPLES:
            print("-" * 20)
            print(f"{request} Input:  {input_text}")
            if len(output_text) > 0:
                print(f"{request} Output: {output_text}")
            else:
                print(f"[WARN] {request} Output was empty!")

    # Compute stats and maybe print them
    gen_time = end_time - start_time
    tok_per_sec = token_count / gen_time
    print("-" * 20)
    print("--- Finished CB Generation Example ---\n")
    print(f"CB generation took: {gen_time:.2f} seconds for {token_count} tokens. {tok_per_sec:.2f}tok/s")
