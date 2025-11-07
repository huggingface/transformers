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
import contextlib
import json
import os
import random
import time
from typing import Optional

import datasets
import torch
from torch.profiler import ProfilerActivity, profile
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers.generation.continuous_batching.requests import logger


SLIDING_WINDOW = 0
MODEL_ID = "google/gemma-2-2b-it" if SLIDING_WINDOW > 0 else "meta-llama/Meta-Llama-3-8B"
FORCE_MAX_LENGTH = False  # should be False unless you are debugging sliding window features
SKIP_SPECIAL_TOKENS = False


def generate_simple(
    attn_impl: str, simple_batch_inputs: list[int], generation_config: GenerationConfig
) -> dict[str, str]:
    attn_impl = {
        "sdpa": "sdpa",
        "eager": "eager",
        "paged_attention": "eager",  # TODO: this does not work on AMD docker
        "flash_paged": "flash_attention_2",  # TODO: this does not work on AMD docker
        "kernels-community/flash-attn": "eager",
    }[attn_impl]

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, attn_implementation=attn_impl)
    model = model.cuda().eval()
    if getattr(model.config, "sliding_window", None) is not None:
        model.config.sliding_window = SLIDING_WINDOW

    decoded_outputs = {}
    for input_ids in tqdm(simple_batch_inputs, desc="Generating outputs without CB"):
        key = " ".join(map(str, input_ids))  # This will be used to identify the output after batched generation
        input_ids = torch.tensor([input_ids]).to("cuda")
        # attention_mask = torch.ones_like(input_ids)
        outputs = model.generate(input_ids, generation_config=generation_config, use_model_defaults=False)
        generated_tokens = outputs[0][input_ids.shape[1] :]
        decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=SKIP_SPECIAL_TOKENS)
        decoded_outputs[key] = decoded_output
    return decoded_outputs


def setup_metrics():
    try:
        from opentelemetry import metrics, trace
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": "transformers"})
        metrics_exporter = PeriodicExportingMetricReader(
            OTLPMetricExporter(
                endpoint="http://localhost:9090/api/v1/otlp/v1/metrics"
            ),  # Uses OTEL_EXPORTER_OTLP_METRICS_ENDPOINT env var
            export_interval_millis=1000,
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metrics_exporter])
        metrics.set_meter_provider(meter_provider)
        trace_exporter = OTLPSpanExporter(
            endpoint="http://localhost:4318/v1/traces"
        )  # Uses OTEL_EXPORTER_OTLP_TRACES_ENDPOINT env var
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
        trace.set_tracer_provider(tracer_provider)
    except Exception as e:
        print(f"Error setting up metrics: {e}")


def batch_generate(
    model: AutoModelForCausalLM,
    simple_batch_inputs: list,
    generation_config: GenerationConfig,
    tokenizer: AutoTokenizer,
    displayed_samples: int = 0,  # -1: no display, 0: display stats, >0: display inputs and some outputs
    output_file: Optional[str] = None,
    expected_outputs: Optional[list[str]] = None,
) -> tuple[float, float]:
    # Actual batch generation
    if displayed_samples >= 0:
        print("--- Running CB Generation Example ---")
    start_time_simple = time.time()
    batch_outputs = model.generate_batch(
        inputs=simple_batch_inputs,
        generation_config=generation_config,
    )
    end_time_simple = time.time()
    if displayed_samples >= 0:
        print("Done with batch generation.")

    # Decode outputs
    token_count = 0
    data = []
    for i, request in enumerate(batch_outputs):
        input_text = tokenizer.decode(batch_outputs[request].prompt_ids, skip_special_tokens=SKIP_SPECIAL_TOKENS)
        # The key is used to tie back to the output of unbatched generation
        key = " ".join(map(str, batch_outputs[request].prompt_ids))
        data.append({"input": input_text, "key": key})

        # Try to decode the output
        try:
            output_text = tokenizer.decode(
                batch_outputs[request].generated_tokens, skip_special_tokens=SKIP_SPECIAL_TOKENS
            )
            token_count += len(batch_outputs[request].generated_tokens[1:])
            data[-1]["cb_outputs"] = output_text
        except Exception as e:
            print(f"Decoding failed for request {request}: {e}")
            data[-1]["cb_outputs"] = "__ERROR__"
            continue

        # Display sample if asked
        if i < displayed_samples:
            if len(output_text) > 0:
                print("-" * 20)
                print(f"{request} Input:  {input_text}")
                print(f"{request} Output: {output_text}")
            else:
                print(f"{request} Input:  {input_text}")
                print("[WARN]")
                print(f"{request} Output was empty!")

        # Compare with classic generate if asked
        if expected_outputs is not None:
            expected_output = expected_outputs.pop(key)
            matches = output_text == expected_output  # TODO: rework this for a better distance metric
            data[-1]["without_cb"] = expected_output
            data[-1]["matches"] = matches
            data[-1].pop("key")
            print(f"Request {i} matches" if matches else f"Request {i} does NOT match!")

    # Compute stats and maybe print them
    gen_time = end_time_simple - start_time_simple
    tok_per_sec = token_count / gen_time
    if displayed_samples >= 0:
        print("-" * 20)
        print("--- Finished CB Generation Example ---\n")
        print(f"CB generation took: {gen_time:.2f} seconds for {token_count} tokens. {tok_per_sec:.2f}tok/s")
    stats = {
        "num_blocks": generation_config.num_blocks,
        "max_batch_tokens": generation_config.max_batch_tokens,
        "gen_time": gen_time,
        "token_count": token_count,
        "tok_per_sec": tok_per_sec,
    }

    # If an output file is provided, save the reordered data to it
    data.sort(key=lambda x: x["input"])
    data = [stats] + data
    if output_file is not None:
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)

    return gen_time, tok_per_sec


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-blocks", "-n", type=int, default=None)
    parser.add_argument("--max-batch-tokens", "-b", type=int, default=None)

    parser.add_argument("--attn", type=str, default="kernels-community/flash-attn", help="Attention implementation")
    parser.add_argument("--matmul-precision", "-mp", type=str, default="high")  # set to "none" to disable
    parser.add_argument("--cuda-graph", "-cg", help="Use cuda graphs", type=str, default=None)
    parser.add_argument("--compile", action="store_true", help="Compile the model using torch.compile")

    parser.add_argument("--samples", type=int, default=500, help="Number of samples to generate")
    parser.add_argument("--add-prefix", action="store_true", help="Add a prefix to the samples")

    parser.add_argument("--displayed", type=int, default=0, help="Number of samples to display")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--profile", type=str, default=None)
    args = parser.parse_args()

    # Set log level
    logger.setLevel(args.log_level.upper())

    # If turned on, we setup metrics
    if args.metrics:
        setup_metrics()

    # Set matmul precision if not none
    if args.matmul_precision != "none":
        torch.set_float32_matmul_precision(args.matmul_precision)
    # Parse cuda graph argument
    if args.cuda_graph is not None:
        use_cuda_graph = {
            "none": None,
            "yes": True, "y": True, "true": True, "t": True, "1": True,
            "no": False, "n": False, "false": False, "f": False, "0": False,
        }[args.cuda_graph.lower()]  # fmt: skip
    else:
        use_cuda_graph = None

    # Prepare model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        attn_implementation=args.attn,
        dtype=torch.bfloat16,
    )
    model = model.cuda().eval()
    if getattr(model.config, "sliding_window", None) is not None:
        print(f"Setting sliding window from {model.config.sliding_window} to {SLIDING_WINDOW}")
        model.config.sliding_window = SLIDING_WINDOW

    # If turned on, we compile the model
    if args.compile:
        model.forward = torch.compile(model.forward, mode="max-autotune-no-cudagraphs")

    # Prepare tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")

    dataset = datasets.load_dataset("openai/gsm8k", "socratic", split="test")
    dataset = dataset.select(range(args.samples))

    def random_prefix() -> str:
        if not args.add_prefix:
            return ""
        prefixes = [
            "Math and reasonning problems are very important to the world. This is a problem, and then you will find the answer.\n",
            "We all know that reasonning can be taught by answering questions, often illustrated with examples. Here is one and its solution, hopefully you will enjoy it!\n",
            "Reasonning a very good metric of intelligence, hence it is regularly trained and tested in both children and AI model like LLMs. This test can look like a math or a logical problem, a riddle or pattern detection task. For instance, this is one of those test. You will find it and the solution associated after. Here it goes:\n",
        ] # fmt: skip
        return random.choice(prefixes)

    random.seed(0)
    simple_batch_inputs = [tokenizer(random_prefix() + item["question"])["input_ids"] for item in dataset]

    # Prepare generation config
    generation_config = GenerationConfig(
        max_new_tokens=512,
        use_cuda_graph=use_cuda_graph,
        eos_token_id=tokenizer.pad_token_id if FORCE_MAX_LENGTH else tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=not args.compare,
        temperature=0.8,
        top_p=0.9,
        num_blocks=args.num_blocks,
        max_batch_tokens=args.max_batch_tokens,
    )

    # If we need to compare, we need to generate the reference outputs
    expected_outputs = generate_simple(args.attn, simple_batch_inputs, generation_config) if args.compare else None

    # If no output file is provided, we pick a name based on the args
    if args.output_file is None:
        os.makedirs("runs/cb", exist_ok=True)
        attn = args.attn.replace("|", "_").replace("/", "_")
        args.output_file = (
            f"runs/cb/{args.num_blocks}_{args.max_batch_tokens}_{attn}_{args.matmul_precision}_{args.samples}.json"
        )

    # Run warmup batch generation # TODO: understand why warmup incurs a large overhead during cache creation
    batch_generate(
        model,
        simple_batch_inputs[: min(5, args.samples)],
        generation_config,
        tokenizer,
        displayed_samples=-1,
    )

    if args.profile is not None:
        cm = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)
    else:
        cm = contextlib.nullcontext()
    with cm as prof:
        # Run batch generation
        gen_time, tok_per_sec = batch_generate(
            model,
            simple_batch_inputs,
            generation_config,
            tokenizer,
            displayed_samples=args.displayed,
            output_file=args.output_file,
            expected_outputs=expected_outputs,
        )
    if args.profile is not None:
        filename = args.profile if args.profile.endswith(".json") else args.profile + ".json"
        prof.export_chrome_trace(filename)

# Example usage:
# python examples/pytorch/continuous_batching.py --attn sdpa_paged -mp none --samples 3 --compare
# python examples/pytorch/continuous_batching.py --num-blocks 369 --max-batch-tokens 23 --attn sdpa_paged -mp none --samples 1 --displayed 0 --output-file sliced.json
