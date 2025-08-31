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
import json
import os
import time
from typing import Optional

import datasets
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def generate_simple(
    attn_implementation: str, simple_batch_inputs: list[int], generation_config: GenerationConfig
) -> list[str]:
    attn_implementation = {
        "sdpa_paged": "sdpa",
        "eager_paged": "eager",
        "flash_paged": "flash_attention_2",
    }[attn_implementation]

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
        .cuda()
        .eval()
    )

    decoded_outputs = []
    for input_ids in simple_batch_inputs:
        input_ids = torch.tensor([input_ids]).to("cuda")
        attention_mask = torch.ones_like(input_ids)
        outputs = model.generate(input_ids, attention_mask=attention_mask, generation_config=generation_config)
        generated_tokens = outputs[0][input_ids.shape[1] :]
        decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        decoded_outputs.append(decoded_output)

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
    slice_inputs: bool = True,
) -> tuple[float, float]:
    # Actual batch generation
    if displayed_samples >= 0:
        print("--- Running CB Generation Example ---")
    start_time_simple = time.time()
    batch_outputs = model.generate_batch(
        inputs=simple_batch_inputs,
        generation_config=generation_config,
        slice_inputs=slice_inputs,  # TODO: move this to the generation config
    )
    end_time_simple = time.time()
    if displayed_samples >= 0:
        print("Done with batch generation.")

    # Decode outputs
    token_count = 0
    data = []
    for i, request in enumerate(batch_outputs):
        input_text = tokenizer.decode(batch_outputs[request].prompt_ids, skip_special_tokens=True)
        data.append({"input": input_text})

        # Try to decode the output
        try:
            output_text = tokenizer.decode(batch_outputs[request].generated_tokens, skip_special_tokens=True)
            token_count += len(batch_outputs[request].generated_tokens[1:])
            data[-1]["output"] = output_text
        except Exception as e:
            print(f"Decoding failed for request {request}: {e}")
            data[-1]["output"] = "__ERROR__"
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
            matches = output_text == expected_outputs[i]
            data[-1]["ref"] = expected_outputs[i]
            data[-1]["matches"] = matches
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

    parser.add_argument(
        "--attn", type=str, default="paged_attention|kernels-community/flash-attn", help="Attention implementation"
    )
    parser.add_argument("--matmul-precision", "-mp", type=str, default="high")  # set to "none" to disable
    parser.add_argument("--slice-inputs", action="store_true", default=False)
    parser.add_argument("--use-cuda-graph", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False)

    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--displayed", type=int, default=0, help="Number of samples to display")
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--compare", action="store_true", default=False)
    parser.add_argument("--metrics", action="store_true", default=False)
    args = parser.parse_args()

    # If turned on, we setup metrics
    if args.metrics:
        setup_metrics()

    # Set matmul precision if not none
    if args.matmul_precision != "none":
        torch.set_float32_matmul_precision(args.matmul_precision)

    # Prepare model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        attn_implementation=args.attn,
        dtype=torch.bfloat16,
    )
    model = model.cuda().eval()

    # If turned on, we compile the model
    if args.compile:
        model.forward = torch.compile(model.forward, mode="max-autotune-no-cudagraphs")
    if args.slice_inputs:
        assert not args.compile, "Slicing inputs requires is not the model to be compiled"
        assert not args.use_cuda_graph, "Slicing inputs is not compatible with cuda graphs"

    # Prepare tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    dataset = datasets.load_dataset("openai/gsm8k", "socratic", split="test")
    dataset = dataset.select(range(args.samples))  # Use only 5 examples for the simple version
    tokenized_datasets = dataset.map(lambda x: tokenizer(x["question"]), batched=True)
    simple_batch_inputs = [item["input_ids"] for item in tokenized_datasets]

    # Prepare generation config
    generation_config = GenerationConfig(
        max_new_tokens=512,
        use_cuda_graph=args.use_cuda_graph,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
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

    # Run warmup batch generation
    batch_generate(
        model,
        simple_batch_inputs[: min(5, args.samples)],
        generation_config,
        tokenizer,
        displayed_samples=-1,
        slice_inputs=args.slice_inputs,
    )

    # Run batch generation
    gen_time, tok_per_sec = batch_generate(
        model,
        simple_batch_inputs,
        generation_config,
        tokenizer,
        displayed_samples=args.displayed,
        output_file=args.output_file,
        expected_outputs=expected_outputs,
        slice_inputs=args.slice_inputs,
    )

# Example usage:
# python examples/pytorch/continuous_batching.py --num-blocks 369 --max-batch-tokens 23 --attn sdpa_paged -mp none --samples 1 --displayed 0 --output-file sliced.json
