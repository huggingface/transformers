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
import logging
import os
import time
from itertools import cycle

import datasets
import torch
from torch.profiler import ProfilerActivity, profile
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig
from transformers.generation import GenerationConfig
from transformers.generation.continuous_batching.requests import logger


def generate_without_cb(
    model_id: str, sliding_window: int, attn_impl: str, batched_inputs: list[int], generation_config: GenerationConfig
) -> dict[str, str]:
    # Setup model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation=attn_impl)
    model = model.cuda().eval()
    if sliding_window > 0 and getattr(model.config, "sliding_window", None) is not None:
        model.config.sliding_window = sliding_window
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Generate one by one
    decoded_outputs = {}
    for input_ids in tqdm(batched_inputs, desc="Generating outputs without CB"):
        key = " ".join(map(str, input_ids))  # This will be used to identify the output after batched generation
        input_ids = torch.tensor([input_ids]).to("cuda")
        attention_mask = torch.ones_like(input_ids)
        outputs = model.generate(input_ids, attention_mask=attention_mask, generation_config=generation_config)
        generated_tokens = outputs[0][input_ids.shape[1] :]
        decoded_outputs[key] = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    return decoded_outputs


def maybe_setup_metrics(use_metrics: bool) -> None:
    if not use_metrics:
        return
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
    output_file: str | None = None,
    expected_outputs: list[str] | None = None,
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
        input_text = tokenizer.decode(batch_outputs[request].prompt_ids, skip_special_tokens=False)
        # The key is used to tie back to the output of unbatched generation
        key = " ".join(map(str, batch_outputs[request].prompt_ids))
        data.append({"input": input_text, "key": key})

        # Try to decode the output
        try:
            output_text = tokenizer.decode(batch_outputs[request].generated_tokens, skip_special_tokens=False)
            token_count += len(batch_outputs[request].generated_tokens[1:])
            data[-1]["cb_outputs"] = output_text
        except Exception as e:
            print(f"Decoding failed for request {request}: {e}")
            data[-1]["cb_outputs"] = "__ERROR__"
            continue

        # Display sample if asked
        if i < displayed_samples:
            print("-" * 20, f"{request} Input:  {input_text}", f"{request} Output: {output_text}", sep="\n")

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
    parser = argparse.ArgumentParser()

    # Continuous batching parameters
    parser.add_argument("--num-blocks", "-n", type=int, default=None)
    parser.add_argument("--max-batch-tokens", "-b", type=int, default=None)

    # Model parameters
    parser.add_argument("--sliding-window", type=int, default=0)
    parser.add_argument("--attn", type=str, default=None, help="Attention implementation")

    # Performance parameters
    parser.add_argument("--matmul-precision", "-mp", type=str, default="high")  # set to "none" to disable
    parser.add_argument("--cuda-graph", "-cg", help="Use cuda graphs", type=str, default=None)
    parser.add_argument("--compile", action="store_true", help="Compile the model using torch.compile")
    parser.add_argument("--do-sample", action="store_true", help="Activate sampling")
    parser.add_argument("--num-return-sequences", type=int, default=1, help="Number of return sequences")

    # Benchmark parameters
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to generate")
    parser.add_argument(
        "--input-length", type=int, default=None, help="Length of input sequences. Leave to None to mimic real eval."
    )
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate")
    parser.add_argument("--force-max-length", action="store_true", help="Force generation to stop at max length")

    parser.add_argument("--add-prefix", action="store_true", help="Add a prefix to the samples")
    parser.add_argument("--compare", action="store_true", help="Compare CB generation with classic generate")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # Display parameters
    parser.add_argument("--displayed", type=int, default=0, help="Number of samples to display")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--output-file", type=str, default=None)

    args = parser.parse_args()

    # Choose attention implementation
    if args.attn is None:
        if args.compile:
            args.attn = "kernels-community/flash-attn3@fake-ops-return-probs"
            logger.warning(
                "No attention implementation was provided and compile is enabled. Using experimental kernel: "
                "kernels-community/flash-attn3@fake-ops-return-probs because compile is not supported on main. Change "
                "this when main supports it."  # TODO: cf comment
            )
        else:
            args.attn = "kernels-community/flash-attn3"

    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Create model
    model_id = "google/gemma-2-2b-it" if args.sliding_window > 0 else "meta-llama/Llama-3.1-8B-Instruct"
    has_system_role = args.sliding_window == 0

    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation=args.attn, dtype=torch.bfloat16)
    model = model.cuda().eval()

    if args.sliding_window > 0 and getattr(model.config, "sliding_window", None) is not None:
        print(f"Setting sliding window from {model.config.sliding_window} to {args.sliding_window}")
        model.config.sliding_window = args.sliding_window

    # Set up diagnostics
    logger.setLevel(args.log_level.upper())
    maybe_setup_metrics(args.metrics)

    # Set up performance
    if args.matmul_precision != "none":
        torch.set_float32_matmul_precision(args.matmul_precision)

    cuda_graph_arg = args.cuda_graph.lower() if args.cuda_graph is not None else None
    use_cuda_graph = {
        "none": None, None: None,
        "yes": True, "y": True, "true": True, "t": True, "1": True,
        "no": False, "n": False, "false": False, "f": False, "0": False,
    }[cuda_graph_arg]  # fmt: skip

    # Prepare tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

    dataset = datasets.load_dataset("openai/gsm8k", "socratic", split="test")
    dataset = dataset.select(range(args.samples))

    if args.add_prefix:
        possible_prefixes = [
            None,
            "You are a bot that solves math problems.",
            "You are a bot who solves math problems. Try to make your answer clear and understandable, and include your stages of reasoning.",
            "You are a bot with the aim to solves math problems. Try to make your answer clear and understandable, and include your stages of reasoning. No loud words or emojis, all responses must be readable by a child. Here is now the problem:",
        ]  # fmt: skip
    else:
        possible_prefixes = [None]

    tokenizer_kwargs = {"add_generation_prompt": True}
    if args.input_length is not None:
        tokenizer_kwargs["max_length"] = args.input_length
        tokenizer_kwargs["truncation"] = True
        tokenizer_kwargs["padding"] = True
        tokenizer.pad_token_id = tokenizer.eos_token_id

    batched_inputs = []
    for item, prefix in zip(dataset, cycle(possible_prefixes)):
        messages = []
        question = item["question"]
        if prefix is not None:
            if has_system_role:
                messages.append({"role": "system", "content": prefix})
            else:
                question = prefix + "\n\n" + question
        messages.append({"role": "user", "content": question})
        inputs = tokenizer.apply_chat_template(messages, **tokenizer_kwargs)
        inputs = inputs if isinstance(inputs, list) else inputs["input_ids"]
        batched_inputs.append(inputs)

    # If num_return_sequences > 1, automatically enable do_sample with a warning
    do_sample = args.do_sample
    if args.num_return_sequences != 1 and not args.do_sample:
        logger.warning(
            f"num_return_sequences={args.num_return_sequences} > 1, automatically enabling do_sample=True. "
            "Set --do-sample explicitly to suppress this warning."
        )
        do_sample = True

    # Prepare generation config
    generation_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        use_cuda_graph=use_cuda_graph,
        eos_token_id=tokenizer.pad_token_id if args.force_max_length else tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=do_sample,
        temperature=0.8,
        top_p=0.9,
        num_blocks=args.num_blocks,
        max_batch_tokens=args.max_batch_tokens,
        num_return_sequences=args.num_return_sequences,
    )

    # Add a compile config if requested
    if args.compile:
        generation_cfg.compile_config = CompileConfig(
            fullgraph=True,
            mode="max-autotune-no-cudagraphs",
            dynamic=True,  # FIXME: if we warmup all graphs, this is not needed anymore
        )

    # If we need to compare, we need to generate the reference outputs
    if args.compare:
        expected_outputs = generate_without_cb(
            model_id, args.sliding_window, args.attn, batched_inputs, generation_cfg
        )
    else:
        expected_outputs = None

    # If no output file is provided, we pick a name based on the args
    if args.output_file is None:
        os.makedirs("runs/cb", exist_ok=True)
        attn = args.attn.replace("|", "_").replace("/", "_")
        args.output_file = (
            f"runs/cb/{args.num_blocks}_{args.max_batch_tokens}_{attn}_{args.matmul_precision}_{args.samples}.json"
        )

    # Run warmup batch generation if log level is above DEBUG # TODO: understand why warmup incurs a large overhead during cache creation
    if logger.level > logging.DEBUG:
        batch_generate(
            model,
            batched_inputs[: min(5, args.samples)],
            generation_cfg,
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
            batched_inputs,
            generation_cfg,
            tokenizer,
            displayed_samples=args.displayed,
            output_file=args.output_file,
            expected_outputs=expected_outputs,
        )
    if args.profile is not None:
        filename = args.profile if args.profile.endswith(".json") else args.profile + ".json"
        prof.export_chrome_trace(filename)

# Example usage:
# python examples/pytorch/continuous_batching.py --attn sdpa --add-prefix --samples 10 --compare
# python examples/pytorch/continuous_batching.py --attn flash_attention_2 -mp none --add-prefix --samples 500
# python examples/pytorch/continuous_batching.py -mp none -cg yes --samples 10 --max-new-tokens 32 --profile profile_wip.json
