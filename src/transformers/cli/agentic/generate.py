# Copyright 2025 The HuggingFace Team. All rights reserved.
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
Text generation CLI commands.

Uses ``AutoModelForCausalLM`` directly to expose the full set of generation
options: streaming, decoding strategies, speculative decoding, watermarking,
tool calling, constrained decoding, and quantization.
"""

from typing import Annotated

import typer

from ._common import resolve_input


def generate(
    # Input
    prompt: Annotated[str | None, typer.Option(help="Prompt text.")] = None,
    file: Annotated[str | None, typer.Option(help="Read prompt from this file.")] = None,
    # Model
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model ID or local path.")] = None,
    assistant_model: Annotated[str | None, typer.Option(help="Draft model for speculative/assisted decoding.")] = None,
    device: Annotated[str | None, typer.Option(help="Device (cpu, cuda, cuda:0, mps).")] = None,
    dtype: Annotated[str, typer.Option(help="Dtype: auto, float16, bfloat16, float32.")] = "auto",
    trust_remote_code: Annotated[bool, typer.Option(help="Trust remote code.")] = False,
    token: Annotated[str | None, typer.Option(help="HF Hub token.")] = None,
    revision: Annotated[str | None, typer.Option(help="Model revision.")] = None,
    # Generation parameters
    max_new_tokens: Annotated[int, typer.Option(help="Maximum new tokens to generate.")] = 256,
    temperature: Annotated[float | None, typer.Option(help="Sampling temperature.")] = None,
    top_k: Annotated[int | None, typer.Option(help="Top-k sampling.")] = None,
    top_p: Annotated[float | None, typer.Option(help="Top-p (nucleus) sampling.")] = None,
    num_beams: Annotated[int | None, typer.Option(help="Number of beams for beam search.")] = None,
    repetition_penalty: Annotated[float | None, typer.Option(help="Repetition penalty (1.0 = no penalty).")] = None,
    no_repeat_ngram_size: Annotated[int | None, typer.Option(help="Prevent repeating n-grams of this size.")] = None,
    do_sample: Annotated[bool | None, typer.Option(help="Use sampling instead of greedy decoding.")] = None,
    # Features
    stream: Annotated[bool, typer.Option(help="Stream output token-by-token.")] = False,
    watermark: Annotated[bool, typer.Option(help="Apply watermark to generated text.")] = False,
    tools: Annotated[str | None, typer.Option(help="Path to a JSON file defining tools for function calling.")] = None,
    grammar: Annotated[str | None, typer.Option(help="Constrain output format: 'json' for valid JSON output.")] = None,
    # Quantization
    quantization: Annotated[str | None, typer.Option(help="Load model quantized: 'bnb-4bit', 'bnb-8bit'.")] = None,
    cache_quantization: Annotated[str | None, typer.Option(help="Quantize KV cache: '4bit', '8bit'.")] = None,
):
    """
    Generate text from a prompt with full control over decoding.

    Loads a causal language model and generates text. Supports all major
    decoding strategies, streaming, speculative decoding, watermarking,
    tool calling, constrained decoding, and quantized inference.

    Examples::

        # Basic generation
        transformers generate --model meta-llama/Llama-3.2-1B-Instruct --prompt "Once upon a time"

        # Streaming output
        transformers generate --model meta-llama/Llama-3.2-1B-Instruct --prompt "Hello" --stream

        # Sampling with temperature and top-p
        transformers generate --prompt "The future of AI" --temperature 0.7 --top-p 0.9

        # Speculative decoding (faster inference with a draft model)
        transformers generate --model meta-llama/Llama-3.1-8B-Instruct \\
            --assistant-model meta-llama/Llama-3.2-1B-Instruct --prompt "Explain gravity."

        # Watermark generated text
        transformers generate --prompt "Write an essay." --watermark

        # Tool/function calling (provide tools as JSON)
        transformers generate --model meta-llama/Llama-3.2-1B-Instruct --prompt "What is the weather?" --tools tools.json

        # Constrained JSON output
        transformers generate --prompt "List 3 items as JSON:" --grammar json

        # 4-bit quantized inference
        transformers generate --model meta-llama/Llama-3.1-8B-Instruct --prompt "Hello" --quantization bnb-4bit

        # Quantized KV cache for long context
        transformers generate --model meta-llama/Llama-3.1-8B-Instruct --prompt "..." --cache-quantization 4bit
    """
    import json as json_mod

    from transformers import AutoModelForCausalLM, AutoTokenizer

    input_text = resolve_input(prompt, file)

    # --- Load model & tokenizer ---
    model_id = model or "HuggingFaceTB/SmolLM2-360M-Instruct"

    tok_kwargs = {}
    model_kwargs = {}
    if trust_remote_code:
        tok_kwargs["trust_remote_code"] = True
        model_kwargs["trust_remote_code"] = True
    if token:
        tok_kwargs["token"] = token
        model_kwargs["token"] = token
    if revision:
        tok_kwargs["revision"] = revision
        model_kwargs["revision"] = revision
    if device and device != "cpu":
        model_kwargs["device_map"] = device
    elif device is None:
        model_kwargs["device_map"] = "auto"
    if dtype != "auto":
        import torch

        model_kwargs["torch_dtype"] = getattr(torch, dtype)

    if quantization == "bnb-4bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    elif quantization == "bnb-8bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    loaded_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    loaded_model.eval()

    # --- Load assistant model for speculative decoding ---
    loaded_assistant = None
    if assistant_model is not None:
        loaded_assistant = AutoModelForCausalLM.from_pretrained(
            assistant_model,
            **{k: v for k, v in model_kwargs.items() if k != "quantization_config"},
        )

    # --- Build generation kwargs ---
    gen_kwargs = {"max_new_tokens": max_new_tokens}

    if temperature is not None:
        gen_kwargs["temperature"] = temperature
    if top_k is not None:
        gen_kwargs["top_k"] = top_k
    if top_p is not None:
        gen_kwargs["top_p"] = top_p
    if num_beams is not None:
        gen_kwargs["num_beams"] = num_beams
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if no_repeat_ngram_size is not None:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    if do_sample is not None:
        gen_kwargs["do_sample"] = do_sample
    elif temperature is not None or top_k is not None or top_p is not None:
        gen_kwargs["do_sample"] = True

    if watermark:
        from transformers import WatermarkingConfig

        gen_kwargs["watermarking_config"] = WatermarkingConfig()

    if cache_quantization is not None:
        from transformers import QuantizedCacheConfig

        nbits = 4 if "4" in cache_quantization else 8
        gen_kwargs["cache_implementation"] = "quantized"
        gen_kwargs["cache_config"] = QuantizedCacheConfig(nbits=nbits)

    if loaded_assistant is not None:
        gen_kwargs["assistant_model"] = loaded_assistant

    # --- Constrained decoding ---
    if grammar == "json":
        from transformers import GrammarConstrainedLogitsProcessor, LogitsProcessorList

        gen_kwargs.setdefault("logits_processor", LogitsProcessorList())
        gen_kwargs["logits_processor"].append(
            GrammarConstrainedLogitsProcessor(tokenizer=tokenizer, grammar_str='root ::= "{" [^}]* "}"')
        )

    # --- Tokenize (with tool calling via chat template if needed) ---
    if tools is not None:
        with open(tools) as f:
            tools_def = json_mod.load(f)
        messages = [{"role": "user", "content": input_text}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tools=tools_def,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
    else:
        inputs = tokenizer(input_text, return_tensors="pt")

    if hasattr(loaded_model, "device"):
        inputs = inputs.to(loaded_model.device)

    # --- Generate ---
    if stream:
        from transformers import TextStreamer

        streamer = TextStreamer(tokenizer, skip_prompt=True)
        gen_kwargs["streamer"] = streamer
        loaded_model.generate(**inputs, **gen_kwargs)
        print()
    else:
        output_ids = loaded_model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        print(tokenizer.decode(new_tokens, skip_special_tokens=True))


def detect_watermark(
    text: Annotated[str | None, typer.Option(help="Text to check for watermark.")] = None,
    file: Annotated[str | None, typer.Option(help="Read text from this file.")] = None,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Model ID (must match the model that generated the text).")
    ] = None,
):
    """
    Detect whether text contains a watermark.

    The ``--model`` must match the model that originally generated the text
    (the watermark is tied to the model's vocabulary and config).

    Example::

        transformers detect-watermark --model meta-llama/Llama-3.2-1B-Instruct --text "The generated essay text..."
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, WatermarkDetector

    input_text = resolve_input(text, file)
    model_id = model or "HuggingFaceTB/SmolLM2-360M-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    detector = WatermarkDetector(
        model_config=AutoModelForCausalLM.from_pretrained(model_id).config,
        device="cpu",
    )

    tokens = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    result = detector(tokens)

    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.4f}")
