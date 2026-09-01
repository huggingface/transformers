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
Quantization CLI command.

Quantize a model and save the result locally or push to the Hub.

Examples::

    # BitsAndBytes 4-bit (NF4)
    transformers quantize --model meta-llama/Llama-3.1-8B --method bnb-4bit --output ./llama-4bit

    # GPTQ with calibration data
    transformers quantize --model meta-llama/Llama-3.1-8B --method gptq --calibration-dataset wikitext --output ./llama-gptq

    # AWQ
    transformers quantize --model meta-llama/Llama-3.1-8B --method awq --output ./llama-awq

Supported methods: bnb-4bit, bnb-8bit, gptq, awq.
"""

from typing import Annotated

import typer


_QUANTIZATION_METHODS = ("bnb-4bit", "bnb-8bit", "gptq", "awq")


def quantize(
    model: Annotated[str, typer.Option("--model", "-m", help="Model ID or local path to quantize.")],
    method: Annotated[str, typer.Option(help=f"Quantization method: {', '.join(_QUANTIZATION_METHODS)}.")],
    output: Annotated[str, typer.Option(help="Output directory for the quantized model.")],
    calibration_dataset: Annotated[
        str | None, typer.Option(help="Calibration dataset for GPTQ/AWQ (Hub name or local path).")
    ] = None,
    calibration_samples: Annotated[int, typer.Option(help="Number of calibration samples.")] = 128,
    bits: Annotated[int, typer.Option(help="Target bit width (for GPTQ/AWQ).")] = 4,
    group_size: Annotated[int, typer.Option(help="Group size for GPTQ/AWQ.")] = 128,
    device: Annotated[str | None, typer.Option(help="Device for quantization.")] = None,
    trust_remote_code: Annotated[bool, typer.Option(help="Trust remote code.")] = False,
    token: Annotated[str | None, typer.Option(help="HF Hub token.")] = None,
    push_to_hub: Annotated[bool, typer.Option(help="Push quantized model to Hub.")] = False,
    hub_model_id: Annotated[str | None, typer.Option(help="Hub repo ID for push.")] = None,
):
    """
    Quantize a model and save it.

    Loads the model with the specified quantization method and saves the
    quantized weights. For GPTQ and AWQ, a calibration dataset is used
    to determine optimal quantization parameters.

    Examples::

        transformers quantize --model meta-llama/Llama-3.1-8B --method bnb-4bit --output ./llama-4bit
        transformers quantize --model meta-llama/Llama-3.1-8B --method gptq --calibration-dataset wikitext --output ./llama-gptq
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if method not in _QUANTIZATION_METHODS:
        raise SystemExit(f"Unknown method '{method}'. Choose from: {', '.join(_QUANTIZATION_METHODS)}")

    common_kwargs = {}
    if trust_remote_code:
        common_kwargs["trust_remote_code"] = True
    if token:
        common_kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(model, **common_kwargs)

    model_kwargs = {**common_kwargs}
    if device:
        model_kwargs["device_map"] = device
    else:
        model_kwargs["device_map"] = "auto"

    # --- BitsAndBytes ---
    if method == "bnb-4bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4",
        )
        print(f"Loading {model} in 4-bit (BitsAndBytes NF4)...")
        loaded_model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        loaded_model.save_pretrained(output)
        tokenizer.save_pretrained(output)
        print(f"Quantized model saved to {output}")

    elif method == "bnb-8bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        print(f"Loading {model} in 8-bit (BitsAndBytes)...")
        loaded_model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        loaded_model.save_pretrained(output)
        tokenizer.save_pretrained(output)
        print(f"Quantized model saved to {output}")

    # --- GPTQ ---
    elif method == "gptq":
        from transformers import GPTQConfig

        if calibration_dataset is None:
            calibration_dataset = "wikitext"
            print("No --calibration-dataset specified, defaulting to 'wikitext'.")

        from datasets import load_dataset

        cal_ds = load_dataset(calibration_dataset, split=f"train[:{calibration_samples}]")
        cal_texts = [ex["text"] for ex in cal_ds if ex.get("text")]

        quantization_config = GPTQConfig(
            bits=bits,
            group_size=group_size,
            dataset=cal_texts,
            tokenizer=tokenizer,
        )
        model_kwargs["quantization_config"] = quantization_config

        print(f"Quantizing {model} with GPTQ ({bits}-bit, group_size={group_size})...")
        loaded_model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        loaded_model.save_pretrained(output)
        tokenizer.save_pretrained(output)
        print(f"GPTQ-quantized model saved to {output}")

    # --- AWQ ---
    elif method == "awq":
        from transformers import AwqConfig

        quantization_config = AwqConfig(
            bits=bits,
            group_size=group_size,
        )
        model_kwargs["quantization_config"] = quantization_config

        print(f"Quantizing {model} with AWQ ({bits}-bit, group_size={group_size})...")
        loaded_model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        loaded_model.save_pretrained(output)
        tokenizer.save_pretrained(output)
        print(f"AWQ-quantized model saved to {output}")

    if push_to_hub:
        repo_id = hub_model_id or output
        loaded_model.push_to_hub(repo_id, token=token)
        tokenizer.push_to_hub(repo_id, token=token)
        print(f"Pushed to Hub: {repo_id}")
