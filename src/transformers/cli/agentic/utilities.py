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
Utility CLI commands for model exploration and analysis.

Commands in this module don't run inference or training — they inspect
models, tokenizers, embeddings, and activations. Useful for debugging,
prototyping, and understanding model behavior.
"""

import json
from typing import Annotated

import typer

from ._common import _load_pretrained, load_image, resolve_input


def embed(
    # Text input
    text: Annotated[str | None, typer.Option(help="Text to embed.")] = None,
    file: Annotated[str | None, typer.Option(help="Read text from this file.")] = None,
    # Image input
    image: Annotated[str | None, typer.Option(help="Path or URL to an image to embed.")] = None,
    # Model & output
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model ID or local path.")] = None,
    output: Annotated[str | None, typer.Option(help="Save embeddings to this file (.npy or .json).")] = None,
    device: Annotated[str | None, typer.Option(help="Device.")] = None,
    dtype: Annotated[str, typer.Option(help="Dtype.")] = "auto",
    trust_remote_code: Annotated[bool, typer.Option(help="Trust remote code.")] = False,
    token: Annotated[str | None, typer.Option(help="HF Hub token.")] = None,
    revision: Annotated[str | None, typer.Option(help="Model revision.")] = None,
):
    """
    Compute embeddings for text or images.

    Uses ``AutoModel`` with ``AutoTokenizer`` (text) or
    ``AutoImageProcessor`` (images). Outputs shape and a preview by
    default. Pass ``--output`` to save as ``.npy`` (NumPy) or ``.json``.

    Examples::

        # Text embeddings
        transformers embed --model BAAI/bge-small-en-v1.5 --text "The quick brown fox." --output embeddings.npy

        # Image embeddings
        transformers embed --model facebook/dinov2-small --image photo.jpg --output features.npy

        # Quick preview (no file saved)
        transformers embed --text "Hello world"
    """
    import numpy as np
    import torch

    from transformers import AutoModel

    if image is not None:
        from transformers import AutoImageProcessor

        model_id = model or "facebook/dinov2-small"
        loaded_model, processor = _load_pretrained(
            AutoModel, AutoImageProcessor, model_id, device, dtype, trust_remote_code, token, revision
        )
        img = load_image(image)
        inputs = processor(images=img, return_tensors="pt")
    elif text is not None or file is not None:
        from transformers import AutoTokenizer

        model_id = model or "BAAI/bge-small-en-v1.5"
        loaded_model, tokenizer = _load_pretrained(
            AutoModel, AutoTokenizer, model_id, device, dtype, trust_remote_code, token, revision
        )
        input_text = resolve_input(text, file)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    else:
        raise SystemExit("Error: provide --text, --file, or --image.")

    if hasattr(loaded_model, "device"):
        inputs = inputs.to(loaded_model.device)

    with torch.no_grad():
        outputs = loaded_model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)[0].cpu().numpy()

    if output is not None:
        if output.endswith(".npy"):
            np.save(output, embedding)
        elif output.endswith(".json"):
            with open(output, "w") as f:
                json.dump(embedding.tolist(), f)
        else:
            np.save(output, embedding)
        print(f"Embedding shape {embedding.shape} saved to {output}")
    else:
        print(f"Embedding shape: {embedding.shape}")
        flat = embedding.flatten()
        preview = ", ".join(f"{v:.6f}" for v in flat[:8])
        if len(flat) > 8:
            preview += ", ..."
        print(f"Values: [{preview}]")


def tokenize(
    text: Annotated[str | None, typer.Option(help="Text to tokenize.")] = None,
    file: Annotated[str | None, typer.Option(help="Read text from this file.")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model ID or local path.")] = None,
    token: Annotated[str | None, typer.Option(help="HF Hub token.")] = None,
    trust_remote_code: Annotated[bool, typer.Option(help="Trust remote code.")] = False,
    show_ids: Annotated[bool, typer.Option("--ids", help="Show token IDs.")] = False,
    output_json: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
):
    """
    Tokenize text and display the resulting tokens.

    Shows how the model's tokenizer breaks text into subword tokens.
    Useful for debugging prompt formatting, checking token counts, and
    understanding tokenizer behavior.

    Examples::

        transformers tokenize --model meta-llama/Llama-3.2-1B-Instruct --text "Hello, world!"
        transformers tokenize --model meta-llama/Llama-3.2-1B-Instruct --text "Hello, world!" --ids
        transformers tokenize --model bert-base-uncased --text "Tokenization is fun." --json
    """
    from transformers import AutoTokenizer

    input_text = resolve_input(text, file)
    model_id = model or "HuggingFaceTB/SmolLM2-360M-Instruct"

    tok_kwargs = {}
    if token is not None:
        tok_kwargs["token"] = token
    if trust_remote_code:
        tok_kwargs["trust_remote_code"] = True

    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    encoding = tokenizer(input_text)

    token_ids = encoding["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    if output_json:
        data = {"tokens": tokens, "token_ids": token_ids, "num_tokens": len(tokens)}
        print(json.dumps(data, indent=2))
    else:
        print(f"Tokens ({len(tokens)}):")
        for i, (tok, tid) in enumerate(zip(tokens, token_ids)):
            if show_ids:
                print(f"  {i:4d}  {tid:8d}  {tok!r}")
            else:
                print(f"  {i:4d}  {tok!r}")


def inspect(
    model: Annotated[str, typer.Argument(help="Model ID or local path to inspect.")],
    token: Annotated[str | None, typer.Option(help="HF Hub token.")] = None,
    trust_remote_code: Annotated[bool, typer.Option(help="Trust remote code.")] = False,
    output_json: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
):
    """
    Inspect a model's configuration without downloading weights.

    Shows architecture, hidden size, number of layers, vocabulary size,
    and other key config values. Use ``--json`` for the full config dict.

    Examples::

        transformers inspect meta-llama/Llama-3.2-1B-Instruct
        transformers inspect meta-llama/Llama-3.2-1B-Instruct --json
    """
    from transformers import AutoConfig

    kwargs = {}
    if token is not None:
        kwargs["token"] = token
    if trust_remote_code:
        kwargs["trust_remote_code"] = True

    config = AutoConfig.from_pretrained(model, **kwargs)

    if output_json:
        print(json.dumps(config.to_dict(), indent=2, default=str))
    else:
        config_dict = config.to_dict()
        print(f"Model: {model}")
        print(f"Architecture: {config_dict.get('architectures', ['unknown'])}")
        print(f"Model type: {config_dict.get('model_type', 'unknown')}")
        print()

        important_keys = [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "vocab_size",
            "max_position_embeddings",
            "hidden_act",
            "torch_dtype",
        ]
        for key in important_keys:
            if key in config_dict:
                print(f"  {key}: {config_dict[key]}")

        remaining = {
            k: v
            for k, v in config_dict.items()
            if k not in important_keys and k not in ("architectures", "model_type", "transformers_version")
        }
        if remaining:
            print(f"\n  ({len(remaining)} additional config keys — use --json for full output)")


def inspect_forward(
    text: Annotated[str, typer.Option(help="Text to run through the model.")],
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model ID or local path.")] = None,
    output: Annotated[str | None, typer.Option(help="Directory to save activations as .npy files.")] = None,
    layers: Annotated[
        str | None, typer.Option(help="Comma-separated layer indices to inspect (default: all).")
    ] = None,
    token: Annotated[str | None, typer.Option(help="HF Hub token.")] = None,
    trust_remote_code: Annotated[bool, typer.Option(help="Trust remote code.")] = False,
    output_json: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
):
    """
    Examine attention weights and hidden states from a forward pass.

    Runs the input through the model with ``output_attentions=True`` and
    ``output_hidden_states=True``, then prints shape and statistics for
    each layer. Pass ``--output ./activations/`` to save attention and
    hidden state tensors as NumPy ``.npy`` files for further analysis.

    Examples::

        # Print summary for all layers
        transformers inspect-forward --model bert-base-uncased --text "The cat sat on the mat."

        # Inspect only layers 0 and 11, save to disk
        transformers inspect-forward --model bert-base-uncased --text "Hello world" --layers 0,11 --output ./activations/
    """
    import numpy as np

    from transformers import AutoModel, AutoTokenizer

    model_id = model or "answerdotai/ModernBERT-base"

    common_kwargs = {}
    if token is not None:
        common_kwargs["token"] = token
    if trust_remote_code:
        common_kwargs["trust_remote_code"] = True

    tokenizer = AutoTokenizer.from_pretrained(model_id, **common_kwargs)
    loaded_model = AutoModel.from_pretrained(model_id, **common_kwargs)
    loaded_model.eval()

    inputs = tokenizer(text, return_tensors="pt")
    import torch

    with torch.no_grad():
        outputs = loaded_model(**inputs, output_attentions=True, output_hidden_states=True)

    attentions = outputs.attentions
    hidden_states = outputs.hidden_states

    layer_indices = None
    if layers is not None:
        layer_indices = [int(i) for i in layers.split(",")]

    print(f"Model: {model_id}")
    print(f"Input tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
    print(f"Hidden state layers: {len(hidden_states)} (including embedding layer)")
    print(f"Attention layers: {len(attentions)}")

    for i, (attn, hs) in enumerate(zip(attentions, hidden_states[1:])):
        if layer_indices is not None and i not in layer_indices:
            continue
        print(f"\n  Layer {i}:")
        print(f"    Attention shape: {list(attn.shape)} (batch, heads, seq, seq)")
        print(f"    Hidden state shape: {list(hs.shape)} (batch, seq, hidden)")
        attn_np = attn[0].cpu().numpy()
        print(f"    Attention mean: {attn_np.mean():.6f}, max: {attn_np.max():.6f}")
        hs_np = hs[0].cpu().numpy()
        print(f"    Hidden state norm (mean): {np.linalg.norm(hs_np, axis=-1).mean():.4f}")

    if output is not None:
        from pathlib import Path

        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, attn in enumerate(attentions):
            if layer_indices is not None and i not in layer_indices:
                continue
            np.save(out_dir / f"attention_layer_{i}.npy", attn[0].cpu().numpy())
        for i, hs in enumerate(hidden_states):
            if layer_indices is not None and i not in layer_indices and i > 0:
                continue
            np.save(out_dir / f"hidden_state_layer_{i}.npy", hs[0].cpu().numpy())
        print(f"\nActivations saved to {output}")


def benchmark_quantization(
    model: Annotated[str, typer.Option("--model", "-m", help="Model ID or local path.")],
    methods: Annotated[
        str, typer.Option(help="Comma-separated quantization methods to compare: none, bnb-4bit, bnb-8bit.")
    ] = "bnb-4bit,bnb-8bit",
    prompt: Annotated[
        str, typer.Option(help="Prompt to use for benchmarking.")
    ] = "The quick brown fox jumps over the lazy dog.",
    max_new_tokens: Annotated[int, typer.Option(help="Tokens to generate per run.")] = 50,
    trust_remote_code: Annotated[bool, typer.Option(help="Trust remote code.")] = False,
    token: Annotated[str | None, typer.Option(help="HF Hub token.")] = None,
    output_json: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
):
    """
    Compare quality and performance across quantization methods.

    Loads the same model under each quantization method, generates text,
    and reports tokens/sec, latency, peak GPU memory, and a preview of
    the output. Use ``none`` as a method to include the unquantized
    baseline.

    Examples::

        # Compare 4-bit vs 8-bit
        transformers benchmark-quantization --model meta-llama/Llama-3.1-8B --methods bnb-4bit,bnb-8bit

        # Include unquantized baseline, output as JSON
        transformers benchmark-quantization --model meta-llama/Llama-3.1-8B --methods none,bnb-4bit,bnb-8bit --json
    """
    import time

    from transformers import AutoModelForCausalLM, AutoTokenizer

    common_kwargs = {}
    if trust_remote_code:
        common_kwargs["trust_remote_code"] = True
    if token:
        common_kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(model, **common_kwargs)
    method_list = [m.strip() for m in methods.split(",")]

    results = []
    for method in method_list:
        print(f"\n--- {method} ---")
        model_kwargs = {**common_kwargs, "device_map": "auto"}

        if method == "bnb-4bit":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        elif method == "bnb-8bit":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif method == "none":
            pass
        else:
            print(f"  Skipping {method} — only none, bnb-4bit, bnb-8bit are supported for benchmarking.")
            continue

        try:
            loaded_model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
            loaded_model.eval()
            inputs = tokenizer(prompt, return_tensors="pt").to(loaded_model.device)

            # Warmup
            loaded_model.generate(**inputs, max_new_tokens=5)

            # Timed run
            start = time.time()
            output_ids = loaded_model.generate(**inputs, max_new_tokens=max_new_tokens)
            elapsed = time.time() - start

            new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            tokens_per_sec = len(new_tokens) / elapsed

            import torch

            mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

            result = {
                "method": method,
                "tokens_per_sec": round(tokens_per_sec, 2),
                "time_sec": round(elapsed, 3),
                "peak_memory_mb": round(mem_mb, 1),
                "output_preview": generated_text[:100],
            }
            results.append(result)

            print(f"  Tokens/sec: {tokens_per_sec:.2f}")
            print(f"  Time: {elapsed:.3f}s")
            if mem_mb > 0:
                print(f"  Peak memory: {mem_mb:.1f} MB")
            print(f"  Output: {generated_text[:100]}...")

            del loaded_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

        except Exception as e:
            print(f"  Error: {e}")
            results.append({"method": method, "error": str(e)})

    if output_json:
        print(json.dumps(results, indent=2))
