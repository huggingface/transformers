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
Model export CLI command.

Export a Transformers model to a deployment-friendly format.

Examples::

    # ONNX (requires: pip install optimum[exporters])
    transformers export onnx --model bert-base-uncased --output ./bert-onnx/

    # GGUF (for llama.cpp)
    transformers export gguf --model meta-llama/Llama-3.2-1B --output llama-1b.gguf

    # ExecuTorch (for mobile/edge; requires: pip install executorch)
    transformers export executorch --model distilbert-base-uncased --output ./model.pte

Supported formats: onnx, gguf, executorch.
"""

from typing import Annotated

import typer


_EXPORT_FORMATS = ("onnx", "gguf", "executorch")


def export(
    fmt: Annotated[str, typer.Argument(help=f"Export format: {', '.join(_EXPORT_FORMATS)}.")],
    model: Annotated[str, typer.Option("--model", "-m", help="Model ID or local path.")],
    output: Annotated[str, typer.Option(help="Output path (directory for ONNX, file for GGUF).")],
    opset: Annotated[int | None, typer.Option(help="ONNX opset version.")] = None,
    task: Annotated[str | None, typer.Option(help="Task for ONNX export (auto-detected if omitted).")] = None,
    trust_remote_code: Annotated[bool, typer.Option(help="Trust remote code.")] = False,
    token: Annotated[str | None, typer.Option(help="HF Hub token.")] = None,
):
    """
    Export a model to a deployment-friendly format.

    The first argument is the target format. Each format has different
    requirements and produces different output.

    Examples::

        transformers export onnx --model bert-base-uncased --output ./bert-onnx/
        transformers export gguf --model meta-llama/Llama-3.2-1B --output llama-1b.gguf
        transformers export executorch --model distilbert-base-uncased --output ./model.pte
    """
    if fmt not in _EXPORT_FORMATS:
        raise SystemExit(f"Unknown format '{fmt}'. Choose from: {', '.join(_EXPORT_FORMATS)}")

    if fmt == "onnx":
        _export_onnx(model, output, opset, task, trust_remote_code, token)
    elif fmt == "gguf":
        _export_gguf(model, output, trust_remote_code, token)
    elif fmt == "executorch":
        _export_executorch(model, output, trust_remote_code, token)


def _export_onnx(
    model: str, output: str, opset: int | None, task: str | None, trust_remote_code: bool, token: str | None
):
    """Export to ONNX via the optimum library."""
    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        raise SystemExit(
            "ONNX export requires the 'optimum' library.\nInstall it with: pip install optimum[exporters]"
        )

    export_kwargs = {
        "model_name_or_path": model,
        "output": output,
    }
    if opset is not None:
        export_kwargs["opset"] = opset
    if task is not None:
        export_kwargs["task"] = task
    if trust_remote_code:
        export_kwargs["trust_remote_code"] = True
    if token is not None:
        export_kwargs["token"] = token

    print(f"Exporting {model} to ONNX at {output}...")
    main_export(**export_kwargs)
    print(f"ONNX model saved to {output}")


def _export_gguf(model: str, output: str, trust_remote_code: bool, token: str | None):
    """Export to GGUF format."""
    from pathlib import Path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    common_kwargs = {}
    if trust_remote_code:
        common_kwargs["trust_remote_code"] = True
    if token:
        common_kwargs["token"] = token

    print(f"Loading {model}...")
    loaded_model = AutoModelForCausalLM.from_pretrained(model, **common_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model, **common_kwargs)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving as GGUF to {output}...")
    loaded_model.save_pretrained(output_path, gguf_file=output_path.name if output.endswith(".gguf") else None)
    tokenizer.save_pretrained(output_path)
    print(f"GGUF model saved to {output}")


def _export_executorch(model: str, output: str, trust_remote_code: bool, token: str | None):
    """Export to ExecuTorch format for mobile/edge deployment."""
    try:
        from executorch.exir import to_edge
        from torch.export import export as torch_export
    except ImportError:
        raise SystemExit(
            "ExecuTorch export requires the 'executorch' library.\nInstall it with: pip install executorch"
        )

    from pathlib import Path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    common_kwargs = {}
    if trust_remote_code:
        common_kwargs["trust_remote_code"] = True
    if token:
        common_kwargs["token"] = token

    print(f"Loading {model}...")
    loaded_model = AutoModelForCausalLM.from_pretrained(model, **common_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model, **common_kwargs)

    loaded_model.eval()

    # Trace with a dummy input
    dummy_input = tokenizer("Hello", return_tensors="pt")
    exported = torch_export(loaded_model, (dummy_input["input_ids"],))
    edge_program = to_edge(exported)
    et_program = edge_program.to_executorch()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    print(f"ExecuTorch model saved to {output}")
