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
Multimodal CLI commands for the transformers agentic CLI.

All commands use Auto* model classes directly (no pipeline abstraction).
Imports of ``torch`` and ``transformers`` are deferred to function bodies
for fast CLI startup.
"""

from typing import Annotated

import typer

from ._common import (
    DeviceOpt,
    DtypeOpt,
    JsonOpt,
    ModelOpt,
    RevisionOpt,
    TokenOpt,
    TrustOpt,
    _load_pretrained,
    format_output,
    load_audio,
    load_image,
)


def vqa(
    image: Annotated[str, typer.Option(help="Path or URL to the image.")],
    question: Annotated[str, typer.Option(help="Question about the image.")],
    model: ModelOpt = None,
    max_new_tokens: Annotated[int, typer.Option(help="Maximum tokens to generate.")] = 256,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Visual question answering using ``AutoModelForImageTextToText``.

    Provide an image and a natural-language question; the model returns an
    answer grounded in the visual content.

    Example::

        transformers vqa --image photo.jpg --question "What color is the car?"
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = model or "vikhyatk/moondream2"
    loaded_model, processor = _load_pretrained(
        AutoModelForImageTextToText, AutoProcessor, model_id, device, dtype, trust_remote_code, token, revision
    )

    img = load_image(image)
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}]

    inputs = processor.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True)
    if hasattr(loaded_model, "device"):
        inputs = inputs.to(loaded_model.device)

    output_ids = loaded_model.generate(**inputs, max_new_tokens=max_new_tokens)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    result = processor.decode(new_tokens, skip_special_tokens=True)

    if output_json:
        print(format_output({"answer": result}, output_json=True))
    else:
        print(result)


def document_qa(
    image: Annotated[str, typer.Option(help="Path or URL to the document image.")],
    question: Annotated[str, typer.Option(help="Question about the document.")],
    model: ModelOpt = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Extractive document question answering using
    ``AutoModelForDocumentQuestionAnswering``.

    The model reads a document image and extracts a span of text that
    answers the given question.

    Example::

        transformers document-qa --image receipt.png --question "What is the total?"
    """
    import torch

    from transformers import AutoModelForDocumentQuestionAnswering, AutoProcessor

    model_id = model or "impira/layoutlm-document-qa"
    loaded_model, processor = _load_pretrained(
        AutoModelForDocumentQuestionAnswering,
        AutoProcessor,
        model_id,
        device,
        dtype,
        trust_remote_code,
        token,
        revision,
    )

    img = load_image(image)
    inputs = processor(images=img, question=question, return_tensors="pt")
    if hasattr(loaded_model, "device"):
        inputs = inputs.to(loaded_model.device)

    with torch.no_grad():
        outputs = loaded_model(**inputs)

    start_idx = outputs.start_logits.argmax(dim=-1).item()
    end_idx = outputs.end_logits.argmax(dim=-1).item()
    answer = processor.tokenizer.decode(inputs["input_ids"][0, start_idx : end_idx + 1], skip_special_tokens=True)

    result = {"answer": answer, "start": start_idx, "end": end_idx}
    if output_json:
        print(format_output(result, output_json=True))
    else:
        print(format_output(result))


def caption(
    image: Annotated[str, typer.Option(help="Path or URL to the image.")],
    model: ModelOpt = None,
    max_new_tokens: Annotated[int, typer.Option(help="Maximum tokens to generate.")] = 64,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Generate a caption for an image using ``AutoModelForImageTextToText``.

    Example::

        transformers caption --image photo.jpg
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = model or "vikhyatk/moondream2"
    loaded_model, processor = _load_pretrained(
        AutoModelForImageTextToText, AutoProcessor, model_id, device, dtype, trust_remote_code, token, revision
    )

    img = load_image(image)
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": img}, {"type": "text", "text": "Describe this image."}],
        }
    ]

    inputs = processor.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True)
    if hasattr(loaded_model, "device"):
        inputs = inputs.to(loaded_model.device)

    output_ids = loaded_model.generate(**inputs, max_new_tokens=max_new_tokens)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    result = processor.decode(new_tokens, skip_special_tokens=True)

    if output_json:
        print(format_output({"caption": result}, output_json=True))
    else:
        print(result)


def ocr(
    image: Annotated[str, typer.Option(help="Path or URL to the document image.")],
    model: ModelOpt = None,
    max_new_tokens: Annotated[int, typer.Option(help="Maximum tokens to generate.")] = 512,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Extract text from an image using ``AutoModelForImageTextToText``.

    Example::

        transformers ocr --image scanned_page.png
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = model or "vikhyatk/moondream2"
    loaded_model, processor = _load_pretrained(
        AutoModelForImageTextToText, AutoProcessor, model_id, device, dtype, trust_remote_code, token, revision
    )

    img = load_image(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Extract all text from this image."},
            ],
        }
    ]

    inputs = processor.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True)
    if hasattr(loaded_model, "device"):
        inputs = inputs.to(loaded_model.device)

    output_ids = loaded_model.generate(**inputs, max_new_tokens=max_new_tokens)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    result = processor.decode(new_tokens, skip_special_tokens=True)

    if output_json:
        print(format_output({"text": result}, output_json=True))
    else:
        print(result)


def multimodal_chat(
    prompt: Annotated[str, typer.Option(help="Text prompt for the conversation.")],
    model: Annotated[str, typer.Option("--model", "-m", help="Model ID or local path.")],
    image: Annotated[str | None, typer.Option(help="Path or URL to an image.")] = None,
    audio: Annotated[str | None, typer.Option(help="Path to an audio file.")] = None,
    max_new_tokens: Annotated[int, typer.Option(help="Maximum tokens to generate.")] = 256,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
):
    """
    Single-turn conversation with a model that accepts mixed inputs.

    Provide any combination of ``--image``, ``--audio``, and ``--prompt``.
    The model must support the input modalities you provide.

    Example::

        transformers multimodal-chat --model meta-llama/Llama-4-Scout-17B-16E-Instruct \\
            --prompt "Describe what you see and hear." --image photo.jpg --audio clip.wav
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    common_kwargs = {}
    if trust_remote_code:
        common_kwargs["trust_remote_code"] = True
    if token:
        common_kwargs["token"] = token
    if revision:
        common_kwargs["revision"] = revision

    processor = AutoProcessor.from_pretrained(model, **common_kwargs)

    model_kwargs = {**common_kwargs}
    if device and device != "cpu":
        model_kwargs["device_map"] = device
    elif device is None:
        model_kwargs["device_map"] = "auto"
    if dtype != "auto":
        import torch

        model_kwargs["torch_dtype"] = getattr(torch, dtype)

    loaded_model = AutoModelForImageTextToText.from_pretrained(model, **model_kwargs)
    loaded_model.eval()

    # Build multimodal message content
    content = []
    if image is not None:
        img = load_image(image)
        content.append({"type": "image", "image": img})
    if audio is not None:
        audio_data = load_audio(audio)
        content.append({"type": "audio", "audio": audio_data})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True)
    if hasattr(loaded_model, "device"):
        inputs = inputs.to(loaded_model.device)

    output_ids = loaded_model.generate(**inputs, max_new_tokens=max_new_tokens)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    print(processor.decode(new_tokens, skip_special_tokens=True))
