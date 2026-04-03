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
Pipeline-based inference CLI commands.

Each public function is a thin wrapper around ``transformers.pipeline()`` and
is registered as a top-level ``transformers`` CLI command via ``app.py``.

All commands accept ``--model`` (Hub ID or local path), ``--device``,
``--dtype``, ``--trust-remote-code``, ``--token``, ``--revision``, and
``--json`` (for machine-readable output).

Input can be provided via ``--text``/``--image``/``--audio``, ``--file``
(read from disk), or piped through stdin.
"""

import json
from typing import Annotated

import typer

from ._common import format_output, load_audio, load_image, resolve_input


# ---------------------------------------------------------------------------
# Shared option definitions (reused across all inference commands)
# ---------------------------------------------------------------------------

ModelOpt = Annotated[str | None, typer.Option("--model", "-m", help="Model ID or local path.")]
DeviceOpt = Annotated[str | None, typer.Option(help="Device to run on (e.g. 'cpu', 'cuda', 'cuda:0', 'mps').")]
DtypeOpt = Annotated[str, typer.Option(help="Dtype for model weights ('auto', 'float16', 'bfloat16', 'float32').")]
TrustOpt = Annotated[bool, typer.Option(help="Trust remote code from the Hub.")]
TokenOpt = Annotated[str | None, typer.Option(help="HF Hub token for gated/private models.")]
RevisionOpt = Annotated[str | None, typer.Option(help="Model revision (branch, tag, or commit SHA).")]
JsonOpt = Annotated[bool, typer.Option("--json", help="Output results as JSON.")]


def _make_pipeline(
    task: str,
    model: str | None,
    device: str | None,
    dtype: str,
    trust_remote_code: bool,
    token: str | None,
    revision: str | None,
    **kwargs,
):
    """Instantiate a ``transformers.pipeline()`` with the common CLI options."""
    from transformers import pipeline

    pipe_kwargs = {}
    if model is not None:
        pipe_kwargs["model"] = model
    if device is not None:
        pipe_kwargs["device"] = device
    if dtype != "auto":
        import torch

        pipe_kwargs["dtype"] = getattr(torch, dtype)
    if trust_remote_code:
        pipe_kwargs["trust_remote_code"] = True
    if token is not None:
        pipe_kwargs["token"] = token
    if revision is not None:
        pipe_kwargs["revision"] = revision
    pipe_kwargs.update(kwargs)

    return pipeline(task, **pipe_kwargs)


# ===========================================================================
# Text commands
# ===========================================================================


def classify(
    text: Annotated[str | None, typer.Option(help="Text to classify.")] = None,
    file: Annotated[str | None, typer.Option(help="Read text from this file.")] = None,
    labels: Annotated[
        str | None, typer.Option(help="Comma-separated candidate labels for zero-shot classification.")
    ] = None,
    model: ModelOpt = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Classify text into categories.

    Uses the ``text-classification`` pipeline by default (requires a
    fine-tuned classification model). Pass ``--labels`` to switch to
    ``zero-shot-classification`` with arbitrary categories.

    Examples::

        # Supervised (model already fine-tuned for sentiment)
        transformers classify --model distilbert/distilbert-base-uncased-finetuned-sst-2-english --text "Great movie!"

        # Zero-shot (any categories, no fine-tuning needed)
        transformers classify --text "The stock market crashed" --labels "politics,finance,sports"

        # Read from file, output as JSON
        transformers classify --file review.txt --json
    """
    input_text = resolve_input(text, file)

    if labels is not None:
        task = "zero-shot-classification"
        pipe = _make_pipeline(task, model, device, dtype, trust_remote_code, token, revision)
        result = pipe(input_text, candidate_labels=labels.split(","))
    else:
        task = "text-classification"
        pipe = _make_pipeline(task, model, device, dtype, trust_remote_code, token, revision)
        result = pipe(input_text)

    print(format_output(result, output_json))


def ner(
    text: Annotated[str | None, typer.Option(help="Text to extract entities from.")] = None,
    file: Annotated[str | None, typer.Option(help="Read text from this file.")] = None,
    model: ModelOpt = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    aggregation_strategy: Annotated[
        str, typer.Option(help="Entity aggregation: 'none', 'simple', 'first', 'average', 'max'.")
    ] = "simple",
    output_json: JsonOpt = False,
):
    """
    Extract named entities from text (NER).

    Uses the ``token-classification`` pipeline with entity aggregation
    enabled by default (``--aggregation-strategy simple``).

    Example::

        transformers ner --model dslim/bert-base-NER --text "Apple CEO Tim Cook met with President Biden in Washington."
    """
    input_text = resolve_input(text, file)
    pipe = _make_pipeline(
        "token-classification",
        model,
        device,
        dtype,
        trust_remote_code,
        token,
        revision,
        aggregation_strategy=aggregation_strategy,
    )
    result = pipe(input_text)
    print(format_output(result, output_json))


def token_classify(
    text: Annotated[str | None, typer.Option(help="Text to tag.")] = None,
    file: Annotated[str | None, typer.Option(help="Read text from this file.")] = None,
    model: ModelOpt = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Tag tokens with labels (POS tagging, chunking, etc.).

    Uses the ``token-classification`` pipeline. The output depends on the
    model — a POS model outputs POS tags, a NER model outputs entity labels.

    Example::

        transformers token-classify --model vblagoje/bert-english-uncased-finetuned-pos --text "The cat sat on the mat."
    """
    input_text = resolve_input(text, file)
    pipe = _make_pipeline("token-classification", model, device, dtype, trust_remote_code, token, revision)
    result = pipe(input_text)
    print(format_output(result, output_json))


def qa(
    question: Annotated[str, typer.Option(help="The question to answer.")],
    context: Annotated[str | None, typer.Option(help="Context paragraph containing the answer.")] = None,
    file: Annotated[str | None, typer.Option(help="Read context from this file.")] = None,
    model: ModelOpt = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Answer a question given a context paragraph (extractive QA).

    Extracts the answer span from ``--context`` (or ``--file``). The model
    does not generate new text — it highlights the relevant substring.

    Example::

        transformers qa --question "Who invented the telephone?" --context "Alexander Graham Bell invented the telephone in 1876."
    """
    ctx = resolve_input(context, file)
    pipe = _make_pipeline("question-answering", model, device, dtype, trust_remote_code, token, revision)
    result = pipe(question=question, context=ctx)
    print(format_output(result, output_json))


def table_qa(
    question: Annotated[str, typer.Option(help="Question about the table.")],
    table: Annotated[str, typer.Option(help="Path to a CSV file containing the table.")],
    model: ModelOpt = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Answer a question about tabular data (CSV).

    Loads a CSV file into a table and uses the ``table-question-answering``
    pipeline (e.g., TAPAS) to answer the question.

    Example::

        transformers table-qa --question "What is the total revenue?" --table financials.csv
    """
    import pandas as pd

    table_data = pd.read_csv(table).astype(str).to_dict(orient="list")
    pipe = _make_pipeline("table-question-answering", model, device, dtype, trust_remote_code, token, revision)
    result = pipe(query=question, table=table_data)
    print(format_output(result, output_json))


def summarize(
    text: Annotated[str | None, typer.Option(help="Text to summarize.")] = None,
    file: Annotated[str | None, typer.Option(help="Read text from this file.")] = None,
    model: ModelOpt = None,
    max_length: Annotated[int | None, typer.Option(help="Maximum summary length in tokens.")] = None,
    min_length: Annotated[int | None, typer.Option(help="Minimum summary length in tokens.")] = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Summarize text.

    Uses the ``summarization`` pipeline (e.g., BART, T5, Pegasus).

    Examples::

        transformers summarize --model facebook/bart-large-cnn --file article.txt
        transformers summarize --text "Long article text here..." --max-length 100
    """
    input_text = resolve_input(text, file)
    pipe = _make_pipeline("summarization", model, device, dtype, trust_remote_code, token, revision)
    gen_kwargs = {}
    if max_length is not None:
        gen_kwargs["max_length"] = max_length
    if min_length is not None:
        gen_kwargs["min_length"] = min_length
    result = pipe(input_text, **gen_kwargs)
    print(format_output(result, output_json))


def translate(
    text: Annotated[str | None, typer.Option(help="Text to translate.")] = None,
    file: Annotated[str | None, typer.Option(help="Read text from this file.")] = None,
    model: ModelOpt = None,
    max_length: Annotated[int | None, typer.Option(help="Maximum translation length.")] = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Translate text between languages.

    The language pair is determined by the model. Use Helsinki-NLP models
    for specific pairs (e.g., ``Helsinki-NLP/opus-mt-en-de`` for English to German).

    Example::

        transformers translate --model Helsinki-NLP/opus-mt-en-de --text "The weather is nice today."
    """
    input_text = resolve_input(text, file)
    pipe = _make_pipeline("translation", model, device, dtype, trust_remote_code, token, revision)
    gen_kwargs = {}
    if max_length is not None:
        gen_kwargs["max_length"] = max_length
    result = pipe(input_text, **gen_kwargs)
    print(format_output(result, output_json))


def fill_mask(
    text: Annotated[str, typer.Option(help="Text with a [MASK] token.")],
    model: ModelOpt = None,
    top_k: Annotated[int, typer.Option(help="Number of predictions to return.")] = 5,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Predict the masked token in a sentence.

    Uses the ``fill-mask`` pipeline. The mask token depends on the model
    (``[MASK]`` for BERT, ``<mask>`` for RoBERTa).

    Example::

        transformers fill-mask --model bert-base-uncased --text "The capital of France is [MASK]."
    """
    pipe = _make_pipeline("fill-mask", model, device, dtype, trust_remote_code, token, revision)
    result = pipe(text, top_k=top_k)
    print(format_output(result, output_json))


# ===========================================================================
# Vision commands
# ===========================================================================


def image_classify(
    image: Annotated[str, typer.Option(help="Path or URL to the image.")],
    labels: Annotated[str | None, typer.Option(help="Comma-separated labels for zero-shot classification.")] = None,
    model: ModelOpt = None,
    top_k: Annotated[int | None, typer.Option(help="Number of top predictions.")] = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Classify an image into categories.

    Uses ``image-classification`` by default. Pass ``--labels`` to switch to
    ``zero-shot-image-classification`` (CLIP-style, no fine-tuning needed).

    Examples::

        # Supervised
        transformers image-classify --model google/vit-base-patch16-224 --image photo.jpg

        # Zero-shot
        transformers image-classify --model openai/clip-vit-base-patch32 --image photo.jpg --labels "cat,dog,bird"
    """
    img = load_image(image)

    if labels is not None:
        task = "zero-shot-image-classification"
        pipe = _make_pipeline(task, model, device, dtype, trust_remote_code, token, revision)
        result = pipe(img, candidate_labels=labels.split(","))
    else:
        task = "image-classification"
        pipe = _make_pipeline(task, model, device, dtype, trust_remote_code, token, revision)
        kwargs = {}
        if top_k is not None:
            kwargs["top_k"] = top_k
        result = pipe(img, **kwargs)

    print(format_output(result, output_json))


def detect(
    image: Annotated[str, typer.Option(help="Path or URL to the image.")],
    text: Annotated[str | None, typer.Option(help="Text query for grounded/zero-shot detection.")] = None,
    model: ModelOpt = None,
    threshold: Annotated[float, typer.Option(help="Confidence threshold.")] = 0.5,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Detect objects in an image with bounding boxes.

    Uses ``object-detection`` by default. Pass ``--text`` to switch to
    ``zero-shot-object-detection`` (grounded detection — find objects
    matching a text description).

    Examples::

        # Standard detection (DETR)
        transformers detect --model facebook/detr-resnet-50 --image street.jpg

        # Grounded detection (find objects by description)
        transformers detect --model IDEA-Research/grounding-dino-base --image kitchen.jpg --text "red mug on the counter"
    """
    img = load_image(image)

    if text is not None:
        task = "zero-shot-object-detection"
        pipe = _make_pipeline(task, model, device, dtype, trust_remote_code, token, revision)
        result = pipe(img, candidate_labels=text.split(","), threshold=threshold)
    else:
        task = "object-detection"
        pipe = _make_pipeline(task, model, device, dtype, trust_remote_code, token, revision)
        result = pipe(img, threshold=threshold)

    print(format_output(result, output_json))


def segment(
    image: Annotated[str, typer.Option(help="Path or URL to the image.")],
    model: ModelOpt = None,
    points: Annotated[
        str | None, typer.Option(help="JSON list of [x,y] points for SAM-style mask generation.")
    ] = None,
    point_labels: Annotated[
        str | None, typer.Option(help="JSON list of point labels (1=foreground, 0=background).")
    ] = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Segment an image.

    Without ``--points``, uses ``image-segmentation`` (semantic or instance
    segmentation depending on the model). With ``--points``, uses
    ``mask-generation`` for interactive SAM-style segmentation.

    Examples::

        # Semantic segmentation
        transformers segment --model nvidia/segformer-b0-finetuned-ade-512-512 --image scene.jpg

        # Instance segmentation
        transformers segment --model facebook/mask2former-swin-base-coco-instance --image crowd.jpg

        # SAM-style interactive (click a point)
        transformers segment --model facebook/sam-vit-base --image photo.jpg --points "[[120,45]]" --point-labels "[1]"
    """
    img = load_image(image)

    if points is not None:
        task = "mask-generation"
        pipe = _make_pipeline(task, model, device, dtype, trust_remote_code, token, revision)
        pts = json.loads(points)
        kwargs = {"input_points": pts}
        if point_labels is not None:
            kwargs["input_labels"] = json.loads(point_labels)
        result = pipe(img, **kwargs)
    else:
        task = "image-segmentation"
        pipe = _make_pipeline(task, model, device, dtype, trust_remote_code, token, revision)
        result = pipe(img)

    print(format_output(result, output_json))


def depth(
    image: Annotated[str, typer.Option(help="Path or URL to the image.")],
    model: ModelOpt = None,
    output: Annotated[str | None, typer.Option(help="Save depth map to this path (PNG).")] = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
):
    """
    Estimate depth from a single image.

    Outputs a depth map. Pass ``--output depth.png`` to save it to disk.

    Example::

        transformers depth --model Intel/dpt-large --image room.jpg --output depth_map.png
    """
    img = load_image(image)
    pipe = _make_pipeline("depth-estimation", model, device, dtype, trust_remote_code, token, revision)
    result = pipe(img)

    if output is not None:
        result["depth"].save(output)
        print(f"Depth map saved to {output}")
    else:
        print(f"Depth map size: {result['depth'].size}")
        print("Pass --output path.png to save the depth map.")


def keypoints(
    images: Annotated[list[str], typer.Option(help="Paths to two images to match.")],
    model: ModelOpt = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Detect and match keypoints across an image pair.

    Requires exactly two images via ``--images``.

    Example::

        transformers keypoints --model magic-leap-community/superglue --images img1.jpg --images img2.jpg
    """
    if len(images) != 2:
        raise SystemExit("Error: --images requires exactly two image paths.")
    img1 = load_image(images[0])
    img2 = load_image(images[1])
    pipe = _make_pipeline("keypoint-matching", model, device, dtype, trust_remote_code, token, revision)
    result = pipe(img1, img2)
    print(format_output(result, output_json))


# ===========================================================================
# Audio commands
# ===========================================================================


def transcribe(
    audio: Annotated[str, typer.Option(help="Path to the audio file.")],
    model: ModelOpt = None,
    timestamps: Annotated[str | None, typer.Option(help="Timestamp granularity: 'word' or 'chunk'.")] = None,
    language: Annotated[str | None, typer.Option(help="Language code (e.g. 'en', 'fr').")] = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Transcribe speech to text.

    Uses the ``automatic-speech-recognition`` pipeline (e.g., Whisper).
    Pass ``--timestamps word`` for word-level timing information.

    Examples::

        transformers transcribe --model openai/whisper-small --audio recording.wav
        transformers transcribe --model openai/whisper-small --audio meeting.wav --timestamps word --json
    """
    pipe = _make_pipeline("automatic-speech-recognition", model, device, dtype, trust_remote_code, token, revision)
    kwargs = {}
    if timestamps is not None:
        kwargs["return_timestamps"] = timestamps
    if language is not None:
        kwargs["generate_kwargs"] = {"language": language}
    result = pipe(audio, **kwargs)
    if output_json:
        print(format_output(result, output_json=True))
    else:
        print(result["text"])


def audio_classify(
    audio: Annotated[str, typer.Option(help="Path to the audio file.")],
    labels: Annotated[str | None, typer.Option(help="Comma-separated labels for zero-shot classification.")] = None,
    model: ModelOpt = None,
    top_k: Annotated[int | None, typer.Option(help="Number of top predictions.")] = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Classify an audio clip.

    Uses ``audio-classification`` by default. Pass ``--labels`` to switch to
    ``zero-shot-audio-classification``.

    Examples::

        transformers audio-classify --model MIT/ast-finetuned-audioset-10-10-0.4593 --audio clip.wav
        transformers audio-classify --model laion/clap-htsat-unfused --audio clip.wav --labels "speech,music,noise"
    """
    if labels is not None:
        task = "zero-shot-audio-classification"
        pipe = _make_pipeline(task, model, device, dtype, trust_remote_code, token, revision)
        result = pipe(audio, candidate_labels=labels.split(","))
    else:
        task = "audio-classification"
        pipe = _make_pipeline(task, model, device, dtype, trust_remote_code, token, revision)
        kwargs = {}
        if top_k is not None:
            kwargs["top_k"] = top_k
        result = pipe(audio, **kwargs)

    print(format_output(result, output_json))


def speak(
    text: Annotated[str, typer.Option(help="Text to speak.")],
    output: Annotated[str, typer.Option(help="Output audio file path (e.g. speech.wav).")],
    model: ModelOpt = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
):
    """
    Generate speech from text (text-to-speech).

    Saves the generated audio as a WAV file.

    Example::

        transformers speak --model suno/bark-small --text "Hello, how are you today?" --output speech.wav
    """
    import numpy as np
    import scipy.io.wavfile

    pipe = _make_pipeline("text-to-audio", model, device, dtype, trust_remote_code, token, revision)
    result = pipe(text)

    audio_data = result["audio"]
    sampling_rate = result["sampling_rate"]

    if isinstance(audio_data, np.ndarray):
        scipy.io.wavfile.write(output, sampling_rate, audio_data)
    else:
        import torch

        if isinstance(audio_data, torch.Tensor):
            scipy.io.wavfile.write(output, sampling_rate, audio_data.cpu().numpy())

    print(f"Audio saved to {output} (sample rate: {sampling_rate})")


def audio_generate(
    text: Annotated[str, typer.Option(help="Text description of the audio to generate.")],
    output: Annotated[str, typer.Option(help="Output audio file path.")],
    model: ModelOpt = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
):
    """
    Generate audio from a text description (music, sound effects).

    Uses the ``text-to-audio`` pipeline (e.g., MusicGen). Saves the result
    as a WAV file.

    Example::

        transformers audio-generate --model facebook/musicgen-small --text "A calm piano melody" --output music.wav
    """
    import numpy as np
    import scipy.io.wavfile

    pipe = _make_pipeline("text-to-audio", model, device, dtype, trust_remote_code, token, revision)
    result = pipe(text)

    audio_data = result["audio"]
    sampling_rate = result["sampling_rate"]

    if isinstance(audio_data, np.ndarray):
        scipy.io.wavfile.write(output, sampling_rate, audio_data)
    else:
        import torch

        if isinstance(audio_data, torch.Tensor):
            scipy.io.wavfile.write(output, sampling_rate, audio_data.cpu().numpy())

    print(f"Audio saved to {output} (sample rate: {sampling_rate})")


# ===========================================================================
# Video commands
# ===========================================================================


def video_classify(
    video: Annotated[str, typer.Option(help="Path to the video file.")],
    model: ModelOpt = None,
    top_k: Annotated[int | None, typer.Option(help="Number of top predictions.")] = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Classify a video clip into categories.

    Example::

        transformers video-classify --model MCG-NJU/videomae-base-finetuned-kinetics --video clip.mp4
    """
    pipe = _make_pipeline("video-classification", model, device, dtype, trust_remote_code, token, revision)
    kwargs = {}
    if top_k is not None:
        kwargs["top_k"] = top_k
    result = pipe(video, **kwargs)
    print(format_output(result, output_json))


# ===========================================================================
# Multimodal commands
# ===========================================================================


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
    Answer a question about an image (visual QA).

    Uses the ``image-text-to-text`` pipeline with a vision-language model.

    Example::

        transformers vqa --model Salesforce/blip2-opt-2.7b --image chart.png --question "What is the trend shown?"
    """
    img = load_image(image)
    pipe = _make_pipeline("image-text-to-text", model, device, dtype, trust_remote_code, token, revision)
    result = pipe(img, text=question, max_new_tokens=max_new_tokens)
    print(format_output(result, output_json))


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
    Answer a question about a document image (invoices, forms, receipts).

    Uses the ``document-question-answering`` pipeline.

    Example::

        transformers document-qa --model impira/layoutlm-document-qa --image invoice.png --question "What is the total amount?"
    """
    img = load_image(image)
    pipe = _make_pipeline("document-question-answering", model, device, dtype, trust_remote_code, token, revision)
    result = pipe(img, question=question)
    print(format_output(result, output_json))


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
    Generate a caption for an image.

    Uses the ``image-text-to-text`` pipeline with a "Describe this image" prompt.

    Example::

        transformers caption --model Salesforce/blip2-opt-2.7b --image sunset.jpg
    """
    img = load_image(image)
    pipe = _make_pipeline("image-text-to-text", model, device, dtype, trust_remote_code, token, revision)
    result = pipe(img, text="Describe this image.", max_new_tokens=max_new_tokens)
    print(format_output(result, output_json))


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
    Extract text from a document image (OCR).

    Uses the ``image-text-to-text`` pipeline with an extraction prompt.
    For best results, use a model trained for document understanding
    (e.g., Donut, Nougat, Florence).

    Example::

        transformers ocr --model naver-clova-ix/donut-base-finetuned-cord-v2 --image receipt.png
    """
    img = load_image(image)
    pipe = _make_pipeline("image-text-to-text", model, device, dtype, trust_remote_code, token, revision)
    result = pipe(img, text="Extract all text from this image.", max_new_tokens=max_new_tokens)
    print(format_output(result, output_json))


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
