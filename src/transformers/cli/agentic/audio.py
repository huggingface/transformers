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
Audio CLI commands for the transformers agentic CLI.

Each function uses Auto* model classes directly (no pipeline) and is
registered as a top-level ``transformers`` CLI command via ``app.py``.
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
)


def transcribe(
    audio: Annotated[str, typer.Option(help="Path or URL to the audio file.")],
    model: ModelOpt = None,
    timestamps: Annotated[str | None, typer.Option(help="Enable timestamp prediction (e.g. 'true').")] = None,
    language: Annotated[str | None, typer.Option(help="Language code for transcription (e.g. 'en', 'fr').")] = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Transcribe speech from an audio file.

    Uses ``AutoModelForSpeechSeq2Seq`` and ``AutoProcessor`` to load a
    speech-to-text model and produce a transcription.

    Examples::

        transformers transcribe --audio recording.wav
        transformers transcribe --audio recording.wav --language fr --json
        transformers transcribe --audio recording.wav --timestamps true
    """

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    model_id = model or "openai/whisper-small"
    loaded_model, processor = _load_pretrained(
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        model_id,
        device,
        dtype,
        trust_remote_code,
        token,
        revision,
    )

    audio_data = load_audio(audio, sampling_rate=processor.feature_extractor.sampling_rate)
    input_features = processor(
        audio_data, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt"
    ).input_features

    if hasattr(loaded_model, "device"):
        input_features = input_features.to(loaded_model.device)

    gen_kwargs = {}
    if timestamps is not None:
        gen_kwargs["return_timestamps"] = True
    if language is not None:
        gen_kwargs["language"] = language

    output_ids = loaded_model.generate(input_features, **gen_kwargs)
    transcription = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    if output_json:
        print(format_output({"text": transcription}, output_json=True))
    else:
        print(transcription)


def audio_classify(
    audio: Annotated[str, typer.Option(help="Path or URL to the audio file.")],
    labels: Annotated[
        str | None, typer.Option(help="Comma-separated candidate labels for zero-shot audio classification.")
    ] = None,
    model: ModelOpt = None,
    top_k: Annotated[int | None, typer.Option(help="Number of top predictions to return.")] = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Classify an audio file into categories.

    Without ``--labels``, uses ``AutoModelForAudioClassification`` and
    ``AutoFeatureExtractor`` with a fine-tuned classification model.
    With ``--labels``, uses ``AutoModel`` and ``AutoProcessor`` for
    zero-shot classification via CLAP.

    Examples::

        transformers audio-classify --audio sound.wav
        transformers audio-classify --audio sound.wav --labels "dog,cat,bird" --json
        transformers audio-classify --audio sound.wav --top-k 3
    """
    import torch

    if labels is None:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        model_id = model or "MIT/ast-finetuned-audioset-10-10-0.4593"
        loaded_model, feature_extractor = _load_pretrained(
            AutoModelForAudioClassification,
            AutoFeatureExtractor,
            model_id,
            device,
            dtype,
            trust_remote_code,
            token,
            revision,
        )

        sr = feature_extractor.sampling_rate
        audio_data = load_audio(audio, sampling_rate=sr)
        inputs = feature_extractor(audio_data, sampling_rate=sr, return_tensors="pt")

        if hasattr(loaded_model, "device"):
            inputs = inputs.to(loaded_model.device)

        with torch.no_grad():
            logits = loaded_model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        k = top_k or 5
        top_probs, top_indices = torch.topk(probs, min(k, probs.size(0)))

        result = [
            {"label": loaded_model.config.id2label[idx.item()], "score": round(prob.item(), 4)}
            for prob, idx in zip(top_probs, top_indices)
        ]
    else:
        from transformers import AutoModel, AutoProcessor

        model_id = model or "laion/clap-htsat-unfused"
        loaded_model, processor = _load_pretrained(
            AutoModel,
            AutoProcessor,
            model_id,
            device,
            dtype,
            trust_remote_code,
            token,
            revision,
        )

        sr = processor.feature_extractor.sampling_rate
        audio_data = load_audio(audio, sampling_rate=sr)
        candidate_labels = [lbl.strip() for lbl in labels.split(",")]
        inputs = processor(
            audios=audio_data, text=candidate_labels, return_tensors="pt", padding=True, sampling_rate=sr
        )

        if hasattr(loaded_model, "device"):
            inputs = inputs.to(loaded_model.device)

        with torch.no_grad():
            outputs = loaded_model(**inputs)

        probs = outputs.logits_per_audio[0].softmax(dim=-1)
        result = [
            {"label": candidate_labels[i], "score": round(probs[i].item(), 4)} for i in range(len(candidate_labels))
        ]
        result.sort(key=lambda x: x["score"], reverse=True)

    print(format_output(result, output_json))


def speak(
    text: Annotated[str, typer.Option(help="Text to synthesize into speech.")],
    output: Annotated[str, typer.Option(help="Output WAV file path.")],
    model: ModelOpt = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
):
    """
    Synthesize speech from text and save to a WAV file.

    Uses ``AutoModelForTextToWaveform`` and ``AutoProcessor`` to generate
    audio from the given text prompt.

    Examples::

        transformers speak --text "Hello world" --output hello.wav
        transformers speak --text "Bonjour le monde" --output bonjour.wav --model suno/bark-small
    """
    import scipy.io.wavfile

    from transformers import AutoModelForTextToWaveform, AutoProcessor

    model_id = model or "suno/bark-small"
    loaded_model, processor = _load_pretrained(
        AutoModelForTextToWaveform,
        AutoProcessor,
        model_id,
        device,
        dtype,
        trust_remote_code,
        token,
        revision,
    )

    inputs = processor(text, return_tensors="pt")

    if hasattr(loaded_model, "device"):
        inputs = inputs.to(loaded_model.device)

    speech_output = loaded_model.generate(**inputs)
    audio_data = speech_output.cpu().float().numpy().squeeze()

    sampling_rate = getattr(loaded_model.generation_config, "sample_rate", None) or getattr(
        getattr(loaded_model.config, "audio_encoder", None), "sampling_rate", 24000
    )

    scipy.io.wavfile.write(output, sampling_rate, audio_data)
    print(f"Saved audio to {output}")


def audio_generate(
    text: Annotated[str, typer.Option(help="Text prompt describing the audio to generate.")],
    output: Annotated[str, typer.Option(help="Output WAV file path.")],
    model: ModelOpt = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
):
    """
    Generate audio (e.g. music) from a text description and save to a WAV file.

    Uses ``AutoModelForTextToWaveform`` and ``AutoProcessor`` to produce
    audio from a text prompt.

    Examples::

        transformers audio-generate --text "a relaxing piano melody" --output music.wav
        transformers audio-generate --text "upbeat electronic beat" --output beat.wav --model facebook/musicgen-small
    """
    import scipy.io.wavfile

    from transformers import AutoModelForTextToWaveform, AutoProcessor

    model_id = model or "facebook/musicgen-small"
    loaded_model, processor = _load_pretrained(
        AutoModelForTextToWaveform,
        AutoProcessor,
        model_id,
        device,
        dtype,
        trust_remote_code,
        token,
        revision,
    )

    inputs = processor(text=[text], return_tensors="pt", padding=True)

    if hasattr(loaded_model, "device"):
        inputs = inputs.to(loaded_model.device)

    audio_values = loaded_model.generate(**inputs, max_new_tokens=256)
    audio_data = audio_values.cpu().float().numpy().squeeze()

    sampling_rate = getattr(loaded_model.generation_config, "sample_rate", None) or getattr(
        getattr(loaded_model.config, "audio_encoder", None), "sampling_rate", 32000
    )

    scipy.io.wavfile.write(output, sampling_rate, audio_data)
    print(f"Saved audio to {output}")
