# Copyright 2026 The HuggingFace Team. All rights reserved.
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
End-to-end tests for the agentic CLI.

Each test invokes the real CLI command against a tiny-random checkpoint from
``hf-internal-testing``. We never rely on README defaults so the suite stays
stable across checkpoint changes. Outputs are random (untrained weights) — we
only assert exit codes and the *shape* of the output, not its content.

Tests gated by ``@require_torch`` download a few-MB checkpoint each and run on
CPU in well under a second. Heavier vision/audio/multimodal cases are gated by
``@slow``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transformers.testing_utils import require_timm, require_torch, require_vision, slow


PNG_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "tests_samples" / "COCO" / "000000039769.png"


def _whisper_processor_importable() -> bool:
    """Some envs ship an old ``mistral_common`` that breaks Whisper's processor import."""
    try:
        from mistral_common.protocol.instruct.request import ReasoningEffort  # noqa: F401
    except Exception:
        return False
    return True


require_compatible_mistral_common = pytest.mark.skipif(
    not _whisper_processor_importable(),
    reason="local env has an incompatible mistral_common (missing ReasoningEffort)",
)


# text


@require_torch
def test_classify_supervised(cli, extract_json):
    result = cli(
        "classify",
        "--model",
        "hf-internal-testing/tiny-random-DistilBertForSequenceClassification",
        "--text",
        "hello world",
        "--json",
    )
    assert result.exit_code == 0, result.output
    payload = extract_json(result.output)
    assert isinstance(payload, list) and payload
    assert {"label", "score"} <= payload[0].keys()


@require_torch
def test_ner(cli, extract_json):
    result = cli(
        "ner",
        "--model",
        "hf-internal-testing/tiny-random-BertForTokenClassification",
        "--text",
        "Alice met Bob in Paris.",
        "--json",
    )
    assert result.exit_code == 0, result.output
    payload = extract_json(result.output)
    # Untrained weights — list may be empty, but shape must be a list of dicts.
    assert isinstance(payload, list)
    for entity in payload:
        assert {"entity_group", "score", "start", "end", "word"} <= entity.keys()


@require_torch
def test_token_classify(cli, extract_json):
    result = cli(
        "token-classify",
        "--model",
        "hf-internal-testing/tiny-random-BertForTokenClassification",
        "--text",
        "The cat sat.",
        "--json",
    )
    assert result.exit_code == 0, result.output
    payload = extract_json(result.output)
    assert isinstance(payload, list)
    if payload:
        assert {"entity", "score", "start", "end", "word"} <= payload[0].keys()


@require_torch
def test_qa(cli, extract_json):
    result = cli(
        "qa",
        "--model",
        "hf-internal-testing/tiny-random-DistilBertForQuestionAnswering",
        "--question",
        "Who?",
        "--context",
        "Alice met Bob in Paris.",
        "--json",
    )
    assert result.exit_code == 0, result.output
    payload = extract_json(result.output)
    assert {"answer", "score", "start", "end"} <= payload.keys()


@require_torch
def test_summarize(cli, extract_json):
    result = cli(
        "summarize",
        "--model",
        "hf-internal-testing/tiny-random-BartForConditionalGeneration",
        "--text",
        "Lorem ipsum dolor sit amet.",
        "--max-length",
        "8",
        "--json",
    )
    assert result.exit_code == 0, result.output
    payload = extract_json(result.output)
    assert isinstance(payload, list) and "summary_text" in payload[0]


@require_torch
def test_translate(cli, extract_json):
    # Translate uses AutoModelForSeq2SeqLM; any seq2seq tiny model works for
    # shape testing (output is gibberish on random weights, which is fine).
    result = cli(
        "translate",
        "--model",
        "hf-internal-testing/tiny-random-T5ForConditionalGeneration",
        "--text",
        "Hello world.",
        "--max-length",
        "8",
        "--json",
    )
    assert result.exit_code == 0, result.output
    payload = extract_json(result.output)
    assert isinstance(payload, list) and "translation_text" in payload[0]


@require_torch
def test_fill_mask(cli, extract_json):
    result = cli(
        "fill-mask",
        "--model",
        "hf-internal-testing/tiny-random-DistilBertForMaskedLM",
        "--text",
        "The capital of France is [MASK].",
        "--top-k",
        "3",
        "--json",
    )
    assert result.exit_code == 0, result.output
    payload = extract_json(result.output)
    assert isinstance(payload, list) and len(payload) == 3
    assert {"score", "token", "token_str", "sequence"} <= payload[0].keys()


@require_torch
def test_fill_mask_without_mask_token_errors(cli):
    result = cli(
        "fill-mask",
        "--model",
        "hf-internal-testing/tiny-random-DistilBertForMaskedLM",
        "--text",
        "No mask here.",
    )
    assert result.exit_code != 0


# generate


@require_torch
def test_generate_basic(cli):
    result = cli(
        "generate",
        "--model",
        "hf-internal-testing/tiny-random-OPTForCausalLM",
        "--prompt",
        "Hello",
        "--max-new-tokens",
        "4",
    )
    assert result.exit_code == 0, result.output


@require_torch
def test_generate_stream(cli):
    result = cli(
        "generate",
        "--model",
        "hf-internal-testing/tiny-random-OPTForCausalLM",
        "--prompt",
        "Hello",
        "--max-new-tokens",
        "4",
        "--stream",
    )
    assert result.exit_code == 0, result.output


# utilities


@require_torch
def test_tokenize(cli, extract_json):
    result = cli(
        "tokenize",
        "--model",
        "hf-internal-testing/tiny-random-gpt2",
        "--text",
        "Hello, world!",
        "--json",
    )
    assert result.exit_code == 0, result.output
    payload = extract_json(result.output)
    assert {"tokens", "token_ids", "num_tokens"} <= payload.keys()
    assert payload["num_tokens"] == len(payload["tokens"]) == len(payload["token_ids"])


def test_inspect(cli, extract_json):
    # Config-only: doesn't even need torch.
    result = cli(
        "inspect",
        "hf-internal-testing/tiny-random-gpt2",
        "--json",
    )
    assert result.exit_code == 0, result.output
    payload = extract_json(result.output)
    assert "model_type" in payload


@require_torch
def test_embed_text(cli, tmp_path):
    output_path = tmp_path / "emb.npy"
    result = cli(
        "embed",
        "--model",
        "hf-internal-testing/tiny-random-BertModel",
        "--text",
        "hello",
        "--output",
        str(output_path),
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists() and output_path.stat().st_size > 0


@require_torch
def test_embed_text_no_output_prints_preview(cli):
    result = cli(
        "embed",
        "--model",
        "hf-internal-testing/tiny-random-BertModel",
        "--text",
        "hello",
    )
    assert result.exit_code == 0, result.output
    assert "Embedding shape" in result.output


@require_torch
def test_embed_no_input_errors(cli):
    result = cli(
        "embed",
        "--model",
        "hf-internal-testing/tiny-random-BertModel",
    )
    assert result.exit_code != 0


# input plumbing through the real CLI


@require_torch
def test_classify_reads_from_file(cli, extract_json, tmp_path):
    file_path = tmp_path / "in.txt"
    file_path.write_text("hello from file")
    result = cli(
        "classify",
        "--model",
        "hf-internal-testing/tiny-random-DistilBertForSequenceClassification",
        "--file",
        str(file_path),
        "--json",
    )
    assert result.exit_code == 0, result.output
    extract_json(result.output)


# slow: vision / audio / multimodal


@slow
@require_torch
@require_vision
def test_image_classify(cli, extract_json):
    # ``top-k`` is capped by the model's number of labels — tiny checkpoints often
    # have only 2 — so we ask for 2 to keep the assertion exact.
    result = cli(
        "image-classify",
        "--model",
        "hf-internal-testing/tiny-random-BeitForImageClassification",
        "--image",
        str(PNG_FIXTURE),
        "--top-k",
        "2",
        "--json",
    )
    assert result.exit_code == 0, result.output
    payload = extract_json(result.output)
    assert isinstance(payload, list) and len(payload) == 2
    assert {"label", "score"} <= payload[0].keys()


@slow
@require_torch
@require_vision
@require_timm  # tiny-random DETR uses a Timm backbone
def test_detect(cli, extract_json):
    result = cli(
        "detect",
        "--model",
        "hf-internal-testing/tiny-random-DetrForObjectDetection",
        "--image",
        str(PNG_FIXTURE),
        "--threshold",
        "0.0",
        "--json",
    )
    assert result.exit_code == 0, result.output
    payload = extract_json(result.output)
    assert isinstance(payload, list)
    for det in payload:
        assert {"label", "score", "box"} <= det.keys()


@slow
@require_torch
@require_vision
def test_depth(cli, tmp_path):
    output_path = tmp_path / "depth.png"
    result = cli(
        "depth",
        "--model",
        "hf-internal-testing/tiny-random-DPTForDepthEstimation",
        "--image",
        str(PNG_FIXTURE),
        "--output",
        str(output_path),
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists() and output_path.stat().st_size > 0


@slow
@require_torch
@require_vision
def test_embed_image(cli, tmp_path):
    output_path = tmp_path / "emb.npy"
    result = cli(
        "embed",
        "--model",
        "hf-internal-testing/tiny-random-ViTModel",
        "--image",
        str(PNG_FIXTURE),
        "--output",
        str(output_path),
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists() and output_path.stat().st_size > 0


@pytest.fixture
def short_wav(tmp_path):
    """Synthesize a 1-second 16 kHz mono silent wav for audio tests."""
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "audio.wav"
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
    return path


@slow
@require_torch
@require_compatible_mistral_common
def test_transcribe(cli, extract_json, short_wav):
    # The tiny-random Whisper checkpoint defaults ``return_timestamps=True`` in
    # its generation config without exposing ``no_timestamps_token_id``, which
    # makes ``generate`` raise. Use the real (small) ``openai/whisper-tiny``
    # — still a slow-marked test, but reliable.
    result = cli(
        "transcribe",
        "--model",
        "openai/whisper-tiny",
        "--audio",
        str(short_wav),
        "--json",
    )
    assert result.exit_code == 0, result.output
    payload = extract_json(result.output)
    assert isinstance(payload, dict) and "text" in payload


@slow
@require_torch
def test_audio_classify(cli, extract_json, short_wav):
    # Whisper-based audio classification has a fixed 60-frame mel input which
    # doesn't match a 1s wav; use a plain audio classifier instead.
    result = cli(
        "audio-classify",
        "--model",
        "hf-internal-testing/tiny-random-Data2VecAudioForSequenceClassification",
        "--audio",
        str(short_wav),
        "--top-k",
        "2",
        "--json",
    )
    assert result.exit_code == 0, result.output
    payload = extract_json(result.output)
    assert isinstance(payload, list)
    if payload:
        assert {"label", "score"} <= payload[0].keys()


@slow
@require_torch
@require_vision
@pytest.mark.xfail(
    reason=(
        "caption (and the other multimodal commands) call "
        "``processor.apply_chat_template(..., return_dict=True)`` which returns "
        "a plain str for tiny-random-LlavaForConditionalGeneration, breaking the "
        "subsequent ``inputs.to(device)``. Needs a stable tiny multimodal "
        "checkpoint or a defensive branch in multimodal.py."
    ),
    strict=False,
)
def test_caption(cli, extract_json):
    result = cli(
        "caption",
        "--model",
        "hf-internal-testing/tiny-random-LlavaForConditionalGeneration",
        "--image",
        str(PNG_FIXTURE),
        "--max-new-tokens",
        "4",
        "--json",
    )
    assert result.exit_code == 0, result.output
    payload = extract_json(result.output)
    assert isinstance(payload, dict) and "caption" in payload
