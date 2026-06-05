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
Unit tests for the pure helpers shared by all agentic CLI commands.

Covers ``_common.py``: input resolution, output formatting, model-loading
kwargs, and media loaders. None of these tests require torch or network.
"""

from __future__ import annotations

import io
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from transformers.cli.agentic import _common


# resolve_input


def test_resolve_input_prefers_text_over_file_and_stdin(tmp_path, monkeypatch):
    file_path = tmp_path / "x.txt"
    file_path.write_text("from-file")
    monkeypatch.setattr(sys, "stdin", io.StringIO("from-stdin"))
    # Even with file and stdin available, ``--text`` wins.
    assert _common.resolve_input(text="from-text", file=str(file_path)) == "from-text"


def test_resolve_input_reads_from_file(tmp_path):
    file_path = tmp_path / "x.txt"
    file_path.write_text("contents\nwith newline\n")
    assert _common.resolve_input(text=None, file=str(file_path)) == "contents\nwith newline\n"


def test_resolve_input_reads_from_stdin_when_piped(monkeypatch):
    fake_stdin = io.StringIO("piped-input")
    fake_stdin.isatty = lambda: False  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "stdin", fake_stdin)
    assert _common.resolve_input() == "piped-input"


def test_resolve_input_raises_when_nothing_provided(monkeypatch):
    fake_stdin = io.StringIO("")
    fake_stdin.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "stdin", fake_stdin)
    with pytest.raises(SystemExit):
        _common.resolve_input()


# format_output


def test_format_output_json_roundtrip():
    out = _common.format_output([{"label": "POS", "score": 0.9}], output_json=True)
    assert json.loads(out) == [{"label": "POS", "score": 0.9}]


def test_format_output_json_handles_non_serializable():
    # ``default=str`` lets things like numpy/torch scalars pass through.
    class Custom:
        def __str__(self):
            return "custom"

    out = _common.format_output({"x": Custom()}, output_json=True)
    assert json.loads(out) == {"x": "custom"}


def test_format_output_list_of_dicts():
    out = _common.format_output([{"label": "A", "score": 0.5}, {"label": "B", "score": 0.5}])
    assert "label: A" in out and "label: B" in out


def test_format_output_list_of_lists_of_dicts():
    # NER-style nested output should still be flattened readably.
    out = _common.format_output([[{"word": "Tim", "entity": "PER"}]])
    assert "word: Tim" in out and "entity: PER" in out


def test_format_output_list_of_scalars():
    out = _common.format_output(["a", "b", "c"])
    assert out.splitlines() == ["a", "b", "c"]


def test_format_output_dict():
    out = _common.format_output({"answer": "42", "score": 0.99})
    assert "answer: 42" in out and "score: 0.99" in out


def test_format_output_scalar():
    assert _common.format_output("hello") == "hello"
    assert _common.format_output(7) == "7"


# Load pretrained
def _fake_classes():
    """Return (model_cls, processor_cls) MagicMocks whose ``from_pretrained`` records kwargs."""
    model_cls = MagicMock()
    processor_cls = MagicMock()
    fake_model = MagicMock()
    fake_model.eval = MagicMock(return_value=None)
    model_cls.from_pretrained.return_value = fake_model
    processor_cls.from_pretrained.return_value = MagicMock()
    return model_cls, processor_cls


def test_load_pretrained_default_device_uses_device_map_auto():
    model_cls, proc_cls = _fake_classes()
    _common._load_pretrained(
        model_cls,
        proc_cls,
        "some/model",
        device=None,
        dtype="auto",
        trust_remote_code=False,
        token=None,
        revision=None,
    )
    kwargs = model_cls.from_pretrained.call_args.kwargs
    assert kwargs.get("device_map") == "auto"
    assert "torch_dtype" not in kwargs
    # Processor should not receive device/dtype kwargs at all.
    assert proc_cls.from_pretrained.call_args.kwargs == {}


def test_load_pretrained_cpu_does_not_set_device_map():
    model_cls, proc_cls = _fake_classes()
    _common._load_pretrained(
        model_cls,
        proc_cls,
        "some/model",
        device="cpu",
        dtype="auto",
        trust_remote_code=False,
        token=None,
        revision=None,
    )
    assert "device_map" not in model_cls.from_pretrained.call_args.kwargs


def test_load_pretrained_explicit_cuda_sets_device_map():
    model_cls, proc_cls = _fake_classes()
    _common._load_pretrained(
        model_cls,
        proc_cls,
        "some/model",
        device="cuda:0",
        dtype="auto",
        trust_remote_code=False,
        token=None,
        revision=None,
    )
    assert model_cls.from_pretrained.call_args.kwargs.get("device_map") == "cuda:0"


def test_load_pretrained_dtype_is_resolved_against_torch():
    pytest.importorskip("torch")
    import torch

    model_cls, proc_cls = _fake_classes()
    _common._load_pretrained(
        model_cls,
        proc_cls,
        "some/model",
        device="cpu",
        dtype="float16",
        trust_remote_code=False,
        token=None,
        revision=None,
    )
    assert model_cls.from_pretrained.call_args.kwargs["torch_dtype"] is torch.float16


def test_load_pretrained_propagates_common_kwargs_to_both_classes():
    model_cls, proc_cls = _fake_classes()
    _common._load_pretrained(
        model_cls,
        proc_cls,
        "some/model",
        device="cpu",
        dtype="auto",
        trust_remote_code=True,
        token="hf_xxx",
        revision="main",
    )
    for cls in (model_cls, proc_cls):
        kw = cls.from_pretrained.call_args.kwargs
        assert kw.get("trust_remote_code") is True
        assert kw.get("token") == "hf_xxx"
        assert kw.get("revision") == "main"


def test_load_pretrained_calls_eval_on_model():
    model_cls, proc_cls = _fake_classes()
    model, _ = _common._load_pretrained(
        model_cls,
        proc_cls,
        "some/model",
        device="cpu",
        dtype="auto",
        trust_remote_code=False,
        token=None,
        revision=None,
    )
    model.eval.assert_called_once()


# Load image
def test_load_image_local_path(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    p = tmp_path / "img.png"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(p)
    out = _common.load_image(str(p))
    assert out.size == (4, 4)


def test_load_image_url_uses_requests():
    pytest.importorskip("PIL")
    from PIL import Image

    fake_response = MagicMock()
    # ``Image.open(requests.get(url, stream=True).raw)`` is the call path.
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), color=(0, 255, 0)).save(buf, format="PNG")
    buf.seek(0)
    fake_response.raw = buf

    with patch.dict(sys.modules, {"requests": MagicMock(get=MagicMock(return_value=fake_response))}):
        out = _common.load_image("https://example.com/img.png")
    assert out.size == (2, 2)


# Load audio
def test_load_audio_falls_back_to_soundfile_when_librosa_missing(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")

    # Write a 16kHz mono wav.
    audio_path = tmp_path / "a.wav"
    sf.write(audio_path, np.zeros(1600, dtype=np.float32), 16000)

    # Pretend librosa is not installed.
    monkeypatch.setitem(sys.modules, "librosa", None)
    out = _common.load_audio(str(audio_path), sampling_rate=16000)
    assert out.shape == (1600,)


def test_load_audio_soundfile_fallback_rejects_sample_rate_mismatch(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")

    audio_path = tmp_path / "a.wav"
    sf.write(audio_path, np.zeros(800, dtype=np.float32), 8000)

    monkeypatch.setitem(sys.modules, "librosa", None)
    with pytest.raises(SystemExit):
        _common.load_audio(str(audio_path), sampling_rate=16000)


# Load video
def test_load_video_systemexit_when_no_backend(monkeypatch):
    # Simulate both backends absent.
    monkeypatch.setitem(sys.modules, "decord", None)
    monkeypatch.setitem(sys.modules, "av", None)
    with pytest.raises(SystemExit):
        _common.load_video("nonexistent.mp4", num_frames=4)
