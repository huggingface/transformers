# Copyright 2026 the HuggingFace Team. All rights reserved.
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

import unittest
from pathlib import Path

import torch

from transformers.testing_utils import slow


_TEXT = "Ada Lovelace wrote notes about the analytical engine in London."
_SCHEMA = {"entities": {"person": {}, "work": {}, "location": {}}}
_DECIDE_SPAN = [-96.554901, 5.955377, -432.44342, -51.917194]
_DECIDE_ENTITIES = {"person": ["Ada Lovelace"], "work": [], "location": ["London"]}
_BASE_START = [5.006553, -2.509621, -4.07003, -2.389648]
_BASE_ENTITIES = {"person": ["Ada Lovelace"], "work": ["analytical engine"], "location": ["London"]}


def _weight_snapshot(repo_id: str) -> Path | None:
    """Return the local snapshot that contains weights, if one is cached."""
    repo = Path.home() / ".cache/huggingface/hub" / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo / "snapshots"
    if not snapshots.is_dir():
        return None
    ordered = []
    ref = repo / "refs" / "main"
    if ref.is_file():
        ordered.append(snapshots / ref.read_text().strip())
    ordered.extend(sorted(snapshots.iterdir()))
    for snapshot in ordered:
        weights = snapshot / "model.safetensors"
        if weights.is_file() and weights.stat().st_size > 0 and (snapshot / "config.json").is_file():
            return snapshot
    return None


_DECIDE = _weight_snapshot("fastino/GLiNER2.5-Decide")
_BASE = _weight_snapshot("fastino/gliner2.5-base-v1")


def _encode(processor, architecture: str):
    """Encode the Ada Lovelace schema, including when kwargs merge is a ForwardRef."""
    try:
        return processor(_TEXT, schema=_SCHEMA, return_tensors="pt", architecture=architecture)
    except AttributeError as exc:
        if "ForwardRef" not in str(exc):
            raise

        def merge(_cls, tokenizer_init_kwargs=None, **kwargs):
            return {
                "text_kwargs": {
                    "max_len": kwargs.get("max_len"),
                    "architecture": kwargs.get("architecture", architecture),
                    "labels": kwargs.get("labels"),
                    "max_gold_per_query": kwargs.get("max_gold_per_query"),
                }
            }

        processor._merge_kwargs = merge
        return processor(_TEXT, schema=_SCHEMA, return_tensors="pt", architecture=architecture)


def _entity_strings(decoded) -> dict:
    row = decoded[0] if isinstance(decoded, list) else decoded
    return row["entities"]


@slow
@unittest.skipUnless(
    _DECIDE is not None and _BASE is not None,
    "fastino/GLiNER2.5-Decide and fastino/gliner2.5-base-v1 are not in the local HF cache",
)
class Gliner2GoldenTest(unittest.TestCase):
    """Logit slices and entity strings for the two cached GLiNER2 checkpoints."""

    def _load(self, snapshot: Path):
        from transformers import Gliner2ForSchemaExtraction, Gliner2Processor

        model = Gliner2ForSchemaExtraction.from_pretrained(
            snapshot, local_files_only=True, attn_implementation="eager"
        ).eval()
        processor = Gliner2Processor.from_pretrained(snapshot, local_files_only=True)
        return model, processor

    def test_decide_span_slice_and_entities(self):
        model, processor = self._load(_DECIDE)
        encoded = _encode(processor, model.config.architecture)
        metadata = encoded.pop("metadata")
        with torch.no_grad():
            output = model(**encoded)
        flat = output.span_logits[0][0].detach().float().reshape(-1)[:4]
        torch.testing.assert_close(flat, torch.tensor(_DECIDE_SPAN), atol=1e-4, rtol=1e-4)
        decoded = processor.post_process_extraction(output, metadata, threshold=0.5)
        self.assertEqual(_entity_strings(decoded), _DECIDE_ENTITIES)

    def test_base_start_slice_and_entities(self):
        model, processor = self._load(_BASE)
        encoded = _encode(processor, model.config.architecture)
        metadata = encoded.pop("metadata")
        with torch.no_grad():
            output = model(**encoded)
        flat = output.boundary.start_logits.detach().float().reshape(-1)[:4]
        torch.testing.assert_close(flat, torch.tensor(_BASE_START), atol=1e-4, rtol=1e-4)
        decoded = processor.post_process_extraction(output, metadata, threshold=0.5)
        self.assertEqual(_entity_strings(decoded), _BASE_ENTITIES)
