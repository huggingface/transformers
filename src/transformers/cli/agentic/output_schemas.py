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
JSON Schema fragments describing each CLI command's stdout shape in ``--format json``.

Each entry documents what the command prints in json mode. The human /
agent / quiet renderings are derived from the same data by the huggingface_hub
``Output`` primitives.

Adding a command? Add its ``id`` here. If missing at manifest generation
time, ``_skill_derive`` flags a ``GAP:`` so it cannot silently drift.
"""

_LABEL_SCORE_ITEM = {
    "type": "object",
    "properties": {"label": {"type": "string"}, "score": {"type": "number"}},
}
_LABEL_SCORE_ARRAY = {"type": "array", "items": _LABEL_SCORE_ITEM}

_ENTITY_ITEM = {
    "type": "object",
    "properties": {
        "entity_group": {"type": "string"},
        "entity": {"type": "string"},
        "score": {"type": "number"},
        "word": {"type": "string"},
        "start": {"type": "integer"},
        "end": {"type": "integer"},
    },
}


OUTPUT_SCHEMAS: dict[str, dict] = {
    # text.py
    "classify": {
        "type": "object",
        "properties": {
            "labels": {"type": "array", "items": {"type": "string"}},
            "scores": {"type": "array", "items": {"type": "number"}},
        },
    },
    "ner": {"type": "array", "items": _ENTITY_ITEM},
    "token-classify": {"type": "array", "items": _ENTITY_ITEM},
    "qa": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "score": {"type": "number"},
            "start": {"type": "integer"},
            "end": {"type": "integer"},
        },
    },
    "table-qa": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "coordinates": {"type": "array"},
            "cells": {"type": "array", "items": {"type": "string"}},
            "aggregator": {"type": "string"},
        },
    },
    "summarize": {"type": "object", "properties": {"summary_text": {"type": "string"}}},
    "translate": {"type": "object", "properties": {"translation_text": {"type": "string"}}},
    "fill-mask": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "token": {"type": "integer"},
                "token_str": {"type": "string"},
                "sequence": {"type": "string"},
            },
        },
    },
    # vision.py
    "image-classify": _LABEL_SCORE_ARRAY,
    "detect": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "score": {"type": "number"},
                "box": {
                    "type": "object",
                    "properties": {
                        "xmin": {"type": "number"},
                        "ymin": {"type": "number"},
                        "xmax": {"type": "number"},
                        "ymax": {"type": "number"},
                    },
                },
            },
        },
    },
    "segment": {
        "description": "Semantic segmentation → array of {label, score}; SAM-style → {num_masks, iou_scores}.",
        "anyOf": [
            _LABEL_SCORE_ARRAY,
            {
                "type": "object",
                "properties": {
                    "num_masks": {"type": "integer"},
                    "iou_scores": {"type": "array", "items": {"type": "number"}},
                },
            },
        ],
    },
    "depth": {
        "type": "object",
        "properties": {
            "size": {"type": "string", "description": "Depth map dimensions, e.g. '480x640'."},
            "output_path": {"type": "string", "description": "Path to saved PNG if --output was given."},
        },
    },
    "keypoints": {"type": "object", "description": "Keypoint matching result."},
    "video-classify": _LABEL_SCORE_ARRAY,
    # audio.py
    "transcribe": {"type": "object", "properties": {"text": {"type": "string"}}},
    "audio-classify": _LABEL_SCORE_ARRAY,
    "speak": {"type": "object", "properties": {"output_path": {"type": "string"}}},
    "audio-generate": {"type": "object", "properties": {"output_path": {"type": "string"}}},
    # multimodal.py
    "vqa": {"type": "object", "properties": {"answer": {"type": "string"}}},
    "document-qa": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "start": {"type": "integer"},
            "end": {"type": "integer"},
        },
    },
    "caption": {"type": "object", "properties": {"caption": {"type": "string"}}},
    "ocr": {"type": "object", "properties": {"text": {"type": "string"}}},
    "multimodal-chat": {"type": "object", "properties": {"text": {"type": "string"}}},
    # generate.py
    "generate": {"type": "object", "properties": {"text": {"type": "string"}}},
    "detect-watermark": {
        "type": "object",
        "properties": {"prediction": {"type": "string"}, "confidence": {"type": "number"}},
    },
    # utilities.py
    "embed": {
        "type": "object",
        "properties": {
            "shape": {"type": "string"},
            "values": {"type": "string"},
            "output_path": {"type": "string"},
        },
    },
    "tokenize": {
        "type": "object",
        "properties": {
            "tokens": {"type": "array", "items": {"type": "string"}},
            "token_ids": {"type": "array", "items": {"type": "integer"}},
            "num_tokens": {"type": "integer"},
        },
    },
    "inspect": {"type": "object", "description": "Model configuration (summary for humans, full dict for agents)."},
    "inspect-forward": {
        "type": "object",
        "description": "Attention and hidden-state shape/stat summary per layer.",
    },
    "benchmark-quantization": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "method": {"type": "string"},
                "tokens_per_sec": {"type": "number"},
                "time_sec": {"type": "number"},
                "peak_memory_mb": {"type": "number"},
                "output_preview": {"type": "string"},
            },
        },
    },
    # train.py, quantize.py, export.py
    "train": {
        "type": "object",
        "properties": {"output_path": {"type": "string"}},
    },
    "quantize": {
        "type": "object",
        "properties": {"method": {"type": "string"}, "output_path": {"type": "string"}},
    },
    "export": {
        "type": "object",
        "properties": {"format": {"type": "string"}, "output_path": {"type": "string"}},
    },
}
