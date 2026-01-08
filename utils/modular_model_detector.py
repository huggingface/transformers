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

# 🔴🔴🔴 THIS IS AN INTERNAL TOOL. It WILL interact with the hub and use significant local compute resources. Use at your own risk.


"""
Modular model detector: utilities for detecting code similarities between model implementations.

This module provides tools to analyze and detect similarities between different model implementations
in the transformers library. It uses both embedding-based and token-based (Jaccard) similarity metrics
to identify similar code patterns across different model definitions.

Its function is to identify which models can be _modular_-ized, meaning, which already existing classes are
present in the codebase and look very similar to the one we have.

Two scores are computed, one is a code embedding, and the other is a simple Jaccard bag-of-tokens index for overlap
of token sets. A score of 1.00 means the code is identical.

Usage:

```bash
cd transformers

# Use directly the util, it will download the index embedding from the hub. It will require some RAM/VRAM.

>>> python utils/modular_model_detector.py --modeling-file my_new_beit3_modeling_file.py
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 33.62it/s]
encoding 21 query definitions with Qwen/Qwen3-Embedding-4B (device=cuda, batch=16, max_length=4096)
stuff.py::Beit3ImageTextMatchingOutput:
embedding:
    blip_2::Blip2ImageTextMatchingModelOutput (0.9994)
    chinese_clip::ChineseCLIPOutput (0.9818)
    owlvit::OwlViTOutput (0.9818)
    aimv2::Aimv2Output (0.9818)
    blip::BlipOutput (0.9818)
jaccard:
    owlv2::Owlv2Output (0.9667)
    metaclip_2::MetaClip2Output (0.9667)
    altclip::AltCLIPOutput (0.9667)
    owlvit::OwlViTOutput (0.9667)
    blip::BlipOutput (0.9667)
intersection:
    blip::BlipOutput
    owlvit::OwlViTOutput

stuff.py::Beit3MLP:
embedding:
    efficientloftr::EfficientLoFTRMLP (0.9718)
    seggpt::SegGptMlp (0.9650)
    mgp_str::MgpstrMlp (0.9646)
    vitpose_backbone::VitPoseBackboneMLP (0.9640)
    granitemoeshared::GraniteMoeSharedMLP (0.9633)
jaccard:
    chinese_clip::ChineseCLIPTextSelfOutput (0.5294)
    convbert::ConvBertSelfOutput (0.5294)
    bert::BertSelfOutput (0.5294)
    roformer::RoFormerSelfOutput (0.5294)
    layoutlmv3::LayoutLMv3SelfOutput (0.5294)
intersection:

stuff.py::Beit3FeedForwardNetwork:
embedding:
    prophetnet::ProphetNetFeedForward (0.9766)
    dab_detr::DabDetrDecoderLayerFFN (0.9730)
    kosmos2::Kosmos2TextFFN (0.9697)
    kosmos2_5::Kosmos2_5TextFFN (0.9697)
    parakeet::ParakeetEncoderFeedForward (0.9678)
jaccard:
    groupvit::GroupViTMLP (0.4898)
    convbert::ConvBertOutput (0.4600)
    chinese_clip::ChineseCLIPTextOutput (0.4565)
    bert::BertOutput (0.4565)
    roformer::RoFormerOutput (0.4565)
intersection:



```


# If you wish to build the index first, you can run

python utils/modular_model_detector.py --build

# You can also change the embedding model for a larger/smaller one.
"""

import argparse
import ast
import json
import logging
import math
import os
import re
from datetime import datetime
from functools import cache
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub import logging as huggingface_hub_logging
from safetensors.numpy import load_file as safetensors_load
from safetensors.numpy import save_file as safetensors_save
from tqdm import tqdm

import transformers
from transformers import AutoModel, AutoTokenizer
from transformers.utils import enable_tf32
from transformers.utils import logging as transformers_logging

# ANSI color codes for CLI output styling
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_HEADER = "\033[1;36m"
ANSI_SECTION = "\033[1;35m"
ANSI_ROW = "\033[0;37m"
ANSI_HIGHLIGHT_TOP = "\033[1;32m"
ANSI_HIGHLIGHT_OLD = "\033[1;33m"
ANSI_HIGHLIGHT_CANDIDATE = "\033[1;34m"

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

MODELS_ROOT = Path("src/transformers/models")
EMBEDDINGS_PATH = "embeddings.safetensors"
EMBEDDINGS_INT8_PATH = "embeddings_int8.safetensors"
INDEX_MAP_PATH = "code_index_map.json"
TOKENS_PATH = "code_index_tokens.json"
INDEX_MAP_METHODS_PATH = "code_index_map_methods.json"
TOKENS_METHODS_PATH = "code_index_tokens_methods.json"
HUB_DATASET_DEFAULT = "hf-internal-testing/transformers_code_embeddings"

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
BATCH_SIZE = 16
MAX_LENGTH = 4096
HYBRID_ALPHA = 0.7


def _sanitize_for_embedding(code: str, model_hint: str | None, symbol_hint: str | None) -> str:
    """
    Sanitize code for embedding by replacing model-specific identifiers with generic placeholder.

    Args:
        code (`str`): The source code to sanitize.
        model_hint (`str` or `None`): Hint about the model name (e.g., 'llama').
        symbol_hint (`str` or `None`): Hint about the symbol name (e.g., 'LlamaAttention').

    Returns:
        `str`: The sanitized code with model-specific identifiers replaced by 'Model'.
    """
    base = "\n".join(
        line
        for line in re.sub(r"#.*", "", re.sub(r'("""|\'\'\')(?:.|\n)*?\1', "", code)).splitlines()
        if not re.match(r"\s*(from|import)\s+", line)
    )
    variants = set()
    if model_hint:
        variants.add(model_hint)
        variants.add(model_hint.replace("_", ""))
        variants.add(re.sub(r"\d+", "", model_hint))
    if symbol_hint:
        match = re.match(r"^([A-Z][a-z0-9]+)", symbol_hint) or re.match(r"^([A-Za-z0-9]+)", symbol_hint)
        prefix = match.group(1) if match else ""
        if prefix:
            variants.add(prefix)
            variants.add(prefix.replace("_", ""))
            variants.add(re.sub(r"\d+", "", prefix))
    variants |= {variant.lower() for variant in list(variants)}
    sanitized = base
    for variant in sorted({x for x in variants if len(x) >= 3}, key=len, reverse=True):
        sanitized = re.sub(re.escape(variant), "Model", sanitized, flags=re.IGNORECASE)
    return sanitized


def _format_table(headers: list[str], rows: list[tuple[str, ...] | None], row_styles: list[str] | None = None) -> str:
    if not rows:
        return f"{ANSI_ROW}(no matches){ANSI_RESET}"

    widths = [len(header) for header in headers]
    for row in rows:
        if row is None:
            continue
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    header_line = " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    divider = "-+-".join("-" * widths[idx] for idx in range(len(headers)))
    total_width = sum(widths) + 3 * (len(headers) - 1)

    styled_rows = []
    style_idx = 0
    for row in rows:
        if row is None:
            styled_rows.append(f"{ANSI_SECTION}{'-' * total_width}{ANSI_RESET}")
            continue

        line = " | ".join(cell.ljust(widths[col_idx]) for col_idx, cell in enumerate(row))
        style = ANSI_ROW
        if row_styles and style_idx < len(row_styles) and row_styles[style_idx]:
            style = row_styles[style_idx]
        styled_rows.append(f"{style}{line}{ANSI_RESET}")
        style_idx += 1

    return "\n".join([f"{ANSI_SECTION}{header_line}{ANSI_RESET}", divider] + styled_rows)


def _parse_release_date(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


@cache
def _load_definition_line_map(relative_path: str) -> dict[str, int]:
    file_path = MODELS_ROOT / relative_path
    try:
        source = file_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return {}

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    line_map: dict[str, int] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            line_map[node.name] = getattr(node, "lineno", None) or 1
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        key = f"{node.name}.{child.name}"
                        line_map[key] = getattr(child, "lineno", None) or 1
    return line_map


def _resolve_definition_location(relative_path: str, definition: str) -> tuple[str, str]:
    full_path = MODELS_ROOT / relative_path
    line = _load_definition_line_map(relative_path).get(definition)
    line_str = str(line) if line is not None else "?"
    return str(full_path), line_str


def _build_display_path(relative_path: str, match_name: str) -> tuple[str, str]:
    full_path, line = _resolve_definition_location(relative_path, match_name)
    return f"{full_path}:{line} ({match_name})", line


def _colorize_heading(text: str) -> str:
    return f"{ANSI_HEADER}{ANSI_BOLD}{text}{ANSI_RESET}"


def build_date_data() -> dict[str, str]:
    root_dir = transformers.__file__.split("src/transformers")[0]
    root = Path(root_dir).joinpath("docs/source/en/model_doc")
    result: dict[str, str] = {}

    for md_path in root.glob("*.md"):
        try:
            text = md_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            logging.info(f"Failed to read md for {md_path}")
            continue

        m = re.search(
            r"(?:^|[\*_`\s>])(?:this|the)\s+model\s+was\s+released\s+on\s+(\d{4}-\d{2}-\d{2})\b",
            text,
            re.IGNORECASE,
        )
        if m:
            result[md_path.stem] = m.group(1)

    return result


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _match_entry(
    identifier: str,
    score: float,
    dates: dict[str, str],
    embedding_scores: dict[str, float] | None = None,
    jaccard_scores: dict[str, float] | None = None,
) -> dict | None:
    if ":" not in identifier:
        return None
    relative_path, match_name = identifier.split(":", 1)
    match_class, match_method = match_name.split(".", 1) if "." in match_name else (None, None)
    model_id = Path(relative_path).parts[0] if Path(relative_path).parts else "?"
    full_path, line = _resolve_definition_location(relative_path, match_name)
    entry = {
        "identifier": identifier,
        "relative_path": relative_path,
        "match_name": match_name,
        "class_name": match_class,
        "method_name": match_method,
        "score": score,
        "release_date": dates.get(model_id, "unknown release date"),
        "full_path": full_path,
        "line": int(line) if line.isdigit() else None,
    }
    if embedding_scores is not None:
        entry["embedding_score"] = embedding_scores.get(identifier)
    if jaccard_scores is not None:
        entry["jaccard_score"] = jaccard_scores.get(identifier)
    return entry


def _match_row(
    identifier: str,
    score: float,
    dates: dict[str, str],
    include_metric_column: bool,
    metric: str,
) -> tuple[tuple[str, ...], str, float, str] | None:
    if ":" not in identifier:
        return None
    relative_path, match_name = identifier.split(":", 1)
    model_id = Path(relative_path).parts[0] if Path(relative_path).parts else "?"
    match_release = dates.get(model_id, "unknown release date")
    display_path, _ = _build_display_path(relative_path, match_name)
    if include_metric_column:
        row = ("", metric, display_path, f"{score:.4f}", match_release)
    else:
        row = ("", display_path, f"{score:.4f}", match_release)
    return row, relative_path, score, match_release


def _compute_idf(tokens_map: dict[str, list[str]]) -> tuple[dict[str, float], float]:
    doc_count = len(tokens_map)
    if doc_count == 0:
        return {}, 1.0
    df: dict[str, int] = {}
    for tokens in tokens_map.values():
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1
    idf = {token: math.log((doc_count + 1) / (count + 1)) + 1.0 for token, count in df.items()}
    default_idf = math.log((doc_count + 1) / 1) + 1.0
    return idf, default_idf


def _weighted_jaccard(
    query_tokens: set[str], candidate_tokens: set[str], idf_map: dict[str, float], default_idf: float
) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    intersection = query_tokens & candidate_tokens
    if not intersection:
        return 0.0
    union = query_tokens | candidate_tokens
    union_weight = sum(idf_map.get(token, default_idf) for token in union)
    if union_weight <= 0:
        return 0.0
    intersection_weight = sum(idf_map.get(token, default_idf) for token in intersection)
    return intersection_weight / union_weight


class CodeSimilarityAnalyzer:
    """
    Analyzer for detecting code similarities between model implementations.

    This class uses embedding-based and token-based similarity metrics to identify similar
    code patterns across different model definitions in the transformers library.

    Args:
        hub_dataset (`str`): The Hub dataset repository ID containing the code embeddings index.
    """

    def __init__(self, hub_dataset: str, precision: str = "float32", granularity: str = "definition"):
        for name in ("huggingface_hub", "httpx", "urllib3", "transformers"):
            logging.getLogger(name).setLevel(logging.ERROR)
        huggingface_hub_logging.set_verbosity_error()
        transformers_logging.set_verbosity_error()
        enable_tf32(True)
        torch.set_grad_enabled(False)

        self.models_root = MODELS_ROOT
        self.hub_dataset = hub_dataset
        self.precision = precision
        self.granularity = granularity
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(EMBEDDING_MODEL, torch_dtype="auto", device_map="auto").eval()

        self.device = self.model.device
        self.dtype = self.model.dtype
        self.index_dir: Path | None = None

    # ---------- HUB IO ----------

    def _embedding_filename(self) -> str:
        suffix = ""
        if self.granularity == "method":
            suffix += "_methods"
        if self.precision == "int8":
            suffix += "_int8"
        if not suffix:
            return EMBEDDINGS_PATH
        return f"embeddings{suffix}.safetensors"

    def _index_map_filename(self) -> str:
        if self.granularity == "method":
            return INDEX_MAP_METHODS_PATH
        return INDEX_MAP_PATH

    def _tokens_filename(self) -> str:
        if self.granularity == "method":
            return TOKENS_METHODS_PATH
        return TOKENS_PATH

    def _resolve_index_path(self, filename: str) -> Path:
        if self.index_dir is None:
            return Path(filename)
        return self.index_dir / filename

    def _iter_filtered_identifiers(
        self,
        indices: np.ndarray,
        identifier_map: dict[int, str],
        self_model_normalized: str,
        self_name: str,
    ) -> list[tuple[int, str]]:
        output = []
        for match_id in indices:
            identifier = identifier_map[int(match_id)]
            if ":" not in identifier:
                continue
            relative_path, match_name = identifier.split(":", 1)
            parent_model = Path(relative_path).parts[0]
            if match_name == self_name:
                continue
            if self_model_normalized and re.sub(r"[^a-z0-9]+", "", parent_model.lower()) == self_model_normalized:
                continue
            output.append((int(match_id), identifier))
        return output

    def _required_index_files(self) -> tuple[str, ...]:
        return (self._embedding_filename(), self._index_map_filename(), self._tokens_filename())

    def ensure_local_index(self) -> None:
        """Ensure index files are available locally, preferring Hub cache snapshots."""
        required_files = self._required_index_files()
        if self.index_dir is not None and all((self.index_dir / fname).exists() for fname in required_files):
            return

        workspace_dir = Path.cwd()
        if all((workspace_dir / fname).exists() for fname in required_files):
            self.index_dir = workspace_dir
            return

        logging.info(f"downloading index from hub cache: {self.hub_dataset}")
        snapshot_path = snapshot_download(repo_id=self.hub_dataset, repo_type="dataset")
        snapshot_dir = Path(snapshot_path)
        missing = [fname for fname in required_files if not (snapshot_dir / fname).exists()]
        if missing:
            raise FileNotFoundError("Missing expected files in Hub snapshot: " + ", ".join(missing))
        self.index_dir = snapshot_dir

    def push_index_to_hub(self) -> None:
        """Upload index files to the Hub dataset repository."""
        api = HfApi()
        api.create_repo(repo_id=self.hub_dataset, repo_type="dataset", exist_ok=True)
        for fname in self._required_index_files():
            logging.info(f"pushing {fname} -> {self.hub_dataset}")
            api.upload_file(
                path_or_fileobj=fname,
                path_in_repo=os.path.basename(fname),
                repo_id=self.hub_dataset,
                repo_type="dataset",
            )

    # ---------- parsing & encoding ----------

    def _extract_definitions(
        self,
        file_path: Path,
        relative_to: Path | None = None,
        model_hint: str | None = None,
        granularity: str = "definition",
    ) -> tuple[dict[str, str], dict[str, str], dict[str, list[str]], dict[str, str]]:
        """
        Extract class and function definitions from a Python file.

        Args:
            file_path (`Path`): Path to the Python file to parse.
            relative_to (`Path` or `None`): Base path for computing relative identifiers.
            model_hint (`str` or `None`): Model name hint for sanitization.

        Returns:
            `tuple[dict[str, str], dict[str, str], dict[str, list[str]], dict[str, str]]`: A tuple containing:
                - definitions_raw: Mapping of identifiers to raw source code
                - definitions_sanitized: Mapping of identifiers to sanitized source code
                - definitions_tokens: Mapping of identifiers to sorted token lists
                - definitions_kind: Mapping of identifiers to either "class" or "function"
        """
        definitions_raw = {}
        definitions_sanitized = {}
        definitions_tokens = {}
        definitions_kind = {}
        source = file_path.read_text(encoding="utf-8")
        lines = source.splitlines()
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and granularity in ("definition", "method"):
                segment = ast.get_source_segment(source, node)
                if segment is None and hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                    start = max(0, node.lineno - 1)
                    end = node.end_lineno
                    segment = "\n".join(lines[start:end])
                if not segment:
                    continue
                identifier = (
                    f"{file_path.relative_to(relative_to)}:{node.name}"
                    if relative_to
                    else f"{file_path.name}:{node.name}"
                )
                definitions_raw[identifier] = segment
                sanitized = _sanitize_for_embedding(segment, model_hint, node.name)
                definitions_sanitized[identifier] = sanitized
                definitions_tokens[identifier] = sorted(
                    set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", sanitized))
                )
                definitions_kind[identifier] = "function"
                continue

            if isinstance(node, ast.ClassDef):
                class_segment = ast.get_source_segment(source, node)
                if class_segment is None and hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                    start = max(0, node.lineno - 1)
                    end = node.end_lineno
                    class_segment = "\n".join(lines[start:end])
                class_header = ""
                if class_segment:
                    class_header = class_segment.splitlines()[0].strip()
                class_docstring = ast.get_docstring(node)
                class_context = class_header
                if class_docstring:
                    first_line = class_docstring.strip().splitlines()[0]
                    class_context = f'{class_header}\n"""{first_line}"""' if class_header else first_line

                if granularity == "definition":
                    if not class_segment:
                        continue
                    identifier = (
                        f"{file_path.relative_to(relative_to)}:{node.name}"
                        if relative_to
                        else f"{file_path.name}:{node.name}"
                    )
                    definitions_raw[identifier] = class_segment
                    sanitized = _sanitize_for_embedding(class_segment, model_hint, node.name)
                    definitions_sanitized[identifier] = sanitized
                    definitions_tokens[identifier] = sorted(
                        set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", sanitized))
                    )
                    definitions_kind[identifier] = "class"
                    continue

                for child in node.body:
                    if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    segment = ast.get_source_segment(source, child)
                    if segment is None and hasattr(child, "lineno") and hasattr(child, "end_lineno"):
                        start = max(0, child.lineno - 1)
                        end = child.end_lineno
                        segment = "\n".join(lines[start:end])
                    if not segment:
                        continue
                    method_name = child.name
                    combined = f"{class_context}\n{segment}" if class_context else segment
                    identifier = (
                        f"{file_path.relative_to(relative_to)}:{node.name}.{method_name}"
                        if relative_to
                        else f"{file_path.name}:{node.name}.{method_name}"
                    )
                    definitions_raw[identifier] = segment
                    sanitized = _sanitize_for_embedding(combined, model_hint, node.name)
                    definitions_sanitized[identifier] = sanitized
                    definitions_tokens[identifier] = sorted(
                        set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", sanitized))
                    )
                    definitions_kind[identifier] = "method"
        return definitions_raw, definitions_sanitized, definitions_tokens, definitions_kind

    def _infer_model_from_relative_path(self, relative_path: Path) -> str | None:
        try:
            relative = relative_path.resolve().relative_to(self.models_root.resolve())
            return relative.parts[0]
        except Exception:
            return None

    def _infer_query_model_name(self, modeling_file: Path) -> str | None:
        model = self._infer_model_from_relative_path(modeling_file)
        if model:
            return model
        stem = modeling_file.stem
        if stem.startswith("modeling_") and len(stem) > len("modeling_"):
            return stem[len("modeling_") :]
        return None

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encode a batch of texts into normalized embeddings.

        Args:
            texts (`list[str]`): List of text strings to encode.

        Returns:
            `np.ndarray`: Normalized embeddings as a float32 numpy array.
        """
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with (
            torch.autocast(device_type=self.device.type, dtype=self.dtype)
            if self.device.type == "cuda"
            else torch.no_grad()
        ):
            output = self.model(**encoded)
            if hasattr(output, "last_hidden_state"):
                embeddings = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                embeddings = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
            elif hasattr(output, "pooler_output"):
                embeddings = output.pooler_output
            else:
                embeddings = output[0].mean(dim=1)
        embeddings = torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)
        return embeddings.cpu().numpy().astype("float32")

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings, processing in batches.

        Args:
            texts (`list[str]`): List of text strings to encode.

        Returns:
            `np.ndarray`: Stacked embeddings for all texts.
        """
        output = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="encode", leave=False):
            output.append(self._encode_batch(texts[i : i + BATCH_SIZE]))
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return np.vstack(output) if output else np.zeros((0, 0), dtype="float32")

    def _quantize_int8(
        self, embeddings: np.ndarray, scales: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if scales is None:
            max_abs = np.max(np.abs(embeddings), axis=0)
            max_abs = np.maximum(max_abs, 1e-8)
            scales = (max_abs / 127.0).astype("float32")
        quantized = np.round(embeddings / scales).clip(-127, 127).astype("int8")
        return quantized, scales

    def _save_embeddings(self, embeddings: np.ndarray, path: Path) -> None:
        if self.precision == "int8":
            quantized, scales = self._quantize_int8(embeddings)
            safetensors_save({"embeddings": quantized, "scale": scales}, path)
        else:
            safetensors_save({"embeddings": embeddings}, path)

    # ---------- build & search ----------

    def build_index(self) -> None:
        """Build the code similarity index from all modeling files and save to disk."""
        logging.info("collecting files")
        files = list(self.models_root.rglob("modeling_*.py"))
        logging.info(f"parsing {len(files)} files")

        identifiers = []
        sanitized_sources = []
        tokens_map = {}

        for file_path in tqdm(files, desc="parse", leave=False):
            model_hint = self._infer_model_from_relative_path(file_path)
            (
                _,
                definitions_sanitized,
                definitions_tokens,
                _,
            ) = self._extract_definitions(file_path, self.models_root, model_hint, self.granularity)
            for identifier in definitions_sanitized.keys():
                identifiers.append(identifier)
                sanitized_sources.append(definitions_sanitized[identifier])
                tokens_map[identifier] = definitions_tokens[identifier]

        logging.info(
            f"encoding {len(sanitized_sources)} definitions with {EMBEDDING_MODEL} (device={self.device.type}, batch={BATCH_SIZE}, max_length={MAX_LENGTH})"
        )
        embeddings = self.encode(sanitized_sources)
        embedding_path = self._embedding_filename()
        self._save_embeddings(embeddings, Path(embedding_path))
        with open(self._index_map_filename(), "w", encoding="utf-8") as file:
            json.dump({int(i): identifiers[i] for i in range(len(identifiers))}, file)
        with open(self._tokens_filename(), "w", encoding="utf-8") as file:
            json.dump(tokens_map, file)

        self.index_dir = Path.cwd()

    def _topk_embedding(
        self,
        query_embedding_row: np.ndarray,
        base_embeddings: np.ndarray,
        identifier_map: dict[int, str],
        self_model_normalized: str,
        self_name: str,
        k: int,
        pool_size: int | None = None,
    ) -> list[tuple[str, float]]:
        similarities = query_embedding_row @ base_embeddings.T
        pool = k + 32 if pool_size is None else max(k, pool_size)
        indices = np.argpartition(-similarities, pool)[:pool]
        indices = indices[np.argsort(-similarities[indices])]
        output = []
        for match_id, identifier in self._iter_filtered_identifiers(
            indices, identifier_map, self_model_normalized, self_name
        ):
            output.append((identifier, float(similarities[match_id])))
            if len(output) >= k:
                break
        return output

    def _combine_hybrid(
        self,
        candidates: list[tuple[str, float]],
        query_tokens: set[str],
        tokens_map: dict[str, list[str]],
        idf_map: dict[str, float],
        default_idf: float,
        k: int,
        hybrid_alpha: float = HYBRID_ALPHA,
    ) -> tuple[list[tuple[str, float]], dict[str, float], dict[str, float]]:
        embedding_scores: dict[str, float] = {}
        jaccard_scores: dict[str, float] = {}
        hybrid_scores = []
        for identifier, embedding_score in candidates:
            tokens = set(tokens_map.get(identifier, []))
            jaccard_score = _weighted_jaccard(query_tokens, tokens, idf_map, default_idf)
            embedding_scores[identifier] = embedding_score
            jaccard_scores[identifier] = jaccard_score
            hybrid = hybrid_alpha * max(0.0, embedding_score) + (1.0 - hybrid_alpha) * jaccard_score
            hybrid_scores.append((identifier, hybrid))
        hybrid_scores.sort(key=lambda item: item[1], reverse=True)
        return hybrid_scores[:k], embedding_scores, jaccard_scores

    def _topk_candidates(
        self,
        query_embedding_row: np.ndarray,
        base_embeddings: np.ndarray,
        scales: np.ndarray | None,
        identifier_map: dict[int, str],
        self_model_normalized: str,
        self_name: str,
        k: int,
        pool_size: int | None = None,
    ) -> list[tuple[str, float]]:
        if self.precision == "int8":
            if scales is None:
                raise ValueError("Missing int8 scales for int8 search.")
            return self._topk_int8(
                query_embedding_row,
                base_embeddings,
                scales,
                identifier_map,
                self_model_normalized,
                self_name,
                k,
                pool_size=pool_size,
            )
        return self._topk_embedding(
            query_embedding_row,
            base_embeddings,
            identifier_map,
            self_model_normalized,
            self_name,
            k,
            pool_size=pool_size,
        )

    def _topk_int8(
        self,
        query_embedding_row: np.ndarray,
        base_embeddings: np.ndarray,
        scales: np.ndarray,
        identifier_map: dict[int, str],
        self_model_normalized: str,
        self_name: str,
        k: int,
        pool_size: int | None = None,
    ) -> list[tuple[str, float]]:
        weighted_query = (query_embedding_row * scales).astype("float32")
        similarities = weighted_query @ base_embeddings.T.astype("float32")
        pool = k + 32 if pool_size is None else max(k, pool_size)
        indices = np.argpartition(-similarities, pool)[:pool]
        indices = indices[np.argsort(-similarities[indices])]
        output = []
        for match_id, identifier in self._iter_filtered_identifiers(
            indices, identifier_map, self_model_normalized, self_name
        ):
            output.append((identifier, float(similarities[match_id])))
            if len(output) >= k:
                break
        return output

    def _topk_jaccard(
        self,
        query_tokens: set[str],
        identifiers: list[str],
        tokens_map: dict[str, list[str]],
        idf_map: dict[str, float],
        default_idf: float,
        self_model_normalized: str,
        self_name: str,
        k: int,
    ) -> list[tuple[str, float]]:
        """
        Find top-k most similar definitions using Jaccard similarity on token sets.

        Args:
            query_tokens (`set[str]`): Set of tokens from the query definition.
            identifiers (`list[str]`): List of all definition identifiers in the index.
            tokens_map (`dict[str, list[str]]`): Mapping of identifiers to their token lists.
            self_model_normalized (`str`): Normalized name of the query model to exclude.
            self_name (`str`): Name of the query definition to exclude.
            k (`int`): Number of top results to return.

        Returns:
            `list[tuple[str, float]]`: List of (identifier, score) tuples.
        """
        scores = []
        for identifier in identifiers:
            if ":" not in identifier:
                continue
            parent_relative_path, match_name = identifier.split(":", 1)
            parent_model = Path(parent_relative_path).parts[0]
            if match_name == self_name:
                continue
            if self_model_normalized and re.sub(r"[^a-z0-9]+", "", parent_model.lower()) == self_model_normalized:
                continue
            tokens = set(tokens_map.get(identifier, []))
            score = _weighted_jaccard(query_tokens, tokens, idf_map, default_idf)
            if score > 0:
                scores.append((identifier, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def analyze_file(
        self,
        modeling_file: Path,
        top_k_per_item: int = 5,
        allow_hub_fallback: bool = True,
        use_jaccard: bool = False,
    ) -> dict[str, dict[str, list]]:
        """
        Analyze a modeling file and find similar code definitions in the index.

        Args:
            modeling_file (`Path`): Path to the modeling file to analyze.
            top_k_per_item (`int`, *optional*, defaults to 5): Number of top matches to return per definition.
            allow_hub_fallback (`bool`, *optional*, defaults to `True`): Whether to download index from Hub if not found locally.

        Returns:
            `dict[str, dict[str, list]]`: Dictionary mapping definition names to their similarity results.
                Each result contains 'embedding', 'jaccard', and 'intersection' keys.
        """
        if allow_hub_fallback:
            self.ensure_local_index()
        hybrid_alpha = HYBRID_ALPHA

        base = None
        base_embeddings = None
        scales = None
        embedding_path = self._resolve_index_path(self._embedding_filename())
        if self.precision == "int8":
            base = safetensors_load(str(embedding_path))
            base_embeddings = base["embeddings"]
            scales = base["scale"]
        else:
            base = safetensors_load(str(embedding_path))
            base_embeddings = base["embeddings"]
        with open(self._resolve_index_path(self._index_map_filename()), "r", encoding="utf-8") as file:
            identifier_map = {int(key): value for key, value in json.load(file).items()}
        identifiers = [identifier_map[i] for i in range(len(identifier_map))]
        with open(self._resolve_index_path(self._tokens_filename()), "r", encoding="utf-8") as file:
            tokens_map = json.load(file)
        idf_map, default_idf = _compute_idf(tokens_map)

        self_model = self._infer_query_model_name(modeling_file)
        definitions_raw, definitions_sanitized, _, definitions_kind = self._extract_definitions(
            modeling_file, None, self_model, self.granularity
        )
        query_identifiers = list(definitions_raw.keys())
        query_sources_sanitized = [definitions_sanitized[key] for key in query_identifiers]
        query_tokens_list = [
            set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", source)) for source in query_sources_sanitized
        ]
        self_model_normalized = re.sub(r"[^a-z0-9]+", "", self_model.lower()) if self_model else ""

        logging.info(
            f"encoding {len(query_sources_sanitized)} query definitions with {EMBEDDING_MODEL} (device={self.device.type}, batch={BATCH_SIZE}, max_length={MAX_LENGTH})"
        )
        query_embeddings = self.encode(query_sources_sanitized)

        output = {}
        for i, query_identifier in enumerate(query_identifiers):
            query_name = query_identifier.split(":")[-1]
            embedding_scores = None
            jaccard_scores = None
            pool_size = max(top_k_per_item * 5, top_k_per_item + 32)
            candidates = self._topk_candidates(
                query_embeddings[i],
                base_embeddings,
                scales,
                identifier_map,
                self_model_normalized,
                query_name,
                pool_size,
                pool_size=pool_size,
            )
            embedding_top, embedding_scores, jaccard_scores = self._combine_hybrid(
                candidates,
                query_tokens_list[i],
                tokens_map,
                idf_map,
                default_idf,
                top_k_per_item,
                hybrid_alpha,
            )
            embedding_set = {identifier for identifier, _ in embedding_top}
            kind = definitions_kind.get(query_identifier, "function")
            entry = {"kind": kind, "embedding": embedding_top}
            if embedding_scores is not None:
                entry["embedding_scores"] = embedding_scores
            if jaccard_scores is not None:
                entry["jaccard_scores"] = jaccard_scores
            if use_jaccard:
                jaccard_top = self._topk_jaccard(
                    query_tokens_list[i],
                    identifiers,
                    tokens_map,
                    idf_map,
                    default_idf,
                    self_model_normalized,
                    query_name,
                    top_k_per_item,
                )
                jaccard_set = {identifier for identifier, _ in jaccard_top}
                intersection = set(embedding_set & jaccard_set)

                entry.update({"jaccard": jaccard_top, "intersection": intersection})
            output[query_name] = entry
        return output


def main():
    """CLI entry point for the modular model detector."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(prog="hf-code-sim")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--modeling-file", type=str, help='You can just specify "vits" if you are lazy like me.')
    parser.add_argument(
        "--push-new-index", action="store_true", help="After --build, push index files to a Hub dataset."
    )
    parser.add_argument(
        "--hub-dataset", type=str, default=HUB_DATASET_DEFAULT, help="Hub dataset repo id to pull/push the index."
    )
    parser.add_argument("--use_jaccard", type=bool, default=False, help="Whether or not to use jaccard index")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k matches to return per symbol.")
    parser.add_argument(
        "--precision",
        choices=["float32", "int8"],
        default="float32",
        help="Embedding precision for building/loading the index.",
    )
    parser.add_argument(
        "--granularity",
        choices=["definition", "method"],
        default="definition",
        help="Index granularity: definitions (classes/functions) or class methods.",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Write results to JSON for tooling.")
    args = parser.parse_args()

    analyzer = CodeSimilarityAnalyzer(
        hub_dataset=args.hub_dataset, precision=args.precision, granularity=args.granularity
    )

    if args.build:
        analyzer.build_index()
        if args.push_new_index:
            analyzer.push_index_to_hub()
        return

    if not args.modeling_file:
        raise SystemExit("Provide --modeling-file or use --build")

    dates = build_date_data()
    modeling_file = args.modeling_file
    if os.sep not in modeling_file:
        modeling_file = os.path.join("src", "transformers", "models", modeling_file, f"modeling_{modeling_file}.py")

    results = analyzer.analyze_file(
        Path(modeling_file),
        top_k_per_item=args.top_k,
        allow_hub_fallback=True,
        use_jaccard=args.use_jaccard,
    )
    modeling_filename = Path(modeling_file).name
    release_key = modeling_filename.split("modeling_")[-1][:-3]
    release_date = dates.get(release_key, "unknown release date")

    aggregate_scores: dict[str, float] = {}
    for data in results.values():
        for identifier, score in data.get("embedding", []):
            if ":" not in identifier:
                continue
            relative_path, _ = identifier.split(":", 1)
            aggregate_scores[relative_path] = aggregate_scores.get(relative_path, 0.0) + score

    best_candidate_path: str | None = None
    if aggregate_scores:
        best_candidate_path = max(aggregate_scores.items(), key=lambda item: item[1])[0]
        best_model = Path(best_candidate_path).parts[0] if Path(best_candidate_path).parts else "?"
        best_release = dates.get(best_model, "unknown release date")
        logging.info(
            f"{ANSI_HIGHLIGHT_CANDIDATE}Closest overall candidate: {MODELS_ROOT / best_candidate_path}"
            f" (release: {best_release}, total score: {aggregate_scores[best_candidate_path]:.4f}){ANSI_RESET}"
        )

    if args.output_json:
        payload = {
            "modeling_file": str(Path(modeling_file)),
            "precision": args.precision,
            "granularity": args.granularity,
            "score_mode": "hybrid",
            "hybrid_alpha": HYBRID_ALPHA,
            "release_date": release_date,
            "best_candidate": None,
            "results": {},
        }
        if best_candidate_path is not None:
            best_model = Path(best_candidate_path).parts[0] if Path(best_candidate_path).parts else "?"
            payload["best_candidate"] = {
                "relative_path": best_candidate_path,
                "full_path": str(MODELS_ROOT / best_candidate_path),
                "release_date": dates.get(best_model, "unknown release date"),
                "score": aggregate_scores.get(best_candidate_path, 0.0),
            }
        for query_name, data in results.items():
            class_name = None
            method_name = None
            if "." in query_name:
                class_name, method_name = query_name.split(".", 1)
            embedding_scores = data.get("embedding_scores", {})
            jaccard_scores = data.get("jaccard_scores", {})
            entry = {
                "kind": data.get("kind", "function"),
                "class_name": class_name,
                "method_name": method_name,
                "embedding": [],
            }
            for identifier, score in data.get("embedding", []):
                match_entry = _match_entry(identifier, score, dates, embedding_scores, jaccard_scores)
                if match_entry:
                    entry["embedding"].append(match_entry)
            if args.use_jaccard:
                entry["jaccard"] = []
                for identifier, score in data.get("jaccard", []):
                    match_entry = _match_entry(identifier, score, dates)
                    if match_entry:
                        entry["jaccard"].append(match_entry)
                entry["intersection"] = sorted(data.get("intersection", []))
            payload["results"][query_name] = entry
        write_json(Path(args.output_json), payload)

    grouped: dict[str, list[tuple[str, dict]]] = {"class": [], "function": []}
    for query_name, data in results.items():
        kind = data.get("kind", "function")
        grouped.setdefault(kind, []).append((query_name, data))

    section_titles = [("class", "Classes"), ("method", "Methods"), ("function", "Functions")]
    legend_shown = False
    for kind, title in section_titles:
        entries = grouped.get(kind, [])
        if not entries:
            continue

        metrics_present: set[str] = set()
        for _, data in entries:
            if data.get("embedding"):
                metrics_present.add("embedding")
            if args.use_jaccard:
                if data.get("jaccard"):
                    metrics_present.add("jaccard")
                if data.get("intersection"):
                    metrics_present.add("intersection")

        include_metric_column = bool(metrics_present - {"embedding"})
        score_label = "Score (hybrid)"
        headers = ["Symbol", "Path", score_label, "Release"]
        if include_metric_column:
            headers = ["Symbol", "Metric", "Path", score_label, "Release"]

        table_rows: list[tuple[str, ...] | None] = []
        row_styles: list[str] = []
        has_metric_rows = False

        logging.info(_colorize_heading(title))

        for query_name, data in entries:
            if table_rows:
                table_rows.append(None)

            symbol_label = query_name

            symbol_row = (symbol_label,) + ("",) * (len(headers) - 1)
            table_rows.append(symbol_row)
            row_styles.append(ANSI_BOLD)

            embedding_details: list[tuple[str, float, str]] = []
            embedding_style_indices: list[int] = []

            for identifier, score in data.get("embedding", []):
                match_row = _match_row(identifier, score, dates, include_metric_column, "embedding")
                if match_row is None:
                    continue
                row, relative_path, score, match_release = match_row

                table_rows.append(row)
                row_styles.append(ANSI_ROW)
                embedding_style_indices.append(len(row_styles) - 1)
                embedding_details.append((relative_path, score, match_release))
                has_metric_rows = True

            if embedding_details:
                highest_idx = max(range(len(embedding_details)), key=lambda idx: embedding_details[idx][1])
                highest_score = embedding_details[highest_idx][1]

                row_styles[embedding_style_indices[highest_idx]] = ANSI_HIGHLIGHT_TOP

                if highest_score is not None:
                    oldest_idx = None
                    oldest_date = None
                    for idx, (_, score, release_value) in enumerate(embedding_details):
                        if highest_score - score > 0.1:
                            continue
                        parsed = _parse_release_date(release_value)
                        if parsed is None:
                            continue
                        if oldest_date is None or parsed < oldest_date:
                            oldest_date = parsed
                            oldest_idx = idx
                    if (
                        oldest_idx is not None
                        and row_styles[embedding_style_indices[oldest_idx]] != ANSI_HIGHLIGHT_TOP
                    ):
                        row_styles[embedding_style_indices[oldest_idx]] = ANSI_HIGHLIGHT_OLD

                if best_candidate_path is not None:
                    for idx, (relative_path, _, _) in enumerate(embedding_details):
                        style_position = embedding_style_indices[idx]
                        if row_styles[style_position] != ANSI_ROW:
                            continue
                        if relative_path == best_candidate_path:
                            row_styles[style_position] = ANSI_HIGHLIGHT_CANDIDATE

            if args.use_jaccard:
                for identifier, score in data.get("jaccard", []):
                    match_row = _match_row(identifier, score, dates, include_metric_column, "jaccard")
                    if match_row is None:
                        continue
                    row, relative_path, _, _ = match_row

                    table_rows.append(row)
                    row_styles.append(ANSI_ROW)
                    has_metric_rows = True
                    if best_candidate_path == relative_path:
                        row_styles[-1] = ANSI_HIGHLIGHT_CANDIDATE

                for identifier in sorted(data.get("intersection", [])):
                    match_row = _match_row(identifier, 0.0, dates, include_metric_column, "intersection")
                    if match_row is None:
                        continue
                    row, relative_path, _, match_release = match_row
                    row = tuple("--" if cell == f"{0.0:.4f}" else cell for cell in row)

                    table_rows.append(row)
                    row_styles.append(ANSI_ROW)
                    has_metric_rows = True
                    if best_candidate_path == relative_path:
                        row_styles[-1] = ANSI_HIGHLIGHT_CANDIDATE

        if table_rows:
            if not legend_shown and has_metric_rows:
                logging.info(
                    "Legend: "
                    f"{ANSI_HIGHLIGHT_TOP}highest match{ANSI_RESET}, "
                    f"{ANSI_HIGHLIGHT_OLD}oldest within 0.1{ANSI_RESET}, "
                    f"{ANSI_HIGHLIGHT_CANDIDATE}closest overall candidate{ANSI_RESET}"
                )
                legend_shown = True

            logging.info(_format_table(headers, table_rows, row_styles))
            logging.info("")


if __name__ == "__main__":
    main()
