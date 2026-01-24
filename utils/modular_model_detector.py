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
INDEX_MAP_PATH = "code_index_map.json"
TOKENS_PATH = "code_index_tokens.json"
HUB_DATASET_DEFAULT = "hf-internal-testing/transformers_code_embeddings"

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
BATCH_SIZE = 16
MAX_LENGTH = 4096


def _normalize(string: str | None) -> str:
    """
    Normalize a string by removing all non-alphanumeric characters and converting to lowercase.

    Args:
        string (`str` or `None`): The string to normalize.

    Returns:
        `str`: The normalized string, or empty string if input is None.
    """
    return re.sub(r"[^a-z0-9]+", "", string.lower()) if string else ""


def _strip_source_for_tokens(code: str) -> str:
    """
    Strip docstrings, comments, and import statements from source code.

    Args:
        code (`str`): The source code to strip.

    Returns:
        `str`: The stripped source code.
    """
    code = re.sub(r'("""|\'\'\')(?:.|\n)*?\1', "", code)
    code = re.sub(r"#.*", "", code)
    return "\n".join(line for line in code.splitlines() if not re.match(r"\s*(from|import)\s+", line))


def _tokenize(code: str) -> set[str]:
    """
    Extract all Python identifiers from source code.

    Args:
        code (`str`): The source code to tokenize.

    Returns:
        `set[str]`: A set of all identifiers found in the code.
    """
    return set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code))


def _leading_symbol_prefix(name: str) -> str:
    """
    Extract the leading prefix from a symbol name (e.g., 'Llama' from 'LlamaAttention').

    Args:
        name (`str`): The symbol name to extract prefix from.

    Returns:
        `str`: The leading prefix, or empty string if no match.
    """
    match = re.match(r"^([A-Z][a-z0-9]+)", name) or re.match(r"^([A-Za-z0-9]+)", name)
    return match.group(1) if match else ""


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
    base = _strip_source_for_tokens(code)
    variants = set()
    if model_hint:
        variants.add(model_hint)
        variants.add(model_hint.replace("_", ""))
        variants.add(re.sub(r"\d+", "", model_hint))
    if symbol_hint:
        prefix = _leading_symbol_prefix(symbol_hint)
        if prefix:
            variants.add(prefix)
            variants.add(prefix.replace("_", ""))
            variants.add(re.sub(r"\d+", "", prefix))
    variants |= {variant.lower() for variant in list(variants)}
    sanitized = base
    for variant in sorted({x for x in variants if len(x) >= 3}, key=len, reverse=True):
        sanitized = re.sub(re.escape(variant), "Model", sanitized, flags=re.IGNORECASE)
    return sanitized


class CodeSimilarityAnalyzer:
    """
    Analyzer for detecting code similarities between model implementations.

    This class uses embedding-based and token-based similarity metrics to identify similar
    code patterns across different model definitions in the transformers library.

    Args:
        hub_dataset (`str`): The Hub dataset repository ID containing the code embeddings index.
    """

    def __init__(self, hub_dataset: str):
        for name in ("huggingface_hub", "httpx", "urllib3", "transformers"):
            logging.getLogger(name).setLevel(logging.ERROR)
        huggingface_hub_logging.set_verbosity_error()
        transformers_logging.set_verbosity_error()
        enable_tf32(True)
        torch.set_grad_enabled(False)

        self.models_root = MODELS_ROOT
        self.hub_dataset = hub_dataset
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(EMBEDDING_MODEL, torch_dtype="auto", device_map="auto").eval()

        self.device = self.model.device
        self.index_dir: Path | None = None

    # ---------- HUB IO ----------

    def _resolve_index_path(self, filename: str) -> Path:
        if self.index_dir is None:
            return Path(filename)
        return self.index_dir / filename

    def ensure_local_index(self) -> None:
        """Ensure index files are available locally, preferring Hub cache snapshots."""
        if self.index_dir is not None and all(
            (self.index_dir / fname).exists() for fname in (EMBEDDINGS_PATH, INDEX_MAP_PATH, TOKENS_PATH)
        ):
            return

        workspace_dir = Path.cwd()
        if all((workspace_dir / fname).exists() for fname in (EMBEDDINGS_PATH, INDEX_MAP_PATH, TOKENS_PATH)):
            self.index_dir = workspace_dir
            return

        logging.info(f"downloading index from hub cache: {self.hub_dataset}")
        snapshot_path = snapshot_download(repo_id=self.hub_dataset, repo_type="dataset")
        snapshot_dir = Path(snapshot_path)
        missing = [
            fname for fname in (EMBEDDINGS_PATH, INDEX_MAP_PATH, TOKENS_PATH) if not (snapshot_dir / fname).exists()
        ]
        if missing:
            raise FileNotFoundError("Missing expected files in Hub snapshot: " + ", ".join(missing))
        self.index_dir = snapshot_dir

    def push_index_to_hub(self) -> None:
        """Upload index files to the Hub dataset repository."""
        api = HfApi()
        api.create_repo(repo_id=self.hub_dataset, repo_type="dataset", exist_ok=True)
        for fname in (EMBEDDINGS_PATH, INDEX_MAP_PATH, TOKENS_PATH):
            logging.info(f"pushing {fname} -> {self.hub_dataset}")
            api.upload_file(
                path_or_fileobj=fname,
                path_in_repo=os.path.basename(fname),
                repo_id=self.hub_dataset,
                repo_type="dataset",
            )

    # ---------- parsing & encoding ----------

    def _extract_definitions(
        self, file_path: Path, relative_to: Path | None = None, model_hint: str | None = None
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
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                segment = ast.get_source_segment(source, node)
                if segment is None and hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                    start = max(0, node.lineno - 1)
                    end = node.end_lineno
                    segment = "\n".join(lines[start:end])
                if segment:
                    identifier = (
                        f"{file_path.relative_to(relative_to)}:{node.name}"
                        if relative_to
                        else f"{file_path.name}:{node.name}"
                    )
                    definitions_raw[identifier] = segment
                    sanitized = _sanitize_for_embedding(segment, model_hint, node.name)
                    definitions_sanitized[identifier] = sanitized
                    definitions_tokens[identifier] = sorted(_tokenize(sanitized))
                    if isinstance(node, ast.ClassDef):
                        definitions_kind[identifier] = "class"
                    else:
                        definitions_kind[identifier] = "function"
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
            ) = self._extract_definitions(file_path, self.models_root, model_hint)
            for identifier in definitions_sanitized.keys():
                identifiers.append(identifier)
                sanitized_sources.append(definitions_sanitized[identifier])
                tokens_map[identifier] = definitions_tokens[identifier]

        logging.info(
            f"encoding {len(sanitized_sources)} definitions with {EMBEDDING_MODEL} (device={self.device.type}, batch={BATCH_SIZE}, max_length={MAX_LENGTH})"
        )
        embeddings = self.encode(sanitized_sources)
        safetensors_save({"embeddings": embeddings}, EMBEDDINGS_PATH)
        with open(INDEX_MAP_PATH, "w", encoding="utf-8") as file:
            json.dump({int(i): identifiers[i] for i in range(len(identifiers))}, file)
        with open(TOKENS_PATH, "w", encoding="utf-8") as file:
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
    ) -> list[tuple[str, float]]:
        similarities = query_embedding_row @ base_embeddings.T
        indices = np.argpartition(-similarities, k + 32)[: k + 32]
        indices = indices[np.argsort(-similarities[indices])]
        output = []
        for match_id in indices:
            identifier = identifier_map[int(match_id)]
            parent_relative_path, match_name = identifier.split(":", 1)
            parent_model = Path(parent_relative_path).parts[0]
            if match_name == self_name:
                continue
            if self_model_normalized and _normalize(parent_model) == self_model_normalized:
                continue
            output.append((identifier, float(similarities[match_id])))
            if len(output) >= k:
                break
        return output

    def _topk_jaccard(
        self,
        query_tokens: set[str],
        identifiers: list[str],
        tokens_map: dict[str, list[str]],
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
            parent_relative_path, match_name = identifier.split(":", 1)
            parent_model = Path(parent_relative_path).parts[0]
            if match_name == self_name:
                continue
            if self_model_normalized and _normalize(parent_model) == self_model_normalized:
                continue
            tokens = set(tokens_map.get(identifier, []))
            if not tokens or not query_tokens:
                continue
            score = len(query_tokens & tokens) / len(query_tokens | tokens)
            if score > 0:
                scores.append((identifier, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def analyze_file(
        self, modeling_file: Path, top_k_per_item: int = 5, allow_hub_fallback: bool = True, use_jaccard=False
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

        base = safetensors_load(str(self._resolve_index_path(EMBEDDINGS_PATH)))
        base_embeddings = base["embeddings"]
        with open(self._resolve_index_path(INDEX_MAP_PATH), "r", encoding="utf-8") as file:
            identifier_map = {int(key): value for key, value in json.load(file).items()}
        identifiers = [identifier_map[i] for i in range(len(identifier_map))]
        with open(self._resolve_index_path(TOKENS_PATH), "r", encoding="utf-8") as file:
            tokens_map = json.load(file)

        self_model = self._infer_query_model_name(modeling_file)
        definitions_raw, definitions_sanitized, _, definitions_kind = self._extract_definitions(
            modeling_file, None, self_model
        )
        query_identifiers = list(definitions_raw.keys())
        query_sources_sanitized = [definitions_sanitized[key] for key in query_identifiers]
        query_tokens_list = [set(_tokenize(source)) for source in query_sources_sanitized]
        self_model_normalized = _normalize(self_model)

        logging.info(
            f"encoding {len(query_sources_sanitized)} query definitions with {EMBEDDING_MODEL} (device={self.device.type}, batch={BATCH_SIZE}, max_length={MAX_LENGTH})"
        )
        query_embeddings = self.encode(query_sources_sanitized)

        output = {}
        for i, query_identifier in enumerate(query_identifiers):
            query_name = query_identifier.split(":")[-1]
            embedding_top = self._topk_embedding(
                query_embeddings[i], base_embeddings, identifier_map, self_model_normalized, query_name, top_k_per_item
            )
            embedding_set = {identifier for identifier, _ in embedding_top}
            kind = definitions_kind.get(query_identifier, "function")
            entry = {"kind": kind, "embedding": embedding_top}
            if use_jaccard:
                jaccard_top = self._topk_jaccard(
                    query_tokens_list[i], identifiers, tokens_map, self_model_normalized, query_name, top_k_per_item
                )
                jaccard_set = {identifier for identifier, _ in jaccard_top}
                intersection = set(embedding_set & jaccard_set)

                entry.update({"jaccard": jaccard_top, "intersection": intersection})
            output[query_name] = entry
        return output


_RELEASE_RE = re.compile(
    r"(?:^|[\*_`\s>])(?:this|the)\s+model\s+was\s+released\s+on\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE
)


def build_date_data() -> dict[str, str]:
    """
    Scan Markdown files in `root_dir` and build {model_id: date_released}.

    - model_id is the filename without extension (e.g., "llama" for "llama.md")
    - date_released is the first YYYY-MM-DD matched after "...was released on ..."
    - Ignores non-*.md files and directories.

    Returns:
        dict[str, str]: mapping of model_id -> ISO date string (YYYY-MM-DD).
                        Files without a match are simply omitted.
    """

    root_dir = transformers.__file__.split("src/transformers")[0]
    root = Path(root_dir).joinpath("docs/source/en/model_doc")
    result: dict[str, str] = {}

    for md_path in root.glob("*.md"):
        try:
            text = md_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            # Skip unreadable files quietly
            logging.info(f"Failed to read md for {md_path}")

        m = _RELEASE_RE.search(text)
        if m:
            model_id = md_path.stem  # e.g., "llama" from "llama.md"
            result[model_id] = m.group(1)

    return result


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
    """Return a datetime parsed from YYYY-MM-DD strings, otherwise None."""
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


@cache
def _load_definition_line_map(relative_path: str) -> dict[str, int]:
    """Return {definition_name: line_number} for top-level definitions in the given file."""
    file_path = MODELS_ROOT / relative_path
    try:
        source = file_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return {}  # gracefully keep going

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    line_map: dict[str, int] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            line_map[node.name] = getattr(node, "lineno", None) or 1
        elif isinstance(node, ast.Assign):
            continue
    return line_map


def _resolve_definition_location(relative_path: str, definition: str) -> tuple[str, str]:
    """Return full path and formatted line number string for the given definition."""
    full_path = MODELS_ROOT / relative_path
    line = _load_definition_line_map(relative_path).get(definition)
    line_str = str(line) if line is not None else "?"
    return str(full_path), line_str


def _colorize_heading(text: str) -> str:
    return f"{ANSI_HEADER}{ANSI_BOLD}{text}{ANSI_RESET}"


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
    args = parser.parse_args()

    analyzer = CodeSimilarityAnalyzer(hub_dataset=args.hub_dataset)

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
        Path(modeling_file), top_k_per_item=5, allow_hub_fallback=True, use_jaccard=args.use_jaccard
    )
    modeling_filename = Path(modeling_file).name
    release_key = modeling_filename.split("modeling_")[-1][:-3]
    release_date = dates.get(release_key, "unknown release date")

    aggregate_scores: dict[str, float] = {}
    for data in results.values():
        for identifier, score in data.get("embedding", []):
            try:
                relative_path, _ = identifier.split(":", 1)
            except ValueError:
                continue
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

    grouped: dict[str, list[tuple[str, dict]]] = {"class": [], "function": []}
    for query_name, data in results.items():
        kind = data.get("kind", "function")
        grouped.setdefault(kind, []).append((query_name, data))

    section_titles = [("class", "Classes"), ("function", "Functions")]
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
        headers = ["Symbol", "Path", "Score", "Release"]
        if include_metric_column:
            headers = ["Symbol", "Metric", "Path", "Score", "Release"]

        table_rows: list[tuple[str, ...] | None] = []
        row_styles: list[str] = []
        has_metric_rows = False

        logging.info(_colorize_heading(title))

        for query_name, data in entries:
            if table_rows:
                table_rows.append(None)

            symbol_label = query_name
            if release_date:
                symbol_label = f"{symbol_label}"

            symbol_row = (symbol_label,) + ("",) * (len(headers) - 1)
            table_rows.append(symbol_row)
            row_styles.append(ANSI_BOLD)

            embedding_details: list[tuple[str, str, str, float, str]] = []
            embedding_style_indices: list[int] = []

            for identifier, score in data.get("embedding", []):
                try:
                    relative_path, match_name = identifier.split(":", 1)
                except ValueError:
                    continue
                model_id = Path(relative_path).parts[0] if Path(relative_path).parts else "?"
                match_release = dates.get(model_id, "unknown release date")
                full_path, line = _resolve_definition_location(relative_path, match_name)
                display_path = f"{full_path}:{line} ({match_name})"

                if include_metric_column:
                    row = ("", "embedding", display_path, f"{score:.4f}", match_release)
                else:
                    row = ("", display_path, f"{score:.4f}", match_release)

                table_rows.append(row)
                row_styles.append(ANSI_ROW)
                embedding_style_indices.append(len(row_styles) - 1)
                embedding_details.append((relative_path, model_id, match_name, score, match_release))
                has_metric_rows = True

            if embedding_details:
                highest_score = None
                highest_idx = None
                for idx, (_, _, _, score, _) in enumerate(embedding_details):
                    if highest_score is None or score > highest_score:
                        highest_score = score
                        highest_idx = idx

                if highest_idx is not None:
                    row_styles[embedding_style_indices[highest_idx]] = ANSI_HIGHLIGHT_TOP

                if highest_score is not None:
                    oldest_idx = None
                    oldest_date = None
                    for idx, (_, model_id, _, score, release_value) in enumerate(embedding_details):
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
                    for idx, (relative_path, _, _, _, _) in enumerate(embedding_details):
                        style_position = embedding_style_indices[idx]
                        if row_styles[style_position] != ANSI_ROW:
                            continue
                        if relative_path == best_candidate_path:
                            row_styles[style_position] = ANSI_HIGHLIGHT_CANDIDATE

            if args.use_jaccard:
                for identifier, score in data.get("jaccard", []):
                    try:
                        relative_path, match_name = identifier.split(":", 1)
                    except ValueError:
                        continue
                    model_id = Path(relative_path).parts[0] if Path(relative_path).parts else "?"
                    match_release = dates.get(model_id, "unknown release date")
                    full_path, line = _resolve_definition_location(relative_path, match_name)
                    display_path = f"{full_path}:{line} ({match_name})"

                    if include_metric_column:
                        row = ("", "jaccard", display_path, f"{score:.4f}", match_release)
                    else:
                        row = ("", display_path, f"{score:.4f}", match_release)

                    table_rows.append(row)
                    row_styles.append(ANSI_ROW)
                    has_metric_rows = True
                    if best_candidate_path == relative_path:
                        row_styles[-1] = ANSI_HIGHLIGHT_CANDIDATE

                for identifier in sorted(data.get("intersection", [])):
                    try:
                        relative_path, match_name = identifier.split(":", 1)
                    except ValueError:
                        continue
                    model_id = Path(relative_path).parts[0] if Path(relative_path).parts else "?"
                    match_release = dates.get(model_id, "unknown release date")
                    full_path, line = _resolve_definition_location(relative_path, match_name)
                    display_path = f"{full_path}:{line} ({match_name})"

                    if include_metric_column:
                        row = ("", "intersection", display_path, "--", match_release)
                    else:
                        row = ("", display_path, "--", match_release)

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
