#!/usr/bin/env python3
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
Run the modular model detector on every model in the eval dataset, save summary results to JSON,
and evaluate whether the ground-truth base model(s) appear in the top-1 / top-k suggestions.

Usage (from repo root):

    # Default: evaluate `original_modeling_code` from Hub dataset rows.
    python utils/run_modular_detector_eval.py --output results.json

    # Legacy JSON mode (local files):
    python utils/run_modular_detector_eval.py --eval-source json --eval-dataset modular_model_eval.json --output results.json

JSON mode expects a list of dicts with keys: model, modular_file, bases (list of model ids).
Hub mode loads rows from a dataset split and expects columns:
    model_name, original_modeling_code, current_modular_code, bases
"""

import argparse
import gc
import json
import logging
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from datasets import load_dataset


# Allow importing modular_model_detector when run as python utils/run_modular_detector_eval.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modular_model_detector import (
    CodeSimilarityAnalyzer,
    compute_model_class_match_summary,
)


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ANSI color codes
ANSI_RESET = "\033[0m"
ANSI_GREEN = "\033[1;32m"


LIGHT_MODELS: set[str] = {
    # Only models that have non-empty `bases` in modular_model_eval.json
    # and have a small number of top-level class/function definitions (<= 15)
    # to keep quick evals fast.
    "colqwen2",
    "glm46v",
    "lfm2_vl",
    "deepseek_vl",
    "fast_vlm",
    "perception_lm",
    "voxtral",
    "audioflamingo3",
    "lighton_ocr",
    "mistral3",
    "biogpt",
    "deepseek_vl_hybrid",
    "llava_next_video",
    "falcon_mamba",
    "prompt_depth_anything",
}


FILTERED_MODELS: set[str] = {
    # Models currently selected by filter_modular_dataset.py in itazap/modeling-dataset
    "biogpt",
    "camembert",
    "conditional_detr",
    "deepseek_v2",
    "deepseek_v3",
    "deformable_detr",
    "falcon_mamba",
    "gpt_neox",
    "granite",
    "granitemoe",
    "hubert",
    "hunyuan_v1_moe",
    "jetmoe",
    "mistral",
    "olmo",
    "olmoe",
    "paddleocr_vl",
    "persimmon",
    "phi",
    "phi3",
    "phimoe",
    "qwen2",
    "qwen2_moe",
    "sew",
    "switch_transformers",
    "unispeech",
    "unispeech_sat",
    "wavlm",
    "xlm_roberta",
}


def load_eval_dataset(path: Path) -> list[dict]:
    """Load eval dataset from a JSON file (list of {model, modular_file, bases})."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Eval dataset JSON must be a list of entries.")
    return data


def load_eval_dataset_from_hub(dataset_repo: str, split: str) -> list[dict]:
    """Load and filter hub dataset rows into eval entries."""
    ds = load_dataset(dataset_repo, split=split)

    entries = []
    for row in ds:
        model_id = row.get("model_name")
        original_modeling_code = row.get("original_modeling_code")
        current_modular_code = row.get("current_modular_code")
        bases = row.get("bases") or []

        if not model_id or not original_modeling_code or not current_modular_code or not bases:
            continue

        if model_id not in FILTERED_MODELS:
            continue

        entries.append(
            {
                "model": model_id,
                "bases": bases,
                "original_modeling_code": original_modeling_code,
                "source": f"hf://datasets/{dataset_repo}/{split}/{model_id}",
            }
        )

    return entries


def clear_runtime_cache() -> None:
    """Best-effort memory cleanup between model evaluations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main():
    parser = argparse.ArgumentParser(
        description="Run modular detector on eval dataset, save results, and compute accuracy."
    )
    parser.add_argument(
        "--eval-source",
        type=str,
        choices=["hub-original", "json"],
        default="hub-original",
        help="Where to load eval entries from: hub dataset original code (default) or legacy JSON file.",
    )
    parser.add_argument(
        "--eval-dataset",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "modular_model_eval_dataset.json",
        help="Path to eval dataset JSON for --eval-source json.",
    )
    parser.add_argument(
        "--eval-hub-repo",
        type=str,
        default="itazap/modeling-dataset",
        help="Hub dataset repo id for --eval-source hub-original.",
    )
    parser.add_argument(
        "--eval-hub-split",
        type=str,
        default="train",
        help="Hub dataset split name for --eval-source hub-original.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("modular_detector_eval_results.json"),
        help="Path to write per-model results and summary.",
    )
    parser.add_argument(
        "--hub-dataset",
        type=str,
        default="itazap/transformers_code_embeddings_v3",
        help="Hub dataset repo id for the code embeddings index.",
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="If set, restrict eval to a small subset of 'light' models for quick runs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, run only on the first N eval entries (for quick tests).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker threads.",
    )
    parser.add_argument(
        "--clear-cache-each-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear Python/CUDA caches after each model run (enabled by default).",
    )
    parser.add_argument(
        "--reload-analyzer-each-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recreate and unload CodeSimilarityAnalyzer for each entry to minimize persistent GPU memory.",
    )
    args = parser.parse_args()

    if args.eval_source == "json":
        if not args.eval_dataset.exists():
            logger.error("Eval dataset not found at %s", args.eval_dataset)
            logger.info("Generate it or download from https://huggingface.co/datasets/itazap/modular-model-eval")
            sys.exit(1)
        eval_entries = load_eval_dataset(args.eval_dataset)
    else:
        eval_entries = load_eval_dataset_from_hub(args.eval_hub_repo, args.eval_hub_split)
        logger.info(
            "Loaded %d eval entries from %s (%s), filtered to predefined modular model set",
            len(eval_entries),
            args.eval_hub_repo,
            args.eval_hub_split,
        )

    if args.light:
        before = len(eval_entries)
        eval_entries = [entry for entry in eval_entries if entry.get("model") in LIGHT_MODELS]
        logger.info(
            "Filtered eval dataset to %d 'light' models (from %d total)",
            len(eval_entries),
            before,
        )
    if args.limit is not None:
        eval_entries = eval_entries[: args.limit]
        logger.info("Limited to first %d entries", args.limit)

    analyzer = None
    if not args.reload_analyzer_each_run:
        analyzer = CodeSimilarityAnalyzer(hub_dataset=args.hub_dataset)
        analyzer.ensure_local_index()

    def process_entry(args_tuple):
        i, entry = args_tuple
        model_id = entry["model"]

        modeling_file: Path | None = None
        temp_dir: tempfile.TemporaryDirectory | None = None

        if entry.get("original_modeling_code"):
            temp_dir = tempfile.TemporaryDirectory(prefix=f"modular_eval_{model_id}_")
            modeling_file = Path(temp_dir.name) / f"modeling_{model_id}.py"
            modeling_file.write_text(entry["original_modeling_code"], encoding="utf-8")
        else:
            repo_root = Path(__file__).resolve().parent.parent
            modular_file = Path(entry["modular_file"])
            if not modular_file.is_absolute():
                modular_file = repo_root / modular_file
            modeling_file = modular_file.parent / f"modeling_{model_id}.py"

        if not modeling_file.exists():
            logger.warning("Skipping %s: modeling file not found at %s", model_id, modeling_file)
            if temp_dir is not None:
                temp_dir.cleanup()
            return model_id, {
                "error": "modeling file not found",
                "modeling_file": str(modeling_file),
                "bases": entry.get("bases", []),
                "source": entry.get("source"),
            }

        logger.info("[%d/%d] Running detector on %s", i + 1, len(eval_entries), model_id)
        local_analyzer = analyzer
        if args.reload_analyzer_each_run:
            local_analyzer = CodeSimilarityAnalyzer(hub_dataset=args.hub_dataset)
            local_analyzer.ensure_local_index()
        try:
            raw_results = local_analyzer.analyze_file(
                modeling_file,
                top_k_per_item=12,
                allow_hub_fallback=True,
                use_jaccard=True,
            )
            total_classes, summary_list = compute_model_class_match_summary(raw_results)
        except Exception as e:
            logger.warning("Detector failed for %s: %s", model_id, e)
            if temp_dir is not None:
                temp_dir.cleanup()
            if args.reload_analyzer_each_run and local_analyzer is not None:
                del local_analyzer
            if args.clear_cache_each_run:
                clear_runtime_cache()
            return model_id, {
                "error": str(e),
                "modeling_file": str(modeling_file),
                "bases": entry.get("bases", []),
                "source": entry.get("source"),
            }

        if temp_dir is not None:
            temp_dir.cleanup()
        if args.reload_analyzer_each_run and local_analyzer is not None:
            del local_analyzer
        if args.clear_cache_each_run:
            clear_runtime_cache()

        return model_id, {
            "modeling_file": str(modeling_file),
            "total_classes": total_classes,
            "models_with_most_matched_classes": summary_list,
            "bases": entry.get("bases", []),
            "source": entry.get("source"),
        }

    results_by_model: dict[str, dict] = {}
    if args.workers == 1:
        for i, entry in enumerate(eval_entries):
            model_id, result = process_entry((i, entry))
            results_by_model[model_id] = result
    else:
        logger.warning(
            "Running with workers=%d may increase peak memory usage and cause OOM.",
            args.workers,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_entry, (i, entry)): entry for i, entry in enumerate(eval_entries)}
            for future in as_completed(futures):
                model_id, result = future.result()
                results_by_model[model_id] = result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "eval_source": args.eval_source,
        "eval_dataset": str(args.eval_dataset) if args.eval_source == "json" else None,
        "eval_hub_repo": args.eval_hub_repo if args.eval_source == "hub-original" else None,
        "eval_hub_split": args.eval_hub_split if args.eval_source == "hub-original" else None,
        "results_by_model": results_by_model,
    }
    args.output.write_text(json.dumps(output_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote results to %s", args.output)

    # Evaluate: top-1 and top-k accuracy
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    total_with_bases = 0
    total_with_summary = 0

    for model_id, data in results_by_model.items():
        bases = data.get("bases", [])
        if not bases:
            continue
        total_with_bases += 1
        summary = data.get("models_with_most_matched_classes", [])
        if not summary:
            if not data.get("error"):
                total_with_summary += 1
            continue
        total_with_summary += 1
        top_ids = [s["model_id"] for s in summary]
        if top_ids and top_ids[0] in bases:
            correct_top1 += 1
        if any(m in bases for m in top_ids[:3]):
            correct_top3 += 1
        if any(m in bases for m in top_ids[:5]):
            correct_top5 += 1

    n = total_with_summary
    logger.info("")
    logger.info("=== Eval summary (models that have bases and a non-empty detector summary) ===")
    logger.info("Total with labels and summary: %d", n)
    if n:
        logger.info(
            "Top-1 accuracy (first suggested model in bases): %.2f%% (%d/%d)", 100 * correct_top1 / n, correct_top1, n
        )
        logger.info("Top-3 accuracy (any base in top 3): %.2f%% (%d/%d)", 100 * correct_top3 / n, correct_top3, n)
        logger.info("Top-5 accuracy (any base in top 5): %.2f%% (%d/%d)", 100 * correct_top5 / n, correct_top5, n)
    logger.info(
        "Total eval entries with bases: %d (skipped/errors: %d)",
        total_with_bases,
        total_with_bases - total_with_summary,
    )

    # Per-model table for quick inspection
    logger.info("")
    logger.info("=== Per-model predictions ===")
    rows = []
    for model_id in sorted(results_by_model):
        data = results_by_model[model_id]
        bases = data.get("bases", [])
        if not bases:
            continue
        summary = data.get("models_with_most_matched_classes", [])
        if summary:
            top_3 = []
            bases_set = set(bases)
            for s in summary[:3]:
                pred_model = s.get("model_id", "-")
                if pred_model in bases_set:
                    colored = f"{ANSI_GREEN}{pred_model}{ANSI_RESET}"
                else:
                    colored = pred_model
                top_3.append(colored)
            predicted = ", ".join(top_3) if top_3 else "-"
        elif data.get("error"):
            predicted = "<error>"
        else:
            predicted = "-"
        rows.append((model_id, ",".join(bases), predicted))

    if rows:
        model_width = max(len("model"), max(len(model) for model, _, _ in rows))
        bases_width = max(len("bases"), max(len(bases) for _, bases, _ in rows))

        # For pred_width, account for ANSI codes by stripping them first
        def strip_ansi(s: str) -> str:
            import re

            return re.sub(r"\033\[[0-9;]*m", "", s)

        pred_width = max(len("predicted"), max(len(strip_ansi(pred)) for _, _, pred in rows))

        header = f"{'model':<{model_width}}  {'bases':<{bases_width}}  {'predicted':<{pred_width}}"
        sep = f"{'-' * model_width}  {'-' * bases_width}  {'-' * pred_width}"
        logger.info(header)
        logger.info(sep)
        for model, bases, predicted in rows:
            # Don't pad predicted column since it has ANSI codes
            logger.info(f"{model:<{model_width}}  {bases:<{bases_width}}  {predicted}")


if __name__ == "__main__":
    main()
