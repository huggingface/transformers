"""
Build a dataset with columns:
    model_name, checkpoint, date_released, model_options,
    original_modeling_code, current_modeling_code, current_modular_code

Priority for original_modeling_code:
  1. Hub repo modeling_*.py (for custom_code models that have one)
  2. GitHub repo linked in model card — searches for model.py / modeling*.py
  3. First git commit of the transformers file
"""

import json
import os
import re
import subprocess
import urllib.request
from datetime import datetime
from pathlib import Path

from huggingface_hub import ModelCard, hf_hub_download, list_repo_files

REPO_ROOT = Path(__file__).parent
MODELS_DIR = REPO_ROOT / "src/transformers/models"
CHECKPOINTS_JSON = REPO_ROOT / "model_first_from_pretrained_checkpoints.json"
RELEASE_DATES_JSONL = REPO_ROOT / "modular-model-eval.full.jsonl"

# Repos that are infrastructure/tooling — not the original model implementation
NOISE_REPOS = {
    "huggingface/transformers",
    "huggingface/huggingface-llama-recipes",
    "vllm-project/vllm",
    "microsoft/Phi-3CookBook",
    "lm-sys/FastChat",
    "EleutherAI/lm-evaluation-harness",
}

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")


def _github_get(url: str) -> dict | list | None:
    headers = {"User-Agent": "Mozilla/5.0"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _github_raw(owner_repo: str, path: str, branch: str = "main") -> str | None:
    """Fetch raw file content from GitHub, trying main then master branch."""
    for ref in ([branch] if branch not in ("main", "master") else ["main", "master"]):
        url = f"https://raw.githubusercontent.com/{owner_repo}/{ref}/{path}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception:
            continue
    return None


def _score_model_file(path: str) -> int:
    """
    Score a file path for likelihood of being the primary model implementation.
    Higher = better candidate.
    """
    name = path.lower().split("/")[-1]
    score = 0
    # Strongly prefer files named model.py or modeling*.py
    if name == "model.py":
        score += 10
    elif name.startswith("modeling") and name.endswith(".py"):
        score += 9
    elif name == "models.py":
        score += 7
    # Penalise test/utility/config/tokenizer files
    for bad in ("test", "util", "config", "tokeniz", "convert", "process", "quantiz", "loader"):
        if bad in name:
            score -= 5
    # Prefer shallower paths (top-level or one directory deep)
    depth = path.count("/")
    score -= depth
    return score


def _parse_release_date(value: str | None) -> datetime | None:
    """Return a datetime parsed from YYYY-MM-DD strings, otherwise None."""
    try:
        return datetime.strptime(value or "", "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def _load_release_dates() -> dict[str, str]:
    """Load {model_name: date_released} from the modular-model-eval dataset."""
    release_dates: dict[str, str] = {}

    if RELEASE_DATES_JSONL.exists():
        with open(RELEASE_DATES_JSONL) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                model_name = row.get("model_name")
                date_released = row.get("date_released") or ""
                if not model_name or not date_released:
                    continue
                existing = release_dates.get(model_name)
                if existing is None:
                    release_dates[model_name] = date_released
                    continue
                existing_dt = _parse_release_date(existing)
                new_dt = _parse_release_date(date_released)
                if existing_dt is None or (new_dt is not None and new_dt < existing_dt):
                    release_dates[model_name] = date_released
        return release_dates

    try:
        from datasets import load_dataset
    except Exception:
        return release_dates

    try:
        dataset = load_dataset("itazap/modular-model-eval", split="train")
    except Exception:
        return release_dates

    for row in dataset:
        model_name = row.get("model_name")
        date_released = row.get("date_released") or ""
        if model_name and date_released:
            release_dates.setdefault(model_name, date_released)

    return release_dates


def _build_model_options(release_dates: dict[str, str]) -> dict[str, list[str]]:
    """Return {model_name: [earlier_model_names...]} ordered by release date."""
    by_date: dict[datetime, list[str]] = {}
    for model_name, date_released in release_dates.items():
        parsed = _parse_release_date(date_released)
        if parsed is None:
            continue
        by_date.setdefault(parsed, []).append(model_name)

    model_options: dict[str, list[str]] = {}
    released_so_far: list[str] = []
    for release_date in sorted(by_date):
        same_day_models = sorted(by_date[release_date])
        for model_name in same_day_models:
            model_options[model_name] = sorted(released_so_far)
        released_so_far.extend(same_day_models)

    return model_options


def get_github_modeling_code(checkpoint: str) -> tuple[str, str] | tuple[None, None]:
    """
    Try to find the original model implementation in a GitHub repo linked from
    the Hub model card.
    Returns (content, url) or (None, None).
    """
    try:
        card = ModelCard.load(checkpoint)
        card_text = card.content
    except Exception:
        return None, None

    gh_repos = re.findall(r"https?://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", card_text)
    seen = set()
    candidates = []
    for r in gh_repos:
        r = r.rstrip("/").split("/blob/")[0].split("/tree/")[0]
        if r not in seen and r not in NOISE_REPOS:
            seen.add(r)
            candidates.append(r)

    for owner_repo in candidates:
        data = _github_get(f"https://api.github.com/repos/{owner_repo}/git/trees/HEAD?recursive=1")
        if not data or "tree" not in data:
            continue
        # Resolve the actual default branch from the sha (HEAD ref)
        ref = data.get("sha", "main")
        py_files = [
            item["path"] for item in data["tree"]
            if item["path"].endswith(".py") and item.get("type") == "blob"
        ]
        if not py_files:
            continue
        scored = sorted(py_files, key=_score_model_file, reverse=True)
        best = scored[0]
        if _score_model_file(best) < 5:
            continue
        content = _github_raw(owner_repo, best)
        if content:
            url = f"https://github.com/{owner_repo}/blob/{ref}/{best}"
            return content, url

    return None, None


def get_modeling_file_path(model_name: str) -> Path | None:
    model_dir = MODELS_DIR / model_name
    candidates = list(model_dir.glob(f"modeling_{model_name}.py"))
    if candidates:
        return candidates[0]
    candidates = list(model_dir.glob("modeling_*.py"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def get_modular_bases(modular_code: str) -> list[str]:
    """
    Extract the model names inherited from in a modular file.
    Looks for imports of the form:
      from ..{model}.modeling_{model} import ...
      from ...models.{model}.modeling_{model} import ...
    Excludes modeling_auto.
    """
    pattern = re.compile(r"from \.\.+(?:[\w.]+\.)?(\w+)\.modeling_(?!\w*auto)(\w+) import", re.IGNORECASE)
    bases = set()
    for m in pattern.finditer(modular_code):
        # group(2) is the model name part after "modeling_"
        bases.add(m.group(2))
    return sorted(bases)


def get_modular_file_path(model_name: str) -> Path | None:
    model_dir = MODELS_DIR / model_name
    candidates = list(model_dir.glob(f"modular_{model_name}.py"))
    if candidates:
        return candidates[0]
    candidates = list(model_dir.glob("modular_*.py"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def get_first_git_commit_content(rel_path: str) -> str | None:
    """Return the file content at the first commit that added it."""
    result = subprocess.run(
        ["git", "log", "--oneline", "--diff-filter=A", "--", rel_path],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    lines = result.stdout.strip().splitlines()
    if not lines:
        return None
    first_commit = lines[-1].split()[0]
    result = subprocess.run(
        ["git", "show", f"{first_commit}:{rel_path}"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    return result.stdout if result.returncode == 0 else None


def get_hub_modeling_code(checkpoint: str) -> tuple[str, str] | tuple[None, None]:
    """Download the modeling_*.py from a Hub repo, if present. Returns (content, url)."""
    try:
        files = list(list_repo_files(checkpoint))
    except Exception:
        return None, None
    modeling_files = [f for f in files if f.startswith("modeling_") and f.endswith(".py")]
    if not modeling_files:
        return None, None
    filename = modeling_files[0]
    try:
        local_path = hf_hub_download(repo_id=checkpoint, filename=filename)
        content = Path(local_path).read_text()
        url = f"https://huggingface.co/{checkpoint}/blob/main/{filename}"
        return content, url
    except Exception:
        return None, None


def build_dataset(use_github: bool = True):
    with open(CHECKPOINTS_JSON) as f:
        checkpoints = json.load(f)

    release_dates = _load_release_dates()
    model_options_by_name = _build_model_options(release_dates)

    rows = []
    for model_name, info in checkpoints.items():
        checkpoint = info["checkpoint"]
        is_custom_code = info.get("custom_code", False)

        modeling_path = get_modeling_file_path(model_name)
        current_modeling_code = modeling_path.read_text() if modeling_path else None

        modular_path = get_modular_file_path(model_name)
        current_modular_code = modular_path.read_text() if modular_path else None

        # --- original modeling code, in priority order ---
        original_modeling_code = None
        original_source = None

        # 1. Hub modeling_*.py (custom_code models only)
        if is_custom_code:
            original_modeling_code, original_source = get_hub_modeling_code(checkpoint)

        # 2. GitHub repo from model card
        if original_modeling_code is None and use_github:
            original_modeling_code, original_source = get_github_modeling_code(checkpoint)

        # 3. First git commit in transformers (skip if auto-generated)
        if original_modeling_code is None and modeling_path is not None:
            rel = str(modeling_path.relative_to(REPO_ROOT))
            content = get_first_git_commit_content(rel)
            if content and "This file was automatically generated" not in content[:500]:
                result = subprocess.run(
                    ["git", "log", "--oneline", "--diff-filter=A", "--", rel],
                    capture_output=True, text=True, cwd=REPO_ROOT,
                )
                commit = result.stdout.strip().splitlines()[-1].split()[0]
                original_modeling_code = content
                original_source = f"https://github.com/huggingface/transformers/blob/{commit}/{rel}"

        bases = get_modular_bases(current_modular_code) if current_modular_code else []
        date_released = release_dates.get(model_name)

        rows.append({
            "model_name": model_name,
            "checkpoint": checkpoint,
            "date_released": date_released,
            "model_options": model_options_by_name.get(model_name, []),
            "original_modeling_code": original_modeling_code,
            "original_source": original_source,
            "current_modeling_code": current_modeling_code,
            "current_modular_code": current_modular_code,
            "bases": bases,
        })

        source_label = original_source.split("/")[2] if original_source else "✗"  # github.com / huggingface.co
        status = f"original={'✓ [' + source_label + ']' if original_modeling_code else '✗'}"
        status += f", modeling={'✓' if current_modeling_code else '✗'}"
        status += f", modular={'✓' if current_modular_code else '✗'}"
        print(f"  {model_name}: {status}")

    return rows


if __name__ == "__main__":
    import datasets

    print("Building dataset...")
    rows = build_dataset(use_github=True)

    ds = datasets.Dataset.from_list(rows)
    print(f"\nDataset: {len(ds)} rows, columns: {ds.column_names}")

    output_path = REPO_ROOT / "modeling_dataset"
    ds.save_to_disk(str(output_path))
    print(f"Saved to {output_path}")

    has_original = sum(1 for r in rows if r["original_modeling_code"])
    has_modular = sum(1 for r in rows if r["current_modular_code"])
    print(f"  Models with original code: {has_original}/{len(rows)}")
    print(f"  Models with modular code:  {has_modular}/{len(rows)}")

    hub_repo = "itazap/modeling-dataset"
    print(f"\nPushing to {hub_repo}...")
    ds.push_to_hub(hub_repo, private=False)
    print(f"Done: https://huggingface.co/datasets/{hub_repo}")
