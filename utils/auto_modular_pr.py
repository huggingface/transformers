#!/usr/bin/env python3
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
End-to-end pipeline to open a modular model integration PR in transformers.

Usage:
    python utils/auto_modular_pr.py \\
        --hub-repo sarvamai/sarvam-105b \\
        --modeling-file modeling_sarvam_moe.py \\
        --model-name sarvam \\
        --dry-run

Steps:
    1. Download modeling file from HF Hub.
    2. Run modular_model_detector to find the best base model and generate an LLM prompt.
    3. Call the HF Inference API to write the modular file.
    4. Run modular_model_converter to regenerate the modeling file from modular.
    5. Fork huggingface/transformers, push a branch, open a PR.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from huggingface_hub import InferenceClient, hf_hub_download


# ── Helpers ───────────────────────────────────────────────────────────────────

TRANSFORMERS_ROOT = Path(__file__).parent.parent
MODELS_ROOT = TRANSFORMERS_ROOT / "src" / "transformers" / "models"
UTILS_DIR = Path(__file__).parent
GEMMA_MODULAR_REF = TRANSFORMERS_ROOT / "src" / "transformers" / "models" / "gemma" / "modular_gemma.py"
LLM_PROMPT_TEMPLATE = UTILS_DIR / "auto_modular_prompt.md"
PR_BODY_TEMPLATE = UTILS_DIR / "auto_modular_pr_body.md"


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Print and run a shell command."""
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=True)


def _strip_code_fence(text: str) -> str:
    """Remove markdown ```python ... ``` fences from LLM output."""
    text = text.strip()
    text = re.sub(r"^```(?:python)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _render_template(path: Path, **kwargs) -> str:
    return path.read_text(encoding="utf-8").format(**kwargs)


# ── Step 1: Fetch modeling file from HF Hub ───────────────────────────────────


def fetch_modeling_file(hub_repo: str, hub_filename: str, model_name: str) -> Path:
    """
    Download *hub_filename* from *hub_repo* and save it as
    ``src/transformers/models/<model_name>/modeling_<model_name>.py``.
    Returns the local Path.
    """
    model_dir = MODELS_ROOT / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    target = model_dir / f"modeling_{model_name}.py"

    print(f"    Downloading {hub_filename} from {hub_repo}...")
    tmp = hf_hub_download(repo_id=hub_repo, filename=hub_filename)
    shutil.copy2(tmp, target)

    print(f"    Saved, {target.relative_to(TRANSFORMERS_ROOT)}")
    return target


# ── Step 2: Run modular model detector ────────────────────────────────────────


def run_detector(modeling_file: Path, model_name: str) -> tuple[str, Path]:
    """
    Import and call the detector to produce a guidance prompt.
    Returns (prompt_text, prompt_file_path).
    """
    # Ensure the utils dir is on sys.path so the detector can import its siblings.
    if str(UTILS_DIR) not in sys.path:
        sys.path.insert(0, str(UTILS_DIR))

    # Change cwd so relative paths inside the detector (MODELS_ROOT etc.) resolve correctly.
    original_cwd = Path.cwd()
    os.chdir(TRANSFORMERS_ROOT)
    try:
        from modular_model_detector import (
            HUB_DATASET_DEFAULT,
            CodeSimilarityAnalyzer,
            compute_model_class_match_summary,
            generate_modular_prompt,
        )

        analyzer = CodeSimilarityAnalyzer(hub_dataset=HUB_DATASET_DEFAULT)
        results = analyzer.analyze_file(
            modeling_file,
            top_k_per_item=12,
            allow_hub_fallback=True,
            use_jaccard=True,
            ignore_models=set(),
        )
        _, ordered_summary = compute_model_class_match_summary(results)

        if not ordered_summary:
            raise RuntimeError("Detector found no matching base model. Check the modeling file.")

        print(
            f"    Top matched base model: {ordered_summary[0]['model_id']} "
            f"({ordered_summary[0]['pct']:.1f}% class match)"
        )

        prompt = generate_modular_prompt(
            modeling_file=modeling_file,
            ordered_summary=ordered_summary,
            results=results,
            models_root=analyzer.models_root,
        )
    finally:
        os.chdir(original_cwd)

    prompt_path = modeling_file.with_name(f"{model_name}_MODULAR_PROMPT")
    prompt_path.write_text(prompt, encoding="utf-8")
    print(f"    Prompt saved, {prompt_path.relative_to(TRANSFORMERS_ROOT)}")
    return prompt, prompt_path


# ── Step 3: Generate modular file with HF Inference API ───────────────────────


def _build_llm_prompt(prompt: str, modeling_file: Path, model_name: str) -> str:
    """Build the full prompt to send to the LLM."""
    modeling_code = modeling_file.read_text(encoding="utf-8")
    ref_code = GEMMA_MODULAR_REF.read_text(encoding="utf-8") if GEMMA_MODULAR_REF.exists() else ""
    return _render_template(
        LLM_PROMPT_TEMPLATE,
        model_name=model_name,
        prompt=prompt,
        modeling_file_name=modeling_file.name,
        modeling_code=modeling_code,
        ref_code=ref_code,
    )


def generate_modular_with_hf(
    prompt: str,
    modeling_file: Path,
    model_name: str,
    hf_model: str,
    hf_token: str | None,
    max_retries: int = 5,
    base_delay: float = 10.0,
) -> str:
    """Call a model via the HuggingFace Inference API (free tier, no local VRAM needed)."""
    full_prompt = _build_llm_prompt(prompt, modeling_file, model_name)
    client = InferenceClient(model=hf_model, token=hf_token)
    print(f"    Using HF Inference API: {hf_model}")

    for attempt in range(max_retries):
        try:
            response = client.chat_completion(
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=16000,
            )
            return _strip_code_fence(response.choices[0].message.content)
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            retryable = status in (429, 500, 502, 503, 504) or status is None
            if not retryable or attempt == max_retries - 1:
                raise
            delay = base_delay * (2**attempt)
            print(
                f"    HF API error ({e.__class__.__name__}: {e}). Retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(delay)


# ── Step 4: Run modular converter ─────────────────────────────────────────────


def run_modular_converter(modular_file: Path) -> None:
    """
    Run modular_model_converter.py to regenerate modeling_<model>.py from the modular file.
    Must be executed from the utils/ directory due to its local imports.
    """
    _run(
        [sys.executable, "modular_model_converter.py", "--files", str(modular_file.resolve())],
        cwd=UTILS_DIR,
    )


# ── Step 5: Fork, branch, commit, push, PR ────────────────────────────────────


def _gh_auth_username() -> str:
    """Return the currently logged-in GitHub username via `gh auth status`."""
    result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
    output = result.stdout + result.stderr
    match = re.search(r"Logged in to \S+ account (\S+)", output)
    if not match:
        match = re.search(r"as (\S+)", output)
    if not match:
        raise RuntimeError("Could not determine GitHub username from `gh auth status`. Pass --fork-owner explicitly.")
    return match.group(1)


def create_pr(
    model_name: str,
    model_dir: Path,
    fork_owner: str,
) -> None:
    """
    Fork huggingface/transformers (idempotent), then create a branch
    from upstream main and push.
    """
    branch = f"add-{model_name}-model"
    fork_url = f"git@github.com:{fork_owner}/transformers.git"
    upstream_url = "https://github.com/huggingface/transformers.git"

    # Ensure the fork exists
    _run(["gh", "repo", "fork", "huggingface/transformers", "--clone=false"])

    with tempfile.TemporaryDirectory(prefix=f"hf-pr-{model_name}-") as tmp:
        clone_dir = Path(tmp) / "transformers"

        # clone of upstream main
        print(f"    Shallow-cloning upstream main into {clone_dir}...")
        _run(
            [
                "git",
                "clone",
                "--depth=1",
                "--branch=main",
                upstream_url,
                str(clone_dir),
            ]
        )

        # Point origin at the fork so we push there
        _run(["git", "remote", "set-url", "origin", fork_url], cwd=clone_dir)

        # Create the PR branch
        _run(["git", "checkout", "-b", branch], cwd=clone_dir)

        # Copy model directory into the clone
        dest = clone_dir / "src" / "transformers" / "models" / model_name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(model_dir, dest)

        # commit
        _run(["git", "add", str(dest)], cwd=clone_dir)
        _run(
            [
                "git",
                "commit",
                "-m",
                f"Add {model_name} model (auto-generated modular integration)",
            ],
            cwd=clone_dir,
        )

        # Push to fork
        _run(["git", "push", "--force", "origin", branch], cwd=clone_dir)

    # PR body
    pr_body = _render_template(PR_BODY_TEMPLATE, model_name=model_name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(pr_body)
        body_file = f.name

    try:
        _run(
            [
                "gh",
                "pr",
                "create",
                "--repo",
                "huggingface/transformers",
                "--head",
                f"{fork_owner}:{branch}",
                "--base",
                "main",
                "--title",
                f"Add {model_name} model",
                "--body-file",
                body_file,
                "--draft",
            ],
            cwd=TRANSFORMERS_ROOT,
        )
    finally:
        os.unlink(body_file)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        prog="auto-modular-pr",
        description="End-to-end pipeline: HF Hub repo, modular PR in huggingface/transformers",
    )
    parser.add_argument(
        "--hub-repo",
        help="HF Hub repo ID containing the modeling file (e.g. sarvamai/sarvam-105b). "
        "Not required when using --from-dir.",
    )
    parser.add_argument(
        "--modeling-file",
        help="Filename of the modeling file in the hub repo (e.g. modeling_sarvam_moe.py). "
        "Not required when using --from-dir.",
    )
    parser.add_argument(
        "--from-dir",
        metavar="PATH",
        help="Skip steps 1-4 and go straight to the PR step using files already in PATH "
        "(e.g. src/transformers/models/sarvam_dry). "
        "Requires --model-name.",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name to use in transformers (e.g. sarvam). Determines the directory and file names.",
    )
    parser.add_argument(
        "--fork-owner",
        help="GitHub username that owns the transformers fork. Defaults to the account returned by `gh auth status`.",
    )
    parser.add_argument(
        "--hf-model",
        metavar="MODEL_ID",
        default="utils/auto_modular_pr.py",
        help="HuggingFace Inference API model id to use for modular code generation. "
        "E.g. 'Qwen/Qwen2.5-Coder-32B-Instruct'"
        "Uses your HF_TOKEN env var or huggingface-cli login credentials.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run steps 1-4 (generate files) but skip all git/PR actions.",
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    fork_owner = args.fork_owner
    if not fork_owner and not args.dry_run:
        fork_owner = _gh_auth_username()
        print(f"  Detected GitHub user: {fork_owner}")

    # ------------------------------------------------------------------
    if args.from_dir:
        # Skip steps 1-4, use pre-generated files directly.
        model_dir = Path(args.from_dir)
        if not model_dir.exists():
            raise SystemExit(f"--from-dir path does not exist: {model_dir}")
        print(f"\n[1-4/5] Skipping generation — using files from {model_dir}")
        for f in sorted(model_dir.iterdir()):
            print(f"    {f.name}")
    else:
        if not args.hub_repo or not args.modeling_file:
            raise SystemExit("Provide --hub-repo and --modeling-file, or use --from-dir.")
        if not args.hf_model:
            raise SystemExit("Provide --hf-model <model_id> for modular generation via the HuggingFace Inference API.")

        print("\n[1/5] Fetching modeling file from HF Hub...")
        modeling_file = fetch_modeling_file(args.hub_repo, args.modeling_file, args.model_name)

        print("\n[2/5] Running modular model detector...")
        prompt, _ = run_detector(modeling_file, args.model_name)

        print(f"\n[3/5] Generating modular file with HF Inference API ({args.hf_model})...")
        modular_code = generate_modular_with_hf(prompt, modeling_file, args.model_name, args.hf_model, hf_token)
        modular_file = modeling_file.with_name(f"modular_{args.model_name}.py")
        modular_file.write_text(modular_code, encoding="utf-8")
        print(f"    Written, {modular_file.relative_to(TRANSFORMERS_ROOT)}")

        print("\n[4/5] Running modular converter...")
        run_modular_converter(modular_file)
        print("    Done.")

        model_dir = modeling_file.parent

    # ------------------------------------------------------------------
    if args.dry_run:
        print("\n[5/5] Dry run — skipping git/PR steps.")
        return

    print(f"\n[5/5] Creating fork, branch, commit, and PR from {model_dir}...")
    create_pr(args.model_name, model_dir, fork_owner)
    print("\nDone!")


if __name__ == "__main__":
    main()
