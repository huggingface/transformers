#!/usr/bin/env python3
"""
Run modified processor test files sequentially and aggregate the
'memory usage — top 30 tests by retained RSS' sections across all outputs.

Usage:
    python analyze_mem.py                      # runs DEFAULT_TEST_FILES below
    python analyze_mem.py tests/models/foo/... # explicit list of test files
"""

import re
import subprocess
import sys
from pathlib import Path

PYTHON = "/c/Users/yih-d/miniconda3/envs/py310/python.exe"
REPO_ROOT = Path(__file__).parent

# All processor test files modified on debug_mem vs main (git diff main...HEAD)
DEFAULT_TEST_FILES = [
    "tests/models/altclip/test_processing_altclip.py",
    "tests/models/aria/test_processing_aria.py",
    "tests/models/aya_vision/test_processing_aya_vision.py",
    "tests/models/clip/test_processing_clip.py",
    "tests/models/colqwen2/test_processing_colqwen2.py",
    "tests/models/csm/test_processing_csm.py",
    "tests/models/donut/test_processing_donut.py",
    "tests/models/ernie4_5_vl_moe/test_processing_ernie4_5_vl_moe.py",
    "tests/models/fuyu/test_processing_fuyu.py",
    "tests/models/gemma3n/test_processing_gemma3n.py",
    "tests/models/glm46v/test_processor_glm46v.py",
    "tests/models/glm4v/test_processor_glm4v.py",
    "tests/models/glm_image/test_processor_glm_image.py",
    "tests/models/grounding_dino/test_processing_grounding_dino.py",
    "tests/models/idefics2/test_processing_idefics2.py",
    "tests/models/idefics3/test_processing_idefics3.py",
    "tests/models/internvl/test_processing_internvl.py",
    "tests/models/janus/test_processing_janus.py",
    "tests/models/kosmos2_5/test_processor_kosmos2_5.py",
    "tests/models/lighton_ocr/test_processor_lighton_ocr.py",
    "tests/models/llava/test_processing_llava.py",
    "tests/models/llava_next_video/test_processing_llava_next_video.py",
    "tests/models/minicpmv4_6/test_processing_minicpmv4_6.py",
    "tests/models/mistral3/test_processing_mistral3.py",
    "tests/models/mllama/test_processing_mllama.py",
    "tests/models/owlv2/test_processing_owlv2.py",
    "tests/models/parakeet/test_processing_parakeet.py",
    "tests/models/pixtral/test_processing_pixtral.py",
    "tests/models/qianfan_ocr/test_processing_qianfan_ocr.py",
    "tests/models/qwen2_5_omni/test_processing_qwen2_5_omni.py",
    "tests/models/qwen2_5_vl/test_processing_qwen2_5_vl.py",
    "tests/models/qwen2_audio/test_processing_qwen2_audio.py",
    "tests/models/qwen2_vl/test_processing_qwen2_vl.py",
    "tests/models/qwen3_omni_moe/test_processing_qwen3_omni_moe.py",
    "tests/models/qwen3_vl/test_processing_qwen3_vl.py",
    "tests/models/smolvlm/test_processing_smolvlm.py",
    "tests/models/trocr/test_processing_trocr.py",
    "tests/models/video_llama_3/test_processing_video_llama_3.py",
    "tests/test_processing_common.py",
]


def run_pytest(test_file: str) -> str:
    """Run pytest -v --tb=no on one file, return combined stdout+stderr."""
    print(f"  running {test_file} ...", flush=True)
    result = subprocess.run(
        [PYTHON, "-m", "pytest", test_file, "-v", "--tb=no"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    return result.stdout + result.stderr


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

MEM_HEADER_RE = re.compile(r"memory usage.*top \d+", re.IGNORECASE)

# "  +   492.8     1608.7     gw1  tests/models/..."
MEM_ROW_RE = re.compile(
    r"^\s+([+-])\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+(tests/\S+)"
)

# verbose line: "PASSED tests/...::Cls::method (0.12s)"
# or with mem plugin: "PASSED tests/... [PASSED] 0.05s mem:+8MB"
DUR_RE = re.compile(
    r"(?:PASSED|FAILED|ERROR|SKIPPED)\s+(tests/\S+)\s+"
    r"(?:\[(?:PASSED|FAILED|ERROR|SKIPPED)\]\s+)?"
    r"([\d.]+)s"
)


def parse_memory_table(output: str) -> list[dict]:
    rows = []
    in_table = False
    for line in output.splitlines():
        if MEM_HEADER_RE.search(line):
            in_table = True
            continue
        if not in_table:
            continue
        m = MEM_ROW_RE.match(line)
        if m:
            sign, delta_str, end_str, worker, test = m.groups()
            rows.append({
                "delta": float(delta_str) * (1 if sign == "+" else -1),
                "end_mb": float(end_str),
                "worker": worker,
                "test": test.strip(),
            })
        elif re.search(r"per-worker RSS|={5}", line):
            in_table = False
    return rows


def parse_durations(output: str) -> dict[str, float]:
    durations: dict[str, float] = {}
    for line in output.splitlines():
        m = DUR_RE.search(line)
        if m:
            durations[m.group(1)] = float(m.group(2))
    return durations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    test_files = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TEST_FILES

    if not test_files:
        print("No modified test files found.")
        return

    print(f"Will run {len(test_files)} test file(s):\n")
    for f in test_files:
        print(f"  {f}")
    print()

    all_rows: list[dict] = []
    all_durations: dict[str, float] = {}
    # track total duration per test file (sum of all per-test durations in that file)
    file_durations: dict[str, float] = {}

    for test_file in test_files:
        output = run_pytest(test_file)
        rows = parse_memory_table(output)
        durations = parse_durations(output)

        for row in rows:
            row["source_file"] = test_file

        all_rows.extend(rows)
        all_durations.update(durations)
        file_durations[test_file] = sum(durations.values())

    if not all_rows:
        print("\nNo memory-usage table found in any output.")
        return

    # Sort by Delta MB descending
    all_rows.sort(key=lambda r: r["delta"], reverse=True)

    # Deduplicate: keep the highest-delta entry per (test, source_file)
    seen: set[tuple] = set()
    unique_rows: list[dict] = []
    for row in all_rows:
        key = (row["test"], row["source_file"])
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)

    # -----------------------------------------------------------------------
    # Table 1: per-test
    # -----------------------------------------------------------------------
    col_test = max(len(r["test"]) for r in unique_rows)
    col_test = max(col_test, 60)

    print(
        f"\n{'='*120}\n"
        f" PER-TEST MEMORY USAGE — {len(unique_rows)} entries across "
        f"{len(test_files)} test file(s) — sorted by Delta MB\n"
        f"{'='*120}"
    )
    print(f"{'Delta MB':>10}  {'End MB':>10}  {'Duration':>10}  {'Worker':>6}  Test")
    print(f"{'':->10}  {'':->10}  {'':->10}  {'':->6}  {'':-<{col_test}}")

    for row in unique_rows:
        test = row["test"]
        dur = all_durations.get(test)
        dur_str = f"{dur:.2f}s" if dur is not None else "n/a"
        sign = "+" if row["delta"] >= 0 else ""
        print(
            f"  {sign}{row['delta']:>8.1f}  {row['end_mb']:>10.1f}"
            f"  {dur_str:>10}  {row['worker']:>6}  {test}"
        )

    # -----------------------------------------------------------------------
    # Table 2: per-model (aggregated per source file)
    # -----------------------------------------------------------------------
    def model_name(path: str) -> str:
        """'tests/models/aria/test_processing_aria.py' -> 'aria'"""
        p = Path(path)
        if p.parent.name == "models":
            # tests/test_processing_common.py
            return p.stem
        return p.parent.name  # tests/models/<model>/...

    # Build one entry per source file: max delta, max end_mb, total duration
    model_stats: dict[str, dict] = {}
    for row in unique_rows:
        sf = row["source_file"]
        if sf not in model_stats:
            model_stats[sf] = {"delta": row["delta"], "end_mb": row["end_mb"]}
        else:
            model_stats[sf]["delta"] = max(model_stats[sf]["delta"], row["delta"])
            model_stats[sf]["end_mb"] = max(model_stats[sf]["end_mb"], row["end_mb"])

    model_rows = [
        {
            "model": model_name(sf),
            "delta": stats["delta"],
            "end_mb": stats["end_mb"],
            "total_dur": file_durations.get(sf, 0.0),
        }
        for sf, stats in model_stats.items()
    ]
    model_rows.sort(key=lambda r: r["delta"], reverse=True)

    col_model = max(len(r["model"]) for r in model_rows)
    col_model = max(col_model, 20)

    print(
        f"\n{'='*80}\n"
        f" PER-MODEL SUMMARY — max Delta MB & End MB per test file, "
        f"total duration — sorted by Delta MB\n"
        f"{'='*80}"
    )
    print(f"{'Delta MB':>10}  {'End MB':>10}  {'Total Dur':>10}  Model")
    print(f"{'':->10}  {'':->10}  {'':->10}  {'':-<{col_model}}")

    for r in model_rows:
        sign = "+" if r["delta"] >= 0 else ""
        dur_str = f"{r['total_dur']:.1f}s" if r["total_dur"] > 0 else "n/a"
        print(
            f"  {sign}{r['delta']:>8.1f}  {r['end_mb']:>10.1f}"
            f"  {dur_str:>10}  {r['model']}"
        )

    print(f"\nTotal unique entries: {len(unique_rows)}")
    print(f"Source files run:     {len(test_files)}")


if __name__ == "__main__":
    main()
