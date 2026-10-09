"""Markdown summary of a contract run, for a CI job summary.

    python example/run.py ... --output results.jsonl 2> run.log
    python example/report.py results.jsonl --log run.log --runner "python utils/contracts/run.py" \
        --lock tests/contracts/fleet-lock.json >> "$GITHUB_STEP_SUMMARY"

One row per target (status, first failure line, baseline), the contracts the
changed files selected and why (read from the runner's stderr), and for each
failure the command that reproduces it and what to do next: a regression is
fixed in the change; an intended change re-records the baseline for review.
"""

import argparse
import json
from pathlib import Path

ICON = {"pass": "✅", "fail": "❌", "error": "💥", "xfail": "⚠️", "unsupported": "⏭️"}
UPDATE = "https://github.com/huggingface/model-integration-contracts/blob/main/docs/improving-contracts.md"


def first_line(failures):
    """The most telling line of the first failure: a traceback's last line, else the failure itself."""
    if not failures:
        return ""
    lines = [line.strip() for line in failures[0].strip().splitlines() if line.strip()]
    return (lines[-1] if len(lines) > 1 else lines[0])[:200].replace("|", "\\|")


def short(contract):
    return contract.split("/")[-1].removesuffix("-integration")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--log", type=Path, help="the runner's stderr, for the selection")
    parser.add_argument("--runner", default="python example/run.py", help="how to invoke the runner in repro commands")
    parser.add_argument("--lock", help="the --lock passed to the runner, repeated in repro commands")
    args = parser.parse_args()

    records = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()] if args.results.exists() else []
    selection = None
    if args.log and args.log.exists():
        for line in args.log.read_text().splitlines():
            if line.startswith('{"selection"'):
                selection = json.loads(line)
    lock = f" --lock {args.lock}" if args.lock else ""
    counts = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    out = ["### Model integration contracts", ""]
    if selection is not None:
        chosen = selection["selection"]
        everything = sorted(selection["code_dirs"])
        if not chosen:
            out.append(f"No contract selected: the changed files touch none of the code the {len(everything)} contracts use.")
        else:
            reasons = sorted({reason for reason in chosen.values()})
            why = reasons[0] if len(reasons) == 1 else "; ".join(f"{name}: {reason}" for name, reason in sorted(chosen.items()))
            skipped = [name for name in everything if name not in chosen]
            out.append(f"Selected {len(chosen)} of {len(everything)} contracts ({why})." + (f" Not selected: {', '.join(skipped)}." if skipped else ""))
        out.append("")
    if records:
        out.append(" · ".join(f"{ICON.get(s, '')} {n} {s}" for s, n in sorted(counts.items())))
        out += ["", "| | Contract | Target | Platform | Detail |", "| --- | --- | --- | --- | --- |"]
        for r in records:
            if r["status"] == "xfail":
                detail = f"known failure: {r.get('known_failure', '')[:160]}"
                if any("Traceback" in f for f in r.get("failures", [])):
                    detail += " (the contract stopped at this error: its later checks did not run)"
            elif r["status"] == "pass" and r.get("skipped_outputs"):
                detail = f"not compared (known failure): {', '.join(r['skipped_outputs'])}"
            else:
                detail = first_line(r.get("failures"))
            out.append(f"| {ICON.get(r['status'], r['status'])} | {short(r['contract'])} | `{r['target']}` | {r.get('platform', '')} | {detail} |")
    bad = [r for r in records if r["status"] in ("fail", "error")]
    if bad:
        out += ["", "#### Failures", ""]
        for r in bad:
            command = f"{args.runner} --framework transformers --fixtures-only{lock} --contract {short(r['contract'])} --target {r['target']}"
            if r.get("variant", "default") != "default":
                command += f" --variant {r['variant']}"
            out += [f"**{short(r['contract'])} / `{r['target']}`** ({r['status']}; baseline `{r.get('baseline', 'none')}`)", "",
                    "```", *(f.rstrip() for f in r.get("failures", [])[:5]), "```", "",
                    f"Reproduce: `{command}`", ""]
        out += ["Is it a regression or an intended change? A regression: fix it in this PR (the fixture runs the contract's real usage code). "
                f"An intended change: re-record the baseline with `--record` and have it reviewed ([how]({UPDATE})).", ""]
    if not records and selection is None:
        out.append("No result: the run produced no records (see the job log).")
    print("\n".join(out))


if __name__ == "__main__":
    main()
