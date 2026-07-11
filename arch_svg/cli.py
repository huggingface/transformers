"""`python -m arch_svg ...`

python -m arch_svg one --model llama --mode full --out out/
python -m arch_svg one --model gemma --mode diff --out out/
python -m arch_svg all --mode diff --out out/ --jobs 8
python -m arch_svg all --mode diff --out out/ --limit 20   # quick subset
"""

from __future__ import annotations

import argparse
import os
import sys


def _cmd_one(args) -> int:
    from .discover import discover_models, model_type_for
    from .gallery import render_one

    model_type = None
    for e in discover_models():
        if e.name == args.model:
            model_type = model_type_for(e)
            break
    svg, record = render_one(args.model, model_type or args.model, args.mode)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.model}.svg")
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"wrote {path}  (status={record['status']})")
    if record.get("build_error"):
        print(f"  note: {record['build_error']}")
    if args.mode == "diff" and record.get("is_modular") is not None:
        print(
            f"  modular={record.get('is_modular')} parent={record.get('parent_model')} totals={record.get('diff_totals')}"
        )
    return 0 if record["status"] != "failed" else 1


def _cmd_all(args) -> int:
    from .gallery import run_all

    res = run_all(out=args.out, mode=args.mode, jobs=args.jobs, limit=args.limit)
    s = res["summary"]
    print(f"\ndone → {os.path.join(args.out, 'index.html')}")
    print(f"  {s}")
    return 0


def _cmd_list(args) -> int:
    from .discover import discover_models

    for e in discover_models():
        tag = "modular" if e.has_modular else "standalone"
        print(f"{e.name:32s} {tag}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="arch_svg", description="transformers architecture SVG generator")
    sub = p.add_subparsers(dest="cmd", required=True)

    one = sub.add_parser("one", help="render a single model")
    one.add_argument("--model", required=True)
    one.add_argument("--mode", choices=["full", "diff"], default="full")
    one.add_argument("--out", default="out")
    one.set_defaults(func=_cmd_one)

    alle = sub.add_parser("all", help="render the whole model zoo + index.html")
    alle.add_argument("--mode", choices=["full", "diff", "both"], default="both")
    alle.add_argument("--out", default="out")
    alle.add_argument("--jobs", type=int, default=4)
    alle.add_argument("--limit", type=int, default=None)
    alle.set_defaults(func=_cmd_all)

    lst = sub.add_parser("list", help="list discovered models")
    lst.set_defaults(func=_cmd_list)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
