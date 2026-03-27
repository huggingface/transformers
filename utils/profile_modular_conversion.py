#!/usr/bin/env python

"""
Benchmark and profile modular conversion.

Examples:
    python utils/profile_modular_conversion.py \
        src/transformers/models/conditional_detr/modular_conditional_detr.py

    python utils/profile_modular_conversion.py \
        src/transformers/models/conditional_detr/modular_conditional_detr.py \
        --repeat 3 \
        --compare-old-wrapper \
        --verify-equal \
        --snakeviz
"""

import argparse
import cProfile
import json
import re
import shutil
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from pstats import Stats

import modular_model_converter as mmc
from libcst.metadata import MetadataWrapper as LibCSTMetadataWrapper


class OldBehaviorMetadataWrapper(LibCSTMetadataWrapper):
    def __init__(self, module, unsafe_skip_copy=False, cache=None):
        if cache is None:
            cache = {}
        super().__init__(module, unsafe_skip_copy=False, cache=cache)


@contextmanager
def metadata_wrapper_context(wrapper_cls):
    original = mmc.MetadataWrapper
    mmc.MetadataWrapper = wrapper_cls
    try:
        yield
    finally:
        mmc.MetadataWrapper = original


def run_conversion(modular_file: str, wrapper_cls) -> dict[str, str]:
    with metadata_wrapper_context(wrapper_cls):
        return mmc.convert_modular_file(modular_file)


def benchmark_conversion(modular_file: str, wrapper_cls, repeat: int) -> list[float]:
    durations = []
    for _ in range(repeat):
        start = time.perf_counter()
        run_conversion(modular_file, wrapper_cls)
        durations.append(time.perf_counter() - start)
    return durations


def profile_conversion(modular_file: str, wrapper_cls, output_path: Path) -> None:
    profile = cProfile.Profile()
    profile.enable()
    run_conversion(modular_file, wrapper_cls)
    profile.disable()
    profile.dump_stats(str(output_path))


def write_snakeviz_bundle(profile_path: Path, bundle_dir: Path, profile_name: str) -> Path:
    try:
        import snakeviz
        from snakeviz.stats import json_stats, table_rows
        from tornado.template import Template
    except ImportError as error:
        raise RuntimeError("SnakeViz bundle generation requires snakeviz and tornado to be installed.") from error

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    static_dir = bundle_dir / "static"
    snakeviz_root = Path(snakeviz.__file__).resolve().parent
    shutil.copytree(snakeviz_root / "static", static_dir)

    stats = Stats(str(profile_path))
    template_text = (snakeviz_root / "templates" / "viz.html").read_text(encoding="utf-8")
    html = Template(template_text).generate(
        profile_name=profile_name,
        table_rows=json.dumps(table_rows(stats)),
        callees=json.dumps(json_stats(stats)),
    ).decode("utf-8")

    static_uri = static_dir.resolve().as_uri()
    html = html.replace('href="/static/', f'href="{static_uri}/')
    html = html.replace("href='/static/", f"href='{static_uri}/")
    html = html.replace('src="/static/', f'src="{static_uri}/')
    html = html.replace("src='/static/", f"src='{static_uri}/")
    html = html.replace(
        'event.data[\'url\'] + "/static/vendor/immutable.min.js",',
        f'"{(static_dir / "vendor" / "immutable.min.js").resolve().as_uri()}",',
    )
    html = html.replace(
        'event.data[\'url\'] + "/static/vendor/lodash.min.js");',
        f'"{(static_dir / "vendor" / "lodash.min.js").resolve().as_uri()}");',
    )
    html = html.replace("'url': window.location.origin", "'url': ''")

    html_path = bundle_dir / "index.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


def make_output_root(output_dir: str | None, modular_file: str) -> Path:
    if output_dir is not None:
        return Path(output_dir)

    match = re.search(r"modular_(.*)(?=\.py$)", modular_file)
    name = match.group(1) if match is not None else Path(modular_file).stem
    return Path("/tmp/codex-profiles") / f"{name}_modular_profile"


def summarize(label: str, durations: list[float]) -> str:
    mean = statistics.mean(durations)
    median = statistics.median(durations)
    minimum = min(durations)
    maximum = max(durations)
    return (
        f"{label}: mean={mean:.3f}s median={median:.3f}s min={minimum:.3f}s "
        f"max={maximum:.3f}s runs={len(durations)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark and profile modular conversion.")
    parser.add_argument("modular_file", help="Path to a modular_*.py file.")
    parser.add_argument("--repeat", type=int, default=3, help="Number of timing runs per mode.")
    parser.add_argument(
        "--compare-old-wrapper",
        action="store_true",
        help="Compare against simulated pre-change MetadataWrapper behavior.",
    )
    parser.add_argument(
        "--verify-equal",
        action="store_true",
        help="Verify that current and old-wrapper outputs are byte-identical.",
    )
    parser.add_argument(
        "--snakeviz",
        action="store_true",
        help="Generate a standalone SnakeViz HTML bundle for the current mode.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for profile outputs. Defaults to /tmp/codex-profiles/<model>_modular_profile.",
    )
    args = parser.parse_args()

    modular_file = args.modular_file
    output_root = make_output_root(args.output_dir, modular_file)
    output_root.mkdir(parents=True, exist_ok=True)

    modes = [("current", LibCSTMetadataWrapper)]
    if args.compare_old_wrapper:
        modes.append(("old_wrapper", OldBehaviorMetadataWrapper))

    for label, wrapper_cls in modes:
        durations = benchmark_conversion(modular_file, wrapper_cls, args.repeat)
        print(summarize(label, durations))

    if args.compare_old_wrapper and args.verify_equal:
        current = run_conversion(modular_file, LibCSTMetadataWrapper)
        old = run_conversion(modular_file, OldBehaviorMetadataWrapper)
        same = current.keys() == old.keys() and all(current[key] == old[key] for key in current)
        print(f"byte_identical={same}")
        if not same:
            return 1

    profile_path = output_root / "current.prof"
    profile_conversion(modular_file, LibCSTMetadataWrapper, profile_path)
    print(f"profile={profile_path}")

    if args.snakeviz:
        html_path = write_snakeviz_bundle(profile_path, output_root / "snakeviz", modular_file)
        print(f"snakeviz={html_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
