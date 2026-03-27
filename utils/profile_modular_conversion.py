#!/usr/bin/env python

"""
Benchmark and profile modular conversion.

Examples:
    python utils/profile_modular_conversion.py \
        src/transformers/models/conditional_detr/modular_conditional_detr.py

    python utils/profile_modular_conversion.py \
        src/transformers/models/conditional_detr/modular_conditional_detr.py \
        src/transformers/models/deepseek_vl/modular_deepseek_vl.py \
        --repeat 3 \
        --compare-old-import-analysis \
        --compare-old-branch \
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


@contextmanager
def module_cache_context(enabled: bool):
    original = mmc.ENABLE_MODULE_SOURCE_CACHE
    mmc.ENABLE_MODULE_SOURCE_CACHE = enabled
    mmc.clear_module_source_cache()
    try:
        yield
    finally:
        mmc.clear_module_source_cache()
        mmc.ENABLE_MODULE_SOURCE_CACHE = original


@contextmanager
def fast_import_analysis_context(enabled: bool):
    original = mmc.ENABLE_FAST_IMPORT_ANALYSIS
    mmc.ENABLE_FAST_IMPORT_ANALYSIS = enabled
    try:
        yield
    finally:
        mmc.ENABLE_FAST_IMPORT_ANALYSIS = original


def run_conversion(
    modular_files: list[str], wrapper_cls, cache_enabled: bool, fast_import_analysis_enabled: bool
) -> dict[str, dict[str, str]]:
    with (
        metadata_wrapper_context(wrapper_cls),
        module_cache_context(cache_enabled),
        fast_import_analysis_context(fast_import_analysis_enabled),
    ):
        return {modular_file: mmc.convert_modular_file(modular_file) for modular_file in modular_files}


def benchmark_conversion(
    modular_files: list[str], wrapper_cls, cache_enabled: bool, fast_import_analysis_enabled: bool, repeat: int
) -> list[float]:
    durations = []
    for _ in range(repeat):
        start = time.perf_counter()
        run_conversion(modular_files, wrapper_cls, cache_enabled, fast_import_analysis_enabled)
        durations.append(time.perf_counter() - start)
    return durations


def profile_conversion(
    modular_files: list[str], wrapper_cls, cache_enabled: bool, fast_import_analysis_enabled: bool, output_path: Path
) -> None:
    profile = cProfile.Profile()
    profile.enable()
    run_conversion(modular_files, wrapper_cls, cache_enabled, fast_import_analysis_enabled)
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


def make_output_root(output_dir: str | None, modular_files: list[str]) -> Path:
    if output_dir is not None:
        return Path(output_dir)

    if len(modular_files) == 1:
        match = re.search(r"modular_(.*)(?=\.py$)", modular_files[0])
        name = match.group(1) if match is not None else Path(modular_files[0]).stem
    else:
        name = f"batch_{len(modular_files)}_modular_files"
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
    parser.add_argument("modular_files", nargs="+", help="One or more paths to modular_*.py files.")
    parser.add_argument("--repeat", type=int, default=3, help="Number of timing runs per mode.")
    parser.add_argument(
        "--compare-old-branch",
        action="store_true",
        help="Compare against simulated pre-change branch behavior (old wrapper + no module cache).",
    )
    parser.add_argument(
        "--compare-old-wrapper",
        action="store_true",
        help="Compare against only the pre-change MetadataWrapper behavior.",
    )
    parser.add_argument(
        "--compare-no-cache",
        action="store_true",
        help="Compare against current wrapper behavior with module source caching disabled.",
    )
    parser.add_argument(
        "--compare-old-import-analysis",
        action="store_true",
        help="Compare against the current branch with the previous get_needed_imports analysis.",
    )
    parser.add_argument(
        "--verify-equal",
        action="store_true",
        help="Verify that current output and all requested comparison modes are byte-identical.",
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

    modular_files = args.modular_files
    output_root = make_output_root(args.output_dir, modular_files)
    output_root.mkdir(parents=True, exist_ok=True)

    modes = [("current", LibCSTMetadataWrapper, True, True)]
    if args.compare_old_branch:
        modes.append(("old_branch", OldBehaviorMetadataWrapper, False, False))
    if args.compare_old_wrapper:
        modes.append(("old_wrapper", OldBehaviorMetadataWrapper, True, True))
    if args.compare_no_cache:
        modes.append(("no_cache", LibCSTMetadataWrapper, False, True))
    if args.compare_old_import_analysis:
        modes.append(("old_import_analysis", LibCSTMetadataWrapper, True, False))

    print(f"files={len(modular_files)}")
    for label, wrapper_cls, cache_enabled, fast_import_analysis_enabled in modes:
        durations = benchmark_conversion(
            modular_files, wrapper_cls, cache_enabled, fast_import_analysis_enabled, args.repeat
        )
        print(summarize(label, durations))

    if args.verify_equal and len(modes) > 1:
        current = run_conversion(modular_files, LibCSTMetadataWrapper, True, True)
        for label, wrapper_cls, cache_enabled, fast_import_analysis_enabled in modes[1:]:
            other = run_conversion(modular_files, wrapper_cls, cache_enabled, fast_import_analysis_enabled)
            same = current.keys() == other.keys() and all(current[key] == other[key] for key in current)
            print(f"{label}_byte_identical={same}")
            if not same:
                return 1

    profile_path = output_root / "current.prof"
    profile_conversion(modular_files, LibCSTMetadataWrapper, True, True, profile_path)
    print(f"profile={profile_path}")

    if args.snakeviz:
        profile_name = modular_files[0] if len(modular_files) == 1 else f"batch:{len(modular_files)} modular files"
        html_path = write_snakeviz_bundle(profile_path, output_root / "snakeviz", profile_name)
        print(f"snakeviz={html_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
