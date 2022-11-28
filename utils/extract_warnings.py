import argparse
import json
import os
import time
import zipfile

from get_ci_error_statistics import download_artifact, get_artifacts_links
from transformers import logging


logger = logging.get_logger(__name__)


def extract_warnings_from_single_artifact(artifact_zip_path, targets):
    """Extract warnings from a downloaded artifact (in .zip format)"""
    selected_warnings = set()
    buffer = []

    try:
        with zipfile.ZipFile(artifact_zip_path) as z:
            for filename in z.namelist():
                if not os.path.isdir(filename):
                    # read the file
                    if filename != "warnings.txt":
                        continue
                    with z.open(filename) as f:
                        for line in f:
                            line = line.decode("UTF-8")
                            if "warnings summary (final)" in line:
                                continue
                            # This means we are outside the body of a warning
                            elif not line.startswith(" "):
                                # process a single warning and move it to `selected_warnings`.
                                if len(buffer) > 0:
                                    warning = "\n".join(buffer)
                                    # Only keep the warnings specified in `targets`
                                    if any(f": {x}: " in warning for x in targets):
                                        selected_warnings.add(warning)
                                    buffer = []
                                continue
                            else:
                                line = line.strip()
                                buffer.append(line)
    except Exception:
        logger.warning(
            f"{artifact_zip_path} is either an invalid zip file or something else wrong. This file is skipped."
        )

    return selected_warnings


def extract_warnings(artifact_dir, targets):
    """Extract warnings from all artifact files"""

    selected_warnings = set()

    paths = [os.path.join(artifact_dir, p) for p in os.listdir(artifact_dir) if p.endswith(".zip")]
    for p in paths:
        selected_warnings.update(extract_warnings_from_single_artifact(p, targets))

    return selected_warnings


if __name__ == "__main__":

    def list_str(values):
        return values.split(",")

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--workflow_run_id", default=None, type=str, required=True, help="A GitHub Actions workflow run id."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="Where to store the downloaded artifacts and other result files.",
    )
    parser.add_argument(
        "--token", default=None, type=str, required=True, help="A token that has actions:read permission."
    )
    parser.add_argument(
        "--targets",
        default="DeprecationWarning,UserWarning,FutureWarning",
        type=list_str,
        help="Comma-separated list of target warning(s) which we want to extract.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # get download links
    artifacts = get_artifacts_links(args.workflow_run_id)
    with open(os.path.join(args.output_dir, "artifacts.json"), "w", encoding="UTF-8") as fp:
        json.dump(artifacts, fp, ensure_ascii=False, indent=4)

    # download artifacts
    for idx, (name, url) in enumerate(artifacts.items()):
        print(name)
        print(url)
        print("=" * 80)
        download_artifact(name, url, args.output_dir, args.token)
        # Be gentle to GitHub
        time.sleep(1)

    # extract warnings from artifacts
    selected_warnings = extract_warnings(args.output_dir, args.targets)
    selected_warnings = sorted(list(selected_warnings))
    with open(os.path.join(args.output_dir, "selected_warnings.json"), "w", encoding="UTF-8") as fp:
        json.dump(selected_warnings, fp, ensure_ascii=False, indent=4)
