# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Default list of files to check (can be overridden via --files)
DEFAULT_FILES_TO_FIND = [
    "kernels/rwkv/wkv_cuda.cu",
    "kernels/rwkv/wkv_op.cpp",
    "kernels/falcon_mamba/selective_scan_with_ln_interface.py",
    "kernels/falcon_mamba/__init__.py",
    "kernels/__init__.py",
    "models/graphormer/algos_graphormer.pyx",
]


def test_custom_files_are_present(transformers_path: Path, files_to_find: List[str], verbose: bool = False) -> bool:
    """
    Check if all specified custom files are present in the given transformers path.

    Args:
        transformers_path (Path): The root path to check for files.
        files_to_find (List[str]): List of relative file paths to verify.
        verbose (bool): If True, print details about missing files.

    Returns:
        bool: True if all files are present, False otherwise.
    """
    if not transformers_path.exists() or not transformers_path.is_dir():
        raise ValueError(f"Transformers path does not exist or is not a directory: {transformers_path}")

    missing_files = []
    for file in files_to_find:
        file_path = transformers_path / file
        if not file_path.exists():
            missing_files.append(file)
            if verbose:
                print(f"Missing file: {file_path}")
        elif verbose:
            print(f"Found file: {file_path}")

    if missing_files:
        print(f"Missing files: {', '.join(missing_files)}")
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if custom files are present in the Transformers package.")
    parser.add_argument(
        "--check_lib",
        action="store_true",
        help="Check the actual installed package instead of the build directory."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES_TO_FIND,
        help="List of files to check (default: predefined list)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging."
    )
    args = parser.parse_args()

    if args.check_lib:
        try:
            import transformers
            transformers_path = Path(transformers.__file__).parent
        except ImportError:
            print("Error: 'transformers' module not found. Install it or use --check_lib=False.")
            sys.exit(1)
    else:
        transformers_path = Path.cwd() / "build/lib/transformers"

    if not test_custom_files_are_present(transformers_path, args.files, args.verbose):
        print("The built release does not contain all required custom files. Fix this before proceeding!")
        sys.exit(1)
    else:
        print("All custom files are present.")