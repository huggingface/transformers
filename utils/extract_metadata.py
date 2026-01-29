#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
Script to extract metadata from setup.py for CI testing.

Usage:
    python utils/extract_metadata.py extras          # Lists all extras, one per line
    python utils/extract_metadata.py python-versions # Outputs JSON array of supported Python versions
"""

import json
import sys
from pathlib import Path


def get_setup_module():
    """Import and return the setup module."""
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))
    import setup
    return setup


def extract_extras():
    """Extract all extras for the current Python version."""
    setup = get_setup_module()
    extras = sorted(setup.extras.keys())
    for extra in extras:
        print(extra)


def extract_python_versions():
    """Extract supported Python versions as JSON array."""
    setup = get_setup_module()
    min_ver, max_ver = setup.SUPPORTED_PYTHON_VERSIONS
    versions = [f"3.{v}" for v in range(min_ver, max_ver + 1)]
    print(json.dumps(versions))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python utils/extract_metadata.py {extras|python-versions}", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]

    if command == "extras":
        extract_extras()
    elif command == "python-versions":
        extract_python_versions()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Usage: python utils/extract_metadata.py {extras|python-versions}", file=sys.stderr)
        sys.exit(1)
