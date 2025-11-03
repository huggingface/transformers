# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Import utilities: Utilities related to imports and our lazy inits.
"""

import importlib.machinery
import importlib.metadata
import importlib.util
import json
import operator
import os
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from collections.abc import Callable
from enum import Enum
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any

from packaging import version

from . import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


PACKAGE_DISTRIBUTION_MAPPING = importlib.metadata.packages_distributions()


def _is_package_available(pkg_name: str, return_version: bool = False) -> tuple[bool, str] | bool:
    """Check if `pkg_name` exist, and optionally try to get its version"""
    spec = importlib.util.find_spec(pkg_name)
    package_exists = spec is not None
    package_version = "N/A"
    if package_exists and return_version:
        try:
            # importlib.metadata works with the distribution package, which may be different from the import
            # name (e.g. `PIL` is the import name, but `pillow` is the distribution name)
            distributions = PACKAGE_DISTRIBUTION_MAPPING[pkg_name]
            # In most cases, the packages are well-behaved and both have the same name. If it's not the case, we
            # pick the first item of the list as best guess (it's almost always a list of length 1 anyway)
            distribution_name = pkg_name if pkg_name in distributions else distributions[0]
            package_version = importlib.metadata.version(distribution_name)
        except (importlib.metadata.PackageNotFoundError, KeyError):
            # If we cannot find the metadata (because of editable install for example), try to import directly.
            # Note that this branch will almost never be run, so we do not import packages for nothing here
            package = importlib.import_module(pkg_name)
            package_version = getattr(package, "__version__", "N/A")
        logger.debug(f"Detected {pkg_name} version: {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists