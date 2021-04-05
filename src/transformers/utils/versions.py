# Copyright 2020 The HuggingFace Team. All rights reserved.
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
Utilities for working with package versions
"""

import operator
import re
import sys
from typing import Optional

from packaging import version


# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


ops = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}


def require_version(requirement: str, hint: Optional[str] = None) -> None:
    """
    Perform a runtime check of the dependency versions, using the exact same syntax used by pip.

    The installed module version comes from the `site-packages` dir via `importlib_metadata`.

    Args:
        requirement (:obj:`str`): pip style definition, e.g.,  "tokenizers==0.9.4", "tqdm>=4.27", "numpy"
        hint (:obj:`str`, `optional`): what suggestion to print in case of requirements not being met
    """

    # note: while pkg_resources.require_version(requirement) is a much simpler way to do it, it
    # fails if some of the dependencies of the dependencies are not matching, which is not necessarily
    # bad, hence the more complicated check - which also should be faster, since it doesn't check
    # dependencies of dependencies.

    hint = f"\n{hint}" if hint is not None else ""

    # non-versioned check
    if re.match(r"^[\w_\-\d]+$", requirement):
        pkg, op, want_ver = requirement, None, None
    else:
        match = re.findall(r"^([^!=<>\s]+)([\s!=<>]{1,2})(.+)", requirement)
        if not match:
            raise ValueError(
                f"requirement needs to be in the pip package format, .e.g., package_a==1.23, or package_b>=1.23, but got {requirement}"
            )
        pkg, op, want_ver = match[0]
        if op not in ops:
            raise ValueError(f"need one of {list(ops.keys())}, but got {op}")

    # special case
    if pkg == "python":
        got_ver = ".".join([str(x) for x in sys.version_info[:3]])
        if not ops[op](version.parse(got_ver), version.parse(want_ver)):
            raise ImportError(
                f"{requirement} is required for a normal functioning of this module, but found {pkg}=={got_ver}."
            )
        return

    # check if any version is installed
    try:
        got_ver = importlib_metadata.version(pkg)
    except importlib_metadata.PackageNotFoundError:
        raise importlib_metadata.PackageNotFoundError(
            f"The '{requirement}' distribution was not found and is required by this application. {hint}"
        )

    # check that the right version is installed if version number was provided
    if want_ver is not None and not ops[op](version.parse(got_ver), version.parse(want_ver)):
        raise ImportError(
            f"{requirement} is required for a normal functioning of this module, but found {pkg}=={got_ver}.{hint}"
        )


def require_version_core(requirement):
    """ require_version wrapper which emits a core-specific hint on failure """
    hint = "Try: pip install transformers -U or pip install -e '.[dev]' if you're working with git master"
    return require_version(requirement, hint)


def require_version_examples(requirement):
    """ require_version wrapper which emits examples-specific hint on failure """
    hint = "Try: pip install -r examples/requirements.txt"
    return require_version(requirement, hint)
