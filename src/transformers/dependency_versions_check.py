import sys

from .dependency_versions_table import deps
from .utils.versions import require_version_core


# define which module versions we always want to check at run time
# (usually the ones defined in `install_requires` in setup.py)
#
# order specific notes:
# - tqdm must be checked before tokenizers

pkgs_to_check_at_runtime = "python tqdm regex sacremoses requests packaging filelock numpy tokenizers".split()
if sys.version_info < (3, 7):
    pkgs_to_check_at_runtime.append("dataclasses")

for pkg in pkgs_to_check_at_runtime:
    if pkg in deps:
        if pkg == "tokenizers":
            # must be loaded here, or else tqdm check may fail
            from .file_utils import is_tokenizers_available

            if not is_tokenizers_available():
                continue  # not required, check version only if installed

        require_version_core(deps[pkg])
    else:
        raise ValueError(f"can't find {pkg} in {deps.keys()}, check dependency_versions_table.py")
