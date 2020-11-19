from .dependency_versions_table import dep_versions
from .utils.versions import require_version_core


# define which module versions we always want to check at run time (usually the ones defined in `install_requires` in setup.py)
pkgs_to_check_at_runtime = "python tokenizers tqdm regex".split()
for k in pkgs_to_check_at_runtime:
    if k in dep_versions:
        require_version_core(dep_versions[k])
    else:
        raise ValueError(f"can't find {k} in {dep_versions.keys()}, check dependency_versions_table.py")
