"""
Functionality for storing and setting the version info for SparseML
"""

from datetime import date


nm_version_base = "1.5.0"
transformers_version_base = "42301"  # 4.23.1
is_release = False  # change to True to set the generated version as a release version


def _generate_version():
    return (
        f"{nm_version_base}.{transformers_version_base}"
        if is_release
        else f"{nm_version_base}.{date.today().strftime('%Y%m%d')}"
    )


__all__ = [
    "__version__",
    "nm_version_base",
    "transformers_version_base",
    "is_release",
    "version",
    "version_major",
    "version_minor",
    "version_bug",
    "version_build",
    "version_major_minor",
]
__version__ = _generate_version()

version = __version__
version_major, version_minor, version_bug, version_build = version.split(".") + (
    [None] if len(version.split(".")) < 4 else []
)  # handle conditional for version being 3 parts or 4 (4 containing build date)
version_major_minor = f"{version_major}.{version_minor}"