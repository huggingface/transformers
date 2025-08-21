"""Small utility to check for DCO (Signed-off-by) in commit messages.

This file is intentionally tiny and dependency-free so adding it won't affect
existing tests or the package's import surface.
"""
from __future__ import annotations


def has_dco_signoff(commit_message: str) -> bool:
    """Return True if ``commit_message`` contains a "Signed-off-by:" line.

    The check is case-insensitive and matches lines that start with
    "Signed-off-by:" ignoring leading whitespace.
    """
    if not commit_message:
        return False

    for line in commit_message.splitlines():
        if line.strip().lower().startswith("signed-off-by:"):
            return True
    return False
