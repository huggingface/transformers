# tests directory-specific settings - this file is run automatically
# by pytest before any tests are run

import sys
from os.path import abspath, dirname, join


# allow having multiple repository checkouts and not needing to remember to rerun
# 'pip install -e .[dev]' when switching between checkouts and running tests.
git_repo_path = abspath(join(dirname(dirname(__file__)), "src"))
sys.path.insert(1, git_repo_path)
