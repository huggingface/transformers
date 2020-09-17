# tests directory-specific settings - this file is run automatically
# by pytest before any tests are run

import sys
from os.path import abspath, dirname


# make it possible to run tests in this directory regardless whether they are
# invoked from this directory or from the parent directories
sys.path.insert(1, abspath(dirname(__file__)))
