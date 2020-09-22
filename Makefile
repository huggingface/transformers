.PHONY: quality style test test-examples docs

# Check that source code meets quality standards

quality:
	black --check --line-length 119 --target-version py35 examples templates tests src utils
	isort --check-only examples templates tests src utils
	flake8 examples templates tests src utils
	python utils/check_copies.py
	python utils/check_repo.py

# Format source code automatically

style:
	black --line-length 119 --target-version py35 examples templates tests src utils
	isort examples templates tests src utils

# Make marked copies of snippets of codes conform to the original

fix-copies:
	python utils/check_copies.py --fix_and_overwrite

# Run tests for the library

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

# Run tests for examples

test-examples:
	python -m pytest -n auto --dist=loadfile -s -v ./examples/

# Check that docs can build

docs:
	cd docs && make html SPHINXOPTS="-W"
