.PHONY: quality style test test-examples docs

# Check that source code meets quality standards

quality:
	black --check examples templates tests src utils
	isort --check-only examples templates tests src utils
	flake8 examples templates tests src utils
	python utils/check_repo.py

# Format source code automatically

style:
	black examples templates tests src utils
	isort examples templates tests src utils

# Run tests for the library

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

# Run tests for examples

test-examples:
	python -m pytest -n auto --dist=loadfile -s -v ./examples/

# Check that docs can build

docs:
	cd docs && make html SPHINXOPTS="-W"
