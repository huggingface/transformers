.PHONY: modified_only_fixup extra_quality_checks quality style fixup fix-copies test test-examples docs


check_dirs := examples tests src utils

# get modified files since the branch was made
fork_point_sha := $(shell git merge-base --fork-point master)
joined_dirs := $(shell echo $(check_dirs) | tr " " "|")
modified_py_files := $(shell git diff --name-only $(fork_point_sha) | egrep '^($(joined_dirs))' | egrep '\.py$$')
#$(info modified files are: $(modified_py_files))

modified_only_fixup:
	@if [ -n "$(modified_py_files)" ]; then \
		echo "Checking/fixing $(modified_py_files)"; \
		black $(modified_py_files); \
		isort $(modified_py_files); \
		flake8 $(modified_py_files); \
	else \
		echo "No library .py files were modified"; \
	fi

# Check that source code meets quality standards

extra_quality_checks:
	python utils/check_copies.py
	python utils/check_dummies.py
	python utils/check_repo.py
	python utils/style_doc.py src/transformers docs/source --max_len 119

# this target runs checks on all files
quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)
	python utils/style_doc.py src/transformers docs/source --max_len 119 --check_only
	${MAKE} extra_quality_checks

# Format source code automatically and check is there are any problems left that need manual fixing

style:
	black $(check_dirs)
	isort $(check_dirs)
	python utils/style_doc.py src/transformers docs/source --max_len 119

# Super fast fix and check target that only works on relevant modified files since the branch was made

fixup: modified_only_fixup extra_quality_checks

# Make marked copies of snippets of codes conform to the original

fix-copies:
	python utils/check_copies.py --fix_and_overwrite
	python utils/check_dummies.py --fix_and_overwrite

# Run tests for the library

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

# Run tests for examples

test-examples:
	python -m pytest -n auto --dist=loadfile -s -v ./examples/

# Check that docs can build

docs:
	cd docs && make html SPHINXOPTS="-W"
