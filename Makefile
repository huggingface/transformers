.PHONY: deps_table_update modified_only_fixup extra_quality_checks quality style fixup fix-copies test test-examples docs


check_dirs := examples tests src utils

modified_only_fixup:
	$(eval modified_py_files := $(shell python utils/get_modified_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo "Checking/fixing $(modified_py_files)"; \
		black $(modified_py_files); \
		isort $(modified_py_files); \
		flake8 $(modified_py_files); \
	else \
		echo "No library .py files were modified"; \
	fi

# Update src/transformers/dependency_versions_table.py

deps_table_update:
	@python setup.py deps_table_update

# Check that source code meets quality standards

extra_quality_checks: deps_table_update
	python utils/check_copies.py
	python utils/check_table.py
	python utils/check_dummies.py
	python utils/check_repo.py
	python utils/style_doc.py src/transformers docs/source --max_len 119
	python utils/class_mapping_update.py

# this target runs checks on all files
quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)
	python utils/style_doc.py src/transformers docs/source --max_len 119 --check_only
	${MAKE} extra_quality_checks

# Format source code automatically and check is there are any problems left that need manual fixing

style: deps_table_update
	black $(check_dirs)
	isort $(check_dirs)
	python utils/style_doc.py src/transformers docs/source --max_len 119

# Super fast fix and check target that only works on relevant modified files since the branch was made

fixup: modified_only_fixup extra_quality_checks

# Make marked copies of snippets of codes conform to the original

fix-copies:
	python utils/check_copies.py --fix_and_overwrite
	python utils/check_table.py --fix_and_overwrite
	python utils/check_dummies.py --fix_and_overwrite

# Run tests for the library

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

# Run tests for examples

test-examples:
	python -m pytest -n auto --dist=loadfile -s -v ./examples/

# Check that docs can build

docs:
	cd docs && make html SPHINXOPTS="-W -j 4"

# Release stuff

pre-release:
	python utils/release.py

pre-patch:
	python utils/release.py --patch

post-release:
	python utils/release.py --post_release

post-patch:
	python utils/release.py --post_release --patch

