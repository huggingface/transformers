# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

.PHONY: style check-repo fix-repo test test-examples benchmark

check_dirs := examples tests src utils scripts benchmark benchmark_v2
exclude_folders :=  ""

# Helper to find all Python files in directories (ty doesn't recursively scan directories)
define get_py_files
$(shell find $(1) -name "*.py" -type f 2>/dev/null)
endef


# this runs all linting/formatting scripts, most notably ruff
style:
	ruff check $(check_dirs) setup.py conftest.py --fix --exclude $(exclude_folders)
	ruff format $(check_dirs) setup.py conftest.py --exclude $(exclude_folders)
	python utils/custom_init_isort.py
	python utils/sort_auto_mappings.py


# Check that the repo is in a good state (both style and consistency CI checks)
# Note: each line is run in its own shell, and doing `-` before the command ignores the errors if any, continuing with next command
check-repo:
	ruff check $(check_dirs) setup.py conftest.py
	ruff format --check $(check_dirs) setup.py conftest.py
	ty check $(call get_py_files,src/transformers/utils) --force-exclude --exclude '**/*_pb2*.py'
	-python utils/custom_init_isort.py --check_only
	-python utils/sort_auto_mappings.py --check_only
	-python -c "from transformers import *" || (echo '🚨 import failed, this means you introduced unprotected imports! 🚨'; exit 1)
	-python utils/check_copies.py
	-python utils/check_modular_conversion.py
	-python utils/check_doc_toc.py
	-python utils/check_docstrings.py
	-python utils/check_dummies.py
	-python utils/check_repo.py
	-python utils/check_modeling_structure.py
	-python utils/check_inits.py
	-python utils/check_pipeline_typing.py
	-python utils/check_config_docstrings.py
	-python utils/check_config_attributes.py
	-python utils/check_doctest_list.py
	-python utils/update_metadata.py --check-only  
	-python utils/add_dates.py --check-only
	-@{ \
		md5sum src/transformers/dependency_versions_table.py > md5sum.saved; \
		python setup.py deps_table_update; \
		md5sum -c --quiet md5sum.saved || (printf "Error: the version dependency table is outdated.\nPlease run 'make fix-repo' and commit the changes. This requires Python 3.10.\n" && exit 1); \
		rm md5sum.saved; \
	}


# Run all repo checks for which there is an automatic fix, most notably modular conversions
# Note: each line is run in its own shell, and doing `-` before the command ignores the errors if any, continuing with next command
fix-repo: style
	-python setup.py deps_table_update
	-python utils/check_doc_toc.py --fix_and_overwrite
	-python utils/check_copies.py --fix_and_overwrite
	-python utils/check_modular_conversion.py --fix_and_overwrite
	-python utils/check_dummies.py --fix_and_overwrite
	-python utils/check_pipeline_typing.py --fix_and_overwrite
	-python utils/check_doctest_list.py --fix_and_overwrite
	-python utils/check_docstrings.py --fix_and_overwrite
	-python utils/add_dates.py


# Run tests for the library, requires pytest-random-order
test:
	python -m pytest -p random_order -n auto --dist=loadfile -s -v --random-order-bucket=module ./tests/

# Run tests for examples, requires pytest-random-order
test-examples:
	python -m pytest -p random_order -n auto --dist=loadfile -s -v --random-order-bucket=module ./examples/pytorch/

# Run benchmark
benchmark:
	python3 benchmark/benchmark.py --config-dir benchmark/config --config-name generation --commit=diff backend.model=google/gemma-2b backend.cache_implementation=null,static backend.torch_compile=false,true --multirun


# Release stuff
pre-release:
	python utils/release.py

pre-patch:
	python utils/release.py --patch

post-release:
	python utils/release.py --post_release

post-patch:
	python utils/release.py --post_release --patch

build-release:
	rm -rf dist
	rm -rf build
	python setup.py bdist_wheel
	python setup.py sdist
