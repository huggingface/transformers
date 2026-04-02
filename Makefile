# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

.PHONY: style typing check-code-quality check-repository-consistency check-repo fix-repo test test-examples benchmark codex claude clean-ai


# Runs all linting/formatting scripts, most notably ruff
style:
	@python utils/checkers.py \
		ruff_check,\
		ruff_format,\
		init_isort,\
		auto_mappings \
		--fix

# Runs ty type checker and model structure rules
typing:
	@python utils/checkers.py \
		types,\
		modeling_structure

# Runs typing, ruff linting/formatting, import-order checks and auto-mappings
check-code-quality:
	@python utils/checkers.py \
		types,\
		modeling_structure,\
		ruff_check,\
		ruff_format,\
		init_isort,\
		auto_mappings

# Runs a full repository consistency check.
check-repository-consistency:
	@python utils/checkers.py \
		imports,\
		import_complexity,\
		copies,\
		modular_conversion,\
		doc_toc,\
		docstrings,\
		dummies,\
		repo,\
		inits,\
		pipeline_typing,\
		config_docstrings,\
		config_attributes,\
		doctest_list,\
		update_metadata,\
		add_dates,\
		deps_table

# Runs typing and formatting checks + repository consistency check (ignores errors)
check-repo:
	@python utils/checkers.py \
		ruff_check,\
		ruff_format,\
		types,\
		modeling_structure,\
		init_isort,\
		auto_mappings,\
		imports,\
		import_complexity,\
		copies,\
		modular_conversion,\
		doc_toc,\
		docstrings,\
		dummies,\
		repo,\
		inits,\
		pipeline_typing,\
		config_docstrings,\
		config_attributes,\
		doctest_list,\
		update_metadata,\
		add_dates,\
		deps_table \
		--keep-going

# Run all repo checks for which there is an automatic fix, most notably modular conversions
fix-repo:
	@python utils/checkers.py \
		ruff_check,\
		ruff_format,\
		init_isort,\
		auto_mappings,\
		doc_toc,\
		copies,\
		modular_conversion,\
		dummies,\
		pipeline_typing,\
		doctest_list,\
		docstrings,\
		add_dates,\
		deps_table \
		--fix --keep-going

# Run tests for the library, requires pytest-random-order
test:
	python -m pytest -p random_order -n auto --dist=loadfile -s -v --random-order-bucket=module ./tests/

# Run tests for examples, requires pytest-random-order
test-examples:
	python -m pytest -p random_order -n auto --dist=loadfile -s -v --random-order-bucket=module ./examples/pytorch/

# Run benchmark
benchmark:
	python3 benchmark/benchmark.py --config-dir benchmark/config --config-name generation --commit=diff backend.model=google/gemma-2b backend.cache_implementation=null,static backend.torch_compile=false,true --multirun

codex:
	mkdir -p .agents
	rm -rf .agents/skills
	ln -snf ../.ai/skills .agents/skills

claude:
	mkdir -p .claude
	rm -rf .claude/skills
	ln -snf ../.ai/skills .claude/skills

clean-ai:
	rm -rf .agents/skills .claude/skills


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
