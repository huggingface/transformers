# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

.PHONY: style typing check-code-quality check-repository-consistency check-repo fix-repo test test-examples benchmark codex claude clean-ai


# Checker lists. The two CI jobs (CircleCI runs `make check-code-quality` and
# `make check-repository-consistency` in parallel) own the canonical sets below.
# Local convenience targets `check-repo` and `fix-repo` are *derived* from them,
# so they can never drift out of sync (e.g. silently dropping `auto_mappings`
# from CI, as happened in #45018 → fixed in #45774).

STYLE_CHECKERS := ruff_check, ruff_format, init_isort, sort_auto_mappings
TYPING_CHECKERS := types, modeling_structure
CODE_QUALITY_CHECKERS := $(TYPING_CHECKERS), $(STYLE_CHECKERS)

REPO_CONSISTENCY_CHECKERS := \
	auto_mappings, \
	imports, \
	import_complexity, \
	copies, \
	modular_conversion, \
	doc_toc, \
	modeling_rules_doc, \
	docstrings, \
	dummies, \
	repo, \
	inits, \
	pipeline_typing, \
	config_docstrings, \
	config_attributes, \
	doctest_list, \
	update_metadata, \
	add_dates, \
	deps_table

ALL_CHECKERS := $(CODE_QUALITY_CHECKERS), $(REPO_CONSISTENCY_CHECKERS)


# Runs all linting/formatting scripts, most notably ruff
style:
	@python utils/checkers.py $(STYLE_CHECKERS) --fix

# Runs ty type checker and model structure rules
typing:
	@python utils/checkers.py $(TYPING_CHECKERS)

# Runs typing, ruff linting/formatting, import-order checks and auto-mappings
check-code-quality:
	@python utils/checkers.py $(CODE_QUALITY_CHECKERS)

# Runs a full repository consistency check.
check-repository-consistency:
	@python utils/checkers.py $(REPO_CONSISTENCY_CHECKERS)

# Runs typing and formatting checks + repository consistency check (ignores errors)
check-repo:
	@python utils/checkers.py $(ALL_CHECKERS) --keep-going

# Run all repo checks for which there is an automatic fix, most notably modular conversions
fix-repo:
	@python utils/checkers.py $(ALL_CHECKERS) --fix --keep-going

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
