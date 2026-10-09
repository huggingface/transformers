# AGENTS.md

Context file for AI agents working on transformers.

**Dual Format**: This file combines Category A (Operations Manual) and Category B (Context Guide) for comprehensive agent guidance.

## Project Overview

transformers is a Python project using Python (pip).

**Key Info:**
- **Primary Language:** Python
- **Build System:** Python (pip)
- **Test Framework:** pytest
- **Total Files:** 6632
- **Test Files:** 1813
- **AI Readiness Score:** 90/100 (Agent-Optimized)

---

## 🚨 AI Policy & Operations

Extracted from CONTRIBUTING.md - operational constraints and procedures.

### AI Policy

- code agents. We are currently bottlenecked by our ability to review and respond to them. As a result,
- **we ask that first-time contributors do not use code agents to create issues or PRs** at this time.
- We'd also ask autonomous agents not to open any PRs or issues for the moment.
- PRs that appear to be agent-written will probably be closed without review, and we may block users who do this
- <summary> Our code agent philosophy in detail </summary>

### Key Requirements

- Unless required by applicable law or agreed to in writing, software
- The *motivation* such as a frustration with the library, a project need, or work you've done that could benefit the community.
- Some model additions need extra integration work.
- [Add vision processing components](https://github.com/huggingface/transformers/blob/main/docs/source/en/add_vision_processing_components.md) if your model requires image or video inputs.
- Hub-first release: ship your model directly on the Hub via Transformers' [remote-code](https://github.com/huggingface/transformers/blob/main/docs/source/en/models.md#custom-models) support, with no upstream PR required.

### Development Procedures

- locating the first commit where it appeared (for example with [git bisect](https://git-scm.com/docs/git-bisect)) is valuable.
- Minimize the diff. Check your PR to eliminate any unnecessary changes. Ensure that you did not commit any testing scripts
- Verify tests locally and in the CI. Before opening a PR, run `make fix-repo` and use `utils/tests_fetcher.py` to
- see a list of tests that cover the files you have changed in your PR branch. Run those tests locally, and make sure
- 4. Install the library in editable mode. Use `[dev]` for most contributions because it includes the development tools needed for style and quality checks. Use `[torch,testing]` for model contributions, or `[quality]` for docs-only changes and small fixes when the full development dependency set is not needed.

### Known Workarounds & Caveats

- limitations under the License.
- The *full* traceback if an exception is raised.



## 🏗️ Architecture & Context Guide

This section provides architectural context and agent-understanding for the codebase.

### Prerequisites

- **Python:** f">=3.{min_version}.0 (or applicable language version)
- **Package Manager:** pip or uv
- **Test Runner:** pytest



### Project Structure

```
transformers/
├── Makefile
├── pyproject.toml
├── setup.py
├── src/                  # Source code
├── tests/                # Test suite (1813 files)
└── README.md             # Project documentation
```

### Architecture Overview

#### Key Components
- **Main Entry:** server.py
- **Test Suite:** 1813 test files
- **Build Configuration:** Makefile, pyproject.toml, setup.py

#### Design Principles

1. **Modularity** - Code organized by functionality with clear separation of concerns
2. **Testability** - Comprehensive test coverage across critical paths
3. **Clarity** - Explicit naming and structure for AI agent understanding
4. **Consistency** - Uniform patterns and conventions throughout codebase
5. **Maintainability** - Well-documented code with clear intent

### Directory Map

| Directory | Purpose |
|-----------|----------|
| `docs/` | Documentation |
| `examples/` | Usage examples |
| `scripts/` | Build and utility scripts |
| `src/` | Source code |
| `tests/` | Test suite |


### Development Workflow

#### Initial Setup

```bash
git clone https://github.com/YOUR_ORG/transformers.git
cd transformers
pip install -e .
# or
uv sync --all-groups
```

#### Development Commands

**Running Tests:**
```bash
pytest                    # Run all tests
pytest tests/             # Run specific test directory
pytest -v                 # Verbose output with test names
pytest -x                 # Stop on first failure
coverage run -m pytest && coverage report  # With coverage report
```

#### Code Quality
```bash
ruff check .              # Lint with ruff
ruff format .             # Format code
mypy .                    # Type checking (if configured)
```

### Code Style & Conventions

- **Naming:** Use Python conventions (snake_case for functions, PascalCase for classes)
- **Type Hints:** Yes (strongly encouraged)
- **Error Handling:** Yes - handle errors at boundaries; let exceptions propagate when another layer owns recovery
- **Logging:** Yes
- **Testing:** Yes - write tests alongside code changes

### Testing Strategy

**Framework:** pytest
**Test Files:** 1813 found

Before committing:
1. Run the full test suite: `pytest`
2. Ensure all tests pass
3. Check type hints: `mypy .`
4. Format code: `ruff format .`

### Writing Documentation

When updating docs:
1. Always include explanatory text before code snippets
2. Describe *why* and *what* before showing *how*
3. Keep sections focused on a single concept
4. Use clear, concrete examples

## Known Gotchas & Warnings

- Avoid small or "busywork" PRs. In the past, we used to accept these, but given the current flood, we simply don't
- Post-release integration: models without strong demand, or that we don't have bandwidth to take on, land after the upstream release.
- Avoid one-off "busywork" PRs (single typo, isolated style cleanup, one mutable default fix, etc.). Bundle mechanical cleanups into a clear, systematic scope.

### Contributing Guidelines

This project has a detailed contribution guide at **`CONTRIBUTING.md`**.

**Key Requirements:**
- Review the contribution guide for all requirements
- Follow established patterns in the codebase
- Ensure alignment with project's contribution policies

### Common Patterns

When contributing to this project:
1. Read existing code in the area you're modifying
2. Follow the established patterns and style
3. Write tests for new functionality
4. Use clear, descriptive variable and function names
5. Add docstrings for public APIs
6. Update tests when changing behavior

### What We Value

✅ Well-tested code with clear intent
✅ Consistent code style and naming conventions
✅ Code that is easy for AI agents to understand
✅ Clear, descriptive commit messages
✅ Modular, reusable components
✅ Comprehensive documentation

### What We Avoid

❌ Large functions doing multiple things
❌ Commented-out dead code
❌ Inconsistent naming or patterns
❌ Unclear error messages
❌ Unexplained magic numbers or strings
❌ Skipped tests or test TODOs

### AI Readiness Dimensions (Scoring)

This project is evaluated across 8 dimensions:

1. **Architecture** (15/100) - Code organization and modularity
2. **Testing** (15/100) - Test coverage and quality
3. **Dependencies** (12/100) - Dependency management
4. **Conventions** (8/100) - Consistent patterns
5. **Entry Points** (7/100) - Clear main/start locations
6. **Security** (15/100) - Input validation and error handling
7. **Build** (10/100) - Clear build/setup instructions
8. **Documentation** (8/100) - Code and project documentation

### Next Steps

Before making changes:
1. Read relevant source files to understand the existing code
2. Look at existing tests for similar functionality
3. Follow the patterns you see in the codebase
4. Write tests for your changes
5. Run `pytest` to verify nothing breaks
6. Run code quality checks: `ruff check . && mypy .`
7. Format your code: `ruff format .`

---

*Generated by Braxis - keeping AI agents in sync with your code*
