# Transformers PR #43805 – Add `set_seed` pytest fixture

`tests/conftest.py` is already in place. Next steps:

1. **Create branch** (from repo root):
   ```bash
   cd /Users/jayzuccarelli/Documents/Projects/transformers
   git fetch origin
   git checkout -b chore/add-set-seed-pytest-fixture origin/main
   ```

2. **Run the test** (from repo root). Your current shell is using the gpt-2-lab venv, which doesn’t have pytest. Use a venv in the transformers repo:
   ```bash
   cd /Users/jayzuccarelli/Documents/Projects/transformers
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   python -m pytest tests/test_training_args.py -v -x
   ```
   (If you already have a transformers `.venv` with dev deps, just `source .venv/bin/activate` and run the pytest line.)
   You should see output like `X passed in Y.XXs`. Add that to your PR description under **Tested with:**.

3. **Commit and push:**
   ```bash
   git add tests/conftest.py
   git commit -m "chore(tests): add set_seed pytest fixture for determinism"
   git push origin chore/add-set-seed-pytest-fixture
   ```

4. **Open PR** from your fork to `huggingface/transformers` main.

---

## PR title

**chore(tests): add `set_seed` pytest fixture for determinism**

## PR description

Fixes #43805

Follow-up to #43794: add a pytest fixture that sets a fixed seed (42) before each test so we always get the same RNG state in model tests and improve determinism.

- **`tests/conftest.py`** (new): `set_seed` fixture with `autouse=True` and `scope="function"`, calling `set_seed(42)` so every test runs with a fixed seed without changing individual tests.

No changes to existing tests; the fixture runs automatically.

**Tested with:** `python -m pytest tests/test_training_args.py -v -x`
