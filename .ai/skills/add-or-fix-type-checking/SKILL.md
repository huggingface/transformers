---
name: add-or-fix-type-checking
description: Fixes broken typing checks detected by ty, make style, or make check-repo. Use when typing errors appear in local runs, CI, or PR logs.
---

# Add Or Fix Type Checking

Use this skill to fix typing failures reported by `ty` or `make style`.

Trigger this skill when command output includes wording like:
- `ty check`
- `make style` failed on typing
- `make check-repo` failed on typing
- `typing` / `type check` failures in CI or PR logs

Use this skill when:
- a user asks to add or fix typing for a module or directory, or
- a PR branch fails typing checks, or
- `make style` or `make check-repo` reports failures from `utils/check_types.py` / `ty`.

## Input

- `<target>`: module or directory to type-check (if known).
- Optional `make style` or CI output showing typing failures.

## Workflow

1. **Identify scope from the failing run**:
   - If you already have `make style` or CI output, extract the failing file/module paths.
   - If not, run:
     ```bash
     make style
     ```
   - Choose the narrowest target that covers the failures.

2. **Run `ty check` for the target** to get a focused baseline:
   ```bash
   ty check --respect-ignore-files --exclude '**/*_pb*' <target>
   ```

3. **Triage errors by category** before fixing anything:
   - Wrong/missing type annotations on signatures
   - Attribute access on union types (for example `X | None`)
   - Mixin/protocol self-type issues
   - Genuinely untyped dynamic access

4. **Apply fixes using this priority order** (simplest first):

   a. **Fix incorrect type hints at the source**. If a parameter is typed `X | None`
      but can never be `None` when actually called, remove `None` from the hint.
      Do not add defensive `if x is not None` guards for impossible cases.

   b. **Annotate untyped attributes**. Add type annotations to instance variables
      set in `__init__` or elsewhere (for example `self.foo: list[int] = []`).
      Declare class-level attributes that are set dynamically later
      (for example `_cache: Cache`, `_token_tensor: torch.Tensor | None`).

   c. **Use `self: "ProtocolType"` for mixins**. When a mixin accesses attributes
      from its host class, define a Protocol in `src/transformers/_typing.py` and
      annotate `self` on methods that need it. Apply this consistently to all methods
      in the mixin. Import under `TYPE_CHECKING` to avoid circular imports.

   d. **`getattr()`/`setattr()` vs Protocol — human decision required**.
      When a type error involves accessing an attribute on a loosely-typed object
      (for example `processor.tokenizer`, `processor.feature_extractor`,
      `model.config.some_flag`), there is a tension between two approaches:

      - **`getattr(obj, "field", default)`**: Quick fix, no new types, but hurts
        readability when the pattern repeats across the codebase.
      - **Protocol**: Cleaner reads, but adds a type definition that must be maintained.

      **Do not auto-decide.** Instead, flag the choice to the user with both options
      and the number of call sites affected. Let the human pick. If they choose a
      Protocol, add it to `src/transformers/_typing.py` (see rule 6 below).

      For runtime-injected fields that are truly one-off (for example config flags
      set by quantizers), `getattr(obj, "field", default)` is usually fine.

   e. **Use `cast()` sparingly and locally**. Only use it when the type checker
      cannot narrow a type that is known to be correct at runtime. Prefer `isinstance()`
      narrowing when the runtime type is checkable. Do not use `cast()` for
      dynamic module attributes — use type guards instead.

   f. **Use `# type: ignore` as a last resort**. Use this for cases where the checker
      cannot prove correctness (for example dynamic dispatch, C extensions, third-party stubs).

5. **Things to never do**:
   - Do not use `getattr(torch, "backend")` to access dynamic device backends
     (`npu`, `xpu`, `hpu`, `musa`, `mlu`, `neuron`, `compiler`) — use hasattr
   - Do not use `cast()` for module attribute narrowing — fix the callee signature or use hasattr
   - Do not add helper methods or abstractions just to satisfy the type checker
     (especially for only 1-2 occurrences)
   - Do not pollute base classes with domain-specific fields; use Protocols
   - Do not add `if x is not None` guards for values guaranteed non-None
     by the call chain; fix the annotation instead
   - Do not use conditional inheritance patterns; annotate `self` instead
   - Do not define Protocols, TypeAliases, or typing helpers inline in feature
     modules — they belong in `src/transformers/_typing.py`
    - Do not use asserts

6. **Organization**:
   - **Always add new Protocols, TypeAliases, and shared typing helpers to
     `src/transformers/_typing.py`** — never define them inline in feature modules.
     This is the single canonical location for all typing constructs in the codebase.
   - Import type-only symbols under `if TYPE_CHECKING:` to avoid circular deps
   - Use `from __future__ import annotations` for PEP 604 syntax (`X | Y`)
   - When importing from `_typing.py` inside subpackages, use relative imports
     (for example `from .._typing import ...`)

7. **Third-party stubs and missing attributes**:
   - Some third-party packages (for example `safetensors`) have incomplete stubs
     that don't declare attributes like `__version__`. For these, use
     `getattr(module, "__version__", "unknown")` or a `hasattr` guard.
   - For `torch` device backends (`torch.hpu`, `torch.npu`, etc.), always use
     inline `hasattr` checks — this should be consistent across all device
     backends, not just the ones that happen to fail.

8. **Verify and close the PR loop**:
   - Re-run `ty check` on the same `<target>`
   - Re-run `make style` to confirm the full style/type step passes
   - If working toward merge readiness, run `make check-repo`
   - Ensure runtime behavior did not change and run relevant tests

9. **Update CI coverage when adding new typed areas**:
   - Update `ty_check_dirs` in `Makefile` to include newly type-checked directories.
