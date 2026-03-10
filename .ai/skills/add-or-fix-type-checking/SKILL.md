---
name: add-or-fix-type-checking
description: Fixes broken typing checks detected by ty, make style, or make check-repo. Use when typing errors appear in local runs, CI, or PR logs.
---

# Add Or Fix Type Checking

Use this skill to fix typing failures reported by `ty`, `make style`, or `make check-repo` (typically through `utils/check_types.py`).

Trigger this skill when command output includes wording like:
- `ty check`
- `make style` failed on typing
- `make check-repo` failed on typing
- `utils/check_types.py`
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

   d. **Use `TypeGuard` functions for dynamic module attributes** (for example
      `torch.npu`, `torch.xpu`, `torch.compiler`). Instead of `getattr(torch, "npu")`
      or `hasattr(torch, "npu") and torch.npu.is_available()`, define a type guard
      function in `src/transformers/_typing.py`:
      ```python
      def has_torch_npu(mod: ModuleType) -> TypeGuard[Any]:
          return hasattr(mod, "npu") and mod.npu.is_available()
      ```
      Then use it as a narrowing check: `if has_torch_npu(torch): torch.npu.device_count()`.
      After the guard, `ty` treats the module as `Any`, allowing attribute access without
      `getattr()` or `cast()`. See existing guards in `_typing.py` for all device backends.

      **Key rules for type guards**:
      - Use `TypeGuard[Any]` (not a Protocol) — this is the simplest form that works
        with `ty` and avoids losing the original module's known attributes.
      - The guard function must be called directly in an `if` condition for narrowing
        to work. `ty` does NOT narrow through `and` conditions or `if not guard: return`.
      - Import guards with `from .._typing import has_torch_xxx` (not via module
        attribute `_typing.has_torch_xxx`) — `ty` only resolves `TypeGuard` from
        direct imports.

   e. **Use `getattr()` / `setattr()` for dynamic model/config attributes**.
      For runtime-injected fields (for example config/model flags), use
      `getattr(obj, "field", default)` for reads and `setattr(obj, "field", value)`
      for writes. Avoid `getattr(torch, "npu")` style — use type guards instead (see above).

   f. **Use `cast()` sparingly and locally**. Only use it when the type checker
      cannot narrow a type that is known to be correct at runtime. Prefer `isinstance()`
      narrowing when the runtime type is checkable. Do not use `cast()` for
      dynamic module attributes — use type guards instead.

   g. **Use `# type: ignore` as a last resort**. Use this for cases where the checker
      cannot prove correctness (for example dynamic dispatch, C extensions, third-party stubs).

5. **Things to never do**:
   - Do not use `getattr(torch, "backend")` to access dynamic device backends
     (`npu`, `xpu`, `hpu`, `musa`, `mlu`, `neuron`, `compiler`) — use type guards
   - Do not use `cast()` for module attribute narrowing — use type guards
   - Do not add helper methods or abstractions just to satisfy the type checker
     (especially for only 1-2 occurrences)
   - Do not pollute base classes with domain-specific fields; use Protocols
   - Do not add `if x is not None` guards for values guaranteed non-None
     by the call chain; fix the annotation instead
   - Do not use conditional inheritance patterns; annotate `self` instead

6. **Organization**:
   - Keep shared Protocols and type aliases in `src/transformers/_typing.py`
   - Import type-only symbols under `if TYPE_CHECKING:` to avoid circular deps
   - Use `from __future__ import annotations` for PEP 604 syntax (`X | Y`)

7. **Verify and close the PR loop**:
   - Re-run `ty check` on the same `<target>`
   - Re-run `make style` to confirm the full style/type step passes
   - If working toward merge readiness, run `make check-repo`
   - Ensure runtime behavior did not change and run relevant tests

8. **Update CI coverage when adding new typed areas**:
   - Update `ty_check_dirs` in `Makefile` to include newly type-checked directories.

