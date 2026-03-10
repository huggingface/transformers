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
   - Functions returning broad unions (for example `str | list | BatchEncoding`)
   - Mixin/protocol self-type issues
   - Dynamic attributes on objects or modules
   - Third-party stub gaps (missing kwargs, missing `__version__`, etc.)

4. **Apply fixes using this priority order** (simplest first):

   a. **Narrow unions with `isinstance()` / `if x is None` / `hasattr()`**.
      This is the primary tool for resolving union-type errors. `ty` narrows
      through all of these patterns, including the negative forms:
      ```python
      # Narrow X | None
      if x is None:
          raise ValueError("x must not be None")
      x.method()  # ty knows x is X here

      # Narrow str | UploadFile
      if isinstance(field, str):
          raise TypeError("Expected file upload, got string")
      await field.read()  # ty knows field is UploadFile here

      # Narrow union from function return (e.g. apply_chat_template -> str | BatchEncoding)
      result = func(return_dict=True)
      if not hasattr(result, "to"):
          raise TypeError("Expected BatchEncoding")
      result.to(device)  # ty treats result as having .to() here
      ```
      **Key**: always use `if ...: raise` (not `assert`) for narrowing. Asserts
      are stripped in optimized mode (`python -O`) and should never be used for
      type narrowing or input validation.

   b. **Use local variables to help ty track narrowing across closures**.
      When `self.x` is `X | None` and you need to pass it to nested functions
      or closures, `ty` cannot track that `self.x` stays non-None. Copy to a
      local variable and narrow the local:
      ```python
      manager = self.batching_manager
      if manager is None:
          raise RuntimeError("Manager not initialized")
      # Use `manager` (not `self.batching_manager`) in nested functions
      ```

   c. **Split chained calls when the intermediate type is a broad union**.
      If `func().method()` fails because `func()` returns a union, split it:
      ```python
      # BAD: ty can't narrow through chained calls
      result = func(return_dict=True).to(device)["input_ids"]

      # GOOD: split, narrow, then chain
      result = func(return_dict=True)
      if not hasattr(result, "to"):
          raise TypeError("Expected dict-like result")
      inputs = result.to(device)["input_ids"]
      ```

   d. **Fix incorrect type hints at the source**. If a parameter is typed `X | None`
      but can never be `None` when actually called, remove `None` from the hint.

   e. **Annotate untyped attributes**. Add type annotations to instance variables
      set in `__init__` or elsewhere (for example `self.foo: list[int] = []`).
      Declare class-level attributes that are set dynamically later
      (for example `_cache: Cache`, `_token_tensor: torch.Tensor | None`).

   f. **Use `self: "ProtocolType"` for mixins**. When a mixin accesses attributes
      from its host class, define a Protocol in `src/transformers/_typing.py` and
      annotate `self` on methods that need it. Apply this consistently to all methods
      in the mixin. Import under `TYPE_CHECKING` to avoid circular imports.

   g. **Use `TypeGuard` functions for dynamic module attributes** (for example
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

   h. **Use `getattr()` / `setattr()` for dynamic model/config attributes**.
      For runtime-injected fields (for example config/model flags), use
      `getattr(obj, "field", default)` for reads and `setattr(obj, "field", value)`
      for writes. Also use `getattr()` for third-party packages missing type stubs
      (for example `getattr(safetensors, "__version__", "unknown")`).
      Avoid `getattr(torch, "npu")` style — use type guards instead (see above).

   i. **Use `cast()` for pattern-matched types after structural validation**.
      When a library like `libcst` validates node structure via pattern matching
      but returns a base type, use `cast()` to tell the checker what the actual
      type is:
      ```python
      # After m.matches(node, pattern) confirms structure:
      stmt = cast(cst.Assign, node.body[0])
      name = cast(cst.Name, stmt.targets[0].target).value
      ```
      Do not use `cast()` for module attribute narrowing — use type guards.

   j. **Use `# type: ignore` only for third-party stub defects**. This means
      cases where the third-party package's type stubs are wrong or incomplete
      and there is no way to narrow or cast around it. Examples:
      - A kwarg that exists at runtime but is missing from the stubs
      - A method that exists but isn't declared in the stubs
      Always add the specific error code: `# type: ignore[call-arg]`, not bare
      `# type: ignore`.

5. **Things to never do**:
   - **Never use `assert` for type narrowing.** Asserts are stripped by `python -O`
     and must not be relied on for correctness. Use `if ...: raise` instead.
   - **Never use `# type: ignore` as a first resort.** Before adding one, try
     every approach above (isinstance, hasattr, local variable, getattr, cast).
     If you find yourself adding more than 2-3 ignores, stop and reconsider.
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
