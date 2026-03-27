# fix(tokenization, modeling): fix non-persistent buffer clobber & DeBERTa post_processor regression

Closes #44534
Closes #44568

---

## What's the problem?

This PR fixes **two independent but related bugs** introduced in the v5 refactor of
`from_pretrained` / `_post_init`. Both bugs result in silently wrong values after loading
a model or tokenizer from a checkpoint.

---

### Bug 1 — `modeling_utils`: non-persistent buffers overwritten with garbage on `from_pretrained()` (issue #44534)

**Root cause:**

`_move_missing_keys_from_meta_to_device()` iterated over all non-persistent buffers and
unconditionally replaced every one of them with `torch.empty_like(buffer, device=target)`,
regardless of whether the buffer was already on a real device with its correct, initialized
value:

```python
# BEFORE (broken)
for key, buffer in self.named_non_persistent_buffers():
    buffer_device = get_device(device_map, key, valid_torch_device=True)
    value = torch.empty_like(buffer, device=buffer_device)   # ← random garbage!
    _load_parameter_into_model(self, key, value)
```

Non-persistent buffers are not saved in the checkpoint (`state_dict`), so the weight-loading
path never touches them. Models that initialize non-persistent buffers with meaningful values
in `__init__` (e.g. sinusoidal position embeddings, precomputed masks, inv_freq tables) had
those values silently overwritten with uninitialized memory on every `from_pretrained()` call.

---

### Bug 2 — `DebertaV2Tokenizer`: `[CLS]`/`[SEP]` tokens stripped after `from_pretrained()` (issue #44568)

**Root cause:**

The custom `TemplateProcessing` post-processor was assigned to `self._tokenizer.post_processor`
**after** `super().__init__()`:

```python
# BEFORE (broken)
super().__init__(...)

# Too late — _post_init() has already run!
self._tokenizer.post_processor = processors.TemplateProcessing(...)
```

In v5, `TokenizersBackend._post_init()` (called from inside `super().__init__()`) invokes
`update_post_processor()`. Because `DebertaV2Tokenizer` does not use `add_bos_token` /
`add_eos_token`, that call installs a **no-op** `TemplateProcessing` template. Any
post-processor set afterwards on `self._tokenizer` was then **silently overwritten** the
next time `from_pretrained()` triggered `_post_init()` again — stripping `[CLS]` and `[SEP]`
from all encoded sequences.

---

## What was changed

### `src/transformers/modeling_utils.py`

- In `_move_missing_keys_from_meta_to_device()`, the non-persistent buffer loop now **skips
  any buffer that is already on a real (non-meta) device**. Such buffers were correctly
  initialized in `__init__` and must not be clobbered.
- For buffers that are still on meta device (the legitimate case this loop was designed for),
  `torch.empty_like` is replaced with **`torch.zeros_like`** so the value is at least
  deterministic zeros, which `_initialize_missing_keys` → `_init_weights` can then fix to
  the correct non-zero value if the model implements that logic.

```python
# AFTER (fixed)
for key, buffer in self.named_non_persistent_buffers():
    if buffer.device.type != "meta":
        continue  # already initialized — do not clobber
    buffer_device = get_device(device_map, key, valid_torch_device=True)
    value = torch.zeros_like(buffer, device=buffer_device)
    _load_parameter_into_model(self, key, value)
```

### `src/transformers/models/deberta_v2/tokenization_deberta_v2.py`

- The `TemplateProcessing` post-processor is now **built before `super().__init__()`** and
  passed via the `post_processor=` kwarg. This causes the base class to install it and set
  `_should_update_post_processor = False`, permanently preventing any subsequent call to
  `update_post_processor()` from overwriting it.
- A `kwargs.pop("post_processor", None)` guard is added to prevent a `TypeError` when
  `from_pretrained()` / `convert_to_native_format()` already injected a `post_processor`
  key from the saved `tokenizer.json`.

```python
# AFTER (fixed)
# Build the post-processor BEFORE super().__init__() so the base class installs
# it and marks _should_update_post_processor=False.
vocab_dict = self._tokenizer.get_vocab()
cls_token_id = vocab_dict.get(str(cls_token), 0)
sep_token_id = vocab_dict.get(str(sep_token), 0)
post_processor = processors.TemplateProcessing(
    single=f"{str(cls_token)}:0 $A:0 {str(sep_token)}:0",
    pair=f"{str(cls_token)}:0 $A:0 {str(sep_token)}:0 $B:1 {str(sep_token)}:1",
    special_tokens=[
        (str(cls_token), cls_token_id),
        (str(sep_token), sep_token_id),
    ],
)
kwargs.pop("post_processor", None)   # prevent "multiple values" TypeError

super().__init__(
    ...,
    post_processor=post_processor,
    **kwargs,
)
```

---

## Tests added

### `tests/models/deberta_v2/test_tokenization_deberta_v2.py`

Added `test_add_special_tokens_regression_issue_44568`, which verifies **all three
failure scenarios** from the original bug report:

1. **Direct instantiation** — `[CLS]` and `[SEP]` are present on a freshly created tokenizer.
2. **Save / load round-trip** — `[CLS]` and `[SEP]` survive `save_pretrained()` +
   `from_pretrained()` (this is the exact execution path that triggered the regression).
3. **Pair encoding after round-trip** — the pair template (`[CLS] A [SEP] B [SEP]`)
   produces exactly two `[SEP]` tokens with the last token being `[SEP]`.

**Test results (local):**
```
58 passed, 2 skipped in 102s
```
The one excluded test (`test_empty_input_string`) is a pre-existing numpy dtype mismatch
in the shared `TokenizerTesterMixin` base class — not caused by or related to this PR.

---

## Breaking changes

None. Both fixes are purely corrective:

- The buffer fix only changes behaviour for models whose non-persistent buffers were
  previously being silently corrupted — restoring the **correct** initialized values.
- The tokenizer fix restores CLS/SEP token insertion that was already broken in v5.

---

## Checklist

- [x] This PR fixes a bug
- [x] Related issue(s) are linked above
- [x] Tests added / updated
- [x] Passes all relevant existing tests
- [ ] Docs updated (no public API change — no docs update needed)
