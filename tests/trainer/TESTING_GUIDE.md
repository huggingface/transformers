# Trainer Testing Guide

## Where to put your test

| File | What it covers |
|---|---|
| `test_trainer.py` | Core integration: mixed precision, gradient accumulation/checkpointing, NEFTune, logging, metrics, step counting, early stopping, double-wrap, end-to-end |
| `test_trainer_checkpointing.py` | Checkpoint save/resume, interrupted training resume, batch size changes, frozen params, gradient accumulation resume, auto batch size |
| `test_trainer_data.py` | Data handling: collators, dynamic shapes, iterable datasets, eval/predict with iterable datasets, label smoothing |
| `test_trainer_optimizers.py` | Optimizers & LR schedulers: custom optimizers, BNB, LOMO, GrokAdamW, schedule-free, GaLore, Apollo, cosine-with-min-lr, reduce-on-plateau, Adafactor |
| `test_trainer_seq2seq.py` | Seq2seq: encoder-decoder fine-tuning, predict with generate |
| `trainer_test_utils.py` | Shared test utilities (models, datasets, helpers) — not a test file itself |

Pick the file that matches the feature you're testing. If none fits, add a new class to `test_trainer.py`.

## Running tests

```bash
# Single test
RUN_SLOW=1 python -m pytest tests/trainer/test_trainer.py::TrainerIntegrationTest::test_double_train_wrap_once -xvs

# Single GPU only
CUDA_VISIBLE_DEVICES=0 RUN_SLOW=1 python -m pytest tests/trainer/test_trainer_data.py -xvs

# All common trainer tests (non-distributed)
RUN_SLOW=1 python -m pytest tests/trainer/test_trainer.py tests/trainer/test_trainer_checkpointing.py tests/trainer/test_trainer_data.py tests/trainer/test_trainer_optimizers.py tests/trainer/test_trainer_seq2seq.py -v --tb=line
```

- `RUN_SLOW=1` is required — most trainer tests are marked `@slow` and skipped without it.
- When using `-k` with multiple test files, the filter applies globally across all files. Use full node IDs instead: `pytest file::Class::test_method`.

---

## Writing tests: best practices

### Use the shared test utilities

All test models, datasets, and helpers live in `trainer_test_utils.py`. **Reuse them** — don't create your own unless you have a specific need.

**Models** (use the smallest one that fits your purpose):

| Model | Type | When to use |
|---|---|---|
| `RegressionModel` | `nn.Module` | Simplest case, no config/pretrained support needed |
| `RegressionPreTrainedModel` | `PreTrainedModel` | Need `.from_pretrained()`, `.save_pretrained()`, or `config` |
| `RegressionPreTrainedModelWithGradientCheckpointing` | `PreTrainedModel` | Testing gradient checkpointing |
| `BasicTextGenerationModel` | `nn.Module` | Need cross-entropy / text generation |

For LLM-style tests (optimizers, memory), use a tiny model from the Hub:
```python
model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
# or
model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
```

**Datasets**:

| Dataset | When to use |
|---|---|
| `RegressionDataset` | Default choice — simple `y = ax + b` with known solution |
| `SampleIterableDataset` | Testing iterable dataset support |
| `FiniteIterableDataset` | Iterable that tracks position (for resume tests) |
| `RepeatDataset` | Token-based tasks (LLM optimizers, text generation) |
| `DynamicShapesDataset` | Variable-length sequences, padding tests |

**The `get_regression_trainer()` factory** — the fastest way to get a working Trainer:
```python
trainer = get_regression_trainer(
    output_dir=tmp_dir,
    learning_rate=0.1,
    num_train_epochs=3,
    save_steps=5,
)
trainer.train()
```
It creates the model, datasets, and args for you. Pass any `TrainingArguments` kwarg directly.

### Test class structure

Every test class should:
1. Inherit from `TestCasePlus` (provides `self.get_auto_remove_tmp_dir()`)
2. Optionally inherit from `TrainerIntegrationCommon` for checkpoint assertion helpers
3. Be decorated with `@require_torch`

```python
@require_torch
class TrainerMyFeatureTest(TestCasePlus, TrainerIntegrationCommon):

    def test_my_feature(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(output_dir=tmp_dir, my_arg=True)
        trainer.train()
        self.assertEqual(trainer.state.global_step, expected_steps)
```

### Temporary directories

Use `self.get_auto_remove_tmp_dir()` — it auto-cleans after the test. Alternatively, `tempfile.TemporaryDirectory()` as a context manager works too.

```python
# Preferred
tmp_dir = self.get_auto_remove_tmp_dir()

# Also fine
with tempfile.TemporaryDirectory() as tmp_dir:
    ...
```

### Decorators

Use the right skip decorators so tests don't fail on machines without the required hardware/packages:

```python
@require_torch                        # PyTorch required (use on all test classes)
@require_torch_accelerator            # Any accelerator (GPU/XPU/HPU)
@require_torch_multi_accelerator      # Multiple GPUs
@require_torch_non_multi_accelerator  # Single GPU only
@require_torch_gpu                    # CUDA GPU specifically
@require_torch_fp16                   # FP16 support
@require_torch_bf16                   # BF16 support
@slow                                 # Skipped unless RUN_SLOW=1
@require_peft                         # PEFT library
@require_bitsandbytes                 # bitsandbytes library
@require_galore_torch                 # galore_torch library
@require_schedulefree                 # schedulefree library
@require_apollo_torch                 # apollo_torch library
```

**With `@parameterized.expand`**: put it **outermost** (on top), above `@require_*`:

```python
# CORRECT — skip applies to each generated test
@parameterized.expand([("adamw",), ("sgd",)])
@require_bitsandbytes
def test_optim(self, optim):
    ...

# WRONG — skip is on the original, parameterized replaces it → ImportError instead of skip
@require_bitsandbytes
@parameterized.expand([("adamw",), ("sgd",)])
def test_optim(self, optim):
    ...
```

### Common assertion patterns

```python
# Model parameters converged
torch.testing.assert_close(model.a, expected_a, atol=1e-5, rtol=1e-5)

# Training state
self.assertEqual(trainer.state.global_step, expected_steps)
self.assertEqual(trainer.state.epoch, expected_epoch)

# Loss values (use places=6 for multi-GPU tolerance)
self.assertAlmostEqual(results["eval_loss"], expected_loss, places=6)

# Checkpoint exists
checkpoint_path = os.path.join(output_dir, "checkpoint-5")
self.assertTrue(os.path.isdir(checkpoint_path))

# Log history
log = trainer.state.log_history[-1]
self.assertIn("train_loss", log)

# Dataloader properties
self.assertEqual(trainer.get_train_dataloader().total_batch_size, expected)
```

### Make tests multi-GPU safe

Tests run on machines with 1 or more GPUs. The Trainer uses `nn.DataParallel` when `n_gpu > 1`. Keep these rules in mind:

1. **Never hardcode batch sizes in assertions.** `state.train_batch_size` is the effective batch size (`per_device * n_gpu`). Use `args.train_batch_size` instead.
   ```python
   # BAD
   self.assertEqual(state.train_batch_size, 2)
   # GOOD
   self.assertEqual(state.train_batch_size, args.train_batch_size)
   ```

2. **Compute steps dynamically.** Fewer steps per epoch with more GPUs.
   ```python
   steps_per_epoch = math.ceil(num_samples / (args.train_batch_size * args.gradient_accumulation_steps))
   self.assertEqual(trainer.state.global_step, steps_per_epoch)
   ```

3. **Use large enough datasets.** With 2 GPUs, a 13-sample dataset with batch_size=4 and grad_accum=3 gives `ceil(13/12) = 2` steps — which may be too few for resume tests. Use 100+ samples.

4. **Tolerate FP rounding.** DataParallel gather introduces ~1e-8 FP32 differences. Use `places=6` (tolerance 5e-7) instead of the default `places=7`.

5. **Don't assume wrapping state.** If your test calls `train()` multiple times, the model may already be wrapped in `DataParallel`. Use `self.assertIs(before, after)` to test identity, not `isinstance`.

6. **Watch out for `model_accepts_loss_kwargs`.** If a test model's forward has `**kwargs`, the Trainer assumes it uses `num_items_in_batch` for token-level loss averaging. If it doesn't, set `model.accepts_loss_kwargs = False` to prevent incorrect loss scaling.

### Keep tests fast

- Use `RegressionModel` / `RegressionDataset` — they train in milliseconds.
- Use `max_steps=10` instead of `num_train_epochs=3` when you just need training to run.
- Set `logging_steps=1` only if you need to inspect logs, otherwise leave the default.
- Use tiny configs for real models: `hidden_size=32, num_hidden_layers=3, num_attention_heads=4`.

### `TorchTracemalloc` for memory tests

```python
with TorchTracemalloc() as tracemalloc:
    trainer.train()

peak_mb = tracemalloc.peaked + bytes2megabytes(tracemalloc.begin)
self.assertLess(peak_mb, upper_bound)
```

Only meaningful on `cuda`/`xpu` — returns zeros on CPU. The context manager must handle non-GPU devices gracefully (set `begin`/`end`/`peak` to 0 in the else branch).
