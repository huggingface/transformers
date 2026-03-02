# Trainer Testing Guide

## Where to put your test

| File | What it covers |
|---|---|
| `test_trainer.py` | Core integration: mixed precision, grad accumulation, logging, metrics, step counting, early stopping |
| `test_trainer_checkpointing.py` | Checkpoint save/resume, interrupted training, batch size changes, frozen params |
| `test_trainer_data.py` | Collators, dynamic shapes, iterable datasets, label smoothing |
| `test_trainer_optimizers.py` | Optimizers & LR schedulers: BNB, LOMO, GaLore, Apollo, schedule-free, etc. |
| `test_trainer_seq2seq.py` | Encoder-decoder fine-tuning, predict with generate |
| `trainer_test_utils.py` | Shared utilities (models, datasets, helpers) — not a test file |

If none fits, add a new class to `test_trainer.py`.

## Running tests

```bash
# Single test
RUN_SLOW=1 python -m pytest tests/trainer/test_trainer.py::TrainerIntegrationTest::test_double_train_wrap_once -xvs

# All common trainer tests
RUN_SLOW=1 python -m pytest tests/trainer/test_trainer.py tests/trainer/test_trainer_checkpointing.py tests/trainer/test_trainer_data.py tests/trainer/test_trainer_optimizers.py tests/trainer/test_trainer_seq2seq.py -v --tb=line
```

- `RUN_SLOW=1` is required — most tests are `@slow`.
- `-k` filter applies globally across files. Use full node IDs: `pytest file::Class::test_method`.

## Writing tests

### Shared utilities in `trainer_test_utils.py`

**Models** — use the smallest one that fits:

| Model | When to use |
|---|---|
| `RegressionModel` (`nn.Module`) | Simplest case, no pretrained support needed |
| `RegressionPreTrainedModel` | Need `.from_pretrained()`, `.save_pretrained()`, or `config` |
| `BasicTextGenerationModel` | Cross-entropy / text generation |

For LLM-style tests, use tiny models from the Hub:
```python
model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
```

**Datasets**: `RegressionDataset` (default), `SampleIterableDataset` (iterable), `RepeatDataset` (token tasks), `DynamicShapesDataset` (variable lengths).

**`get_regression_trainer()`** — fastest way to get a working Trainer. Creates model, datasets, and args. Pass any `TrainingArguments` kwarg directly.

### Test class structure

```python
@require_torch
class TrainerMyFeatureTest(TestCasePlus, TrainerIntegrationCommon):
    def test_my_feature(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(output_dir=tmp_dir, my_arg=True)
        trainer.train()
        self.assertEqual(trainer.state.global_step, expected_steps)
```

### Decorators

`@parameterized.expand` must be **outermost** (on top), above `@require_*`:

```python
# CORRECT
@parameterized.expand([("adamw",), ("sgd",)])
@require_bitsandbytes
def test_optim(self, optim): ...

# WRONG — tests fail with ImportError instead of skip
@require_bitsandbytes
@parameterized.expand([("adamw",), ("sgd",)])
def test_optim(self, optim): ...
```

### Multi-GPU safety

The Trainer uses `nn.DataParallel` when `n_gpu > 1`. Key rules:

1. **Batch size depends on GPU count.** `args.train_batch_size` equals `per_device_train_batch_size * n_gpu`, so assertions with hardcoded values break on multi-GPU. Use `args.per_device_train_batch_size` (GPU-invariant) or `args.train_batch_size` (GPU-aware) instead.
2. **Compute steps dynamically.** `steps = math.ceil(num_samples / (batch_size * grad_accum))`.
3. **Use large enough datasets.** Small datasets (13 samples) can leave zero steps to resume on multi-GPU. Use 100+.
4. **Tolerate FP rounding.** DataParallel gather introduces ~1e-8 differences. Use `places=6` for loss assertions.
5. **Watch `model_accepts_loss_kwargs`.** If a test model has `**kwargs` but doesn't use `num_items_in_batch`, set `model.accepts_loss_kwargs = False`.

### Keep tests fast

- Use `RegressionModel` / `RegressionDataset` — trains in milliseconds.
- Use `max_steps=10` instead of `num_train_epochs=3` when you just need training to run.
- Use tiny Hub models for LLM tests.
