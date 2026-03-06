# Trainer Testing Guide

## Test files

| File | What it covers |
|---|---|
| `test_trainer.py` | Core: mixed precision, grad accumulation, logging, metrics, early stopping |
| `test_trainer_checkpointing.py` | Checkpoint save/resume, interrupted training, frozen params |
| `test_trainer_data.py` | Collators, dynamic shapes, iterable datasets, label smoothing |
| `test_trainer_optimizers.py` | Optimizers & LR schedulers |
| `test_trainer_seq2seq.py` | Encoder-decoder fine-tuning |
| `trainer_test_utils.py` | Shared utilities (models, datasets, helpers) — not a test file |
| `distributed/` | DDP, FSDP, DeepSpeed (see [below](#distributed-tests)) |

## Running tests

Always use `RUN_SLOW=1` — most trainer tests are `@slow` and will be skipped without it.

### Debugging workflow

**Never run the full suite until the specific failing test passes.** Work from smallest scope outward:

1. **Single GPU** — fastest feedback:
   ```bash
   CUDA_VISIBLE_DEVICES=0 RUN_SLOW=1 python -m pytest tests/trainer/test_trainer.py::Class::test_name -xvs
   ```
2. **Fix and re-run** that same test until it passes.
3. **2 GPUs** — catch DataParallel issues:
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 python -m pytest tests/trainer/test_trainer.py::Class::test_name -xvs
   ```
4. **Full test class** — check for regressions:
   ```bash
   RUN_SLOW=1 python -m pytest tests/trainer/test_trainer.py::Class -xvs
   ```
5. **All tests in that file — only at the very end**:
   ```bash
   RUN_SLOW=1 python -m pytest tests/trainer/test_trainer.py -v --tb=line
   ```

Same for distributed tests — single failing test first, fix, confirm, then widen scope.

**Tip**: `-k` filter applies globally across files. Use full node IDs instead: `pytest file::Class::test`.

## Writing tests

**`get_regression_trainer()`** is the fastest way to get a working Trainer. Pass any `TrainingArguments` kwarg directly. Uses `RegressionModel` + `RegressionDataset` (trains in milliseconds).

For LLM tests, use tiny Hub models: `AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")`.

Use `max_steps=10` instead of `num_train_epochs=3` when you just need training to run.

### Multi-GPU safety

The Trainer uses `nn.DataParallel` when `n_gpu > 1`:

- `train_batch_size = per_device_train_batch_size * n_gpu` — don't hardcode batch sizes in assertions.
- Compute steps dynamically: `math.ceil(num_samples / (batch_size * grad_accum))`.
- Use 100+ samples — small datasets can leave zero resume steps on multi-GPU.
- DataParallel gather introduces ~1e-8 FP differences — use `places=6` for loss assertions.
- If a test model has `**kwargs` but ignores `num_items_in_batch`, set `model.accepts_loss_kwargs = False`.

### Decorators

`@parameterized.expand` must be **outermost** (top), above `@require_*`.

---

## Distributed tests

### Directory layout

```
distributed/
  test_trainer_distributed.py           # Base: path constants, TrainerDistributedCommon ABC
  test_trainer_distributed_ddp.py       # DDP tests
  test_trainer_distributed_fsdp.py      # FSDP tests (config parsing + distributed)
  test_trainer_distributed_deepspeed.py # DeepSpeed tests (single-GPU + distributed)
  accelerate_configs/                   # YAML configs for `accelerate launch`
  scripts/                              # Scripts launched as subprocesses
    train.py                            # Main training script (synthetic data, tiny Qwen2)
    torchrun_env_check.py               # Dumps distributed env info to JSON per rank
    ds_config_zero2.json, ds_config_zero3.json
```

### Architecture

Each framework has three pieces:

1. **`{Framework}CommandsMixin`** — `get_torchrun_cmd()` and `get_accelerate_cmd()`.
2. **`TestTrainerDistributed{Framework}`** — framework-specific tests (env parity, etc.). NOT `@slow`.
3. **`TestTrainerDistributed{Framework}Common`** — inherits `TrainerDistributedCommon` for shared scenarios. `@slow`.

MRO: `class Foo(Mixin, TrainerDistributedCommon, TestCasePlus)` — Mixin before ABC.

`TrainerDistributedCommon` provides: `check_training`, `check_mixed_precision`, `check_gradient_accumulation`, `check_resume`, `check_eval`. Subclasses call these with `config_file=...`.

### Env parity tests

Both torchrun and accelerate sides must use the framework:

- **DDP**: no extra args (both `DistributedType.MULTI_GPU`)
- **FSDP**: `--fsdp full_shard --fsdp_config '{"fsdp_version": 1}'` (JSON string, no file)
- **DeepSpeed**: `--deepspeed path/to/ds_config_zero2.json`

`torchrun_env_check.py` uses `HfArgumentParser(TrainingArguments)` — accepts any TrainingArguments flag.

### Adding a distributed test

1. Shared scenario → add `check_*` to `TrainerDistributedCommon`, wire from each Common class.
2. Framework-specific → add to `TestTrainerDistributed{Framework}`.
3. New scripts → `distributed/scripts/`, reference via `SCRIPTS_DIR`.

### Pitfalls

- `str(args.parallel_mode)` → `"ParallelMode.DISTRIBUTED"`, not `"DISTRIBUTED"`.
- FSDP `cpu_offload` is not JSON-serializable — use `str()`.
- `train.py` defaults to `do_train=True`. Pass `--do_eval` explicitly for eval. Auto-enables when `--eval_output_file` is passed.
- DeepSpeed eval only works with ZeRO-3.
- `--fsdp_config` accepts a file path OR JSON string starting with `{`. Same for `--deepspeed`, `--accelerator_config`.
- `args.local_rank` may be -1 before framework consumes it — use `assertIn(val, (rank, -1))`.
- `@parameterized.expand` + ABC: can't use `@abstractmethod` on methods that subclasses decorate with expand.
