# Copyright 2018 the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Core Trainer integration tests: reproducibility, gradient accumulation, gradient checkpointing,
mixed precision, logging, NEFTune, memory metrics, and end-to-end training.
"""

import math
import os
import tempfile
from functools import partial

import datasets
import numpy as np
import pytest
import torch
from torch import nn

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    GPT2Config,
    GPT2LMHeadModel,
    IntervalStrategy,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    logging,
    set_seed,
)
from transformers.integrations import activate_neftune
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.testing_utils import (
    CaptureLogger,
    LoggingLevel,
    TestCasePlus,
    backend_device_count,
    execute_subprocess_async,
    require_bitsandbytes,
    require_liger_kernel,
    require_non_hpu,
    require_peft,
    require_torch,
    require_torch_accelerator,
    require_torch_bf16,
    require_torch_fp16,
    require_torch_gpu,
    require_torch_multi_accelerator,
    require_torch_non_multi_accelerator,
    require_torch_tf32,
    run_first,
    slow,
    torch_device,
)

from .trainer_test_utils import (
    ATOL,
    PATH_SAMPLE_TEXT,
    RTOL,
    AlmostAccuracy,
    BasicTextGenerationModel,
    RegressionDataset,
    RegressionModel,
    RepeatDataset,
    StoreLossCallback,
    TrainerIntegrationCommon,
    get_dataset,
    get_regression_trainer,
)


<<<<<<< HEAD
# ---------------------------------------------------------------------------
# Mixed precision tests
# ---------------------------------------------------------------------------
=======
if torch_device == "hpu":
    RTOL = 1e-3
    ATOL = 1e-3
else:
    RTOL = 1e-5
    ATOL = 1e-5

if is_torch_available():
    import safetensors.torch
    import torch
    from torch import nn
    from torch.utils.data import IterableDataset

    import transformers.optimization
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        EarlyStoppingCallback,
        GlueDataset,
        GlueDataTrainingArguments,
        GPT2Config,
        GPT2LMHeadModel,
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedModel,
        Trainer,
        TrainerState,
    )
    from transformers.trainer_pt_utils import AcceleratorConfig

if is_datasets_available():
    import datasets

# for version specific tests in TrainerIntegrationTest
if is_accelerate_available():
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState


PATH_SAMPLE_TEXT = f"{get_tests_dir()}/fixtures/sample_text.txt"


def get_dataset(file_path, tokenizer, max_len):
    dataset = datasets.load_dataset("text", data_files=file_path)

    # Filter out empty lines
    dataset = dataset.filter(lambda example: len(example["text"].strip()) > 0)

    # Define tokenization function
    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], add_special_tokens=True, truncation=True, max_length=max_len)
        # Add labels as a copy of input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Apply tokenization and remove original text column
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    return tokenized_dataset["train"]


class StoreLossCallback(TrainerCallback):
    """
    Simple callback to store the loss.
    """

    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.losses.append(logs["loss"])


class MockCudaOOMCallback(TrainerCallback):
    """
    Simple callback to simulate CUDA OOM error if
    the batch size is >= to `batch_size_limit`.
    """

    def __init__(self, batch_size_limit=16):
        self.batch_size_limit = batch_size_limit

    def on_step_end(self, args, state, control, **kwargs):
        # simulate OOM on the first step
        if state.train_batch_size >= self.batch_size_limit:
            raise RuntimeError("CUDA out of memory.")


def ForCausalLMLoss(logits, labels, vocab_size, num_items_in_batch, disable_num_items_in_batch=False):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    if num_items_in_batch is None or disable_num_items_in_batch:
        loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="mean")
    else:
        loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="sum")
        loss = loss / num_items_in_batch
    return loss


class RegressionDataset:
    def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
        np.random.seed(seed)
        self.label_names = ["labels"] if label_names is None else label_names
        self.length = length
        self.x = np.random.normal(size=(length,)).astype(np.float32)
        self.ys = [a * self.x + b + np.random.normal(scale=0.1, size=(length,)) for _ in self.label_names]
        self.ys = [y.astype(np.float32) for y in self.ys]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        result = {name: y[i] for name, y in zip(self.label_names, self.ys)}
        result["input_x"] = self.x[i]
        return result


# Converting Bytes to Megabytes
def bytes2megabytes(x):
    return int(x / 2**20)


# Copied from accelerate: https://github.com/huggingface/accelerate/blob/ee163b66fb7848892519e804688cb4ae981aacbe/src/accelerate/test_utils/scripts/external_deps/test_peak_memory_usage.py#L40C1-L73C68
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        if torch_device in ["cuda", "xpu"]:
            backend_empty_cache(torch_device)
            backend_reset_max_memory_allocated(torch_device)  # reset the peak gauge to zero
            self.begin = backend_memory_allocated(torch_device)
        return self

    def __exit__(self, *exc):
        gc.collect()
        if torch_device in ["cuda", "xpu"]:
            backend_empty_cache(torch_device)
            self.end = backend_memory_allocated(torch_device)
            self.peak = backend_max_memory_allocated(torch_device)
        self.used = bytes2megabytes(self.end - self.begin)
        self.peaked = bytes2megabytes(self.peak - self.begin)


@dataclasses.dataclass
class RegressionTrainingArguments(TrainingArguments):
    a: float = 0.0
    b: float = 0.0
    keep_report_to: bool = False

    def __post_init__(self):
        super().__post_init__()
        # save resources not dealing with reporting unless specified (also avoids the warning when it's not set)
        # can be explicitly disabled via `keep_report_to`
        if not self.keep_report_to:
            self.report_to = []


class RepeatDataset:
    def __init__(self, x, length=64):
        self.x = x
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {"input_ids": self.x, "labels": self.x}


class SequenceClassificationDataset:
    def __init__(self, length=64, vocab_size=100, num_labels=5):
        self.length = length
        self.sequences = [torch.randint(0, vocab_size, (64,)).tolist() for _ in range(length)]
        self.labels = torch.randint(0, num_labels, (length,)).tolist()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {"input_ids": self.sequences[i], "label": self.labels[i]}


class DynamicShapesDataset:
    def __init__(self, length=64, seed=42, batch_size=8):
        self.length = length
        np.random.seed(seed)
        sizes = np.random.randint(1, 20, (length // batch_size,))
        # For easy batching, we make every batch_size consecutive samples the same size.
        self.xs = [np.random.normal(size=(s,)).astype(np.float32) for s in sizes.repeat(batch_size)]
        self.ys = [np.random.normal(size=(s,)).astype(np.float32) for s in sizes.repeat(batch_size)]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {"input_x": self.xs[i], "labels": self.ys[i]}


class AlmostAccuracy:
    def __init__(self, thresh=0.25):
        self.thresh = thresh

    def __call__(self, eval_pred):
        predictions, labels = eval_pred
        true = np.abs(predictions - labels) <= self.thresh
        return {"accuracy": true.astype(np.float32).mean().item()}


class AlmostAccuracyBatched:
    def __init__(self, thresh=0.25):
        self.thresh = thresh
        self.batch_acc = []

    def __call__(self, eval_pred, compute_result):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if isinstance(labels, tuple):
            labels = labels[0]
        batch_size = len(predictions)
        true = torch.abs(predictions - labels) <= self.thresh
        acc = true.type(torch.FloatTensor).mean().item()
        self.batch_acc.extend([acc] * batch_size)
        if compute_result:
            result = {"accuracy": np.mean(self.batch_acc).item()}
            self.batch_acc = []
            return result


class RegressionModelConfig(PreTrainedConfig):
    def __init__(self, a=0, b=0, double_output=False, random_torch=True, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.double_output = double_output
        self.random_torch = random_torch
        self.hidden_size = 1


if is_torch_available():

    class SampleIterableDataset(IterableDataset):
        def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
            self.dataset = RegressionDataset(a=a, b=b, length=length, seed=seed, label_names=label_names)

        def __iter__(self):
            for i in range(len(self.dataset)):
                yield self.dataset[i]

    class FiniteIterableDataset(SampleIterableDataset):
        def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
            super().__init__(a, b, length, seed, label_names)
            self.current_sample = 0

        def __iter__(self):
            while self.current_sample < len(self.dataset):
                yield self.dataset[self.current_sample]
                self.current_sample += 1

    class MultiLoader:
        def __init__(self, loaders):
            self.loaders = loaders

        def __len__(self):
            return sum(len(loader) for loader in self.loaders)

        def __iter__(self):
            for loader in self.loaders:
                yield from loader

    class CustomDataloaderTrainer(Trainer):
        def get_train_dataloader(self):
            dataloaders = [super().get_train_dataloader(), super().get_train_dataloader()]
            return MultiLoader(dataloaders)

        def get_eval_dataloader(self, eval_dataset):
            dataloaders = [super().get_eval_dataloader(eval_dataset), super().get_eval_dataloader(eval_dataset)]
            return MultiLoader(dataloaders)

    class RegressionModel(nn.Module):
        def __init__(self, a=0, b=0, double_output=False):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(a).float())
            self.b = nn.Parameter(torch.tensor(b).float())
            self.double_output = double_output
            self.config = None

        def forward(self, input_x, labels=None, **kwargs):
            y = input_x * self.a + self.b
            if labels is None:
                return (y, y) if self.double_output else (y,)
            loss = nn.functional.mse_loss(y, labels)
            return (loss, y, y) if self.double_output else (loss, y)

    class RegressionDictModel(nn.Module):
        def __init__(self, a=0, b=0):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(a).float())
            self.b = nn.Parameter(torch.tensor(b).float())
            self.config = None

        def forward(self, input_x, labels=None, **kwargs):
            y = input_x * self.a + self.b
            result = {"output": y}
            if labels is not None:
                result["loss"] = nn.functional.mse_loss(y, labels)
            return result

    class RegressionPreTrainedModel(PreTrainedModel):
        config_class = RegressionModelConfig
        base_model_prefix = "regression"

        def __init__(self, config):
            super().__init__(config)
            self.a = nn.Parameter(torch.tensor(config.a).float())
            self.b = nn.Parameter(torch.tensor(config.b).float())
            self.double_output = config.double_output
            self.post_init()

        def forward(self, input_x, labels=None, **kwargs):
            y = input_x * self.a + self.b
            if labels is None:
                return (y, y) if self.double_output else (y,)
            loss = nn.functional.mse_loss(y, labels)
            return (loss, y, y) if self.double_output else (loss, y)

    class RegressionPreTrainedModelWithGradientCheckpointing(PreTrainedModel):
        config_class = RegressionModelConfig
        base_model_prefix = "regression"
        supports_gradient_checkpointing = True

        def __init__(self, config):
            super().__init__(config)
            self.layers = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(4)])
            self.head = nn.Linear(config.hidden_size, 1)
            self.gradient_checkpointing = False
            self.double_output = config.double_output
            self.post_init()

        def forward(self, input_x, labels=None, **kwargs):
            y = input_x.unsqueeze(0)

            for layer in self.layers:
                if self.training and self.gradient_checkpointing:
                    outputs = self._gradient_checkpointing_func(layer.__call__, y)
                else:
                    outputs = layer(y)

                y = outputs * 3

            logits = self.head(y)

            if labels is None:
                return (logits, logits) if self.double_output else (logits,)

            loss = nn.functional.mse_loss(logits, labels)

            return (loss, y, y) if self.double_output else (loss, y)

    class RegressionRandomPreTrainedModel(PreTrainedModel):
        config_class = RegressionModelConfig
        base_model_prefix = "regression"

        def __init__(self, config):
            super().__init__(config)
            self.a = nn.Parameter(torch.tensor(config.a).float())
            self.b = nn.Parameter(torch.tensor(config.b).float())
            self.random_torch = config.random_torch
            self.post_init()

        def forward(self, input_x, labels=None, **kwargs):
            y = input_x * self.a + self.b
            if self.random_torch:
                torch_rand = torch.randn(1).squeeze()
            np_rand = np.random.rand()
            rand_rand = random.random()

            if self.random_torch:
                y += 0.05 * torch_rand
            y += 0.05 * torch.tensor(np_rand + rand_rand)

            if labels is None:
                return (y,)
            loss = nn.functional.mse_loss(y, labels)
            return (loss, y)

    class BasicTextGenerationModel(nn.Module):
        def __init__(self, vocab_size, hidden_size):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids, labels=None, **kwargs):
            embedded = self.embedding(input_ids)
            lstm_out, _ = self.lstm(embedded)
            logits = self.fc(lstm_out)
            if labels is None:
                return logits

            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits

    def create_dummy_dataset_for_text_generation(vocab_size, seq_length, num_samples):
        import numpy as np

        # Create random input sequences
        input_ids = np.random.randint(0, vocab_size, (num_samples, seq_length))

        # Create a datasets.Dataset
        dataset = datasets.Dataset.from_dict({"input_ids": input_ids, "labels": input_ids})

        return dataset

    class TstLayer(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear1 = nn.Linear(hidden_size, hidden_size)
            self.ln1 = nn.LayerNorm(hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.ln2 = nn.LayerNorm(hidden_size)
            self.bias = nn.Parameter(torch.zeros(hidden_size))

        def forward(self, x):
            h = self.ln1(nn.functional.relu(self.linear1(x)))
            h = nn.functional.relu(self.linear2(x))
            return self.ln2(x + h + self.bias)

    def get_regression_trainer(
        a=0,
        b=0,
        double_output=False,
        train_len=64,
        eval_len=64,
        pretrained=True,
        keep_report_to=False,
        output_dir=None,
        **kwargs,
    ):
        label_names = kwargs.get("label_names")
        gradient_checkpointing = kwargs.get("gradient_checkpointing", False)
        train_dataset = RegressionDataset(length=train_len, label_names=label_names)
        eval_dataset = RegressionDataset(length=eval_len, label_names=label_names)

        model_init = kwargs.pop("model_init", None)
        if model_init is not None:
            model = None
        else:
            if pretrained:
                config = RegressionModelConfig(a=a, b=b, double_output=double_output)
                # We infer the correct model class if one uses gradient_checkpointing or not
                target_cls = (
                    RegressionPreTrainedModel
                    if not gradient_checkpointing
                    else RegressionPreTrainedModelWithGradientCheckpointing
                )
                model = target_cls(config)
            else:
                model = RegressionModel(a=a, b=b, double_output=double_output)

        compute_metrics = kwargs.pop("compute_metrics", None)
        data_collator = kwargs.pop("data_collator", None)
        optimizers = kwargs.pop("optimizers", (None, None))
        preprocess_logits_for_metrics = kwargs.pop("preprocess_logits_for_metrics", None)
        assert output_dir is not None, "output_dir should be specified for testing"
        args = RegressionTrainingArguments(output_dir, a=a, b=b, keep_report_to=keep_report_to, **kwargs)
        trainer = Trainer(
            model,
            args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            optimizers=optimizers,
            model_init=model_init,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        # TODO: loss function defined in RegressionModel doesn't accept num_item_per_batch, to fix later
        trainer.model_accepts_loss_kwargs = False
        return trainer

    def get_language_model_trainer(**kwargs):
        dataset = datasets.load_dataset("fka/awesome-chatgpt-prompts")
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        def _tokenize_function(examples):
            model_inputs = tokenizer(examples["prompt"], padding="max_length", truncation=True)
            model_inputs["labels"] = np.array(model_inputs["input_ids"]).astype(np.int64)
            return model_inputs

        tokenized_datasets = dataset.map(_tokenize_function, batched=True)
        training_args = TrainingArguments(**kwargs)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
        )

        return trainer


class TrainerIntegrationCommon:
    def check_saved_checkpoints(self, output_dir, freq, total, is_pretrained=True, use_scaler=False):
        weights_file = SAFE_WEIGHTS_NAME
        file_list = [weights_file, "training_args.bin", "optimizer.pt", "scheduler.pt", "trainer_state.json"]
        if is_pretrained:
            file_list.append("config.json")
        if use_scaler:
            file_list.append("scaler.pt")
        for step in range(freq, total, freq):
            checkpoint = os.path.join(output_dir, f"checkpoint-{step}")
            self.assertTrue(os.path.isdir(checkpoint))
            for filename in file_list:
                self.assertTrue(os.path.isfile(os.path.join(checkpoint, filename)))

    def check_best_model_has_been_loaded(
        self,
        output_dir,
        freq,
        total,
        trainer,
        metric,
        greater_is_better=False,
        is_pretrained=True,
    ):
        # Get log history from the final checkpoint (could be at total if not divisible by freq)
        final_checkpoint_step = total if total % freq != 0 else (total // freq) * freq
        checkpoint = os.path.join(output_dir, f"checkpoint-{final_checkpoint_step}")
        log_history = TrainerState.load_from_json(os.path.join(checkpoint, "trainer_state.json")).log_history

        values = [d[metric] for d in log_history if metric in d]
        best_value = max(values) if greater_is_better else min(values)
        best_idx = values.index(best_value)

        # Determine which checkpoint corresponds to the best metric
        # Evals happen at freq intervals, plus potentially at the final step
        eval_steps = list(range(freq, total + 1, freq))
        if total % freq != 0:
            eval_steps.append(total)
        best_checkpoint = eval_steps[best_idx]

        checkpoint = os.path.join(output_dir, f"checkpoint-{best_checkpoint}")
        if is_pretrained:
            best_model = RegressionPreTrainedModel.from_pretrained(checkpoint)
            best_model.to(trainer.args.device)
        else:
            best_model = RegressionModel()
            state_dict = safetensors.torch.load_file(os.path.join(checkpoint, SAFE_WEIGHTS_NAME))
            best_model.load_state_dict(state_dict)
            best_model.to(trainer.args.device)
        torch.testing.assert_close(best_model.a, trainer.model.a)
        torch.testing.assert_close(best_model.b, trainer.model.b)

        metrics = trainer.evaluate()
        self.assertEqual(metrics[metric], best_value)

    def remove_nan_logs(self, log):
        for key in list(log.keys()):
            if log[key] != log[key]:  # Check if the value is NaN
                del log[key]

    def check_trainer_state_are_the_same(self, trainer_state, trainer_state1):
        # We'll pop things so operate on copies.
        state = trainer_state.copy()
        state1 = trainer_state1.copy()
        # Log history main contain different logs for the time metrics (after resuming a training).
        log_history = state.pop("log_history", None)
        log_history1 = state1.pop("log_history", None)
        self.assertEqual(state, state1)
        skip_log_keys = ["train_runtime", "train_samples_per_second", "train_steps_per_second", "train_loss"]
        for log, log1 in zip(log_history, log_history1):
            for key in skip_log_keys:
                _ = log.pop(key, None)
                _ = log1.pop(key, None)

            self.remove_nan_logs(log)
            self.remove_nan_logs(log1)

            self.assertEqual(log, log1)

    def convert_to_sharded_checkpoint(self, folder):
        # Converts a checkpoint of a regression model to a sharded checkpoint.
        loader = safetensors.torch.load_file
        weights_file = os.path.join(folder, SAFE_WEIGHTS_NAME)

        extension = "safetensors"
        saver = safetensors.torch.save_file
        index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)
        shard_name = SAFE_WEIGHTS_NAME

        state_dict = loader(weights_file)

        os.remove(weights_file)
        keys = list(state_dict.keys())

        shard_files = [
            shard_name.replace(f".{extension}", f"-{idx + 1:05d}-of-{len(keys):05d}.{extension}")
            for idx in range(len(keys))
        ]
        index = {"metadata": {}, "weight_map": {key: shard_files[i] for i, key in enumerate(keys)}}

        with open(index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)

        for param_name, shard_file in zip(keys, shard_files):
            saver({param_name: state_dict[param_name]}, os.path.join(folder, shard_file))
>>>>>>> 81125cdccb (redundant check for end of training evaluation removed)


@require_torch
class TrainerMixedPrecisionTest(TestCasePlus, TrainerIntegrationCommon):
    """Tests for FP16, BF16, and TF32 mixed precision training."""

    def setUp(self):
        super().setUp()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(learning_rate=0.1, output_dir=tmp_dir)
            trainer.train()
            self.default_trained_model = (trainer.model.a, trainer.model.b)

    def check_trained_model(self, model, **kwargs):
        (a, b) = self.default_trained_model
        torch.testing.assert_close(model.a, a, **kwargs)
        torch.testing.assert_close(model.b, b, **kwargs)

    @require_torch_fp16
    @require_torch_accelerator
    def test_mixed_fp16(self):
        # very basic test
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(learning_rate=0.1, fp16=True, logging_steps=1, output_dir=tmp_dir)
            trainer.train()
            self.check_trained_model(trainer.model, atol=ATOL, rtol=RTOL)
            log_0 = trainer.state.log_history[:-1][0]
            # check that the grads were properly clipped due to the grad scaler. Otherwise, we get huge values
            self.assertEqual(log_0["grad_norm"] < 100, True)

    @require_torch_bf16
    @require_torch_accelerator
    def test_mixed_bf16(self):
        # very basic test
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(learning_rate=0.1, bf16=True, output_dir=tmp_dir)
            trainer.train()
            self.check_trained_model(trainer.model, atol=ATOL, rtol=RTOL)

    @require_torch_gpu
    @require_torch_tf32
    def test_tf32(self):
        # very basic test
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(learning_rate=0.1, tf32=True, output_dir=tmp_dir)
            trainer.train()
            self.check_trained_model(trainer.model)


# ---------------------------------------------------------------------------
# Gradient accumulation tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerGradientAccumulationTest(TestCasePlus, TrainerIntegrationCommon):
    """Tests for gradient accumulation loss alignment and batch counting."""

    def setUp(self):
        super().setUp()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(learning_rate=0.1, output_dir=tmp_dir)
            trainer.train()
            self.default_trained_model = (trainer.model.a, trainer.model.b)

    def check_trained_model(self, model, **kwargs):
        (a, b) = self.default_trained_model
        torch.testing.assert_close(model.a, a, **kwargs)
        torch.testing.assert_close(model.b, b, **kwargs)

    def test_gradient_accumulation(self):
        # Training with half the batch size but accumulation steps as 2 should give the same training losses.
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                gradient_accumulation_steps=2, per_device_train_batch_size=4, learning_rate=0.1, output_dir=tmp_dir
            )
            trainer.train()
            self.check_trained_model(trainer.model)

    @slow
    def test_gradient_accumulation_loss_alignment_with_model_loss(self):
        set_seed(42)

        model_name = "nickypro/tinyllama-15M"
        dataset_name = "wikitext"
        dataset_config = "wikitext-2-raw-v1"
        dataset = datasets.load_dataset(dataset_name, dataset_config, split="train[:40]")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokenizer.pad_token = tokenizer.eos_token

        def tokenize_function(examples):
            return tokenizer(examples["text"], max_length=16, padding="max_length", truncation=True)

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        args_kwargs = {
            "logging_steps": 1,
            "max_steps": 5,
            "learning_rate": 3e-4,
            "disable_tqdm": True,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                tmp_dir,
                **args_kwargs,
            )
            # train with base loss
            set_seed(42)
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
            base_loss_callback = StoreLossCallback()
            trainer = Trainer(
                model,
                args,
                train_dataset=tokenized_dataset,
                callbacks=[base_loss_callback],
                data_collator=data_collator,
            )
            assert trainer.model_accepts_loss_kwargs
            trainer.train()

            args = TrainingArguments(
                tmp_dir,
                **args_kwargs,
                gradient_accumulation_steps=2,
                per_device_train_batch_size=4,
            )

            # train with gradient accumulation
            set_seed(42)
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
            grad_accum_loss_callback = StoreLossCallback()
            trainer = Trainer(
                model,
                args,
                train_dataset=tokenized_dataset,
                callbacks=[grad_accum_loss_callback],
                data_collator=data_collator,
            )
            assert trainer.model_accepts_loss_kwargs
            trainer.train()

            # train with broken loss
            set_seed(42)
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
            broken_loss_callback = StoreLossCallback()
            trainer = Trainer(
                model,
                args,
                train_dataset=tokenized_dataset,
                callbacks=[broken_loss_callback],
                data_collator=data_collator,
            )
            # disable model_accepts_loss_kwargs so that "num_items_in_batch" is not passed to the model
            trainer.model_accepts_loss_kwargs = False
            trainer.train()

        # Calculate the difference between the base loss and the grad_accum loss
        diff_truth = [
            abs(base - grad) for base, grad in zip(base_loss_callback.losses, grad_accum_loss_callback.losses)
        ]
        diff_broken = [abs(base - grad) for base, grad in zip(base_loss_callback.losses, broken_loss_callback.losses)]

        # all diff truth should be quite close
        self.assertLess(max(diff_truth), 0.01, f"Difference {max(diff_truth)} is not within 0.01")
        # max diff broken should be very off ("very off" is arbitrary, but as long as it's bigger than 0.1, it's fine)
        self.assertGreater(max(diff_broken), 0.7, f"Difference {max(diff_broken)} is not greater than 0.7")

        loss_base = sum(base_loss_callback.losses)
        loss_broken = sum(broken_loss_callback.losses)

        # mean/sum loss should not vary too much.
        relative_diff = abs(loss_base - loss_broken) / max(loss_base, loss_broken)
        self.assertLess(relative_diff, 0.2, f"Relative difference {relative_diff} is not within 0.2")

    def test_gradient_accumulation_loss_alignment_with_loss_func(self):
        set_seed(42)

        model_name = "roneneldan/TinyStories-33M"
        dataset_name = "Salesforce/wikitext"
        dataset_config = "wikitext-2-raw-v1"
        dataset = datasets.load_dataset(dataset_name, dataset_config, split="train[:40]")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokenizer.pad_token = tokenizer.eos_token

        def tokenize_function(examples):
            return tokenizer(examples["text"], max_length=16, padding="max_length", truncation=True)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        tokenizer.pad_token = tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        model = AutoModelForCausalLM.from_pretrained(model_name)

        def compute_loss(logits, labels, vocab_size, num_items_in_batch, disable_num_items_in_batch=False):
            if disable_num_items_in_batch:
                num_items_in_batch = None
            return ForCausalLMLoss(logits["logits"], labels, vocab_size, num_items_in_batch)

        loss_fn = partial(compute_loss, vocab_size=model.config.vocab_size)

        base_loss_callback = StoreLossCallback()

        args_kwargs = {
            "logging_steps": 1,
            "max_steps": 5,
            "learning_rate": 3e-4,
            "disable_tqdm": True,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                tmp_dir,
                **args_kwargs,
            )
            trainer = Trainer(
                model,
                args,
                train_dataset=tokenized_dataset,
                callbacks=[base_loss_callback],
                compute_loss_func=loss_fn,
                data_collator=data_collator,
            )
            trainer.train()

        grad_accum_loss_callback = StoreLossCallback()
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                tmp_dir,
                **args_kwargs,
                gradient_accumulation_steps=2,
                per_device_train_batch_size=4,
            )
            set_seed(42)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            trainer = Trainer(
                model,
                args,
                train_dataset=tokenized_dataset,
                callbacks=[grad_accum_loss_callback],
                compute_loss_func=loss_fn,
                data_collator=data_collator,
            )
            trainer.train()

            set_seed(42)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            broken_loss_callback = StoreLossCallback()
            # we need to disable num_items_in_batch because since we are passing a custom loss,
            # we make the assumption that num_items_in_batch is handled correctly
            loss_fn = partial(compute_loss, vocab_size=model.config.vocab_size, disable_num_items_in_batch=True)
            trainer = Trainer(
                model,
                args,
                train_dataset=tokenized_dataset,
                callbacks=[broken_loss_callback],
                compute_loss_func=loss_fn,
                data_collator=data_collator,
            )
            trainer.train()

            # Calculate the difference between the base loss and the grad_accum loss
            diff_truth = [
                abs(base - grad) for base, grad in zip(base_loss_callback.losses, grad_accum_loss_callback.losses)
            ]
            diff_broken = [
                abs(base - grad) for base, grad in zip(base_loss_callback.losses, broken_loss_callback.losses)
            ]

            # all diff truth should be quite close
            self.assertLess(max(diff_truth), 0.01, f"Difference {max(diff_truth)} is not within 0.01")

            # max diff broken should be very off
            self.assertGreater(max(diff_broken), 3, f"Difference {max(diff_broken)} is not greater than 3")

    @require_torch_multi_accelerator
    def test_num_batches_in_training_with_gradient_accumulation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            for num_train_epochs in [1, 2]:
                for train_len in [123, 120]:
                    trainer = get_regression_trainer(
                        train_len=train_len,
                        per_device_train_batch_size=4,
                        gradient_accumulation_steps=5,
                        num_train_epochs=num_train_epochs,
                        output_dir=tmp_dir,
                    )

                    total_batch_samples = []

                    def wrap_get_batch_samples(fn):
                        def wrapped_fn(epoch_iterator, num_batches, device):
                            self.assertGreater(num_batches, 0)
                            batch_samples, num_items_in_batch = fn(epoch_iterator, num_batches, device)
                            self.assertEqual(len(batch_samples), num_batches)
                            total_batch_samples.append(num_batches)
                            return batch_samples, num_items_in_batch

                        return wrapped_fn

                    trainer.get_batch_samples = wrap_get_batch_samples(trainer.get_batch_samples)

                    trainer.train()

                    self.assertEqual(len(trainer.get_train_dataloader()) * num_train_epochs, sum(total_batch_samples))


# ---------------------------------------------------------------------------
# Gradient checkpointing tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerGradientCheckpointingTest(TestCasePlus):
    """Tests for gradient checkpointing during training."""

    def test_gradient_checkpointing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                per_device_train_batch_size=1,
                learning_rate=0.1,
                gradient_checkpointing=True,
                output_dir=tmp_dir,
            )
            previous_params = {k: v.detach().clone() for k, v in trainer.model.named_parameters()}

            trainer.train()

            # Check if model weights have been updated
            for k, v in trainer.model.named_parameters():
                self.assertFalse(
                    torch.allclose(previous_params[k], v, rtol=1e-4, atol=1e-4),
                    f"Model weights for {k} have not been updated",
                )


# ---------------------------------------------------------------------------
# NEFTune tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerNEFTuneTest(TestCasePlus):
    """Tests for NEFTune noise injection during training."""

    def test_neftune(self):
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=32, n_layer=3, n_head=4)
        tiny_gpt2 = GPT2LMHeadModel(config)
        x = torch.randint(0, 100, (128,))
        train_dataset = RepeatDataset(x)

        # Trainer without inf/nan filter
        args = TrainingArguments(
            self.get_auto_remove_tmp_dir(),
            learning_rate=1e-9,
            logging_steps=5,
            logging_nan_inf_filter=False,
            neftune_noise_alpha=0.4,
        )
        trainer = Trainer(tiny_gpt2, args, train_dataset=train_dataset)

        activate_neftune(trainer.model, trainer.args.neftune_noise_alpha)

        dummy_input = torch.LongTensor([[1, 0, 1]]).to(torch_device)

        emb1 = trainer.model.get_input_embeddings()(dummy_input)
        emb2 = trainer.model.get_input_embeddings()(dummy_input)

        self.assertFalse(torch.allclose(emb1, emb2), "Neftune noise is not applied!")

        # redefine the model
        tiny_gpt2 = GPT2LMHeadModel(config)
        # Trainer without inf/nan filter
        args = TrainingArguments(
            self.get_auto_remove_tmp_dir(),
            learning_rate=1e-9,
            logging_steps=5,
            logging_nan_inf_filter=False,
            neftune_noise_alpha=0.4,
        )
        trainer = Trainer(tiny_gpt2, args, train_dataset=train_dataset)

        # Check that it trains without errors
        trainer.train()

        # Make sure forward pass works fine
        _ = trainer.model(dummy_input)
        self.assertTrue(len(trainer.model.get_input_embeddings()._forward_hooks) == 0)

        trainer.model.eval()

        # Check that we get identical embeddings just in case
        emb1 = trainer.model.get_input_embeddings()(dummy_input)
        emb2 = trainer.model.get_input_embeddings()(dummy_input)
        torch.testing.assert_close(emb1, emb2)


# ---------------------------------------------------------------------------
# Logging tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerLoggingTest(TestCasePlus):
    """Tests for logging behavior: inf/nan filter and log levels."""

    def test_logging_inf_nan_filter(self):
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=32, n_layer=3, n_head=4)
        tiny_gpt2 = GPT2LMHeadModel(config)
        x = torch.randint(0, 100, (128,))
        train_dataset = RepeatDataset(x)

        # Trainer without inf/nan filter
        args = TrainingArguments(
            self.get_auto_remove_tmp_dir(),
            learning_rate=1e9,
            logging_steps=5,
            logging_nan_inf_filter=False,
        )
        trainer = Trainer(tiny_gpt2, args, train_dataset=train_dataset)
        trainer.train()
        log_history_no_filter = trainer.state.log_history

        # Trainer with inf/nan filter
        args = TrainingArguments(
            self.get_auto_remove_tmp_dir(),
            learning_rate=1e9,
            logging_steps=5,
            logging_nan_inf_filter=True,
        )
        trainer = Trainer(tiny_gpt2, args, train_dataset=train_dataset)
        trainer.train()
        log_history_filter = trainer.state.log_history

        def is_any_loss_nan_or_inf(log_history):
            losses = [l["loss"] for l in log_history[:-1]]
            return any(math.isnan(x) for x in losses) or any(math.isinf(x) for x in losses)

        self.assertTrue(is_any_loss_nan_or_inf(log_history_no_filter))
        self.assertFalse(is_any_loss_nan_or_inf(log_history_filter))

    def test_log_level(self):
        # testing only --log_level (--log_level_replica requires multiple gpus and DDP and is tested elsewhere)
        logger = logging.get_logger()
        log_info_string = "Running training"

        # test with the default log_level - should be the same as before and thus we test depending on is_info
        is_info = logging.get_verbosity() <= 20

        with tempfile.TemporaryDirectory() as tmp_dir:
            with CaptureLogger(logger) as cl:
                trainer = get_regression_trainer(output_dir=tmp_dir)
                trainer.train()
            if is_info:
                self.assertIn(log_info_string, cl.out)
            else:
                self.assertNotIn(log_info_string, cl.out)

            with LoggingLevel(logging.INFO):
                # test with low log_level - lower than info
                with CaptureLogger(logger) as cl:
                    trainer = get_regression_trainer(log_level="debug", output_dir=tmp_dir)
                    trainer.train()
                self.assertIn(log_info_string, cl.out)

            with LoggingLevel(logging.INFO):
                # test with high log_level - should be quiet
                with CaptureLogger(logger) as cl:
                    trainer = get_regression_trainer(log_level="error", output_dir=tmp_dir)
                    trainer.train()
                self.assertNotIn(log_info_string, cl.out)


# ---------------------------------------------------------------------------
# Metrics tests (FLOS, memory, input tokens)
# ---------------------------------------------------------------------------


@require_torch
class TrainerMetricsTest(TestCasePlus):
    """Tests for FLOS extraction, memory metrics, and input token counting."""

    def test_flos_extraction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(learning_rate=0.1, output_dir=tmp_dir)

            def assert_flos_extraction(trainer, wrapped_model_to_check):
                self.assertEqual(trainer.model, trainer.accelerator.unwrap_model(wrapped_model_to_check))
                self.assertGreaterEqual(
                    getattr(trainer.accelerator.unwrap_model(wrapped_model_to_check).config, "total_flos", 0), 0
                )

            # with plain model
            assert_flos_extraction(trainer, trainer.model)

            # with enforced DataParallel
            assert_flos_extraction(trainer, nn.DataParallel(trainer.model))

            trainer.train()
            self.assertTrue(isinstance(trainer.state.total_flos, float))

    def check_mem_metrics(self, trainer, check_func):
        metrics = trainer.train().metrics
        check_func("init_mem_cpu_alloc_delta", metrics)
        check_func("train_mem_cpu_alloc_delta", metrics)
        if backend_device_count(torch_device) > 0:
            check_func("init_mem_gpu_alloc_delta", metrics)
            check_func("train_mem_gpu_alloc_delta", metrics)

        metrics = trainer.evaluate()
        check_func("eval_mem_cpu_alloc_delta", metrics)
        if backend_device_count(torch_device) > 0:
            check_func("eval_mem_gpu_alloc_delta", metrics)

        metrics = trainer.predict(RegressionDataset()).metrics
        check_func("test_mem_cpu_alloc_delta", metrics)
        if backend_device_count(torch_device) > 0:
            check_func("test_mem_gpu_alloc_delta", metrics)

    def test_mem_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # with mem metrics enabled
            trainer = get_regression_trainer(skip_memory_metrics=False, output_dir=tmp_dir)
            self.check_mem_metrics(trainer, self.assertIn)

            # with mem metrics disabled
            trainer = get_regression_trainer(skip_memory_metrics=True, output_dir=tmp_dir)
            self.check_mem_metrics(trainer, self.assertNotIn)

    def test_include_num_input_tokens_seen(self):
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        tokenizer.pad_token = "[PAD]"
        model.config.pad_token_id = tokenizer.pad_token_id

        sentences = ["This is a short sentence.", "This is a much longer sentence that will require padding."]
        labels = torch.tensor([0, 1])

        # 1. Test with attention_mask
        tokenized_dataset_with_mask = tokenizer(sentences, truncation=True, padding="longest", return_tensors="pt")
        tokenized_dataset_with_mask["labels"] = labels
        dataset_with_mask = datasets.Dataset.from_dict(tokenized_dataset_with_mask)

        # 2. Test without attention_mask
        tokenized_dataset_no_mask = {k: v for k, v in tokenized_dataset_with_mask.items() if k != "attention_mask"}
        dataset_no_mask = datasets.Dataset.from_dict(tokenized_dataset_no_mask)

        # 3. Test with no padding information
        tokenizer_no_pad = AutoTokenizer.from_pretrained("bert-base-cased")
        tokenizer_no_pad.pad_token = None

        data_collator = default_data_collator

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test case 1: "non_padding" with attention_mask
            args = TrainingArguments(
                output_dir=tmp_dir,
                include_num_input_tokens_seen="non_padding",
                per_device_train_batch_size=2,
                max_steps=1,
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=dataset_with_mask,
                data_collator=data_collator,
                processing_class=tokenizer,
            )
            trainer.train()
            attention_mask = tokenized_dataset_with_mask["attention_mask"]
            non_padded_tokens_with_mask = attention_mask.sum().item()
            self.assertEqual(trainer.state.num_input_tokens_seen, non_padded_tokens_with_mask)

            # Test case 2: "non_padding" without attention_mask (fallback to pad_token_id)
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=dataset_no_mask,
                data_collator=data_collator,
                processing_class=tokenizer,
            )
            trainer.train()
            input_ids = tokenized_dataset_with_mask["input_ids"]  # use original to compute expected
            non_padded_tokens_no_mask = (input_ids != tokenizer.pad_token_id).sum().item()
            self.assertEqual(trainer.state.num_input_tokens_seen, non_padded_tokens_no_mask)

            # Test case 3: "non_padding" with no padding info (fallback to numel)
            with self.assertLogs("transformers.trainer", level="WARNING") as cm:
                trainer = Trainer(
                    model=model,
                    args=args,
                    train_dataset=dataset_no_mask,  # still has input_ids
                    data_collator=data_collator,
                    processing_class=tokenizer_no_pad,  # tokenizer without pad token
                )
                trainer.train()
                self.assertTrue(
                    any("Could not determine method to count non-padding tokens" in log for log in cm.output)
                )
            total_tokens = input_ids.numel()
            self.assertEqual(trainer.state.num_input_tokens_seen, total_tokens)

            # Test case 4: "all"
            args.include_num_input_tokens_seen = "all"
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=dataset_with_mask,
                data_collator=data_collator,
                processing_class=tokenizer,
            )
            trainer.train()
            self.assertEqual(trainer.state.num_input_tokens_seen, total_tokens)

            # Test case 5: True (backward compatibility)
            args.include_num_input_tokens_seen = True
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=dataset_with_mask,
                data_collator=data_collator,
                processing_class=tokenizer,
            )
            trainer.train()
            self.assertEqual(trainer.state.num_input_tokens_seen, total_tokens)

    def test_get_num_trainable_parameters(self):
        model = nn.Sequential(nn.Linear(128, 64), nn.Linear(64, 32))
        # in_features * out_features + bias
        layer_1 = 128 * 64 + 64
        layer_2 = 64 * 32 + 32
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(model=model, args=TrainingArguments(output_dir=tmp_dir))
            self.assertEqual(trainer.get_num_trainable_parameters(), layer_1 + layer_2)
            # Freeze the last layer
            for param in model[-1].parameters():
                param.requires_grad = False
            self.assertEqual(trainer.get_num_trainable_parameters(), layer_1)


# ---------------------------------------------------------------------------
# Step counting and training loss tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerStepCountingTest(TestCasePlus):
    """Tests for training loss computation, step counting, and epoch handling."""

    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    def test_training_loss(self):
        n_gpus = max(1, backend_device_count(torch_device))

        # With even logs
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(logging_steps=64 / (8 * n_gpus), output_dir=tmp_dir)
            trainer.train()
            log_history = trainer.state.log_history

            losses = [log["loss"] for log in log_history if "loss" in log]
            train_loss = log_history[-1]["train_loss"]
            self.assertAlmostEqual(sum(losses) / len(losses), train_loss, places=4)

        # With uneven logs
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(logging_steps=5, output_dir=tmp_dir)
            trainer.train()
            log_history = trainer.state.log_history

            # Training loss should be the same as before
            new_train_loss = log_history[-1]["train_loss"]
            self.assertAlmostEqual(train_loss, new_train_loss, places=4)

    def test_number_of_steps_in_training(self):
        # Regular training has n_epochs * len(train_dl) steps
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(learning_rate=0.1, output_dir=tmp_dir)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, self.n_epochs * 64 / self.batch_size)

        # Check passing num_train_epochs works (and a float version too):
        trainer = get_regression_trainer(learning_rate=0.1, num_train_epochs=1.5, output_dir=tmp_dir)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, int(1.5 * 64 / self.batch_size))

        # If we pass a max_steps, num_train_epochs is ignored
        trainer = get_regression_trainer(learning_rate=0.1, max_steps=10, output_dir=tmp_dir)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, 10)

    def test_num_train_epochs_in_training(self):
        # len(train_dl) < gradient_accumulation_steps shouldn't give ``ZeroDivisionError`` when ``max_steps`` is given.
        # It should give 1 update step for each epoch.
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                max_steps=3,
                train_len=64,
                per_device_train_batch_size=16,
                gradient_accumulation_steps=5,
                output_dir=tmp_dir,
            )
            train_output = trainer.train()
            self.assertEqual(train_output.global_step, 3)

            # Even ``max_steps`` is not specified, we still expect 1 update step for each epoch if
            # len(train_dl) < gradient_accumulation_steps.
            trainer = get_regression_trainer(
                train_len=64, per_device_train_batch_size=16, gradient_accumulation_steps=5, output_dir=tmp_dir
            )
            train_output = trainer.train()
            self.assertEqual(train_output.global_step, int(self.n_epochs))


# ---------------------------------------------------------------------------
# Reproducibility tests (pre-run training to check determinism across configs)
# ---------------------------------------------------------------------------


@require_torch
class TrainerIntegrationPrerunTest(TestCasePlus, TrainerIntegrationCommon):
    """
    Only tests that want to tap into the auto-pre-run 2 trainings:
    - self.default_trained_model
    - self.alternate_trained_model
    directly, or via check_trained_model
    """

    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(learning_rate=0.1, output_dir=tmp_dir)
            trainer.train()
            self.default_trained_model = (trainer.model.a, trainer.model.b)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(learning_rate=0.1, seed=314, output_dir=tmp_dir)
            trainer.train()
            self.alternate_trained_model = (trainer.model.a, trainer.model.b)

    def check_trained_model(self, model, alternate_seed=False, **kwargs):
        (a, b) = self.alternate_trained_model if alternate_seed else self.default_trained_model
        torch.testing.assert_close(model.a, a, **kwargs)
        torch.testing.assert_close(model.b, b, **kwargs)

    def test_reproducible_training(self):
        # Checks that training worked, model trained and seed made a reproducible training.
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(learning_rate=0.1, output_dir=tmp_dir)
            trainer.train()
            self.check_trained_model(trainer.model)

        # Checks that a different seed gets different (reproducible) results.
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(learning_rate=0.1, seed=314, output_dir=tmp_dir)
            trainer.train()
            self.check_trained_model(trainer.model, alternate_seed=True)

    def test_trainer_with_datasets(self):
        np.random.seed(42)
        x = np.random.normal(size=(64,)).astype(np.float32)
        y = 2.0 * x + 3.0 + np.random.normal(scale=0.1, size=(64,)).astype(np.float32)
        train_dataset = datasets.Dataset.from_dict({"input_x": x, "label": y})

        # Base training. Should have the same results as test_reproducible_training
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(tmp_dir, learning_rate=0.1)
            trainer = Trainer(model, args, train_dataset=train_dataset)
            trainer.train()
            self.check_trained_model(trainer.model)

            # Can return tensors.
            train_dataset.set_format(type="torch", dtype=torch.float32)
            model = RegressionModel()
            trainer = Trainer(model, args, train_dataset=train_dataset)
            trainer.train()
            self.check_trained_model(trainer.model)

            # Adding one column not used by the model should have no impact
            z = np.random.normal(size=(64,)).astype(np.float32)
            train_dataset = datasets.Dataset.from_dict({"input_x": x, "label": y, "extra": z})
            model = RegressionModel()
            trainer = Trainer(model, args, train_dataset=train_dataset)
            trainer.train()
            self.check_trained_model(trainer.model)

    def test_model_init(self):
        train_dataset = RegressionDataset()
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(tmp_dir, learning_rate=0.1)
            trainer = Trainer(args=args, train_dataset=train_dataset, model_init=lambda: RegressionModel())
            trainer.train()
            self.check_trained_model(trainer.model)

            # Re-training should restart from scratch, thus lead the same results.
            trainer.train()
            self.check_trained_model(trainer.model)

            # Re-training should restart from scratch, thus lead the same results and new seed should be used.
            trainer.args.seed = 314
            trainer.train()
            self.check_trained_model(trainer.model, alternate_seed=True)


# ---------------------------------------------------------------------------
# Torch compile tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerTorchCompileTest(TestCasePlus):
    @pytest.mark.torch_compile_test
    def test_torch_compile_loss_func_compatibility(self):
        config = LlamaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=3, num_attention_heads=4)
        tiny_llama = LlamaForCausalLM(config)

        x = torch.randint(0, 100, (128,))
        train_dataset = RepeatDataset(x)

        args = TrainingArguments(
            self.get_auto_remove_tmp_dir(),
            per_device_train_batch_size=2,
            torch_compile=True,
            max_steps=1,  # compile happens on the first step
        )
        trainer = Trainer(model=tiny_llama, args=args, train_dataset=train_dataset)  # noqa
        trainer.train()

    @require_peft
    @require_bitsandbytes
    @pytest.mark.torch_compile_test
    def test_bnb_compile(self):
        from peft import LoraConfig, get_peft_model

        # Simply tests if initializing a Trainer with a PEFT + compiled model works out of the box
        # QLoRA + torch compile is not really supported yet, but we should at least support the model
        # loading and let torch throw the
        tiny_model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        )

        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        tiny_model = get_peft_model(tiny_model, peft_config)

        tiny_model = torch.compile(tiny_model)

        x = torch.randint(0, 100, (128,))
        train_dataset = RepeatDataset(x)

        args = TrainingArguments(
            self.get_auto_remove_tmp_dir(),
            learning_rate=1e-9,
            logging_steps=5,
        )
        with self.assertRaises(ValueError):
            _ = Trainer(tiny_model, args, train_dataset=train_dataset)  # noqa

    @require_torch_accelerator
    @pytest.mark.torch_compile_test
    def test_torch_compile_train(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(output_dir=tmp_dir)
            metrics = trainer.train()
            original_train_loss = metrics.training_loss

            trainer = get_regression_trainer(torch_compile=True, output_dir=tmp_dir)
            metrics = trainer.train()
            self.assertAlmostEqual(metrics.training_loss, original_train_loss)

    @require_torch_accelerator
    @pytest.mark.torch_compile_test
    def test_torch_compile_eval(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(output_dir=tmp_dir)
            metrics = trainer.evaluate()
            original_eval_loss = metrics["eval_loss"]

            trainer = get_regression_trainer(torch_compile=True, output_dir=tmp_dir)
            metrics = trainer.evaluate()

            self.assertAlmostEqual(metrics["eval_loss"], original_eval_loss, delta=1e-6)


# ---------------------------------------------------------------------------
# Early stopping tests
# ---------------------------------------------------------------------------
# Early stopping tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerEarlyStoppingTest(TestCasePlus):
    def test_early_stopping_callback(self):
        # early stopping stops training before num_training_epochs
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                num_train_epochs=20,
                gradient_accumulation_steps=1,
                per_device_train_batch_size=16,
                load_best_model_at_end=True,
                eval_strategy=IntervalStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
                compute_metrics=AlmostAccuracy(),
                metric_for_best_model="accuracy",
            )
            trainer.add_callback(EarlyStoppingCallback(1, 0.0001))
            train_output = trainer.train()
            self.assertLess(train_output.global_step, 20 * 64 / 16)

        # Invalid inputs to trainer with early stopping callback result in assertion error
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                num_train_epochs=20,
                gradient_accumulation_steps=1,
                per_device_train_batch_size=16,
                eval_strategy=IntervalStrategy.EPOCH,
                compute_metrics=AlmostAccuracy(),
                metric_for_best_model="accuracy",
            )
            trainer.add_callback(EarlyStoppingCallback(1))
            self.assertEqual(trainer.state.global_step, 0)
            try:
                trainer.train()
            except AssertionError:
                self.assertEqual(trainer.state.global_step, 0)

        # even if load_best_model_at_end is False, `best_model_checkpoint` should be set
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                num_train_epochs=20,
                gradient_accumulation_steps=1,
                per_device_train_batch_size=16,
                load_best_model_at_end=False,
                eval_strategy=IntervalStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
                compute_metrics=AlmostAccuracy(),
                metric_for_best_model="accuracy",
            )
            trainer.add_callback(EarlyStoppingCallback(1, 0.0001))
            train_output = trainer.train()
            self.assertIsNotNone(trainer.state.best_model_checkpoint)


# ---------------------------------------------------------------------------
# Liger kernel tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerLigerKernelTest(TestCasePlus):
    @require_liger_kernel
    def test_use_liger_kernel_patching(self):
        import importlib

        from liger_kernel.transformers import liger_rotary_pos_emb

        from transformers.integrations.liger import apply_liger_kernel
        from transformers.models.llama import modeling_llama

        config = LlamaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=3, num_attention_heads=4)
        tiny_llama = LlamaForCausalLM(config)

        # Spot check that modeling code and model instance variables are not yet patched
        self.assertNotEqual(modeling_llama.apply_rotary_pos_emb, liger_rotary_pos_emb)
        self.assertFalse("LigerRMSNorm" in tiny_llama.model.norm.__repr__())

        apply_liger_kernel(tiny_llama, {})

        # Spot check that modeling code and model instance variables are patched
        self.assertEqual(modeling_llama.apply_rotary_pos_emb, liger_rotary_pos_emb)
        self.assertTrue("LigerRMSNorm" in tiny_llama.model.norm.__repr__())

        # Restore the original module to avoid leaking monkey patches to other tests
        importlib.reload(modeling_llama)

    @require_liger_kernel
    def test_use_liger_kernel_custom_config_patching(self):
        import importlib

        from liger_kernel.transformers import LigerRMSNorm

        from transformers.integrations.liger import apply_liger_kernel
        from transformers.models.llama import modeling_llama

        config = LlamaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=3, num_attention_heads=4)
        tiny_llama = LlamaForCausalLM(config)

        apply_liger_kernel(tiny_llama, {"rms_norm": False})

        # Check that the RMSNorm kernel is not applied as specified in the config
        self.assertFalse(isinstance(tiny_llama.model.norm, LigerRMSNorm))

        # Restore the original module to avoid leaking monkey patches to other tests
        importlib.reload(modeling_llama)

    @require_liger_kernel
    @require_torch_accelerator
    @require_torch_non_multi_accelerator  # Don't work with DP
    def test_use_liger_kernel_trainer(self):
        import importlib

        from transformers.models.llama import modeling_llama

        config = LlamaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=3, num_attention_heads=4)
        tiny_llama = LlamaForCausalLM(config)
        x = torch.randint(0, 100, (128,))
        train_dataset = RepeatDataset(x)
        args = TrainingArguments(
            self.get_auto_remove_tmp_dir(),
            learning_rate=1e-2,
            logging_steps=5,
            max_steps=20,
            use_liger_kernel=True,
        )
        Trainer(tiny_llama, args, train_dataset=train_dataset).train()

        # Restore the original module to avoid leaking monkey patches to other tests
        importlib.reload(modeling_llama)

    @require_liger_kernel
    @require_torch_accelerator
    @require_torch_non_multi_accelerator  # don't work with DP
    def test_use_liger_kernel_custom_config_trainer(self):
        import importlib

        from transformers.models.llama import modeling_llama

        config = LlamaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=3, num_attention_heads=4)
        tiny_llama = LlamaForCausalLM(config)
        x = torch.randint(0, 100, (128,))
        train_dataset = RepeatDataset(x)
        args = TrainingArguments(
            self.get_auto_remove_tmp_dir(),
            learning_rate=1e-2,
            logging_steps=5,
            max_steps=20,
            use_liger_kernel=True,
            liger_kernel_config={"rms_norm": False, "cross_entropy": True, "fused_linear_cross_entropy": False},
        )
        Trainer(tiny_llama, args, train_dataset=train_dataset).train()

        # Restore the original module to avoid leaking monkey patches to other tests
        importlib.reload(modeling_llama)


# ---------------------------------------------------------------------------
# Miscellaneous integration tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerIntegrationTest(TestCasePlus):
    """Integration tests: compatibility, and e2e."""

    @slow
    @run_first
    @require_non_hpu
    @require_torch_multi_accelerator
    def test_end_to_end_example(self):
        # Tests that `translation.py` will run without issues
        script_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "examples", "pytorch", "translation", "run_translation.py"
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            command = [
                "accelerate",
                "launch",
                script_path,
                "--model_name_or_path",
                "google-t5/t5-small",
                "--per_device_train_batch_size",
                "1",
                "--output_dir",
                tmpdir,
                "--do_train",
                "--max_train_samples",
                "64",
                "--num_train_epochs",
                "1",
                "--dataset_name",
                "wmt16",
                "--dataset_config",
                "ro-en",
                "--source_lang",
                "en",
                "--target_lang",
                "ro",
                "--do_predict",
                "--max_predict_samples",
                "64",
                "--predict_with_generate",
                "--ddp_timeout",
                "60",
            ]
            execute_subprocess_async(command)
            # successful return here == success - any errors would have caused an error or a timeout in the sub-call

    def test_special_token_alignment(self):
        """
        Tests that special token changes in the tokenizer result in model configs updates when using the trainer, to
        ensure special tokens are aligned across configs
        """

        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")

        # add new special tokens to tokenizer, so we can test that trainer aligns the model configs with the tokenizer
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.pad_token = "<|im_end|>"
        tokenizer.bos_token = "<|im_start|>"
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_end|>", "<|im_start|>"]})

        # the model needs to have its embedding layer resized accordingly
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

        # create a random dataset from the **new** vocab size
        x = torch.randint(0, len(tokenizer), (64,))
        dataset = RepeatDataset(x, length=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = TrainingArguments(output_dir=tmpdir, max_steps=1, per_device_train_batch_size=1)
            trainer = Trainer(
                model=model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dataset,
            )

            # We haven't started training -> not yet aligned
            self.assertNotEqual(trainer.model.config.eos_token_id, tokenizer.eos_token_id)
            self.assertNotEqual(trainer.model.config.pad_token_id, tokenizer.pad_token_id)
            self.assertNotEqual(trainer.model.config.bos_token_id, tokenizer.bos_token_id)

            trainer.train()

            # Must be aligned as soon as we start training
            self.assertEqual(trainer.model.config.eos_token_id, tokenizer.eos_token_id)
            self.assertEqual(trainer.model.config.pad_token_id, tokenizer.pad_token_id)
            self.assertEqual(trainer.model.config.bos_token_id, tokenizer.bos_token_id)

    def test_trainer_works_without_model_config(self):
        """
        Tests that models without a `config` parameter can still be trained.
        This is useful for preserving compatibility with third parties that train different models using the
        transformers Trainer.

        If this test fails, it doesn't imply that there's issues with transformers, but perhaps with third
        parties.
        """

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        model = BasicTextGenerationModel(vocab_size=tokenizer.vocab_size, hidden_size=32)

        train_dataset = get_dataset(PATH_SAMPLE_TEXT, tokenizer, 100)

        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = TrainingArguments(
                output_dir=tmpdir, max_steps=5, per_device_train_batch_size=1, use_cpu=True
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=train_dataset,
            )
            trainer.train()

    def test_training_arguments_are_left_untouched(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(output_dir=tmp_dir)
        trainer.train()
        args = TrainingArguments(tmp_dir)
        dict1, dict2 = args.to_dict(), trainer.args.to_dict()
        for key in dict1:
            self.assertEqual(dict1[key], dict2[key])

    def test_double_train_wrap_once(self):
        # test that we don't wrap the model more than once
        # since wrapping primarily happens on multi-gpu setup we want multiple gpus to test for
        # example DataParallel(DataParallel(model))

        trainer = get_regression_trainer(output_dir=self.get_auto_remove_tmp_dir())
        trainer.train()
        model_wrapped_before = trainer.model_wrapped
        trainer.train()
        model_wrapped_after = trainer.model_wrapped
        self.assertIs(model_wrapped_before, model_wrapped_after, "should be not wrapped twice")
