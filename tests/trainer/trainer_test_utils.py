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

"""Shared test infrastructure for the Trainer test suite."""

import dataclasses
import gc
import json
import os
import random

import numpy as np

from transformers import (
    AutoTokenizer,
    PreTrainedConfig,
    TrainerCallback,
    TrainingArguments,
    is_datasets_available,
    is_torch_available,
)
from transformers.testing_utils import (
    backend_empty_cache,
    backend_max_memory_allocated,
    backend_memory_allocated,
    backend_reset_max_memory_allocated,
    get_tests_dir,
    torch_device,
)
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    is_accelerate_available,
)


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

    from transformers import (
        AutoModelForCausalLM,
        PreTrainedModel,
        Trainer,
        TrainerState,
    )

if is_datasets_available():
    import datasets

# for version specific tests in TrainerIntegrationTest
if is_accelerate_available():
    pass


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
        else:
            self.begin = 0
        return self

    def __exit__(self, *exc):
        gc.collect()
        if torch_device in ["cuda", "xpu"]:
            backend_empty_cache(torch_device)
            self.end = backend_memory_allocated(torch_device)
            self.peak = backend_max_memory_allocated(torch_device)
        else:
            self.end = 0
            self.peak = 0
        self.used = bytes2megabytes(self.end - self.begin)
        self.peaked = bytes2megabytes(self.peak - self.begin)


@dataclasses.dataclass
class RegressionTrainingArguments(TrainingArguments):
    a: float = 0.0
    b: float = 0.0


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
        args = RegressionTrainingArguments(output_dir, a=a, b=b, **kwargs)
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
        checkpoint = os.path.join(output_dir, f"checkpoint-{(total // freq) * freq}")
        log_history = TrainerState.load_from_json(os.path.join(checkpoint, "trainer_state.json")).log_history

        values = [d[metric] for d in log_history]
        best_value = max(values) if greater_is_better else min(values)
        best_checkpoint = (values.index(best_value) + 1) * freq
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
