import os


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import dataclasses
import tempfile

import datasets
import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, enable_full_determinism
from transformers.testing_utils import TestCasePlus, require_torch, require_torch_up_to_2_accelerators, slow


"""
Creating this file was necessary, instead of adding this function to test_trainer.py. The reason is that, to run
these tests with determinism, we would need to set `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"` before importing
pytorch, otherwise the call to any function using CuBLAS after using `torch.use_deterministic_algorithms(True)`
would throw an error. If this is run together with other workers, these workers can start CuBLAS before the variable is
set. If, instead, we had set this variable at the beginning of test_trainer.py, this might interfere with the other
tests. Details at https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility.
"""


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


@require_torch
class SaveResumeTest(TestCasePlus):
    def setUp(self):
        super().setUp()
        # Imported here to avoid pytest fetching and running these tests
        from .test_trainer import TrainerIntegrationTest

        self._inner = TrainerIntegrationTest()

    @require_torch_up_to_2_accelerators
    @slow
    def test_can_resume_training_lm(self):
        # Check if it works for a simple language modeling example
        with tempfile.TemporaryDirectory() as tmpdir:
            enable_full_determinism(0)
            kwargs = {
                "output_dir": tmpdir,
                "fp16": True,
                "max_steps": 20,
                "per_device_train_batch_size": 4,
                "learning_rate": 1e-5,
                "lr_scheduler_type": "cosine",
                "save_strategy": "steps",
                "save_steps": 1,
                "logging_strategy": "steps",
                "logging_steps": 1,
                "report_to": "none",
            }

            trainer = get_language_model_trainer(**kwargs)
            trainer.train(resume_from_checkpoint=False)
            # Get the parameter length of the model
            model_params = torch.cat([p.flatten() for p in trainer.model.parameters()])
            model_param_len = len(model_params)
            # Sample 1000 uniform index and save the values of the parameters (considering an unrolled vector with
            # all of them)
            indices = torch.randint(0, model_param_len, (1000,))
            # Save the values of the parameters for later comparison
            model_params_sample = model_params[indices].detach().clone()
            state1 = dataclasses.asdict(trainer.state)
            # Delete the reference
            del model_params, trainer
            # Checks if all checkpoints are there
            self._inner.check_saved_checkpoints(
                tmpdir, freq=1, total=20, is_pretrained=True, safe_weights=True, use_scaler=True
            )

            # Checkpoint at step 11
            enable_full_determinism(0)
            checkpoint = os.path.join(tmpdir, "checkpoint-11")
            trainer = get_language_model_trainer(**kwargs)
            trainer.train(resume_from_checkpoint=checkpoint)
            model_params = torch.cat([p.flatten() for p in trainer.model.parameters()])

            # Check that the parameters are the same
            self._inner.assertTrue(torch.allclose(model_params[indices], model_params_sample))
            state2 = dataclasses.asdict(trainer.state)
            self._inner.check_trainer_state_are_the_same(state1, state2)

    def tearDown(self):
        super().tearDown()


if __name__ == "__main__":
    SaveResumeTest().test_can_resume_training_lm()
