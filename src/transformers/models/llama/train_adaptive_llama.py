import pytest

import torch

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_adaptive_llama import AdaptiveFanIn, AdaptiveFanOut, AdaptiveFanInOutput, AdaptiveFanOutOutput, AdaptiveLlamaModel


VOCAB_SIZE = 20
MAX_SEQ_LEN = 10

import random
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM

from transformers import Trainer
from transformers import TrainingArguments

from accelerate.tracking import LOGGER_TYPE_TO_CLASS, GeneralTracker, filter_trackers

class SequentialNumbersDataset():
    def __init__(self, length=1000, num_numbers=100, max_sequence_length=10):
        self.length = length
        self.num_numbers = num_numbers
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return self.length

    def __getitem__(self, i):

        current_length = random.randint(5, self.max_sequence_length)
        start_from = random.randint(3, self.num_numbers - current_length - 3) # pad + bos + eos tokens

        max_padding = self.max_sequence_length - current_length

        inputs_ids = [1] + list(range(start_from, start_from + current_length)) + [2] + ([0] * max_padding)
        labels = [1] + list(range(start_from, start_from + current_length)) + [2] + ([-100] * max_padding)
        attention_mask_length = current_length + 2
        attention_mask = ([1] * attention_mask_length) + ([0] * max_padding)
        attention_mask = torch.tensor(attention_mask)
        special_embeddings_mask = torch.zeros_like(attention_mask)
        special_embeddings_mask[0] = 1
        special_embeddings_mask[attention_mask_length - 1] = 1

        assert len(attention_mask) == len(inputs_ids)

        return {
            "input_ids": inputs_ids,
            "labels": labels,
            "special_embeddings_mask": special_embeddings_mask,
            "attention_mask": attention_mask,
        }


class AdaptiveLlamaTrainer(Trainer):
    def compute_loss(self, model: AdaptiveLlamaForCausalLM, inputs, return_outputs=False, log_metrics=True):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model.forward(
            input_ids=inputs['input_ids'],
            labels=inputs['labels'],
            special_embeddings_mask=inputs['special_embeddings_mask'],
            attention_mask=inputs['attention_mask'],
        )

        outputs_inversed = model.forward(
            input_ids=inputs['input_ids'],
            labels=inputs['labels'],
            special_embeddings_mask=inputs['special_embeddings_mask'],
            attention_mask=inputs['attention_mask'],
            use_cache=False,
            invert_merging_maps=True
        )

        # TODO Lambda for inversed outputs?
        loss = outputs.loss + (outputs_inversed.loss / 10)

        log_info = {
            "debug/straight_loss": loss.detach().item(),
            "debug/inverse_loss": outputs_inversed.loss.item() / 10,
            "debug/mean_merged_tokens": outputs.mean_merged_tokens,
        }
        self.log(log_info)

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: AdaptiveLlamaForCausalLM, *args, **kwargs):
        result = super().training_step(model, *args, **kwargs)

        merger_mpl_grad = model.model.adaptive_down[0].fan_in_mlp.weight.grad.norm(2)

        assert merger_mpl_grad is not None, "merger_mpl_grad is expected to be not none"

        # if merger_mpl_grad > 5:
        #     breakpoint()

        extra_log = dict()
        extra_log["train/merger_mpl_grad_norm"] = merger_mpl_grad.detach().item()

        self.log(extra_log)

        return result



if __name__ == "__main__":
    snd = SequentialNumbersDataset(length=1000, num_numbers=VOCAB_SIZE, max_sequence_length=MAX_SEQ_LEN)
    snd_eval = SequentialNumbersDataset(length=100, num_numbers=VOCAB_SIZE, max_sequence_length=MAX_SEQ_LEN)

    llama_config = LlamaConfig(
        hidden_size=128,
        vocab_size=VOCAB_SIZE,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=MAX_SEQ_LEN,
        use_cache=False,
        attn_implementation = 'eager'
    )

    model = AdaptiveLlamaForCausalLM(llama_config)
    model.train()

    print("num model parameters:", sum(p.numel() for p in model.parameters()))

    training_args = TrainingArguments(
        output_dir="llama_for_sequential_numbers",
        learning_rate=1e-4,
        warmup_steps=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        push_to_hub=False,
        optim="adamw_torch",
        report_to="wandb",
        logging_steps=5,
    )

    trainer = AdaptiveLlamaTrainer(
        model,
        args=training_args,
        train_dataset=snd,
        eval_dataset=snd_eval,
    )

    # trainer.accelerator.log_with = filter_trackers("wandb", training_args.output_dir)
    # trackers = filter_trackers(log_with, self.logging_dir)
    # if len(trackers) < 1 and log_with is not None:
    #     warnings.warn(f"`log_with={log_with}` was passed but no supported trackers are currently installed.")
    # self.log_with = trackers

    trainer.accelerator.init_trackers(
        project_name="llama_for_sequential_numbers",
    )

    trainer.train()
