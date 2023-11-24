# Copyright 2022 - Intel Corp. All rights reserved.
# Authors: Mayank Kumar Raunak, Javier Turek, Nicole Beckage

"""
Implementation of a new method for fine-tuning transformer models that we call
Information Gain Filtration 'IGF' on WikiText data set and compared the results
with the standard fine-tuning method

Steps followed in the code:

1) Generate a objective dataset of pairs (X, IG(X)). IG(X)--Informativeness of context 'X'.
Our IG (information gain) model is learning to predict the ‘informativeness’ of a particular
context. Informativeness is the change in metric between the model’s accuracy on an
objective set before and after seeing that context. For casual language modeling, the
metric is perplexity.

2) A secondary learner is trained to infer a function approximation for IG using the dataset
created in (1).

3) The learner created in (2) is used to inform the fine-tuning process and filter out low informative samples.

Last, a plot is generated to compare the performance of IGF to standard fine-tuning without any filtering

"""

# Prerequisite libraries:

import argparse
import random

import joblib
import numpy as np
import torch
from igf.igf import (
    SecondaryLearner,
    collect_objective_set,
    compute_perplexity,
    generate_datasets,
    load_gpt2,
    recopy_gpt2,
    set_seed,
    train_secondary_learner,
)
from torch.utils.data import DataLoader, RandomSampler

from transformers import GPT2LMHeadModel


def generate_n_pairs(
    context_len=32,
    max_steps=10,
    size_objective_set=100,
    min_len=1026,
    trim=True,
    data_file="data/tokenized_stories_train_wikitext103.jbl",
    igf_data_file="igf_context_pairs.jbl",
):
    """
    Collecting *n* pairs for training the secondary learner
    Args:
        context_len: The maximum total input sequence length after tokenization. Sequences longer
                    than this will be truncated, sequences shorter will be padded
        max_steps: To calculate training epochs of secondary learner
        size_objective_set: size of objective data set used to create (X,IG(X)) pairs which is the training data for secondary learner
        min_len: The minimum length of the article to be used as objective set
        trim: If True truncate the context if it exceeds context length
        data_file: Tokenized data set split for training and evaluation of model
        igf_data_file: file to store (I,IG(X)) paired data set to train secondary learner

    Returns:
        Data stored in igf_data_file

    """
    # generates same data everytime
    set_seed(3)
    # generate train_data and objective_set
    train_data, objective_set = generate_datasets(
        context_len, data_file, number=size_objective_set, min_len=1026, trim=True
    )
    # keeps model same across runs
    set_seed(4)
    # model, lm_optimizer, lm_scheduler = recopy_gpt2(model, device, max_steps) # store original model weights
    # can we train on GPU?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load pretrained model
    model = load_gpt2("gpt2").to(device)
    print("computing perplexity on objective set")
    orig_perp = compute_perplexity(model, objective_set, context_len).item()
    print("perplexity on objective set:", orig_perp)

    # collect igf pairs and save to file demo.jbl
    collect_objective_set(model, orig_perp, context_len, train_data, objective_set, max_steps, device, igf_data_file)

    # clean up, delete model and data we don't need anymore
    del model, train_data, objective_set
    torch.cuda.empty_cache()


def training_secondary_learner(
    secondary_learner_train_data,
    secondary_learner_max_epochs=15,
    secondary_learner_batch_size=128,
    eval_freq=100,
    igf_model_path="igf_model.pt",
):
    """
    Train the secondary learner

    Args:
        secondary_learner_train_data: Data set with (X,IG(X)) pairs to train secondary learner where IG(X) - measure of informativeness and X- context
        secondary_learner_max_epochs: Number of epochs to train secondary learner
        secondary_learner_batch_size: Batch size to train secondary learner
        eval_freq (object): secondary model evaluation can be triggered at eval_freq
        igf_model_path: path to store trained secondary learner

    Returns:
        Trained secondary learner
    """

    set_seed(42)

    # Load pre-trained model
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Initialize secondary learner to use embedding weights of model
    secondary_learner = SecondaryLearner(model)

    # Train secondary learner
    secondary_learner = train_secondary_learner(
        secondary_learner,
        secondary_learner_train_data,
        max_epochs=secondary_learner_max_epochs,
        batch_size=secondary_learner_batch_size,
        eval_freq=100,
        igf_model_path=igf_model_path,
    )

    del model, secondary_learner_train_data
    torch.cuda.empty_cache()

    return secondary_learner


def finetune(
    model,
    train_dataset,
    test_dataset,
    context_len=32,
    max_steps=1000,
    batch_size=16,
    threshold=1.0,
    recopy_model=recopy_gpt2,
    secondary_learner=None,
    eval_interval=10,
    finetuned_model_name="gpt2_finetuned.pt",
):
    """
    fine-tune with IGF if secondary_learner is not None, else standard fine-tuning

    Args:
        model: pre-trained GPT-2 model
        train_dataset: Data set to train GPT-2 model
        test_dataset: Evaluate GPT-2 model
        context_len: The maximum total input sequence length after tokenization. Sequences longer
                    than this will be truncated, sequences shorter will be padded
        max_steps: To calculate training epochs
        batch_size: Batch size to train GPT-2 model
        threshold: The threshold value used by secondary learner to filter the train_data and allow only"
                    informative data as input to the model
        recopy_model: Reset the model to the original pretrained GPT-2 weights after each iteration
        secondary_learner: Selection of IGF as fine-tuning method if not None
        eval_interval: number of batches after which decay the selectivity of our secondary learner filter from
                        1 standard deviation above average to 1 below average
        fine-tuned_model_name: name of the final final-tuned GPT-2 model

    Returns:
        Fine-tuned GPT-2 model

    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler)

    num_train_epochs = max_steps // (len(train_dataset)) + 1
    global_step = 0
    context = torch.zeros((1, context_len), dtype=torch.long, device=device)
    model, lm_optimizer, lm_scheduler = recopy_model(model, device, max_steps)

    model.train()
    if secondary_learner is not None:
        secondary_learner.to(device)
        secondary_learner.eval()
    contexts = []
    examples = 0

    observed_qs = []
    test_perps = []

    # Compute the performance of the transformer model at the beginning
    real_perp = compute_perplexity(model, test_dataset, context_len)
    test_perps.append(real_perp)
    print("Test perplexity, step", global_step, ":", real_perp)
    for epoch in range(int(num_train_epochs)):
        for step, example in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            start = random.randint(0, example.size(2) - context_len - 1)
            context[0, :] = example[0, 0, start : start + context_len]
            lm_optimizer.zero_grad()
            outputs = model(context, labels=context)
            do_backprop = True

            if secondary_learner is not None:
                predicted_q = secondary_learner.forward(
                    torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
                )[0].item()
                observed_qs.append(float(predicted_q))

                # Here we implement the simple non-constant threshold for the predicted IG(X) value
                # We will decay the selectivity of our secondary learner filter from
                # 1 standard deviation above average to 1 below average after 10 batches.

                if global_step == 10:
                    threshold = -1
                if predicted_q < threshold:
                    do_backprop = False

            # If we passed the filter, add the context to the batch!
            if do_backprop:
                contexts.append(np.array(context.cpu()))
                lm_loss = outputs[0]
                lm_loss.backward()
                examples += 1

            del outputs

            # Once the batch is filled with enough contexts, backprop on the batch.
            if examples == batch_size:
                torch.cuda.empty_cache()
                examples = 0
                # Do LM backprop
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                lm_optimizer.step()
                lm_scheduler.step()  # Update learning rate schedule
                global_step += 1
                # Compute the performance of the transformer model at this batch
                if global_step % eval_interval == 0:
                    real_perp = compute_perplexity(model, test_dataset, context_len)
                    test_perps.append(real_perp)

                    print("Test perplexity, step", global_step, ":", real_perp)
            # Break out of the loop after 60 batches
            if max_steps > 0 and global_step > 60:
                break
        if max_steps > 0 and global_step > 60:
            break

    # save finetuned transformer model
    torch.save(model.state_dict(), finetuned_model_name)
    torch.cuda.empty_cache()
    # Do some cleaning up so we can reinitialize for the next run of this function
    del lm_optimizer
    del lm_scheduler
    return model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a transformer model with IGF on a language modeling task")

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain data files for WikiText.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help=(
            "A jbl file containing tokenized data which can be split as objective dataset, "
            "train_dataset and test_dataset."
        ),
    )

    parser.add_argument(
        "--igf_data_file",
        type=str,
        default=None,
        help="A jbl file containing the context and information gain pairs to train secondary learner.",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the final fine-tuned model is stored.",
    )

    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--context_len",
        default=32,
        type=int,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        ),
    )

    parser.add_argument(
        "--size_objective_set",
        default=100,
        type=int,
        help="number of articles that are long enough to be used as our objective set",
    )
    parser.add_argument(
        "--eval_freq", default=100, type=int, help="secondary model evaluation is triggered at eval_freq"
    )

    parser.add_argument("--max_steps", default=1000, type=int, help="To calculate training epochs")

    parser.add_argument(
        "--secondary_learner_batch_size",
        default=128,
        type=int,
        help="batch size of training data for secondary learner",
    )

    parser.add_argument(
        "--batch_size", default=16, type=int, help="batch size of training data of language model(gpt2) "
    )

    parser.add_argument(
        "--eval_interval",
        default=10,
        type=int,
        help=(
            "decay the selectivity of our secondary learner filter from "
            "1 standard deviation above average to 1 below average after 10 batches"
        ),
    )

    parser.add_argument(
        "--number", default=100, type=int, help="The number of examples split to be used as objective_set/test_data"
    )

    parser.add_argument(
        "--min_len", default=1026, type=int, help="The minimum length of the article to be used as objective set"
    )

    parser.add_argument(
        "--secondary_learner_max_epochs", default=15, type=int, help="number of epochs to train secondary learner"
    )

    parser.add_argument("--trim", default=True, type=bool, help="truncate the example if it exceeds context length")

    parser.add_argument(
        "--threshold",
        default=1.0,
        type=float,
        help=(
            "The threshold value used by secondary learner to filter the train_data and allow only"
            " informative data as input to the model"
        ),
    )

    parser.add_argument("--finetuned_model_name", default="gpt2_finetuned.pt", type=str, help="finetuned_model_name")

    parser.add_argument(
        "--recopy_model",
        default=recopy_gpt2,
        type=str,
        help="Reset the model to the original pretrained GPT-2 weights after each iteration",
    )

    # function calls
    # Collecting *n* pairs of context and information gain(X, IG(X)) for training the secondary learner
    generate_n_pairs(
        context_len=32,
        max_steps=10,
        size_objective_set=100,
        min_len=1026,
        trim=True,
        data_file="data/tokenized_stories_train_wikitext103.jbl",
        igf_data_file="igf_context_pairs.jbl",
    )

    # Load train data for secondary learner
    secondary_learner_train_data = joblib.load("data/IGF_values.jbl")

    # Train secondary learner
    secondary_learner = training_secondary_learner(
        secondary_learner_train_data,
        secondary_learner_max_epochs=15,
        secondary_learner_batch_size=128,
        eval_freq=100,
        igf_model_path="igf_model.pt",
    )

    # load pretrained gpt2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    set_seed(42)

    # Generate train and test data to train and evaluate gpt2 model
    train_dataset, test_dataset = generate_datasets(
        context_len=32, file="data/tokenized_stories_train_wikitext103.jbl", number=100, min_len=1026, trim=True
    )

    # fine-tuning of the gpt2 model using igf (Information Gain Filtration)
    finetune(
        model,
        train_dataset,
        test_dataset,
        context_len=32,
        max_steps=1000,
        batch_size=16,
        threshold=1.0,
        recopy_model=recopy_gpt2,
        secondary_learner=secondary_learner,
        eval_interval=10,
        finetuned_model_name="gpt2_finetuned.pt",
    )


if __name__ == "__main__":
    main()
