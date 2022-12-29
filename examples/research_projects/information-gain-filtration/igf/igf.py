# Copyright 2022 - Intel Corp. All rights reserved.
# Authors: Mayank Kumar Raunak, Javier Turek, Nicole Backage

import copy
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import joblib
from transformers import AdamW, GPT2LMHeadModel, get_linear_schedule_with_warmup


logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    For reproducible training

    Args:
        seed: A seed for reproducible training

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_perplexity(model, test_data, context_len):
    """
    Computes perplexity of the transformer model on data in test_data

    Args:
        model: Pre-trained GPT2 model
        test_data: Data on which perplexity calculation is required
        context_len: The maximum total input sequence length after tokenization. Sequences longer
                     than this will be truncated, sequences shorter will be padded

    Returns:
        Perplexity on input test data

    """

    model.eval()
    device = next(model.parameters()).device
    eval_batch_size = 1
    context = torch.zeros((eval_batch_size, context_len), dtype=torch.long, device=device)
    eval_dataloader = DataLoader(test_data, shuffle=False, batch_size=eval_batch_size)
    eval_loss = torch.zeros(1, device=device)
    nb_eval_examples = 0
    for batch in eval_dataloader:
        batch.to(device)
        # pad
        context.zero_()
        for i in range(eval_batch_size):
            context[i, :] = batch[i]
        outputs = model(context, labels=context)
        eval_loss += outputs[0].sum().item()
        nb_eval_examples += batch.size(0)
    eval_loss = eval_loss / nb_eval_examples
    perplexity = torch.exp(eval_loss)
    model.train()
    return perplexity


def load_gpt2(model_name="gpt2"):
    """
    load original gpt2 and save off for quicker loading

    Args:
        model_name: GPT-2

    Returns:
        GPT-2 model

    """

    model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
    torch.save(model.state_dict(), model_name + "local.pt")
    return model


def recopy_gpt2(orig_model, device, max_steps):
    """
    Reset the model to the original pretrained GPT-2 weights after each iteration

    Args:
        orig_model: Original pretrained GPT-2 model imported from Transformers library
        device: CPU/GPU
        max_steps: number of training steps

    Returns:
        Original PreTrained GPT-2 model,
        lm_optimizer: Adam optimizer with Decoupled weight decay
        lm_scheduler: linear scheduler with the appropriate schedule

    """
    model = copy.deepcopy(orig_model)
    model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    lm_optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
    lm_scheduler = get_linear_schedule_with_warmup(lm_optimizer, 0, max_steps)
    torch.cuda.empty_cache()
    return model, lm_optimizer, lm_scheduler


def intermittent_save(contexts, real_perps, past_perps, filename):

    """
    save the perplexity differences to filename

    Args:
        contexts: Example on which the perplexity is calculated
        real_perps: Perplexity after back-propagating on the selected context
        past_perps: Perplexity of model before training on the context
        filename: File to store perplexity differences

    Returns:
        file with perplexity differences

    """
    # save the perplexity differences to filename
    avg = np.array(real_perps).mean()
    std = np.array(real_perps).std()
    perp_diff = (real_perps - avg) / std
    data_final = list(zip(contexts, perp_diff, past_perps))
    joblib.dump(data_final, filename)


def collect_objective_set(
    model,
    orig_perp,
    context_len,
    train_data,
    objective_set,
    max_steps,
    device,
    filename="dev.jbl",
    recopy_model=recopy_gpt2,
):

    """
    Collect individual IGF values from pre-trained transformer model
    max_steps samples of training data to train secondary model

    Args:
        model: Pre-trained GPT2 model
        orig_perp: Perplexity of original pretrained GPT-2 model
        context_len: The maximum total input sequence length after tokenization. Sequences longer
                    than this will be truncated, sequences shorter will be padded
        train_data: Data to train model
        objective_set: Contexts used to create (X,IG(X)) pairs which is the training data for secondary learner
        max_steps: To calculate training epochs of model
        device: GPU/CPU
        filename: To store intermediate perplexity differences
        recopy_model: Reset the model to the original pretrained GPT-2 weights after each iteration

    Returns:
        file stored intermediate perplexity differences in intermediate stages

    """

    # initialize variables to record relevant information
    contexts = []
    real_perps = []
    past_perps = []

    # Initialize the transformer model
    orig_model = copy.deepcopy(model)
    orig_model.to(device="cpu")
    torch.cuda.empty_cache()

    # Compute perplexity of initial transformer model for comparison
    model.train()
    model, lm_optimizer, lm_scheduler = recopy_model(orig_model, device, max_steps)

    for step in tqdm(range(max_steps)):
        context = torch.zeros((1, context_len), dtype=torch.long, device=device)
        story = random.choice(train_data)
        start = random.randint(0, len(story[0]) - context_len - 1)
        context[0, :] = story[0][start : start + context_len]
        lm_optimizer.zero_grad()
        outputs = model(context, labels=context)
        lm_loss = outputs[0]
        past_perp = compute_perplexity(model, context, context_len)
        model.train()
        lm_loss.backward()
        # Do LM backprop
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        lm_optimizer.step()
        lm_scheduler.step()  # Update learning rate schedule

        # Compute perplexity after back-propagating on the selected context
        real_perp = compute_perplexity(model, objective_set, context_len)

        # Periodically save the stored (X, IG(X)) pairs
        if step % 1000 == 0 and step > 1:
            intermittent_save(contexts, real_perps, past_perps, filename)

        # Reset the pretrained model to the original pretrained GPT-2 weights after each iteration
        model, lm_optimizer, lm_scheduler = recopy_model(orig_model, device, max_steps)

        past_perps.append(past_perp.item())
        real_perps.append(orig_perp - real_perp.item())
        contexts.append(np.array(context.cpu()))

    intermittent_save(contexts, real_perps, past_perps, filename)


def generate_datasets(
    context_len, file="data/tokenized_stories_train_wikitext103.jbl", number=100, min_len=1026, trim=True
):
    """
    Generate objective set and training set

    Args:
        context_len: The maximum total input sequence length after tokenization. Sequences longer
                than this will be truncated, sequences shorter will be padded
        file: Tokenized data split into training set and objective set
        number: size of objective dataset
        min_len: minimum length of a context in objective set
        trim: If True truncate the context if it exceeds context length

    Returns:
        Generated objective set and training data


    """
    # Generate objective set and training set
    # Designate the first number (100) articles that are long enough to be used
    # as our objective set, rest (that are long enough) are training data for
    # secondary learner

    data = joblib.load(file)
    print("data loaded")
    objective_set = []
    if trim:
        for i, example in enumerate(data):
            if len(example[0]) > min_len:
                start = random.randint(0, len(example[0]) - context_len - 1)
                objective_set.append(example[0, start : start + context_len])
            if len(objective_set) >= number:
                break
        train_data = []
        for j in range(i + 1, len(data)):
            if len(data[j][0]) > min_len:
                train_data.append(data[j])
    else:
        objective_set = data[0:number]
        train_data = data[number:]

    joblib.dump(objective_set, "objective_set.jbl")
    print("objective set saved")
    return train_data, objective_set


def train_secondary_learner(
    secondary_learner, train_dataset, max_epochs, batch_size, eval_freq=50, igf_model_path="secondary_learner.pt"
):

    """
    Train the secondary learner (igf_model)

    Args:
        secondary_learner: secondary learner
        train_dataset: data to train secondary learner
        max_epochs: number of epochs to train secondary learner
        batch_size: batch size of training data of secondary learner
        eval_freq: secondary model evaluation can be triggered at eval_freq
        igf_model_path: path to store trained secondary learner

    Returns:
        Trained secondary learner

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # We will use the first 512 pairs from our dataset as a test set for
    # our secondary learner and the rest to train
    test_dataset = train_dataset[:512]
    train_dataset = train_dataset[512:]
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    # secondary learner model set up
    loss = nn.MSELoss()
    test_loss = nn.MSELoss(reduction="sum")
    secondary_learner.to(device)
    q_optimizer = torch.optim.Adam(secondary_learner.parameters(), lr=0.00001)
    secondary_learner.train()

    # TODO in original code this is written as number of actual batches seen
    # not number of items seen but other places it is number of items instead.
    # improve consistency! changed this to epochs for clarity
    best_test_loss = float("inf")
    # Iterate through batches until we've used max_steps batches
    for epoch in range(int(max_epochs)):
        tr_q_loss = 0.0
        secondary_learner.train()
        for step, batch in enumerate(train_dataloader):
            context = batch[0].to(device)
            real_q = batch[1].to(device)
            predicted_q = secondary_learner(context)
            q_optimizer.zero_grad()
            q_loss = loss(predicted_q, real_q.float())
            q_loss.backward()
            q_optimizer.step()
            tr_q_loss += q_loss.item()

            # model trains fairly quickly so we won't wait for a full epoch
            # eval is triggered at eval_freq and end of epochs
            if (step % eval_freq == 0 and step > 0) or ((step + 1) == len(train_dataloader)):
                tr_loss = tr_q_loss / (step + 1)

                secondary_learner.eval()
                q_loss2 = 0.0
                sum_q2 = 0.0
                predicted = []
                actual = []
                # Compute performance of the secondary learner after this batch
                for step2, batch2 in enumerate(test_dataloader):
                    features2 = batch2[0].to(device)
                    real_q2 = batch2[1].to(device)
                    predicted_q2 = secondary_learner(features2)
                    q_loss2 += test_loss(predicted_q2, real_q2).item()
                    sum_q2 += torch.sum(predicted_q2).item()
                    for ei, i in enumerate(predicted_q2.cpu().detach().numpy()):
                        predicted.append(i.item())
                    for ei, i in enumerate(real_q2.cpu().detach().numpy()):
                        actual.append(i.item())

                q_loss2 /= len(test_dataset)
                print(
                    "Epoch: ",
                    epoch,
                    "step: ",
                    step,
                    "Avg. q:",
                    sum_q2 / len(test_dataset),
                    "Train Loss: ",
                    tr_loss,
                    "Test Loss: ",
                    q_loss2,
                )
                if q_loss2 < best_test_loss:
                    joblib.dump((predicted, actual), "pred_vs_actual.jbl")
                    torch.save(secondary_learner.state_dict(), igf_model_path)
                    best_test_loss = q_loss2

            secondary_learner.train()
    return secondary_learner


class SecondaryLearner(nn.Module):
    """
    Our secondary learner
    """

    def __init__(self, model):
        """
        We use a simple convolutional network as our secondary learner

        Args:
            model: Pre-trained GPT2 model
        """
        # embeddings are from the pretrained model
        super(SecondaryLearner, self).__init__()
        self.embeddings = model.transformer.wte
        self.embeddings.weight = copy.deepcopy(model.transformer.wte.weight)
        self.conv = nn.Conv1d(self.embeddings.weight.size(1), 256, 3, padding=1)
        self.fc = nn.Sequential(nn.Linear(256, 32), nn.Dropout(p=0.1), nn.Linear(32, 32), nn.Linear(32, 1))

    def forward(self, context):
        """
        Forward pass through the secondary learner

        Args:
            context: Context input to the secondary learner

        Returns:
            tensor after squeeze operation

        """
        pooled = torch.max(self.conv(self.embeddings(context).squeeze(1).transpose(1, 2)), 2)[0]
        qs = self.fc(pooled)
        return qs.squeeze(1)

    @classmethod
    def from_pretrained(cls, state_path, model):
        """
        Load the secondary learner

        Args:
            state_path: Path to save secondary learner
            model: Pretrained GPT-2

        Returns:
            secondary learner
        """

        secondary_learner = cls(model)  # this calls __init__
        state_dict = torch.load(state_path)
        secondary_learner.load_state_dict(state_dict)
        secondary_learner.embeddings = model.transformer.wte
        secondary_learner.embeddings.weight = copy.deepcopy(model.transformer.wte.weight)
        return secondary_learner
