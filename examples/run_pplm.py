# coding=utf-8
# Copyright 2018 The Uber AI Team Authors.
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

# TODO: add code for training a custom discriminator

"""
Example command with bag of words:
python examples/run_pplm.py -B space --cond_text "The president" --length 100 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.01 --window_length 5 --kl_scale 0.01 --gm_scale 0.95

Example command with discriminator:
python examples/run_pplm.py -D sentiment --label_class 3 --cond_text "The lake" --length 10 --gamma 1.0 --num_iterations 30 --num_samples 10 --stepsize 0.01 --kl_scale 0.01 --gm_scale 0.95
"""

import argparse
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange

from transformers import GPT2Tokenizer
from transformers.file_utils import cached_path
from transformers.modeling_gpt2 import GPT2LMHeadModel
from IPython import embed

PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2-medium")

BAG_OF_WORDS_ARCHIVE_MAP = {
    'kitchen': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/kitchen.txt",
    'legal': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/legal.txt",
    'military': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/military.txt",
    'monsters': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/monsters.txt",
    'politics': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/politics.txt",
    'positive_words': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/positive_words.txt",
    'religion': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/religion.txt",
    'science': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/science.txt",
    'space': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/space.txt",
    'technology': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/technology.txt",
}

DISCRIMINATOR_MODELS_PARAMS = {
    "clickbait": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/clickbait_classifierhead.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
    },
    "sentiment": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/sentiment_classifierhead.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
    },
    "toxicity": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/toxicity_classifierhead.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_toxic": 0, "toxic": 1},
        "default_class": 0,
    },
}


class ClassificationHead(torch.nn.Module):
    """ Classification Head for the transformer """

    def __init__(self, class_size=5, embed_size=2048):
        super(ClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        # self.mlp1 = torch.nn.Linear(embed_size, embed_size)
        # self.mlp2 = (torch.nn.Linear(embed_size, class_size))
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        # hidden_state = F.relu(self.mlp1(hidden_state))
        # hidden_state = self.mlp2(hidden_state)
        logits = self.mlp(hidden_state)
        return logits


def to_var(x, requires_grad=False, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k <= 0:
        return logits

    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)

        if probs:
            return torch.where(
                logits < batch_mins,
                torch.ones_like(logits) * 0.0,
                logits
            )

        return torch.where(
            logits < batch_mins,
            torch.ones_like(logits) * -1e10,
            logits
        )


def perturb_past(
        past,
        model,
        last,
        unpert_past=None,
        unpert_logits=None,
        accumulated_hidden=None,
        grad_norms=None,
        stepsize=0.01,
        classifier=None,
        label_class=None,
        one_hot_bows_vectors=None,
        loss_type=0,
        num_iterations=3,
        kl_scale=0.01,
        window_length=0,
        horizon_length=1,
        decay=False,
        gamma=1.5,
):
    # initializie perturbation accumulator
    grad_accumulator = [
        (np.zeros(p.shape).astype("float32"))
        for p in past
    ]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(
            0.0,
            1.0 + SMALL_CONST,
            1.0 / (window_length)
        )[1:]
    else:
        decay_mask = 1.0

    # TODO fix this comment (SUMANTH)
    # generate a mask if perturbated gradient is based on a past window
    _, _, _, curr_length, _ = past[0].shape
    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([window_length])
                + tuple(past[0].shape[-1:])
        )

        zeros_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([curr_length - window_length])
                + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_val_shape)),
            dim=-2
        ).cuda()

    else:
        window_mask = torch.ones_like(past[0]).cuda()

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    for i in range(num_iterations):
        print("Iteration ", i + 1)

        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True)
            for p_ in grad_accumulator
        ]

        # Compute hidden using perturbed past
        curr_pert_past = list(map(add, past, curr_perturbation))
        all_logits, _, all_hidden = model(last, past=curr_pert_past)
        hidden = all_hidden[-1]
        accumulated_hidden += torch.sum(hidden, dim=1).detach()
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        # compute loss
        bow_loss = 0.0
        discrim_loss = 0.0
        kl_loss = 0.0

        if loss_type == PPLM_BOW or loss_type == PPLM_BOW_DISCRIM:
            for one_hot_bow in one_hot_bows_vectors:
                bow_logits = torch.mm(probs, torch.t(one_hot_bow))
                bow_loss += -torch.log(torch.sum(bow_logits))
            print(" pplm_bow_loss:", bow_loss.data.cpu().numpy())

        if loss_type == PPLM_DISCRIM or loss_type == PPLM_BOW_DISCRIM:
            ce_loss = torch.nn.CrossEntropyLoss()
            # TODO all there are for (SUMANTH)
            # TODO why we need to do this assignment and not just using unpert_past?
            curr_unpert_past = unpert_past
            # Get the model's token embeddings in order to compute our own embeds from curr_probs:
            wte = model.resize_token_embeddings()
            # TODO i is never used, why do we need to do this i times instead multiplying
            #   torch.sum(unpert_hidden, dim=1) * horizon_length?
            for i in range(horizon_length):
                # TODO the next two lines can be done only one time, and why not using probs instead as they do not change at each iteration?
                curr_probs = F.softmax(logits, dim=-1)  # get softmax
                curr_probs = torch.unsqueeze(curr_probs, dim=1)
                inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
                _, curr_unpert_past, curr_all_hidden = model(
                    past=curr_unpert_past,
                    inputs_embeds=inputs_embeds
                )
                # get expected hidden states
                unpert_hidden = curr_all_hidden[-1]
                accumulated_hidden += torch.sum(unpert_hidden, dim=1).detach()

            prediction = classifier(
                accumulated_hidden / (curr_length + 1 + horizon_length)
            )

            label = torch.tensor([label_class], device="cuda", dtype=torch.long)
            discrim_loss += ce_loss(prediction, label)
            print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())

        if kl_scale >= 0.0:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = (
                    unpert_probs + SMALL_CONST *
                    (unpert_probs <= SMALL_CONST).type(
                        torch.FloatTensor
                    ).cuda().detach()
            )

            correction = SMALL_CONST * (probs <= SMALL_CONST).type(
                torch.FloatTensor
            ).cuda().detach()
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * (
                (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
            )
            print(' kl_loss', (kl_loss).data.cpu().numpy())

        loss = bow_loss + discrim_loss + kl_loss
        loss_per_iter.append(loss.data.cpu().numpy())
        print(' pplm_loss', (loss - kl_loss).data.cpu().numpy())

        # compute gradients
        loss.backward()

        # calculate gradient norms
        if grad_norms is not None and loss_type == PPLM_BOW:
            grad_norms = [
                torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
                for index, p_ in enumerate(curr_perturbation)
            ]
        else:
            grad_norms = [
                (torch.norm(p_.grad * window_mask) + SMALL_CONST)
                for index, p_ in enumerate(curr_perturbation)
            ]

        # normalize gradients
        grad = [
            -stepsize
            * (p_.grad * window_mask / grad_norms[
                index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        # accumulate gradients
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [
        to_var(torch.from_numpy(p_), requires_grad=True)
        for p_ in grad_accumulator
    ]
    pert_past = list(map(add, past, grad_accumulator))

    return pert_past, accumulated_hidden, grad_norms, loss_per_iter


def get_classifier(
        name: Optional[str], label_class: Union[str, int], device: Union[str, torch.device]
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(
        class_size=params['class_size'],
        embed_size=params['embed_size']
    ).to(device)
    resolved_archive_file = cached_path(params["url"])
    classifier.load_state_dict(torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    if isinstance(label_class, str):
        if label_class in params["class_vocab"]:
            label_id = params["class_vocab"][label_class]
        else:
            label_id = params["default_class"]
            print("label_class {} not in class_vocab".format(label_class))
            print("available values are: {}".format(params["class_vocab"]))
            print("using default class {}".format(label_id))

    elif isinstance(label_class, int):
        if label_class in set(params["class_vocab"].values()):
            label_id = label_class
        else:
            label_id = params["default_class"]
            print("label_class {} not in class_vocab".format(label_class))
            print("available values are: {}".format(params["class_vocab"]))
            print("using default class {}".format(label_id))

    else:
        label_id = params["default_class"]

    return classifier, label_id


def get_bag_of_words_indices(bag_of_words_ids_or_paths: List[str]) -> List[List[List[int]]]:
    bow_indices = []
    for id_or_path in bag_of_words_ids_or_paths:
        if id_or_path in BAG_OF_WORDS_ARCHIVE_MAP:
            filepath = cached_path(BAG_OF_WORDS_ARCHIVE_MAP[id_or_path])
        else:
            filepath = id_or_path
        with open(filepath, "r") as f:
            words = f.read().split("\n")
        bow_indices.append([TOKENIZER.encode(word) for word in words])
    return bow_indices


def build_bows_one_hot_vectors(bow_indices):
    if bow_indices is None:
        return None

    one_hot_bows_vectors = []
    for single_bow in bow_indices:
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        single_bow = torch.tensor(single_bow).cuda()
        num_words = single_bow.shape[0]
        one_hot_bow = torch.zeros(num_words, TOKENIZER.vocab_size).cuda()
        one_hot_bow.scatter_(1, single_bow, 1)
        one_hot_bows_vectors.append(one_hot_bow)
    return one_hot_bows_vectors


def full_text_generation(
        model,
        context=None,
        num_samples=1,
        device="cuda",
        sample=True,
        discrim=None,
        label_class=None,
        bag_of_words=None,
        length=100,
        grad_length=10000,
        stepsize=0.02,
        num_iterations=3,
        temperature=1.0,
        gm_scale=0.9,
        kl_scale=0.01,
        top_k=10,
        window_length=0,
        horizon_length=1,
        decay=False,
        gamma=1.5,
        **kwargs
):
    classifier, class_id = get_classifier(
        discrim,
        label_class,
        device
    )

    bow_indices = []
    if bag_of_words:
        bow_indices = get_bag_of_words_indices(bag_of_words.split(";"))

    if bag_of_words and classifier:
        print("Both PPLM-BoW and PPLM-Discrim are on. This is not optimized.")
        loss_type = PPLM_BOW_DISCRIM

    elif bag_of_words:
        loss_type = PPLM_BOW
        print("Using PPLM-BoW")

    elif classifier is not None:
        loss_type = PPLM_DISCRIM
        print("Using PPLM-Discrim")

    else:
        raise Exception("Specify either --bag_of_words (-B) or --discrim (-D)")

    unpert_gen_tok_text, _, _ = generate_text_pplm(
        model=model,
        context=context,
        device=device,
        length=length,
        perturb=False
    )
    torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []

    for i in range(num_samples):
        pert_gen_tok_text, discrim_loss, loss_in_time = generate_text_pplm(
            model=model,
            context=context,
            device=device,
            sample=sample,
            perturb=True,
            bow_indices=bow_indices,
            classifier=classifier,
            label_class=class_id,
            loss_type=loss_type,
            length=length,
            grad_length=grad_length,
            stepsize=stepsize,
            num_iterations=num_iterations,
            temperature=temperature,
            gm_scale=gm_scale,
            kl_scale=kl_scale,
            top_k=top_k,
            window_length=window_length,
            horizon_length=horizon_length,
            decay=decay,
            gamma=gamma,
        )
        pert_gen_tok_texts.append(pert_gen_tok_text)
        if classifier is not None:
            discrim_losses.append(discrim_loss.data.cpu().numpy())
        losses_in_time.append(loss_in_time)

    torch.cuda.empty_cache()

    return unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time


def generate_text_pplm(
        model,
        context=None,
        past=None,
        device="cuda",
        sample=True,
        perturb=True,
        classifier=None,
        label_class=None,
        bow_indices=None,
        loss_type=0,
        length=100,
        grad_length=10000,
        stepsize=0.02,
        num_iterations=3,
        temperature=1.0,
        gm_scale=0.9,
        kl_scale=0.01,
        top_k=10,
        window_length=0,
        horizon_length=1,
        decay=False,
        gamma=1.5,
):
    output_so_far = (
        torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0)
        if context
        else None
    )

    # collect one hot vectors for bags of words
    one_hot_bows_vectors = build_bows_one_hot_vectors(bow_indices)

    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []
    for i in trange(length, ascii=True):

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current_token

        # run model forward to obtain unperturbed
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            if output_so_far.shape[1] > 1:
                _, past, _ = model(output_so_far[:, :-1])

            unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
            unpert_last_hidden = unpert_all_hidden[-1]

        else:
            unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
            unpert_last_hidden = unpert_all_hidden[-1]

        # check if we are abowe grad max length
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary
        if not perturb or num_iterations == 0:
            pert_past = past

        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            if past is not None:
                pert_past, _, grad_norms, loss_this_iter = perturb_past(
                    past,
                    model,
                    last,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    classifier=classifier,
                    label_class=label_class,
                    one_hot_bows_vectors=one_hot_bows_vectors,
                    loss_type=loss_type,
                    num_iterations=num_iterations,
                    kl_scale=kl_scale,
                    window_length=window_length,
                    horizon_length=horizon_length,
                    decay=decay,
                    gamma=gamma,
                )
                loss_in_time.append(loss_this_iter)
            else:
                pert_past = past

        pert_logits, past, pert_all_hidden = model(last, past=pert_past)
        pert_logits = pert_logits[:, -1, :] / temperature
        pert_probs = F.softmax(pert_logits, dim=-1)

        # compute the discriminator loss using unperturbed hidden
        if classifier is not None:
            prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
            label = torch.tensor([label_class], device="cuda", dtype=torch.long)
            unpert_discrim_loss = torch.nn.CrossEntropyLoss()(prediction, label)
            print(
                "unperturbed discrim loss",
                unpert_discrim_loss.data.cpu().numpy()
            )
        else:
            unpert_discrim_loss = 0

        # Fuse the modified model and original model probabilities
        if perturb:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

            pert_probs = (pert_probs ** gm_scale) * (
                    unpert_probs ** (1 - gm_scale)
            )

            pert_probs = top_k_filter(pert_probs, k=top_k, probs=True)

            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

        else:
            pert_logits = top_k_filter(pert_logits, k=top_k)
            pert_probs = F.softmax(pert_logits, dim=-1)

        # sample or greedy
        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)

        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)

        # update context/output_so_far appending the new token
        output_so_far = (
            last if output_so_far is None
            else torch.cat((output_so_far, last), dim=1)
        )
        print(TOKENIZER.decode(output_so_far.tolist()[0]))

    return output_so_far, unpert_discrim_loss, loss_in_time


def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        "-M",
        type=str,
        default="gpt2-medium",
        help="pretrained model name or path to local checkpoint",
    )
    parser.add_argument(
        "--bag_of_words",
        "-B",
        type=str,
        default=None,
        help="Bags of words used for PPLM-BoW. Either a BOW id (see list in code) or a filepath. Multiple BoWs separated by ;",
    )
    parser.add_argument(
        "--discrim",
        "-D",
        type=str,
        default=None,
        choices=("clickbait", "sentiment", "toxicity"),
        help="Discriminator to use for loss-type 2",
    )
    parser.add_argument(
        "--label_class",
        type=int,
        default=-1,
        help="Class label used for the discriminator",
    )
    parser.add_argument("--stepsize", type=float, default=0.02)
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--gm_scale", type=float, default=0.9)
    parser.add_argument("--kl_scale", type=float, default=0.01)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument(
        "--uncond", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument(
        "--cond_text", type=str, default="The lake",
        help="Prefix texts to condition on"
    )
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=1,
        help="Length of future to optimize over",
    )
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; "
             "0 corresponds to infinite window length",
    )
    parser.add_argument("--decay", action="store_true",
                        help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.5)

    args = parser.parse_args()

    # set Random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # load pretrained model
    model = GPT2LMHeadModel.from_pretrained(
        args.model_path,
        output_hidden_states=True
    )
    model.to(device)
    model.eval()

    # freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # figure out conditioning text
    if args.uncond:
        tokenized_cond_text = TOKENIZER.encode(
            [TOKENIZER.bos_token]
        )
    else:
        raw_text = args.cond_text
        while not raw_text:
            print("Did you forget to add `--cond_text`? ")
            raw_text = input("Model prompt >>> ")
        tokenized_cond_text = TOKENIZER.encode(TOKENIZER.bos_token + raw_text)

    print("= Prefix of sentence =")
    print(TOKENIZER.decode(tokenized_cond_text))
    print()

    # generate unperturbed and perturbed texts

    # full_text_generation returns:
    # unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time
    unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
        model=model, context=tokenized_cond_text, device=device, **vars(args)
    )

    # untokenize unperturbed text
    unpert_gen_text = TOKENIZER.decode(unpert_gen_tok_text.tolist()[0])

    print("=" * 80)
    print("= Unperturbed generated text =")
    print(unpert_gen_text)
    print()

    generated_texts = []

    # iterate through the perturbed texts
    for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
        try:
            # untokenize unperturbed text
            unpert_gen_text = TOKENIZER.decode(pert_gen_tok_text.tolist()[0])

            print("= Perturbed generated text {} =".format(i + 1))
            print(unpert_gen_text)
            print()
        except:
            pass

        # keep the prefix, perturbed seq, original seq for each index
        generated_texts.append(
            (tokenized_cond_text, pert_gen_tok_text, unpert_gen_tok_text)
        )

    return generated_texts


if __name__ == "__main__":
    run_model()
