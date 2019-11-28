#! /usr/bin/env python3
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

from examples.run_pplm_discrim_train import ClassificationHead
from transformers import GPT2Tokenizer
from transformers.file_utils import cached_path
from transformers.modeling_gpt2 import GPT2LMHeadModel

PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
SmallConst = 1e-15
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
        "url": "http://s.yosinski.com/SST_classifier_head.pt",
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
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10,
                           logits)


def perturb_past(
        past,
        model,
        prev,
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
    # def perturb_past(past, model, prev, classifier, good_index=None,
    #             stepsize=0.01, vocab_size=50257,
    #             original_probs=None, accumulated_hidden=None, true_past=None,
    #             grad_norms=None):

    # one_hot_bows_vectors = []
    # for good_list in good_index:
    #     good_list = list(filter(lambda x: len(x) <= 1, good_list))
    #     good_list = torch.tensor(good_list).cuda()
    #     num_good = good_list.shape[0]
    #     one_hot_good = torch.zeros(num_good, vocab_size).cuda()
    #     one_hot_good.scatter_(1, good_list, 1)
    #     one_hot_bows_vectors.append(one_hot_good)

    # Generate inital perturbed past
    past_perturb_orig = [
        (np.random.uniform(0.0, 0.0, p.shape).astype('float32'))
        for p in past]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(0., 1.0 + SmallConst, 1.0 / (window_length))[
                     1:]
    else:
        decay_mask = 1.0

    # Generate a mask is gradient perturbated is based on a past window
    _, _, _, current_length, _ = past[0].shape

    if current_length > window_length and window_length > 0:
        ones_key_val_shape = tuple(past[0].shape[:-2]) + tuple(
            [window_length]) + tuple(
            past[0].shape[-1:])

        zeros_key_val_shape = tuple(past[0].shape[:-2]) + tuple(
            [current_length - window_length]) + tuple(
            past[0].shape[-1:])

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat((ones_mask, torch.zeros(zeros_key_val_shape)),
                                dim=-2).cuda()
    else:
        window_mask = torch.ones_like(past[0]).cuda()

    loss_per_iter = []
    for i in range(num_iterations):
        print("Iteration ", i + 1)
        past_perturb = [torch.from_numpy(p_) for p_ in past_perturb_orig]
        past_perturb = [to_var(p_, requires_grad=True) for p_ in past_perturb]

        perturbed_past = list(map(add, past, past_perturb))

        _, _, _, current_length, _ = past_perturb[0].shape

        # _, future_past = model(prev, past=perturbed_past)
        # hidden = model.hidden_states

        # Piero modified model call
        logits, _, all_hidden = model(prev, past=perturbed_past)
        hidden = all_hidden[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(hidden,
                                                                dim=1).detach()

        # TODO: Check the layer-norm consistency of this with trained discriminator
        logits = logits[:, -1, :]
        probabs = F.softmax(logits, dim=-1)
        loss = 0.0
        loss_list = []
        if loss_type == 1 or loss_type == 3:
            for one_hot_good in one_hot_bows_vectors:
                good_logits = torch.mm(probabs, torch.t(one_hot_good))
                loss_word = good_logits
                loss_word = torch.sum(loss_word)
                loss_word = -torch.log(loss_word)
                # loss_word = torch.sum(loss_word) /torch.sum(one_hot_good)
                loss += loss_word
                loss_list.append(loss_word)
            print(" pplm_bow_loss:", loss.data.cpu().numpy())

        if loss_type == 2 or loss_type == 3:
            ce_loss = torch.nn.CrossEntropyLoss()
            new_true_past = unpert_past
            for i in range(horizon_length):
                future_probabs = F.softmax(logits, dim=-1)  # Get softmax
                future_probabs = torch.unsqueeze(future_probabs, dim=1)

                # _, new_true_past = model(future_probabs, past=new_true_past)
                # future_hidden = model.hidden_states  # Get expected hidden states

                # Piero modified model call
                wte = model.resize_token_embeddings()
                inputs_embeds = torch.matmul(future_probabs, wte.weight.data)
                _, new_true_past, future_hidden = model(
                    past=new_true_past,
                    inputs_embeds=inputs_embeds
                )
                future_hidden = future_hidden[-1]

                new_accumulated_hidden = new_accumulated_hidden + torch.sum(
                    future_hidden, dim=1)

            predicted_sentiment = classifier(new_accumulated_hidden / (
                    current_length + 1 + horizon_length))

            label = torch.tensor([label_class], device='cuda',
                                 dtype=torch.long)
            discrim_loss = ce_loss(predicted_sentiment, label)
            print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())
            loss += discrim_loss
            loss_list.append(discrim_loss)

        kl_loss = 0.0
        if kl_scale > 0.0:
            p = (F.softmax(unpert_logits[:, -1, :], dim=-1))
            p = p + SmallConst * (p <= SmallConst).type(
                torch.FloatTensor).cuda().detach()
            correction = SmallConst * (probabs <= SmallConst).type(
                torch.FloatTensor).cuda().detach()
            corrected_probabs = probabs + correction.detach()
            kl_loss = kl_scale * (
                (corrected_probabs * (corrected_probabs / p).log()).sum())
            print(' kl_loss', (kl_loss).data.cpu().numpy())
            loss += kl_loss  # + discrim_loss

        loss_per_iter.append(loss.data.cpu().numpy())

        print(' pplm_loss', (loss - kl_loss).data.cpu().numpy())

        loss.backward()
        if grad_norms is not None and loss_type == 1:
            grad_norms = [
                torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
                for index, p_ in
                enumerate(past_perturb)]
        else:
            grad_norms = [(torch.norm(p_.grad * window_mask) + SmallConst) for
                          index, p_ in enumerate(past_perturb)]

        grad = [
            -stepsize * (p_.grad * window_mask / grad_norms[
                index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(past_perturb)]
        past_perturb_orig = list(map(add, grad, past_perturb_orig))

        for p_ in past_perturb:
            p_.grad.data.zero_()

        new_past = []
        for p in past:
            new_past.append(p.detach())

        past = new_past

    past_perturb = [torch.from_numpy(p_) for p_ in past_perturb_orig]
    past_perturb = [to_var(p_, requires_grad=True) for p_ in past_perturb]
    perturbed_past = list(map(add, past, past_perturb))

    return perturbed_past, new_accumulated_hidden, grad_norms, loss_per_iter


def get_classifier(
        name: Optional[str], label_class: Union[str, int],
        device: Union[str, torch.device]
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(
        class_size=params['class_size'],
        embed_size=params['embed_size']
    ).to(device)
    resolved_archive_file = cached_path(params["url"])
    classifier.load_state_dict(
        torch.load(resolved_archive_file, map_location=device))
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


def get_bag_of_words_indices(bag_of_words_ids_or_paths: List[str]) -> List[
    List[List[int]]]:
    bow_indices = []
    for id_or_path in bag_of_words_ids_or_paths:
        if id_or_path in BAG_OF_WORDS_ARCHIVE_MAP:
            filepath = cached_path(BAG_OF_WORDS_ARCHIVE_MAP[id_or_path])
        else:
            filepath = id_or_path
        with open(filepath, "r") as f:
            words = f.read().strip().split("\n")
        bow_indices.append(
            [TOKENIZER.encode(word.strip(), add_prefix_space=True) for word in
             words])
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
        raise Exception("Specify either a bag of words or a discriminator")

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
    output = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(
        0) if context else None

    # collect one hot vectors for bags of words
    one_hot_bows_vectors = build_bows_one_hot_vectors(bow_indices)

    grad_norms = None
    loss_in_time = []
    for i in trange(length, ascii=True):

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current-token
        # Therefore, use everything from before current i/p token to generate relevant past

        if past is None and output is not None:
            prev = output[:, -1:]
            # _, past = model(output[:, :-1])
            # original_probs, true_past = model(output)
            # true_hidden = model.hidden_states

            # Piero modified model call
            _, past, _ = model(output[:, :-1])
            unpert_logits, unpert_past, unpert_all_hidden = model(output)
            true_hidden = unpert_all_hidden[-1]

        else:
            # original_probs, true_past = model(output)
            # true_hidden = model.hidden_states

            # Piero modified model call
            unpert_logits, unpert_past, unpert_all_hidden = model(output)
            true_hidden = unpert_all_hidden[-1]

        # Modify the past if necessary

        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        if not perturb or num_iterations == 0:
            perturbed_past = past

        else:
            # Piero modified model call
            # accumulated_hidden = model.hidden_states[:, :-1, :]
            accumulated_hidden = true_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            perturbed_past, _, grad_norms, loss_per_iter = perturb_past(
                past,
                model,
                prev,
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
            loss_in_time.append(loss_per_iter)

        # Piero modified model call
        logits, past, pert_all_hidden = model(prev, past=perturbed_past)
        # test_logits = F.softmax(test_logits[:, -1, :], dim=-1)
        # likelywords = torch.topk(test_logits, k=10, dim=-1)
        # print(TOKENIZER.decode(likelywords[1].tolist()[0]))

        if classifier is not None:
            ce_loss = torch.nn.CrossEntropyLoss()
            predicted_sentiment = classifier(torch.mean(true_hidden, dim=1))
            label = torch.tensor([label_class], device='cuda',
                                 dtype=torch.long)
            true_discrim_loss = ce_loss(predicted_sentiment, label)
            print("true discrim loss", true_discrim_loss.data.cpu().numpy())
        else:
            true_discrim_loss = 0

        # Piero modified model call
        # hidden = model.hidden_states  # update hidden
        # logits = model.forward_hidden(hidden)
        logits = logits[:, -1, :] / temperature  # + SmallConst

        # logits = top_k_filter(logits, k=args.top_k)  # + SmallConst

        log_probs = F.softmax(logits, dim=-1)

        # Fuse the modified model and original model
        if perturb:

            # original_probs = top_k_filter(original_probs[:, -1, :]) #+ SmallConst
            unpert_logits = F.softmax(unpert_logits[:, -1, :], dim=-1)
            # likelywords = torch.topk(original_probs, k=10, dim=-1)
            # print(TOKENIZER.decode(likelywords[1].tolist()[0]))

            log_probs = ((log_probs ** gm_scale) * (
                    unpert_logits ** (1 - gm_scale)))  # + SmallConst

            log_probs = top_k_filter(log_probs, k=top_k,
                                     probs=True)  # + SmallConst

            if torch.sum(log_probs) <= 1:
                log_probs = log_probs / torch.sum(log_probs)

        else:
            logits = top_k_filter(logits, k=top_k)  # + SmallConst
            log_probs = F.softmax(logits, dim=-1)

        if sample:
            # likelywords = torch.topk(log_probs, k=args.top_k, dim=-1)
            # print(TOKENIZER.decode(likelywords[1].tolist()[0]))
            # print(likelywords[0].tolist())
            prev = torch.multinomial(log_probs, num_samples=1)
        else:
            _, prev = torch.topk(log_probs, k=1, dim=-1)
        # if perturb:
        #     prev = future
        output = prev if output is None else torch.cat((output, prev),
                                                       dim=1)  # update output
        print(TOKENIZER.decode(output.tolist()[0]))

    return output, true_discrim_loss, loss_in_time


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
        help="Bags of words used for PPLM-BoW. "
             "Either a BOW id (see list in code) or a filepath. "
             "Multiple BoWs separated by ;",
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
    parser.add_argument("--colorama", action="store_true", help="colors keywords")

    args = parser.parse_args()

    # set Random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # set the device
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    # load pretrained model
    model = GPT2LMHeadModel.from_pretrained(
        args.model_path,
        output_hidden_states=True
    )
    model.to(device)
    model.eval()

    # Freeze GPT-2 weights
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

    bow_word_ids = set()
    if args.bag_of_words and args.colorama:
        bow_indices = get_bag_of_words_indices(args.bag_of_words.split(";"))
        for single_bow_list in bow_indices:
            # filtering all words in the list composed of more than 1 token
            filtered = list(filter(lambda x: len(x) <= 1, single_bow_list))
            # w[0] because we are sure w has only 1 item because previous fitler
            bow_word_ids.update(w[0] for w in filtered)

    # iterate through the perturbed texts
    for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
        try:
            # untokenize unperturbed text
            if args.colorama:
                import colorama

                pert_gen_text = ''
                for word_id in pert_gen_tok_text.tolist()[0]:
                    if word_id in bow_word_ids:
                        pert_gen_text += '{}{}{}'.format(
                            colorama.Fore.RED,
                            TOKENIZER.decode([word_id]),
                            colorama.Style.RESET_ALL
                        )
                    else:
                        pert_gen_text += TOKENIZER.decode([word_id])
            else:
                pert_gen_text = TOKENIZER.decode(pert_gen_tok_text.tolist()[0])

            print("= Perturbed generated text {} =".format(i + 1))
            print(pert_gen_text)
            print()
        except:
            pass

        # keep the prefix, perturbed seq, original seq for each index
        generated_texts.append(
            (tokenized_cond_text, pert_gen_tok_text, unpert_gen_tok_text)
        )

    return


if __name__ == '__main__':
    run_model()
