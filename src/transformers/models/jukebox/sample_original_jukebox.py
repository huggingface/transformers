# coding=utf-8
# Copyright 2022 The OpenAI Team Authors the HuggingFace Inc. team.
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

# in order to be used, the following git repo has to be used :
# git clone --branch adaptive_device https://github.com/ArthurZucker/jukebox.git

import os
import random

import numpy as np
import torch
import torch as t

from jukebox.hparams import HPARAMS_REGISTRY, Hyperparams, setup_hparams
from jukebox.make_models import MODELS, make_prior, make_vqvae
from jukebox.sample import _sample
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.torch_utils import empty_cache


rank, local_rank, device = setup_dist_from_mpi()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.enabled = False


def log_zs(zs, level, model, save_dir="logits"):
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/{model}_{level}.txt", "w") as file:
        file.write(str(zs[level][0].cpu()))


def get_args(model):
    sampling_temperature = 0.98
    lower_batch_size = 16
    max_batch_size = 1 if model == "5b_lyrics" else 16
    lower_level_chunk_size = 32
    chunk_size = 16 if model == "5b_lyrics" else 32
    sampling_kwargs = [
        dict(
            temp=0.99,
            fp16=False,
            max_batch_size=lower_batch_size,
            chunk_size=lower_level_chunk_size,
            sample_tokens=10,
        ),
        dict(
            temp=0.99,
            fp16=False,
            max_batch_size=lower_batch_size,
            chunk_size=lower_level_chunk_size,
            sample_tokens=10,
        ),
        dict(
            temp=sampling_temperature,
            fp16=False,
            max_batch_size=max_batch_size,
            chunk_size=chunk_size,
            sample_tokens=10,
        ),
    ]
    return sampling_kwargs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test_sampling(model, device, tokens=40):
    hps = Hyperparams()
    hps.device = device
    hps.sr = 44100
    hps.n_samples = 1
    hps.name = "samples"
    hps.levels = 3
    hps.hop_fraction = [0.5, 0.5, 0.125]
    HPARAMS_REGISTRY[f"prior_{model}"][
        "min_duration"
    ] = 0  # set the minium duration of the model to 0 to generate only 40 tokens
    vqvae, *priors = MODELS[model]
    vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length=264576)), device)  # before : 1048576, 2645888
    top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)
    hps.sample_length = tokens * top_prior.raw_to_tokens
    metas = (
        [
            dict(
                artist="Zac Brown Band",
                genre="Country",
                total_length=hps.sample_length,
                offset=0,
                lyrics="""I met a traveller from an antique land,
    Who said "Two vast and trunkless legs of stone Stand in the desert. . . . Near them, on the sand, Half sunk a
    shattered visage lies, whose frown, And wrinkled lip, and sneer of cold command, Tell that its sculptor well those
    passions read Which yet survive, stamped on these lifeless things, The hand that mocked them, and the heart that
    fed; And on the pedestal, these words appear: My name is Ozymandias, King of Kings; Look on my Works, ye Mighty,
    and despair! Nothing beside remains. Round the decay Of that colossal Wreck, boundless and bare The lone and level
    sands stretch far away
    """,
            ),
        ]
        * hps.n_samples
    )

    labels = [None, None, top_prior.labeller.get_batch_labels(metas, device)]
    sampling_kwargs = get_args(model)
    hps.sample_length = tokens * top_prior.raw_to_tokens

    set_seed(0)
    zs = [t.zeros(hps.n_samples, 0, dtype=t.long, device=device) for _ in range(len(priors))]
    zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)
    log_zs(zs, 2, f"{model}-{device}")

    del top_prior
    empty_cache()
    upsamplers = [make_prior(setup_hparams(prior, dict()), vqvae, device) for prior in priors[:-1]]
    labels[:2] = [prior.labeller.get_batch_labels(metas, device) for prior in upsamplers]

    set_seed(0)
    zs[-1] = torch.cat((zs[-1], torch.zeros(1, 1000000 - zs[-1].shape[-1]).to(device)), dim=-1).long()
    hps.sample_length = tokens * upsamplers[1].raw_to_tokens
    zs = _sample(zs, labels, sampling_kwargs, [None, upsamplers[1], None], [1], hps)
    log_zs(zs, 1, f"{model}-{device}")

    set_seed(0)
    hps.sample_length = tokens * upsamplers[0].raw_to_tokens
    zs[-2] = torch.cat((zs[-2], torch.zeros(1, 1000000 - zs[-2].shape[-1]).to(device)), dim=-1).long()
    zs = _sample(zs, labels, sampling_kwargs, [upsamplers[0], None, None], [0], hps)
    log_zs(zs, 0, f"{model}-{device}")

    empty_cache()
    del upsamplers


def test_prime_samling(model, device, tokens=40):
    hps = Hyperparams()
    hps.device = device
    hps.sr = 44100
    hps.n_samples = 1
    hps.name = "samples"
    hps.levels = 3
    hps.hop_fraction = [0.5, 0.5, 0.125]
    HPARAMS_REGISTRY[f"prior_{model}"]["min_duration"] = 0
    vqvae, *priors = MODELS[model]
    vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length=264576)), device)  # before : 1048576, 2645888
    top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)
    hps.sample_length = tokens * top_prior.raw_to_tokens
    metas = (
        [
            dict(
                artist="Zac Brown Band",
                genre="Country",
                total_length=hps.sample_length,
                offset=0,
                lyrics="""I met a traveller from an antique land,
    Who said "Two vast and trunkless legs of stone Stand in the desert. . . . Near them, on the sand, Half sunk a
    shattered visage lies, whose frown, And wrinkled lip, and sneer of cold command, Tell that its sculptor well those
    passions read Which yet survive, stamped on these lifeless things, The hand that mocked them, and the heart that
    fed; And on the pedestal, these words appear: My name is Ozymandias, King of Kings; Look on my Works, ye Mighty,
    and despair! Nothing beside remains. Round the decay Of that colossal Wreck, boundless and bare The lone and level
    sands stretch far away
    """,
            ),
        ]
        * hps.n_samples
    )
    labels = [None, None, top_prior.labeller.get_batch_labels(metas, device)]
    sampling_kwargs = get_args(model)

    set_seed(0)
    x = torch.rand((1, 5120, 1)).to(device)
    vqvae.to(device)
    zs = [None, None, top_prior.encode(x, start_level=2, bs_chunks=x.shape[0])[0].to(device)]
    zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)
    log_zs(zs, 2, f"primed-{model}-{device}")

    del top_prior
    empty_cache()

    upsamplers = [make_prior(setup_hparams(prior, dict()), vqvae, device) for prior in priors[:-1]]
    labels = [
        upsamplers[0].labeller.get_batch_labels(metas, device),
        upsamplers[0].labeller.get_batch_labels(metas, device),
        None,
    ]

    set_seed(0)
    hps.sample_length = tokens * upsamplers[1].raw_to_tokens
    zs = [
        None,
        upsamplers[-1].encode(x, start_level=1, bs_chunks=x.shape[0])[0].to(device),
        torch.cat((zs[-1], torch.zeros(1, 1000000 - zs[-1].shape[-1]).to(device)), dim=-1).long(),
    ]
    zs = _sample(zs, labels, sampling_kwargs, [None, upsamplers[1], None], [1], hps)
    log_zs(zs, 1, f"primed-{model}-{device}")

    set_seed(0)
    hps.sample_length = tokens * upsamplers[0].raw_to_tokens
    zs = [
        upsamplers[-1].encode(x, start_level=0, bs_chunks=x.shape[0])[0].to(device),
        torch.cat((zs[1], torch.zeros(1, 1000000 - zs[1].shape[1]).to(device)), dim=-1).long(),
        torch.zeros(1, 1000000).to(device).long(),
    ]
    zs = _sample(zs, labels, sampling_kwargs, [upsamplers[0], None, None], [0], hps)
    log_zs(zs, 0, f"primed-{model}-{device}")


test_sampling("1b_lyrics", "cpu")
test_sampling("1b_lyrics", "cuda")
test_sampling("5b_lyrics", "cpu", tokens=60)
test_sampling("5b_lyrics", "cuda", tokens=60)

test_prime_samling("1b_lyrics", "cpu")
test_prime_samling("1b_lyrics", "cuda")
test_prime_samling("5b_lyrics", "cpu", tokens=60)
test_prime_samling("5b_lyrics", "cuda", tokens=60)
