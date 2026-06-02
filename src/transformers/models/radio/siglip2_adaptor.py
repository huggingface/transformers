# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import string
from argparse import Namespace

import torch.nn.functional as F

from .adaptor_generic import GenericAdaptor
from .adaptor_registry import adaptor_registry, dict_t, state_t
from .utils import rank_gate


_VERSION_MAP = {
    "siglip2-g-384": "google/siglip2-giant-opt-patch16-384",
    "siglip2-so400m": "google/siglip2-so400m-patch16-naflex",
}


class SigLIP2Adaptor(GenericAdaptor):
    def __init__(self, main_config: Namespace, adaptor_config: dict_t, state: state_t):
        super().__init__(main_config, adaptor_config, state)

        version = adaptor_config["model"]
        version = _VERSION_MAP[version]

        from transformers import AutoModel, AutoProcessor

        with rank_gate():
            model = AutoModel.from_pretrained(version, trust_remote_code=True)
            proc = AutoProcessor.from_pretrained(version, trust_remote_code=True)

        self.tokenizer = SigLIP2WrappedTokenizer(proc)
        self.text_model = model.text_model

        del model

    def encode_text(self, text, normalize: bool = False):
        output = self.text_model(**text, return_dict=True)
        token = output.pooler_output

        if normalize:
            token = F.normalize(token, dim=-1)

        return token


class SigLIP2WrappedTokenizer:
    def __init__(self, proc):
        self._proc = proc

    def __call__(self, text: list[str]):
        text = [canonicalize_text(t) for t in text]
        ret = self._proc(text=text, return_tensors="pt", max_length=64, padding="max_length", truncation=True)
        return ret


def canonicalize_text(
    text: str,
    *,
    keep_punctuation_exact_string=None,
    trans_punctuation: dict = str.maketrans("", "", string.punctuation),
):
    """Returns canonicalized `text` (lowercase and punctuation removed).

    From: https://github.com/google-research/big_vision/blob/53f18caf27a9419231bbf08d3388b07671616d3d/big_vision/evaluators/proj/image_text/prompt_engineering.py#L94

    Args:
      text: string to be canonicalized.
      keep_punctuation_exact_string: If provided, then this exact string kept.
        For example providing '{}' will keep any occurrences of '{}' (but will
        still remove '{' and '}' that appear separately).
    """
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(trans_punctuation) for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(trans_punctuation)
    text = text.lower()
    text = " ".join(text.split())
    return text.strip()


@adaptor_registry.register_adaptor("siglip2")
def create_siglip2_adaptor(main_config: Namespace, adaptor_config: dict_t, state: state_t):
    return SigLIP2Adaptor(main_config, adaptor_config, state)
