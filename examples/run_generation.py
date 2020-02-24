#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""


import argparse
import logging

import numpy as np
import torch
import typer

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(seed: int, n_gpu: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(_, tokenizer, prompt_text, **kwargs):
    temperature = kwargs.get("temperature")
    if temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(model, tokenizer, prompt_text, **kwargs):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    xlm_language = kwargs.get("xlm_language")
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if xlm_language in available_languages:
            language = xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(_, tokenizer, prompt_text, **kwargs):
    padding_text = kwargs.get("padding_text")
    prompt_text = (padding_text if padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


def prepare_transfoxl_input(_, tokenizer, prompt_text, **kwargs):
    padding_text = kwargs.get("padding_text")
    prompt_text = (padding_text if padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main(model_type: str,
         model_name_or_path: str,
         prompt: str = "",
         length: int = 20,
         stop_token: str = typer.Option(None, help="Token at which text generation is stopped"),
         temperature: float = typer.Option(1.0, help="temperature of 1.0 has no effect, lower tend toward greedy sampling"),
         repetition_penalty: float = typer.Option(1.0, help="primarily useful for CTRL model; in that case, use 1.2"),
         k: int = 0,
         p: float = 0.9,
         padding_text: str = typer.Option("", help="Padding text for Transfo-XL and XLNet."),
         xlm_language: str = typer.Option("", help="Optional language when used with the XLM model"),
         seed: int = typer.Option(42, help="random seed for initialization"),
         no_cuda: bool = typer.Option(False, '--no-cuda', help="Don't use CUDA and run on CPU."),
         num_return_sequences: int = typer.Option(1, help="The number of samples to generate.")):
    """Generate text based on a prompt using one of
    [gpt2, ctrl, openai-gpt, xlnet, transfo-xl, xlm] as the model_type
    and a a supported model name or path for that model_type

    e.g.

        $ python examples/run_generation.py gpt2 distilgpt2
    """
     
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    set_seed(seed, n_gpu)

    # Initialize the model and tokenizer
    try:
        model_class, tokenizer_class = MODEL_CLASSES[model_type.lower()]
    except KeyError:
        raise KeyError(
            "the model {} you specified is not supported. You are welcome to add it and open a PR :)"
        )

    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    model = model_class.from_pretrained(model_name_or_path)
    model.to(device)

    length = adjust_length_to_model(length, max_sequence_length=model.config.max_position_embeddings)

    prompt_text = prompt if prompt else typer.prompt("Model prompt >>> ")

    # Different models need different input formatting and/or extra arguments
    requires_preprocessing = model_type in PREPROCESSING_FUNCTIONS.keys()
    if requires_preprocessing:
        prepare_input = PREPROCESSING_FUNCTIONS.get(model_type)
        preprocessed_prompt_text = prepare_input(model, tokenizer, prompt_text, xlm_language=xlm_language, padding_text=padding_text)
        encoded_prompt = tokenizer.encode(preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt")
    else:
        encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(device)

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=length + len(encoded_prompt[0]),
        temperature=temperature,
        top_k=k,
        top_p=p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        num_return_sequences=num_return_sequences,
    )

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(stop_token) if stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )

        generated_sequences.append(total_sequence)
        print(total_sequence)


if __name__ == "__main__":
    typer.run(main)
