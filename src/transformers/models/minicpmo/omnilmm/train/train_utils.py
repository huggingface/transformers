import os
import gc
import copy
import time

import torch
import warnings
import transformers

import numpy as np

from typing import Dict, Optional, Sequence
from omnilmm import conversation as conversation_lib

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )



def omni_preprocess(sources,
                      tokenizer: transformers.PreTrainedTokenizer,
                      generation=False):
    system_content = 'You are an artificial intelligence assistant, which gives helpful, detailed, and polite answers to the human\'s questions.'
    ignore_index = -100

    response_template = '\n<|assistant|>\n'
    instruction_template = '\n<|user|>\n'
    response_token_ids = tokenizer.encode(
        response_template, add_special_tokens=False)
    instruction_token_ids = tokenizer.encode(
        instruction_template, add_special_tokens=False)

    batch_input_ids = []
    batch_labels = []
    for i in range(len(sources)):
        new_source = []
        prev_role = 'unexpect'
        for conv_turn in sources[i]:
            role = conv_turn['from'] if 'from' in conv_turn else conv_turn['role']
            content = conv_turn['value'] if 'value' in conv_turn else conv_turn['content']

            role = 'user' if role == 'human' else role
            role = 'assistant' if role == 'gpt' else role

            assert role in ['user', 'assistant']
            assert role != prev_role, f'role={role}, prev_role={prev_role}'
            prev_role = role

            new_turn = {
                'role': role,
                'content': content
            }
            new_source.append(new_turn)
        if new_source[0]['role'] != 'system':
            new_source.insert(0, {'role': 'system', 'content': system_content})

        # TODO: this automatically add '\n' to the end
        res_text = tokenizer.apply_chat_template(
            new_source, tokenize=False, add_generation_prompt=generation)
        if not generation:
            res_text = res_text.strip()

        conversations_tokenized = _tokenize_fn([res_text], tokenizer)
        res_input_ids = conversations_tokenized["input_ids"][0]

        # since labels and input_ids are reference towards the same object
        res_labels = copy.deepcopy(conversations_tokenized["labels"][0])

        response_token_ids_idxs = []
        human_token_ids_idxs = []

        for assistant_idx in np.where(res_labels == response_token_ids[0])[0]:
            # find the indexes of the start of a response.
            if (response_token_ids == res_labels[assistant_idx: assistant_idx + len(
                        response_token_ids)].tolist()
                    ):
                response_token_ids_idxs.append(
                    assistant_idx + len(response_token_ids))

        if len(response_token_ids_idxs) == 0:
            warnings.warn(
                f"Could not find response key `{response_template}` in the "
                f'following instance: @===>{tokenizer.decode(res_input_ids)}<===@ '
                f'Raw text is @===>{res_text}<===@'
                f'Raw source is @===>{new_source}<===@'
                f"This instance will be ignored in loss calculation. "
                f"Note, if this happens often, consider increasing the `max_seq_length`."
            )
            res_labels[:] = ignore_index

        human_token_ids = instruction_token_ids
        for human_idx in np.where(res_labels == human_token_ids[0])[0]:
            # find the indexes of the start of a human answer.
            if human_token_ids == res_labels[human_idx: human_idx + len(human_token_ids)].tolist():
                human_token_ids_idxs.append(human_idx)

        if len(human_token_ids_idxs) == 0:
            warnings.warn(
                f"Could not find instruction key `{instruction_template}` in the "
                f'following instance: @===>{tokenizer.decode(res_input_ids)}<===@ '
                f'Raw text is @===>{res_text}<===@'
                f'Raw source is @===>{new_source}<===@'
                f"This instance will be ignored in loss calculation. "
                f"Note, if this happens often, consider increasing the `max_seq_length`."
            )
            res_labels[:] = ignore_index

        for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
            # Make pytorch loss function ignore all non response tokens
            if idx != 0:
                res_labels[start:end] = ignore_index
            else:
                res_labels[:end] = ignore_index

        if len(response_token_ids_idxs) < len(human_token_ids_idxs):
            res_labels[human_token_ids_idxs[-1]:] = ignore_index

        batch_input_ids.append(res_input_ids)
        batch_labels.append(res_labels)

    return dict(input_ids=batch_input_ids, labels=batch_labels)


