from typing import List, Optional, Union

import numpy as np
import pandas as pd


def tokenize_sent(input_text_list, tokenizer):
    sent1_token_ids = None
    sent2_token_ids = None

    text1 = input_text_list[0]
    sent1_token_ids = tokenizer.encode(text1, add_special_tokens=False)
    if len(input_text_list) == 2:
        text2 = input_text_list[1]
        text2 = text2[:1].lower() + text2[1:]
        sent2_token_ids = tokenizer.encode(" " + text2, add_special_tokens=False)

    return sent1_token_ids, sent2_token_ids


def _tokenize_multipart_input(
    input_text_list,
    template_list,
    special_token_mapping,
    label_word_list,
    tokenizer,
    max_length,
    first_sent_limit,
    other_sent_limit,
):
    input_ids = []
    attention_mask = []
    token_type_ids = []

    segment_id = 0
    # pre-process sentence
    sent1_token_ids, sent2_token_ids = tokenize_sent(input_text_list, tokenizer=tokenizer)

    # truncate
    if first_sent_limit is not None:
        sent1_token_ids = sent1_token_ids[:first_sent_limit]
    if other_sent_limit is not None and sent2_token_ids is not None:
        sent2_token_ids = sent2_token_ids[:other_sent_limit]
    if max_length is not None:
        # default to truncate first sentence if have two sentence in a instance.
        # for avoiding truncate special tokens, only truncate real text.
        len_sent1 = len(sent1_token_ids)
        len_sent2 = len(sent2_token_ids) if sent2_token_ids is not None else 0
        if len_sent1 + len_sent2 > max_length:
            if len_sent2 > 0:
                sent2_token_ids = sent2_token_ids[: max_length - len_sent1]
            else:
                sent1_token_ids = sent1_token_ids[:max_length]

    for _, part in enumerate(template_list):
        new_tokens = []
        segment_plus_1_flag = False
        if part in special_token_mapping:
            new_tokens_ = []
            new_tokens_.append(special_token_mapping[part])
            new_tokens.extend(new_tokens_)
            if part == "sep+":
                segment_plus_1_flag = True
        elif part[:6] == "label_":
            label_id = int(part.split("_")[1])
            label_word = label_word_list[label_id]
            new_tokens.append(label_word)
        elif part[:5] == "sent_":
            # Lower case the first token and discard the last token
            sent_id = int(part.split("_")[1])
            if sent_id == 0:
                new_tokens += sent1_token_ids
            else:
                new_tokens += sent2_token_ids
        else:
            # Just natural language prompt
            part = part.replace("_", " ")
            # handle special case when T5 tokenizer might add an extra space
            if len(part) == 1:
                new_tokens.append(tokenizer._convert_token_to_id(part))
            else:
                new_tokens += tokenizer.encode(part, add_special_tokens=False)

        input_ids += new_tokens
        attention_mask += [1 for i in range(len(new_tokens))]
        token_type_ids += [segment_id for i in range(len(new_tokens))]

        if segment_plus_1_flag:
            segment_id += 1

    return input_ids, attention_mask, token_type_ids


def tokenize_multipart_input(
    input_text_list,
    max_length,
    tokenizer,
    prompt=None,
    len_special_tokens_in_template=0,
    template=None,
    label_word_list=None,
    first_sent_limit=None,
    other_sent_limit=None,
    num_seq_per_example=None,
    demo_num=0,
    max_num_tokens_in_label=1,
    mode=None,
    use_demo=False,
):
    # demo_num = demo_num if mode == 'train' else 1
    input_ids = []
    attention_mask = []
    token_type_ids = []
    mask_pos = None

    # reset `demo_max_length` and `max_length` according to `len_special_tokens_in_template`
    total_demo_length = len(label_word_list) * (demo_num + max_num_tokens_in_label)
    if use_demo:
        real_max_length = max_length + len_special_tokens_in_template + total_demo_length
    else:
        real_max_length = max_length

    if prompt:
        assert template is not None

        special_token_mapping = {
            "cls": tokenizer.cls_token_id,
            "mask": tokenizer.mask_token_id,
            "sep": tokenizer.sep_token_id,
            "sep+": tokenizer.sep_token_id,
            "prompt": tokenizer.pad_token_id,
        }

        template_list = template.split("*")  # Get variable list in the template

        input_ids, attention_mask, token_type_ids = _tokenize_multipart_input(
            input_text_list[:num_seq_per_example],
            max_length=max_length,
            template_list=template_list,
            special_token_mapping=special_token_mapping,
            label_word_list=label_word_list,
            tokenizer=tokenizer,
            first_sent_limit=first_sent_limit,
            other_sent_limit=other_sent_limit,
        )
        input_ids = input_ids[:real_max_length]
        attention_mask = attention_mask[:real_max_length]
        token_type_ids = token_type_ids[:real_max_length]
        if not use_demo:
            block_flag_for_demo = None
        else:
            block_flag_for_demo = [0] * len(input_ids)
            # add token
            for label in label_word_list:  # add label
                if isinstance(label, List):
                    # multi-token label
                    input_ids += [999] * demo_num + label
                    block_flag_for_demo += [1] * demo_num + [0] * len(label)
                else:
                    # single-token label
                    input_ids += [999] * demo_num + [label]
                    block_flag_for_demo += [1] * demo_num + [0]
            attention_mask += [1] * total_demo_length
            token_type_ids += [1] * total_demo_length
            block_flag_for_demo += [0] * (real_max_length - len(block_flag_for_demo))

        ## pad
        input_ids += [tokenizer.pad_token_id] * (real_max_length - len(input_ids))
        attention_mask += [0] * (real_max_length - len(attention_mask))
        token_type_ids += [0] * (real_max_length - len(token_type_ids))
        # end
        # Find mask token
        mask_pos = [input_ids.index(tokenizer.mask_token_id)]
        # Make sure that the masked position is inside the max_length
        assert mask_pos[0] < real_max_length
    result = {"input_ids": input_ids, "attention_mask": attention_mask, "block_flag_for_demo": block_flag_for_demo}
    if "BERT" in type(tokenizer).__name__:
        # Only provide token type ids for BERT
        result["token_type_ids"] = token_type_ids

    if prompt:
        result["mask_pos"] = mask_pos

    return result
