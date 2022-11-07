import dataclasses
import json
import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_processor import median_mapping, processors_mapping
from src.tokenizer import tokenize_multipart_input
from src.utils import count_special_tokens_in_template
from transformers.data import InputFeatures


logger = logging.getLogger(__name__)


class FewShotDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.task_name = args.task_name
        self.processor = processors_mapping[args.task_name]
        self.tokenizer = tokenizer
        self.mode = mode
        self.demo_mode = "get"  # save / get
        self.demo_num = args.demo_num
        self.demo_topk = args.demo_topk

        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        # only for prompt tuning
        self.total_label_tokens = 0

        self.len_special_tokens_in_template = 0
        if args.prompt:
            assert args.mapping is not None
            self.label_to_word = eval(args.mapping)
            self.label2id = {key: idx for idx, key in enumerate(self.label_to_word)}
            self.max_num_tokens_in_label = 1
            self.multi_token_in_label = True if isinstance(list(self.label_to_word.values())[0], List) else False

            for key in self.label_to_word:
                if self.multi_token_in_label:
                    label_to_word_ = []
                    for word in self.label_to_word[key]:
                        label_to_word_.append(tokenizer._convert_token_to_id(tokenizer.tokenize(" " + word)[0]))
                    self.label_to_word[key] = label_to_word_
                    self.total_label_tokens += len(label_to_word_)
                    if self.max_num_tokens_in_label < len(label_to_word_):
                        self.max_num_tokens_in_label = len(label_to_word_)
                    logger.info(
                        "Label {} to word {} ({})".format(
                            key,
                            [tokenizer._convert_id_to_token(token_id) for token_id in self.label_to_word[key]],
                            [self.label_to_word[key]],
                        )
                    )
                else:
                    if self.label_to_word[key][0] not in ["<", "[", ".", ","]:
                        assert len(tokenizer.tokenize(" " + self.label_to_word[key])) == 1
                        self.label_to_word[key] = tokenizer._convert_token_to_id(
                            tokenizer.tokenize(" " + self.label_to_word[key])[0]
                        )
                    else:
                        self.label_to_word[key] = tokenizer._convert_token_to_id(self.label_to_word[key])
                    logger.info(
                        "Label {} to word {} ({})".format(
                            key, tokenizer._convert_id_to_token(self.label_to_word[key]), self.label_to_word[key]
                        )
                    )

            if self.num_labels > 1:
                if self.multi_token_in_label:
                    self.label_word_list = []
                    for label in self.label_list:
                        self.label_word_list.append(
                            self.label_to_word[label]
                            + [tokenizer.pad_token_id]
                            * (self.max_num_tokens_in_label - len(self.label_to_word[label]))
                        )
                else:
                    self.label_word_list = [self.label_to_word[label] for label in self.label_list]
            else:
                self.label_word_list = [self.label_to_word[label] for label in ["0", "1"]]

            self.len_special_tokens_in_template = count_special_tokens_in_template(
                self.args.template, tokenizer=tokenizer, max_len_label_tokens=self.max_num_tokens_in_label,
            )
        else:
            self.label_to_word = None
            self.label_word_list = None

        logger.info(f"Length of special tokens in template: {self.len_special_tokens_in_template}")
        logger.info(f"Creating examples from dataset file at {args.data_dir}")
        self.support_features, self.support_labelids = None, None
        self.query_features, self.query_labelids = None, None
        if mode == "dev":
            self.query_examples = self.processor.get_dev_examples(args.data_dir)
        elif mode == "test":
            self.query_examples = self.processor.get_test_examples(args.data_dir)
        else:
            self.query_examples = self.processor.get_train_examples(args.data_dir)
        self.size = len(self.query_examples)

    def __len__(self):
        return self.size

    def get_labels(self):
        return self.label_list

    def get_demos(self):
        assert self.support_features and self.query_features, (self.support_features, self.query_features)
        self.examples = []
        np_query_features = self.query_features.reconstruct_n(0, self.query_features.ntotal)
        from tqdm import tqdm

        for i, query_feature in tqdm(enumerate(np_query_features), desc="Demo", total=len(np_query_features)):
            # sim: (1, n) instance: (1, n)
            similarity, instance = self.support_features.search(
                query_feature.reshape(1, -1), self.support_features.ntotal
            )
            label2features = {label: [] for label in range(len(self.label_list))}
            if self.mode == "train":  # train: leave-one-out
                similarity = similarity[0][1:].tolist()
                instance = instance[0][1:].tolist()
            else:
                similarity = similarity[0].tolist()
                instance = instance[0].tolist()

            label2sim = {label: [] for label in range(len(self.label_list))}
            for j, idx in enumerate(instance):  # topk
                label = self.support_labelids[idx].item()
                label2features[label].append(self.support_features.reconstruct(idx))
                label2sim[label].append(similarity[j])

            for label in label2features.keys():
                # topk
                label2features[label] = label2features[label][: self.demo_topk]
                label2sim[label] = label2sim[label][: self.demo_topk]
                # weighted sum
                label2features[label] = np.vstack(label2features[label])
                label_sim = torch.tensor(label2sim[label])
                label_sim = (label_sim - label_sim.mean()) / label_sim.std()  # norm
                alpha = torch.softmax(label_sim, dim=-1).view(1, -1).numpy()  # distance to weight
                label2features[label] = np.dot(alpha, label2features[label]).reshape(-1)  # weighted sum
            self.examples.append(list(label2features.values()))

    def __getitem__(self, index):
        # The input (query) example
        example = self.query_examples[index]
        template = self.args.template
        demo_mask_features = None
        if self.demo_mode == "get" and self.args.use_demo:
            assert len(self.examples) > 0
            demo_mask_features = self.examples[index]

        features = self.convert_fn(
            example=example,
            label_list=self.label_list,
            prompt=self.args.prompt,
            template=template,
            label_word_list=self.label_word_list,
            demo_mask_features=demo_mask_features,
            demo_num=self.demo_num,
            mode=self.mode,
            use_demo=self.args.use_demo,
        )
        return features

    def convert_fn(
        self,
        example,
        label_list=None,
        prompt=False,
        template=None,
        label_word_list=None,
        demo_mask_features=None,
        demo_num=None,
        mode=None,
        use_demo=False,
    ):
        """
        return a list of processed "InputFeatures".
        """
        max_length = self.args.max_seq_length

        label_map = {label: i for i, label in enumerate(label_list)}

        num_seq_per_example = 1
        if example.text_b is not None:
            num_seq_per_example = 2

        # convert label to integer
        if example.label is None:
            example_label = None
        elif len(label_list) == 1:
            example_label = float(example.label)
        else:
            example_label = label_map[example.label]

        inputs = tokenize_multipart_input(
            input_text_list=input_example_to_tuple(example),
            max_length=max_length,
            tokenizer=self.tokenizer,
            prompt=prompt,
            len_special_tokens_in_template=self.len_special_tokens_in_template,
            template=template,
            label_word_list=label_word_list,
            first_sent_limit=self.args.first_sent_limit,
            other_sent_limit=self.args.other_sent_limit,
            num_seq_per_example=num_seq_per_example,
            demo_num=demo_num,
            max_num_tokens_in_label=self.max_num_tokens_in_label,
            mode=mode,
            use_demo=use_demo,
        )
        features = OurInputFeatures(**inputs, label=example_label, demo_mask_features=demo_mask_features)
        return features


def input_example_to_tuple(example):
    if example.text_b is None:
        if pd.isna(example.text_a) or example.text_a is None:
            return [""]
            logger.warn("Empty input")
        else:
            return [example.text_a]
    else:
        return [example.text_a, example.text_b]


@dataclass(frozen=True)
class OurInputFeatures(InputFeatures):
    """
    Inherit from Transformers' InputFeatuers.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    mask_pos: Optional[List[int]] = None  # Position of the mask token
    label_word_list: Optional[List[int]] = None  # Label word mapping (dynamic)
    block_flag_for_demo: Optional[List[int]] = None
    demo_mask_features: Optional[Any] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"
