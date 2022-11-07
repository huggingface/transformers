import csv
import json
import logging
import os
import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


keyword_files = ["keyword_train.txt", "keyword_dev.txt", "keyword_test.txt"]


def tokenize(text, tokenizer):
    # berts tokenize ways
    # tokenize the [unused12345678910]
    D = [f"[unused{i}]" for i in range(10)]
    textraw = [text]
    for delimiter in D:
        ntextraw = []
        for i in range(len(textraw)):
            t = textraw[i].split(delimiter)
            for j in range(len(t)):
                ntextraw += [t[j]]
                if j != len(t) - 1:
                    ntextraw += [delimiter]
        textraw = ntextraw
    text = []
    for t in textraw:
        if t in D:
            text += [t]
        else:
            tokens = tokenizer.tokenize(t, add_special_tokens=False)
            for tok in tokens:
                text += [tok]

    for idx, t in enumerate(text):
        if idx + 3 < len(text) and t == "[" and text[idx + 1] == "[UNK]" and text[idx + 2] == "]":
            text = text[:idx] + ["[MASK]"] + text[idx + 3 :]

    return text


n_class = 1


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None, entity=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label
        self.entity = entity


class InputExampleWiki80(object):
    """A single training/test example for span pair classification."""

    def __init__(self, guid, sentence, span1, span2, ner1, ner2, label):
        self.guid = guid
        self.sentence = sentence
        self.span1 = span1
        self.span2 = span2
        self.ner1 = ner1
        self.ner2 = ner2
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, entity=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.entity = entity


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class wiki80Processor(DataProcessor):
    """Processor for the TACRED data set."""

    def __init__(self, data_path, use_prompt):
        super().__init__()
        self.data_dir = data_path

    @classmethod
    def _read_json(cls, input_file):
        data = []
        with open(input_file, "r", encoding="utf-8") as reader:
            all_lines = reader.readlines()
            for line in all_lines:
                ins = eval(line)
                data.append(ins)
        return data

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "val.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self, negative_label="no_relation"):
        data_dir = self.data_dir
        """See base class."""
        with open(os.path.join(data_dir, "rel2id.json"), "r", encoding="utf-8") as reader:
            re2id = json.load(reader)
        return re2id

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for example in dataset:
            sentence = example["token"]
            examples.append(
                InputExampleWiki80(
                    guid=None,
                    sentence=sentence,
                    # maybe some bugs here, I don't -1
                    span1=(example["h"]["pos"][0], example["h"]["pos"][1]),
                    span2=(example["t"]["pos"][0], example["t"]["pos"][1]),
                    ner1=None,
                    ner2=None,
                    label=example["relation"],
                )
            )
        return examples


class wiki80DemoDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels, so, rel2id, tokenizer, mode, args) -> None:
        super().__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.so = so

        self.support_features, self.support_labelids = None, None
        self.query_features, self.query_labelids = None, None
        self.demo_mode = "get"  # save / get
        self.demo_topk = args.demo_topk
        self.temp = args.temp
        self.mode = mode
        self.args = args
        self.labelid_list = [
            a[0]
            for a in tokenizer([f"[class{i}]" for i in range(1, len(rel2id) + 1)], add_special_tokens=False)[
                "input_ids"
            ]
        ]

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
            label2features = {label: [] for label in range(len(self.labelid_list))}
            if self.mode == "train":  # train: leave-one-out
                similarity = similarity[0][1:].tolist()
                instance = instance[0][1:].tolist()
            else:
                similarity = similarity[0].tolist()
                instance = instance[0].tolist()
            label2sim = {label: [] for label in range(len(self.labelid_list))}
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
                if "semeval" in self.args.data_dir:
                    label_sim = torch.tensor(label2sim[label])
                    label_sim = (label_sim - label_sim.mean()) / label_sim.std()  # norm
                    alpha = torch.softmax(label_sim, dim=-1).view(1, -1).numpy()  # distance to weight
                else:
                    alpha = torch.softmax(-label_sim / self.temp, dim=-1).numpy()
                label2features[label] = np.dot(alpha, label2features[label]).reshape(-1)  # weighted sum
            self.examples.append(list(label2features.values()))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        demo_mask_features = None
        if self.demo_mode == "get":
            # demo
            demo_mask_features = self.examples[index]
            input_id = self.input_ids[index].tolist()
            attn_mask = self.attention_mask[index].tolist()
            label = self.labels[index].item()
            so_item = self.so[index].tolist()

            max_seq, real_seq = len(input_id), sum(attn_mask)
            max_demo_seq = max_seq + 2 * len(self.labelid_list)
            input_id = input_id[:real_seq]
            attn_mask = attn_mask[:real_seq]
            block_flag_for_demo = [0] * real_seq

            for labelid in self.labelid_list:
                input_id += [999, labelid]
                block_flag_for_demo += [1, 0]
                attn_mask += [1, 1]

            input_id += [1] * (max_demo_seq - len(input_id))
            attn_mask += [0] * (max_demo_seq - len(attn_mask))
            block_flag_for_demo += [0] * (max_demo_seq - len(block_flag_for_demo))
            assert len(input_id) == len(attn_mask) == len(block_flag_for_demo)
            return tuple(
                [
                    torch.tensor(input_id),
                    torch.tensor(attn_mask),
                    torch.tensor(label),
                    torch.tensor(so_item),
                    torch.tensor(block_flag_for_demo),
                    torch.tensor(demo_mask_features),
                ]
            )
        else:
            return tuple([self.input_ids[index], self.attention_mask[index], self.labels[index], self.so[index]])


def convert_examples_to_features_normal(examples, max_seq_length, tokenizer):
    print("#examples", len(examples))
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenize(example.text_a, tokenizer)
        tokens_b = tokenize(example.text_b, tokenizer)
        tokens_c = tokenize(example.text_c, tokenizer)

        _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        tokens_b = tokens_b + ["[SEP]"] + tokens_c

        inputs = tokenizer(
            example.text_a,
            example.text_b + tokenizer.sep_token + example.text_c,
            truncation="longest_first",
            max_length=max_seq_length,
            padding="max_length",
            add_special_tokens=True,
        )

        label_id = example.label

        if ex_index == 0:
            logger.info(f"input_text : {tokens_a} {tokens_b} {tokens_c}")
            logger.info(f"input_ids : {inputs['input_ids']}")
            # logger.info(f"token_type_ids : {inputs['token_type_ids']}")

        # append 1 sample with 2 input
        features.append(
            InputFeatures(
                input_ids=inputs["input_ids"],
                input_mask=inputs["attention_mask"],
                segment_ids=inputs["attention_mask"],
                # segment_ids=inputs['token_type_ids'],
                label_id=label_id,
            )
        )

    print("#features", len(features))
    return features


def convert_examples_to_features(examples, max_seq_length, tokenizer, args, rel2id, mode):
    """Loads a data file into a list of `InputBatch`s."""

    save_file = "./dataset/cached_wiki80.pkl"
    mode = "text"

    num_tokens = 0
    num_fit_examples = 0
    instances = []

    print("loading..")
    for (ex_index, example) in enumerate(examples):
        """
            the relation between SUBJECT and OBJECT is .
            
        """
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = []
        SUBJECT_START = "[subject_start]"
        SUBJECT_END = "[subject_end]"
        OBJECT_START = "[object_start]"
        OBJECT_END = "[object_end]"

        if mode.startswith("text"):
            for i, token in enumerate(example.sentence):
                if i == example.span1[0]:
                    tokens.append(SUBJECT_START)
                if i == example.span2[0]:
                    tokens.append(OBJECT_START)
                # for sub_token in tokenizer.tokenize(token):
                #     tokens.append(sub_token)
                if i == example.span1[1]:
                    tokens.append(SUBJECT_END)
                if i == example.span2[1]:
                    tokens.append(OBJECT_END)

                tokens.append(token)

        SUBJECT = " ".join(example.sentence[example.span1[0] : example.span1[1]])
        OBJECT = " ".join(example.sentence[example.span2[0] : example.span2[1]])
        SUBJECT_ids = tokenizer(" " + SUBJECT, add_special_tokens=False)["input_ids"]
        OBJECT_ids = tokenizer(" " + OBJECT, add_special_tokens=False)["input_ids"]

        # add prompt [T_n] and entity marker [obj] to enrich the context.

        if args.use_template_words:
            prompt = f"[sub] {SUBJECT} [sub] {tokenizer.mask_token} [obj] {OBJECT} [obj] ."
        else:
            prompt = f"{SUBJECT} {tokenizer.mask_token} {OBJECT}."

        if ex_index == 0:
            input_text = " ".join(tokens)
            logger.info(f"input text : {input_text}")
            logger.info(f"prompt : {prompt}")
            logger.info(f"label : {example.label}")
        inputs = tokenizer(
            prompt,
            " ".join(tokens),
            truncation="longest_first",
            max_length=max_seq_length,
            padding="max_length",
            add_special_tokens=True,
        )

        # find the subject and object tokens, choose the first ones
        sub_st = sub_ed = obj_st = obj_ed = -1
        for i in range(len(inputs["input_ids"])):
            if sub_st == -1 and inputs["input_ids"][i : i + len(SUBJECT_ids)] == SUBJECT_ids:
                sub_st = i
                sub_ed = i + len(SUBJECT_ids)
            if obj_st == -1 and inputs["input_ids"][i : i + len(OBJECT_ids)] == OBJECT_ids:
                obj_st = i
                obj_ed = i + len(OBJECT_ids)
        if sub_st == -1 or obj_st == -1:
            continue
        assert sub_st != -1 and obj_st != -1, (
            inputs["input_ids"],
            tokenizer.convert_ids_to_tokens(inputs["input_ids"]),
            example.sentence,
            example.span1,
            example.span2,
            SUBJECT,
            OBJECT,
            SUBJECT_ids,
            OBJECT_ids,
            sub_st,
            obj_st,
        )

        num_tokens += sum(inputs["attention_mask"])

        if sum(inputs["attention_mask"]) > max_seq_length:
            pass
            # tokens = tokens[:max_seq_length]
        else:
            num_fit_examples += 1

        x = OrderedDict()
        x["input_ids"] = inputs["input_ids"]
        x["attention_mask"] = inputs["attention_mask"]
        x["label"] = rel2id[example.label]
        x["so"] = [sub_st, sub_ed, obj_st, obj_ed]

        instances.append(x)

    with open(file=save_file, mode="wb") as fw:
        pickle.dump(instances, fw)
    print("Finish save preprocessed data to {}.".format(save_file))

    input_ids = [o["input_ids"] for o in instances]
    attention_mask = [o["attention_mask"] for o in instances]
    labels = [o["label"] for o in instances]
    so = torch.tensor([o["so"] for o in instances])

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)

    logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
    logger.info(
        "%d (%.2f %%) examples can fit max_seq_length = %d"
        % (num_fit_examples, num_fit_examples * 100.0 / len(examples), max_seq_length)
    )

    if args.use_demo:
        dataset = wiki80DemoDataset(input_ids, attention_mask, labels, so, rel2id, tokenizer, mode, args)
    else:
        dataset = TensorDataset(input_ids, attention_mask, labels, so)

    return dataset


def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


def get_dataset(mode, args, tokenizer, processor):

    if mode == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif mode == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    elif mode == "test":
        examples = processor.get_test_examples(args.data_dir)
    else:
        raise Exception("mode must be in choice [trian, dev, test]")

    if "wiki80" in args.task_name:
        # normal relation extraction task
        dataset = convert_examples_to_features(
            examples, args.max_seq_length, tokenizer, args, processor.get_labels(), mode
        )
        return dataset
    else:
        train_features = convert_examples_to_features_normal(examples, args.max_seq_length, tokenizer)

    input_ids = []
    input_mask = []
    segment_ids = []
    label_id = []

    for f in train_features:
        input_ids.append(f.input_ids)
        input_mask.append(f.input_mask)
        segment_ids.append(f.segment_ids)
        label_id.append(f.label_id)

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    all_label_ids = torch.tensor(label_id, dtype=torch.float)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return train_data


processors = {
    "wiki80": wiki80Processor,
}
