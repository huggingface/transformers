import os
import ujson

from collections import defaultdict
from colbert.utils.utils import print_message, file_tqdm


def load_collection_(path, retain_titles):
    with open(path) as f:
        collection = []

        for line in file_tqdm(f):
            _, passage, title = line.strip().split('\t')

            if retain_titles:
                passage = title + ' | ' + passage

            collection.append(passage)

    return collection


def load_qas_(path):
    print_message("#> Loading the reference QAs from", path)

    triples = []

    with open(path) as f:
        for line in f:
            qa = ujson.loads(line)
            triples.append((qa['qid'], qa['question'], qa['answers']))

    return triples
