import os
import time
import torch
import ujson

from colbert.utils.utils import f7, print_message, timestamp


def load_contexts(first_hop_topk_path):
    qid2backgrounds = {}

    with open(first_hop_topk_path) as f:
        print_message(f"#> Loading backgrounds from {f.name} ..")

        last = None
        for line in f:
            qid, facts = ujson.loads(line)
            facts = [(tuple(f) if type(f) is list else f) for f in facts]
            qid2backgrounds[qid] = facts
            last = (qid, facts)

    # assert len(qid2backgrounds) in [0, len(queries)], (len(qid2backgrounds), len(queries))
    print_message(f"#> {first_hop_topk_path} has {len(qid2backgrounds)} qids. Last = {last}")

    return qid2backgrounds

def load_collectionX(collection_path, dict_in_dict=False):
    print_message("#> Loading collection...")

    collectionX = {}

    with open(collection_path) as f:
        for line_idx, line in enumerate(f):
            line = ujson.loads(line)

            assert type(line['text']) is list
            assert line['pid'] == line_idx, (line_idx, line)

            passage = [line['title'] + ' | ' + sentence for sentence in line['text']]

            if dict_in_dict:
                collectionX[line_idx] = {}

            for idx, sentence in enumerate(passage):
                if dict_in_dict:
                    collectionX[line_idx][idx] = sentence
                else:
                    collectionX[(line_idx, idx)] = sentence

    return collectionX
