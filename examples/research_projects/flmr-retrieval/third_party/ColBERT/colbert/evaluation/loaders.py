import os
import ujson
import torch
import random

from collections import defaultdict, OrderedDict

from colbert.parameters import DEVICE
from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message, load_checkpoint
from colbert.evaluation.load_model import load_model
from colbert.utils.runs import Run


def load_queries(queries_path):
    queries = OrderedDict()

    print_message("#> Loading the queries from", queries_path, "...")

    with open(queries_path) as f:
        for line in f:
            qid, query, *_ = line.strip().split('\t')
            qid = int(qid)

            assert (qid not in queries), ("Query QID", qid, "is repeated!")
            queries[qid] = query

    print_message("#> Got", len(queries), "queries. All QIDs are unique.\n")

    return queries


def load_qrels(qrels_path):
    if qrels_path is None:
        return None

    print_message("#> Loading qrels from", qrels_path, "...")

    qrels = OrderedDict()
    with open(qrels_path, mode='r', encoding="utf-8") as f:
        for line in f:
            qid, x, pid, y = map(int, line.strip().split('\t'))
            assert x == 0 and y == 1
            qrels[qid] = qrels.get(qid, [])
            qrels[qid].append(pid)

    # assert all(len(qrels[qid]) == len(set(qrels[qid])) for qid in qrels)
    for qid in qrels:
        qrels[qid] = list(set(qrels[qid]))

    avg_positive = round(sum(len(qrels[qid]) for qid in qrels) / len(qrels), 2)

    print_message("#> Loaded qrels for", len(qrels), "unique queries with",
                  avg_positive, "positives per query on average.\n")

    return qrels


def load_topK(topK_path):
    queries = OrderedDict()
    topK_docs = OrderedDict()
    topK_pids = OrderedDict()

    print_message("#> Loading the top-k per query from", topK_path, "...")

    with open(topK_path) as f:
        for line_idx, line in enumerate(f):
            if line_idx and line_idx % (10*1000*1000) == 0:
                print(line_idx, end=' ', flush=True)

            qid, pid, query, passage = line.split('\t')
            qid, pid = int(qid), int(pid)

            assert (qid not in queries) or (queries[qid] == query)
            queries[qid] = query
            topK_docs[qid] = topK_docs.get(qid, [])
            topK_docs[qid].append(passage)
            topK_pids[qid] = topK_pids.get(qid, [])
            topK_pids[qid].append(pid)

        print()

    assert all(len(topK_pids[qid]) == len(set(topK_pids[qid])) for qid in topK_pids)

    Ks = [len(topK_pids[qid]) for qid in topK_pids]

    print_message("#> max(Ks) =", max(Ks), ", avg(Ks) =", round(sum(Ks) / len(Ks), 2))
    print_message("#> Loaded the top-k per query for", len(queries), "unique queries.\n")

    return queries, topK_docs, topK_pids


def load_topK_pids(topK_path, qrels):
    topK_pids = defaultdict(list)
    topK_positives = defaultdict(list)

    print_message("#> Loading the top-k PIDs per query from", topK_path, "...")

    with open(topK_path) as f:
        for line_idx, line in enumerate(f):
            if line_idx and line_idx % (10*1000*1000) == 0:
                print(line_idx, end=' ', flush=True)

            qid, pid, *rest = line.strip().split('\t')
            qid, pid = int(qid), int(pid)

            topK_pids[qid].append(pid)

            assert len(rest) in [1, 2, 3]

            if len(rest) > 1:
                *_, label = rest
                label = int(label)
                assert label in [0, 1]

                if label >= 1:
                    topK_positives[qid].append(pid)

        print()

    assert all(len(topK_pids[qid]) == len(set(topK_pids[qid])) for qid in topK_pids)
    assert all(len(topK_positives[qid]) == len(set(topK_positives[qid])) for qid in topK_positives)

    # Make them sets for fast lookups later
    topK_positives = {qid: set(topK_positives[qid]) for qid in topK_positives}

    Ks = [len(topK_pids[qid]) for qid in topK_pids]

    print_message("#> max(Ks) =", max(Ks), ", avg(Ks) =", round(sum(Ks) / len(Ks), 2))
    print_message("#> Loaded the top-k per query for", len(topK_pids), "unique queries.\n")

    if len(topK_positives) == 0:
        topK_positives = None
    else:
        assert len(topK_pids) >= len(topK_positives)

        for qid in set.difference(set(topK_pids.keys()), set(topK_positives.keys())):
            topK_positives[qid] = []

        assert len(topK_pids) == len(topK_positives)

        avg_positive = round(sum(len(topK_positives[qid]) for qid in topK_positives) / len(topK_pids), 2)

        print_message("#> Concurrently got annotations for", len(topK_positives), "unique queries with",
                      avg_positive, "positives per query on average.\n")

    assert qrels is None or topK_positives is None, "Cannot have both qrels and an annotated top-K file!"

    if topK_positives is None:
        topK_positives = qrels

    return topK_pids, topK_positives


def load_collection(collection_path):
    print_message("#> Loading collection...")

    collection = []

    with open(collection_path) as f:
        for line_idx, line in enumerate(f):
            if line_idx % (1000*1000) == 0:
                print(f'{line_idx // 1000 // 1000}M', end=' ', flush=True)

            pid, passage, *rest = line.strip('\n\r ').split('\t')
            assert pid == 'id' or int(pid) == line_idx

            if len(rest) >= 1:
                title = rest[0]
                passage = title + ' | ' + passage

            collection.append(passage)

    print()

    return collection


def load_colbert(args, do_print=True):
    colbert, checkpoint = load_model(args, do_print)

    # TODO: If the parameters below were not specified on the command line, their *checkpoint* values should be used.
    # I.e., not their purely (i.e., training) default values.

    for k in ['query_maxlen', 'doc_maxlen', 'dim', 'similarity', 'amp']:
        if 'arguments' in checkpoint and hasattr(args, k):
            if k in checkpoint['arguments'] and checkpoint['arguments'][k] != getattr(args, k):
                a, b = checkpoint['arguments'][k], getattr(args, k)
                Run.warn(f"Got checkpoint['arguments']['{k}'] != args.{k} (i.e., {a} != {b})")

    if 'arguments' in checkpoint:
        if args.rank < 1:
            print(ujson.dumps(checkpoint['arguments'], indent=4))

    if do_print:
        print('\n')

    return colbert, checkpoint
