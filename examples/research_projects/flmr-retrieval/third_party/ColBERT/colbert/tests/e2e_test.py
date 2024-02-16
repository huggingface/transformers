import os
import argparse
from collections import namedtuple
from datasets import load_dataset
from utility.utils.dpr import has_answer, DPR_normalize
import tqdm

from colbert import Indexer, Searcher
from colbert.infra import ColBERTConfig, RunConfig, Run

SquadExample = namedtuple("SquadExample", "id title context question answers")


def build_index_and_init_searcher(checkpoint, collection, experiment_dir):
    nbits = 1  # encode each dimension with 1 bits
    doc_maxlen = 180  # truncate passages at 180 tokens
    experiment = f"e2etest.nbits={nbits}"

    with Run().context(RunConfig(nranks=1)):
        config = ColBERTConfig(
            doc_maxlen=doc_maxlen,
            nbits=nbits,
            root=experiment_dir,
            experiment=experiment,
        )
        indexer = Indexer(checkpoint, config=config)
        indexer.index(name=experiment, collection=collection, overwrite=True)

        config = ColBERTConfig(
            root=experiment_dir,
            experiment=experiment,
        )
        searcher = Searcher(
            index=experiment,
            config=config,
        )

        return searcher


def success_at_k(searcher, examples, k):
    scores = []
    for ex in tqdm.tqdm(examples):
        scores.append(evaluate_retrieval_example(searcher, ex, k))
    return sum(scores) / len(scores)


def evaluate_retrieval_example(searcher, ex, k):
    results = searcher.search(ex.question, k=k)
    for passage_id, passage_rank, passage_score in zip(*results):
        passage = searcher.collection[passage_id]
        score = has_answer([DPR_normalize(ans) for ans in ex.answers], passage)
        if score:
            return 1
    return 0


def get_squad_split(squad, split="validation"):
    fields = squad[split].features
    data = zip(*[squad[split][field] for field in fields])
    return [
        SquadExample(eid, title, context, question, answers["text"])
        for eid, title, context, question, answers in data
    ]


def main(args):
    checkpoint = args.checkpoint
    collection = args.collection
    experiment_dir = args.expdir

    # Start the test
    k = 5
    searcher = build_index_and_init_searcher(checkpoint, collection, experiment_dir)

    squad = load_dataset("squad")
    squad_dev = get_squad_split(squad)
    success_rate = success_at_k(searcher, squad_dev[:1000], k)
    assert success_rate > 0.93, f"success rate at {success_rate} is lower than expected"
    print(f"test passed with success rate {success_rate}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start end-to-end test.")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint"
    )
    parser.add_argument(
        "--collection", type=str, required=True, help="Path to collection"
    )
    parser.add_argument(
        "--expdir", type=str, required=True, help="Experiment directory"
    )
    args = parser.parse_args()
    main(args)
