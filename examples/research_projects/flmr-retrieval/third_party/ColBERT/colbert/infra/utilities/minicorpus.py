import os
import random

from colbert.utils.utils import create_directory

from colbert.data import Collection, Queries, Ranking


def sample_minicorpus(name, factor, topk=30, maxdev=3000):
    """
    Factor:
        * nano=1
        * micro=10
        * mini=100
        * small=100 with topk=100
        * medium=150 with topk=300
    """

    random.seed(12345)

    # Load collection
    collection = Collection(path='/dfs/scratch0/okhattab/OpenQA/collection.tsv')

    # Load train and dev queries
    qas_train = Queries(path='/dfs/scratch0/okhattab/OpenQA/NQ/train/qas.json').qas()
    qas_dev = Queries(path='/dfs/scratch0/okhattab/OpenQA/NQ/dev/qas.json').qas()

    # Load train and dev C3 rankings
    ranking_train = Ranking(path='/dfs/scratch0/okhattab/OpenQA/NQ/train/rankings/C3.tsv.annotated').todict()
    ranking_dev = Ranking(path='/dfs/scratch0/okhattab/OpenQA/NQ/dev/rankings/C3.tsv.annotated').todict()

    # Sample NT and ND queries from each, keep only the top-k passages for those
    sample_train = random.sample(list(qas_train.keys()), min(len(qas_train.keys()), 300*factor))
    sample_dev = random.sample(list(qas_dev.keys()), min(len(qas_dev.keys()), maxdev, 30*factor))

    train_pids = [pid for qid in sample_train for qpids in ranking_train[qid][:topk] for pid in qpids]
    dev_pids = [pid for qid in sample_dev for qpids in ranking_dev[qid][:topk] for pid in qpids]

    sample_pids = sorted(list(set(train_pids + dev_pids)))
    print(f'len(sample_pids) = {len(sample_pids)}')

    # Save the new query sets: train and dev
    ROOT = f'/future/u/okhattab/root/unit/data/NQ-{name}'

    create_directory(os.path.join(ROOT, 'train'))
    create_directory(os.path.join(ROOT, 'dev'))

    new_train = Queries(data={qid: qas_train[qid] for qid in sample_train})
    new_train.save(os.path.join(ROOT, 'train/questions.tsv'))
    new_train.save_qas(os.path.join(ROOT, 'train/qas.json'))

    new_dev = Queries(data={qid: qas_dev[qid] for qid in sample_dev})
    new_dev.save(os.path.join(ROOT, 'dev/questions.tsv'))
    new_dev.save_qas(os.path.join(ROOT, 'dev/qas.json'))

    # Save the new collection
    print(f"Saving to {os.path.join(ROOT, 'collection.tsv')}")
    Collection(data=[collection[pid] for pid in sample_pids]).save(os.path.join(ROOT, 'collection.tsv'))

    print('#> Done!')


if __name__ == '__main__':
    sample_minicorpus('medium', 150, topk=300)
