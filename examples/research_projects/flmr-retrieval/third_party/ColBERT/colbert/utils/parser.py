import os
import copy
import faiss

from argparse import ArgumentParser

import colbert.utils.distributed as distributed
from colbert.utils.runs import Run
from colbert.utils.utils import print_message, timestamp, create_directory


class Arguments():
    def __init__(self, description):
        self.parser = ArgumentParser(description=description)
        self.checks = []

        self.add_argument('--root', dest='root', default='experiments')
        self.add_argument('--experiment', dest='experiment', default='dirty')
        self.add_argument('--run', dest='run', default=Run.name)

        self.add_argument('--local_rank', dest='rank', default=-1, type=int)

    def add_model_parameters(self):
        # Core Arguments
        self.add_argument('--similarity', dest='similarity', default='cosine', choices=['cosine', 'l2'])
        self.add_argument('--dim', dest='dim', default=128, type=int)
        self.add_argument('--query_maxlen', dest='query_maxlen', default=32, type=int)
        self.add_argument('--doc_maxlen', dest='doc_maxlen', default=180, type=int)

        # Filtering-related Arguments
        self.add_argument('--mask-punctuation', dest='mask_punctuation', default=False, action='store_true')

    def add_model_training_parameters(self):
        # NOTE: Providing a checkpoint is one thing, --resume is another, --resume_optimizer is yet another.
        self.add_argument('--resume', dest='resume', default=False, action='store_true')
        self.add_argument('--resume_optimizer', dest='resume_optimizer', default=False, action='store_true')
        self.add_argument('--checkpoint', dest='checkpoint', default=None, required=False)

        self.add_argument('--lr', dest='lr', default=3e-06, type=float)
        self.add_argument('--maxsteps', dest='maxsteps', default=400000, type=int)
        self.add_argument('--bsize', dest='bsize', default=32, type=int)
        self.add_argument('--accum', dest='accumsteps', default=2, type=int)
        self.add_argument('--amp', dest='amp', default=False, action='store_true')

    def add_model_inference_parameters(self):
        self.add_argument('--checkpoint', dest='checkpoint', required=True)
        self.add_argument('--bsize', dest='bsize', default=128, type=int)
        self.add_argument('--amp', dest='amp', default=False, action='store_true')

    def add_training_input(self):
        self.add_argument('--triples', dest='triples', required=True)
        self.add_argument('--queries', dest='queries', default=None)
        self.add_argument('--collection', dest='collection', default=None)

        def check_training_input(args):
            assert (args.collection is None) == (args.queries is None), \
                "For training, both (or neither) --collection and --queries must be supplied." \
                "If neither is supplied, the --triples file must contain texts (not PIDs)."

        self.checks.append(check_training_input)

    def add_ranking_input(self):
        self.add_argument('--queries', dest='queries', default=None)
        self.add_argument('--collection', dest='collection', default=None)
        self.add_argument('--qrels', dest='qrels', default=None)

    def add_reranking_input(self):
        self.add_ranking_input()
        self.add_argument('--topk', dest='topK', required=True)
        self.add_argument('--shortcircuit', dest='shortcircuit', default=False, action='store_true')

    def add_indexing_input(self):
        self.add_argument('--collection', dest='collection', required=True)
        self.add_argument('--index_root', dest='index_root', required=True)
        self.add_argument('--index_name', dest='index_name', required=True)

    def add_compressed_index_input(self):
        self.add_argument('--compression_level', dest='compression_level',
                          choices=[1, 2], type=int, default=None)


    def add_index_use_input(self):
        self.add_argument('--index_root', dest='index_root', required=True)
        self.add_argument('--index_name', dest='index_name', required=True)
        self.add_argument('--partitions', dest='partitions', default=None, type=int, required=False)

    def add_retrieval_input(self):
        self.add_index_use_input()
        self.add_argument('--nprobe', dest='nprobe', default=10, type=int)
        self.add_argument('--retrieve_only', dest='retrieve_only', default=False, action='store_true')

    def add_argument(self, *args, **kw_args):
        return self.parser.add_argument(*args, **kw_args)

    def check_arguments(self, args):
        for check in self.checks:
            check(args)

    def parse(self):
        args = self.parser.parse_args()
        self.check_arguments(args)

        args.input_arguments = copy.deepcopy(args)

        args.nranks, args.distributed = distributed.init(args.rank)

        args.nthreads = int(max(os.cpu_count(), faiss.omp_get_max_threads()) * 0.8)
        args.nthreads = max(1, args.nthreads // args.nranks)

        if args.nranks > 1:
            print_message(f"#> Restricting number of threads for FAISS to {args.nthreads} per process",
                          condition=(args.rank == 0))
            faiss.omp_set_num_threads(args.nthreads)

        Run.init(args.rank, args.root, args.experiment, args.run)
        Run._log_args(args)
        Run.info(args.input_arguments.__dict__, '\n')

        return args
