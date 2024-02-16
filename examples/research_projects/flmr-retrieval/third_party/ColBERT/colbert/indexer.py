import os
import time

import torch.multiprocessing as mp

from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert.infra.launcher import Launcher

from colbert.utils.utils import create_directory, print_message

from colbert.indexing.collection_indexer import encode


class Indexer:
    def __init__(self, checkpoint, config=None):
        """
           Use Run().context() to choose the run's configuration. They are NOT extracted from `config`.
        """

        self.index_path = None
        self.checkpoint = checkpoint
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(checkpoint)

        self.config = ColBERTConfig.from_existing(self.checkpoint_config, config, Run().config)
        self.configure(checkpoint=checkpoint)


    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def get_index(self):
        return self.index_path

    def erase(self):
        assert self.index_path is not None
        directory = self.index_path
        deleted = []

        for filename in sorted(os.listdir(directory)):
            filename = os.path.join(directory, filename)

            delete = filename.endswith(".json")
            delete = delete and ('metadata' in filename or 'doclen' in filename or 'plan' in filename)
            delete = delete or filename.endswith(".pt")
            
            if delete:
                deleted.append(filename)
        
        if len(deleted):
            print_message(f"#> Will delete {len(deleted)} files already at {directory} in 3 seconds...")
            time.sleep(3)

            for filename in deleted:
                os.remove(filename)

        return deleted

    def index(self, name, collection, batch_size=64, overwrite=False):
        assert overwrite in [True, False, 'reuse', 'resume']

        self.configure(collection=collection, index_name=name, resume=overwrite=='resume')
        self.configure(bsize=batch_size, partitions=None)

        self.index_path = self.config.index_path_
        index_does_not_exist = (not os.path.exists(self.config.index_path_))

        assert (overwrite in [True, 'reuse', 'resume']) or index_does_not_exist, self.config.index_path_
        create_directory(self.config.index_path_)

        if overwrite is True:
            self.erase()

        if index_does_not_exist or overwrite != 'reuse':
            self.__launch(collection)

        return self.index_path

    def __launch(self, collection):
        manager = mp.Manager()
        shared_lists = [manager.list() for _ in range(self.config.nranks)]
        shared_queues = [manager.Queue(maxsize=1) for _ in range(self.config.nranks)]

        launcher = Launcher(encode)
        launcher.launch(self.config, collection, shared_lists, shared_queues)
