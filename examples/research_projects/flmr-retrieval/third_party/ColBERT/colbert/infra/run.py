import os
import atexit

from colbert.utils.utils import create_directory, print_message, timestamp
from contextlib import contextmanager

from colbert.infra.config import RunConfig


class Run(object):
    _instance = None

    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # NOTE: If a deadlock arises, switch to false!!

    def __new__(cls):
        """
        Singleton Pattern. See https://python-patterns.guide/gang-of-four/singleton/
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.stack = []

            # TODO: Save a timestamp here! And re-use it! But allow the user to override it on calling Run().context a second time.
            run_config = RunConfig()
            run_config.assign_defaults()
            
            cls._instance.__append(run_config)

        # TODO: atexit.register(all_done)

        return cls._instance

    @property
    def config(self):
        return self.stack[-1]

    def __getattr__(self, name):
        if hasattr(self.config, name):
            return getattr(self.config, name)

        super().__getattr__(name)

    def __append(self, runconfig: RunConfig):
        # runconfig.disallow_writes(readonly=True)
        self.stack.append(runconfig)

    def __pop(self):
        self.stack.pop()

    @contextmanager
    def context(self, runconfig: RunConfig, inherit_config=True):
        if inherit_config:
            runconfig = RunConfig.from_existing(self.config, runconfig)

        self.__append(runconfig)

        try:
            yield
        finally:
            self.__pop()
        
    def open(self, path, mode='r'):
        path = os.path.join(self.path_, path)

        if not os.path.exists(self.path_):
            create_directory(self.path_)

        if ('w' in mode or 'a' in mode) and not self.overwrite:
            assert not os.path.exists(path), (self.overwrite, path)

        return open(path, mode=mode)
    
    def print(self, *args):
        print_message("[" + str(self.rank) + "]", "\t\t", *args)

    def print_main(self, *args):
        if self.rank == 0:
            self.print(*args)


if __name__ == '__main__':
    print(Run().root, '!')

    with Run().context(RunConfig(rank=0, nranks=1)):
        with Run().context(RunConfig(experiment='newproject')):
            print(Run().nranks, '!')

        print(Run().config, '!')
        print(Run().rank)


# TODO: Handle logging all prints to a file. There should be a way to determine the level of logs that go to stdout.