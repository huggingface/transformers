import os
import sys
import ujson
# import mlflow
import traceback

# from torch.utils.tensorboard import SummaryWriter
from colbert.utils.utils import print_message, create_directory


class Logger():
    def __init__(self, rank, run):
        self.rank = rank
        self.is_main = self.rank in [-1, 0]
        self.run = run
        self.logs_path = os.path.join(self.run.path, "logs/")

        if self.is_main:
            # self._init_mlflow()
            # self.initialized_tensorboard = False
            create_directory(self.logs_path)

    # def _init_mlflow(self):
    #     mlflow.set_tracking_uri('file://' + os.path.join(self.run.experiments_root, "logs/mlruns/"))
    #     mlflow.set_experiment('/'.join([self.run.experiment, self.run.script]))
        
    #     mlflow.set_tag('experiment', self.run.experiment)
    #     mlflow.set_tag('name', self.run.name)
    #     mlflow.set_tag('path', self.run.path)

    # def _init_tensorboard(self):
    #     root = os.path.join(self.run.experiments_root, "logs/tensorboard/")
    #     logdir = '__'.join([self.run.experiment, self.run.script, self.run.name])
    #     logdir = os.path.join(root, logdir)

    #     self.writer = SummaryWriter(log_dir=logdir)
    #     self.initialized_tensorboard = True

    def _log_exception(self, etype, value, tb):
        if not self.is_main:
            return

        output_path = os.path.join(self.logs_path, 'exception.txt')
        trace = ''.join(traceback.format_exception(etype, value, tb)) + '\n'
        print_message(trace, '\n\n')

        self.log_new_artifact(output_path, trace)

    def _log_all_artifacts(self):
        if not self.is_main:
            return

        # mlflow.log_artifacts(self.logs_path)

    def _log_args(self, args):
        if not self.is_main:
            return

        # for key in vars(args):
        #     value = getattr(args, key)
        #     if type(value) in [int, float, str, bool]:
        #         mlflow.log_param(key, value)

        # with open(os.path.join(self.logs_path, 'args.json'), 'w') as output_metadata:
        #     # TODO: Call provenance() on the values that support it
        #     ujson.dump(args.input_arguments.__dict__, output_metadata, indent=4)
        #     output_metadata.write('\n')

        with open(os.path.join(self.logs_path, 'args.txt'), 'w') as output_metadata:
            output_metadata.write(' '.join(sys.argv) + '\n')

    def log_metric(self, name, value, step, log_to_mlflow=True):
        if not self.is_main:
            return

        # if not self.initialized_tensorboard:
        #     self._init_tensorboard()

        # if log_to_mlflow:
        #     mlflow.log_metric(name, value, step=step)
        # self.writer.add_scalar(name, value, step)

    def log_new_artifact(self, path, content):
        with open(path, 'w') as f:
            f.write(content)

        # mlflow.log_artifact(path)

    def warn(self, *args):
        msg = print_message('[WARNING]', '\t', *args)

        with open(os.path.join(self.logs_path, 'warnings.txt'), 'a') as output_metadata:
            output_metadata.write(msg + '\n\n\n')

    def info_all(self, *args):
        print_message('[' + str(self.rank) + ']', '\t', *args)

    def info(self, *args):
        if self.is_main:
            print_message(*args)
