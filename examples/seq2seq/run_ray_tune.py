from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from functools import partial
import os
import sys
# from durbango import *

def fix_import_paths():
    import os
    import sys
    import transformers
    sys.path.append(
        os.path.join(
            os.path.dirname(transformers.__file__),
            "../../examples/seq2seq/"))

    sys.path.append(
        os.path.join(
            os.path.dirname(transformers.__file__),
            "../../examples/"))

fix_import_paths()


from pathlib import Path


def get_ray_slug(cfg):
    strang = ''
    for k,v in cfg.items():

        strang += f'{k}_{v}'
    for i in range(10000):
        test = f'rayruns/run_{i}'
        try:
            Path(test).mkdir(exist_ok=True,parents=True)
            break
        except Exception:
            continue

    return os.path.expanduser(test)

from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger
def ray_main(args, config):
    fix_import_paths()
    for k,v in config.items():
        #assert hasattr(args, k), k
        setattr(args, k, v)
    args.output_dir = get_ray_slug(config)
    from finetune import main as ft_main
    ft_main(args)


def tune_helsinki_(args, num_samples=100, num_epochs=1):
    args.num_train_epochs = num_epochs
    # args.n_train = 10000
    search_space = {
        "learning_rate": tune.loguniform(1e-6, 0.5),
        "gradient_accumulation_steps": tune.choice([1, 8, 32, 128, 256]),
        "dropout": tune.uniform(0, 0.4),
        'label_smoothing': tune.uniform(0, 0.4),
    }
    # scheduler = ASHAScheduler(
    #      metric="val_avg_bleu",
    #      mode="max",
    #      max_t=num_epochs* int(1/args.val_check_interval),  # max number of reports until termination
    #      grace_period=1,
    #      reduction_factor=4,  # cut 1/4 of trials really quickly, and another 1/4 pretty soon
    #  )

    from ray.tune.suggest.hyperopt import HyperOptSearch

    searcher = HyperOptSearch(metric="val_avg_bleu", mode="max")
    reporter = CLIReporter(
        parameter_columns={
            "learning_rate": "lr",
            "gradient_accumulation_steps": "grad_accum",
            "dropout": "dropout",
            "label_smoothing": "l_smooth",

        },
        metric_columns={
            "val_avg_loss": "loss",
            "val_avg_bleu": "bleu",
            "global_step": "step",
            "training_iteration": "iter"
        }
    )
    config = search_space.copy()

    def datetime_now():
        import datetime
        return datetime.datetime.now().strftime("%H:%M:%S")

    config["wandb"] = {
        "project": "RAY",
        "group": f"gcp_sep16_wmt-{datetime_now()}",
        "api_key": "REMOVE_THIS" # consider setting env var too
    }
    ray.init(log_to_driver=True, address="auto")
    tune.run(
        partial(
            ray_main,
            args,
        ),
        resources_per_trial={"gpu": args.gpus},
        config=config,
        num_samples=10,
        search_alg=searcher,
        # scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_helsinki_asha",
        # loggers=DEFAULT_LOGGERS + (WandbLogger,),
        # max_failures=3,
        fail_fast=True,
    )

# Make default args
import argparse
DEFAULTS = {
    'logger': True,
    'checkpoint_callback': True,
    'early_stop_callback': False,
    'default_root_dir': None,
    'gradient_clip_val': 0,
    'eval_beams': 2,
    'process_position': 0,
    "eval_max_gen_length": 128,
    'num_nodes': 1,
    'num_processes': 1,
    'gpus': 1,
    'auto_select_gpus': False,
    'adafactor': False,
    #'tpu_cores': 0,
    'log_gpu_memory': None,
    'progress_bar_refresh_rate': 0,
    'overfit_batches': 0.0,
    'track_grad_norm': -1,
    'check_val_every_n_epoch': 1,
    'fast_dev_run': False,
    'accumulate_grad_batches': 1,
    'max_epochs': 1,
    'min_epochs': 1,
    'max_steps': None,
    'min_steps': None,
    'limit_train_batches': 1.0,
    'limit_val_batches': 1.0,
    'limit_test_batches': 1.0,
    'val_check_interval': 0.01,
    'log_save_interval': 100,
    'row_log_interval': 50,
    'distributed_backend': None,
    'precision': 32,
    'print_nan_grads': False,
    'weights_summary': 'top',
    'weights_save_path': None,
    'num_sanity_val_steps': 0,
    'truncated_bptt_steps': None,
    'resume_from_checkpoint': None,
    'profiler': None,
    'benchmark': False,
    'deterministic': False,
    'reload_dataloaders_every_epoch': True,
    'auto_lr_find': False,
    'replace_sampler_ddp': True,
    'terminate_on_nan': False,
    'auto_scale_batch_size': False,
    'prepare_data_per_node': True,
    'amp_level': 'O2',
    'val_percent_check': None,
    'test_percent_check': None,
    'train_percent_check': None,
    'overfit_pct': None,
    'model_name_or_path': 'sshleifer/student_marian_en_ro_6_3',
    'config_name': '',
    'tokenizer_name': 'sshleifer/student_marian_en_ro_6_3',
    'cache_dir': '',
    'encoder_layerdrop': None,
    'decoder_layerdrop': None,
    'dropout': None,
    'attention_dropout': None,
    'learning_rate': 0.0003,
    'lr_scheduler': 'linear',
    'weight_decay': 0.0,
    'adam_epsilon': 1e-08,
    'warmup_steps': 500,
    'num_workers': 4,
    'train_batch_size': 64,
    'eval_batch_size': 64,
    'output_dir': 'tmp',
    'fp16': True,
    'fp16_opt_level': 'O1',
    'do_train': True,
    'do_predict': True,
    'seed': 42,
    'data_dir': os.path.expanduser('~/wmt_en_ro'),
    'max_source_length': 128,
    'max_target_length': 128,
    'val_max_target_length': 128,
    'test_max_target_length': 128,
    'freeze_encoder': True,
    'freeze_embeds': True,
    'sortish_sampler': True,
    'logger_name': 'wandb',
    'n_train': -1,
    'n_val': 500,
    'n_test': -1,
    'task': 'translation',
    'label_smoothing': 0.1,
    'src_lang': '',
    'tgt_lang': '',
    'early_stopping_patience': -1,
    'val_metric': None,
    'save_top_k': 1,
}
args = argparse.Namespace(**DEFAULTS)

tune_helsinki_(args)
