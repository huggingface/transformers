from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from functools import partial
from durbango import *
from finetune import main as ft_main
from pathlib import Path
import os

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


def ray_main(args, config):

    for k,v in config.items():
        #assert hasattr(args, k), k
        setattr(args, k, v)
    args.n_train = 64
    args.output_dir = get_ray_slug(config)
    args.num_train_epochs = 3
    ft_main(args)


def tune_helsinki_(args, num_samples=1, num_epochs=3):

    search_space = {
        "learning_rate": tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
        "gradient_accumulation_steps": tune.choice([1, 8, 32, 128, 256]),
        "dropout": tune.choice([0, 0.1, 0.2, 0.4]),
    }
    scheduler = ASHAScheduler(
        metric="val_avg_bleu",
        mode="min",
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=list(search_space.keys()),
        metric_columns=["val_avg_loss", "val_avg_bleu", "global_step"])
    tune.run(
        partial(
            ray_main,
            args,
            ),
        resources_per_trial={"cpu": 0, "gpu": 1},
        config=search_space,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_helsinki_asha")


# Make default args
import argparse

args = argparse.Namespace(**{
    'logger': True,
    'checkpoint_callback': True,
    'early_stop_callback': False,
    'default_root_dir': None,
    'gradient_clip_val': 0,
    'eval_beams': 2,
    'process_position': 0,
    'num_nodes': 1,
    'num_processes': 1,
    'gpus': 1,
    'auto_select_gpus': False,
    'tpu_cores': 0,
    'log_gpu_memory': None,
    'progress_bar_refresh_rate': 1,
    'overfit_batches': 0.0,
    'track_grad_norm': -1,
    'check_val_every_n_epoch': 1,
    'fast_dev_run': False,
    'accumulate_grad_batches': 1,
    'max_epochs': 1000,
    'min_epochs': 1,
    'max_steps': None,
    'min_steps': None,
    'limit_train_batches': 1.0,
    'limit_val_batches': 1.0,
    'limit_test_batches': 1.0,
    'val_check_interval': 0.25,
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
    'reload_dataloaders_every_epoch': False,
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
    'train_batch_size': 32,
    'eval_batch_size': 32,
    'output_dir': 'tmp',
    'fp16': True,
    'fp16_opt_level': 'O1',
    'do_train': True,
    'do_predict': True,
    'seed': 42,
    'data_dir': '/home/shleifer/transformers_fork/examples/seq2seq/wmt_en_ro',
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
    'early_stopping_patience': -1
})

tune_helsinki_(args)
