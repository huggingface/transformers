import argparse
import random
from typing import Dict

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM
import gc

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=100,
        help="",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased",
        help="",
    )

    parser.add_argument(
        "--use-half",
        action="store_true",
    )

    parser.add_argument(
        "--use-cuda",
        action="store_true",
    )

    return parser


def seed_init_fn(x):
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return


def benchmark_training(model, inputs: Dict, num_training_steps: int):
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    # warmup
    for _ in range(10):
        model.zero_grad()
        outputs = model(**inputs)
        loss = outputs.logits.sum()
        loss.backward()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(num_training_steps):
        model.zero_grad()
        outputs = model(**inputs)
        loss = outputs.logits.sum()
        loss.backward()

        progress_bar.update(1)
    end_event.record()
    torch.cuda.synchronize()

    max_memory = torch.cuda.max_memory_allocated(device)

    return (start_event.elapsed_time(end_event) * 1.0e-3) / num_training_steps, max_memory


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    BATCH_SIZES = [1, 2, 4, 8]
    SEQ_LEN = [128, 256]
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

    output_file = open("log_{}_train.csv".format(args.model_name.replace("/", "-")), "w")
    output_file.write(
        "num_training_steps, batch_size, seq_len, is cuda, Time per batch (eager - s), Time per batch (sdpa - s), Speedup (%), Eager peak mem (MB), sdpa peak mem (MB), Mem saving (%)\n"
    )
    all_eager_time_per_batch = {}
    all_eager_max_mem = {}
    all_sdpa_max_mem = {}
    all_sdpa_time_per_batch = {}

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with torch.device(device):
        hf_model = AutoModelForMaskedLM.from_pretrained(
            args.model_name, torch_dtype=torch.float16 if args.use_half else None, attn_implementation="sdpa"
        )
    hf_model = hf_model.to(device)
    for batch_size in BATCH_SIZES:
        for sequence_length in SEQ_LEN:
            print(f"Benchmark sdpa on: bs={batch_size}, seq_len={sequence_length}")

            vocab_size = hf_model.config.vocab_size
            inputs = {
                "input_ids": torch.randint(vocab_size - 1, (batch_size, sequence_length), dtype=torch.int64).to(
                    device
                ),
                #"attention_mask": torch.ones(batch_size, sequence_length, dtype=torch.int64).to(device),
            }

            # raise error if no optimized kernel is available
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                sdpa_time_per_batch, sdpa_max_mem = benchmark_training(
                    hf_model, inputs=inputs, num_training_steps=args.num_training_steps
                )
            
            all_sdpa_max_mem[(batch_size, sequence_length)] = sdpa_max_mem
            all_sdpa_time_per_batch[(batch_size, sequence_length)] = sdpa_time_per_batch
            print(f"PT SDPA: {sdpa_time_per_batch:.3f} s, peak {sdpa_max_mem:.2f} MB")

    del hf_model
    gc.collect()

    with torch.device(device):
        hf_model = AutoModelForMaskedLM.from_pretrained(
            args.model_name, torch_dtype=torch.float16 if args.use_half else None, attn_implementation="eager"
        )
    hf_model = hf_model.to(device)

    for batch_size in BATCH_SIZES:
        for sequence_length in SEQ_LEN:
            print(f"Benchmark eager on: bs={batch_size}, seq_len={sequence_length}")

            vocab_size = hf_model.config.vocab_size
            inputs = {
                "input_ids": torch.randint(vocab_size - 1, (batch_size, sequence_length), dtype=torch.int64).to(
                    device
                ),
                #"attention_mask": torch.ones(batch_size, sequence_length, dtype=torch.int64).to(device),
            }

            eager_time_per_batch, eager_max_mem = benchmark_training(
                hf_model, inputs=inputs, num_training_steps=args.num_training_steps
            )

            all_eager_time_per_batch[(batch_size, sequence_length)] = eager_time_per_batch
            all_eager_max_mem[(batch_size, sequence_length)] = eager_max_mem

            eager_max_mem = all_eager_max_mem[(batch_size, sequence_length)] * 1e-6
            sdpa_max_mem = all_sdpa_max_mem[(batch_size, sequence_length)] * 1e-6

            eager_time_per_batch = all_eager_time_per_batch[(batch_size, sequence_length)]
            sdpa_time_per_batch = all_sdpa_time_per_batch[(batch_size, sequence_length)]

            print(f"PT eager: {eager_time_per_batch:.3f} s, peak {eager_max_mem:.2f} MB")
            print(f"PT SDPA: {sdpa_time_per_batch:.3f} s, peak {sdpa_max_mem:.2f} MB")
            speedup = (eager_time_per_batch / sdpa_time_per_batch - 1) * 100
            mem_saved = (eager_max_mem / sdpa_max_mem - 1) * 100

            output_file.write(
                "{},{},{},{},{},{},{},{},{},{}\n".format(
                    args.num_training_steps,
                    batch_size,
                    sequence_length,
                    args.use_cuda,
                    f"{eager_time_per_batch:.3f}",
                    f"{sdpa_time_per_batch:.3f}",
                    f"{speedup:.3f}",
                    f"{eager_max_mem:.3f}",
                    f"{sdpa_max_mem:.3f}",
                    f"{mem_saved:.3f}",
                )
            )

    output_file.close()
