import copy
import gc
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
import transformers
from black import nullcontext
from torch import optim
from torch.nn.utils import _stateless
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoModelForMaskedLM
from transformers import AutoModelForSeq2SeqLM
from transformers import BertConfig
from transformers import BigBirdConfig
from transformers import ReformerConfig

import torchdynamo
from torchdynamo.optimizations.backends import aot_autograd
from torchdynamo.optimizations.training import aot_autograd_speedup_strategy
from torchdynamo.testing import clone_me
from torchdynamo.testing import same

benchmarks = [
    (AutoConfig.from_pretrained("albert-base-v2"), AutoModelForMaskedLM, (8, 512), []),
    # (AutoConfig.from_pretrained("gpt2"), AutoModelForCausalLM, (4, 512), []),
    # (BertConfig(), AutoModelForMaskedLM, (4, 512), []),
    # (
    #     AutoConfig.from_pretrained("allenai/longformer-base-4096"),
    #     AutoModelForMaskedLM,
    #     (2, 1024),
    #     [torch.float16, torch.bfloat16],
    # ),  # hmm, nans with float16
    # (AutoConfig.from_pretrained("t5-small"), AutoModelForSeq2SeqLM, (4, 1024), [torch.float16, torch.bfloat16]), # Doesn't work with nn.utils._stateless for some reason...
    # (AutoConfig.from_pretrained("facebook/bert-base"), AutoModelForSeq2SeqLM, (4, 512), []), # Doesn't work with nn.utils._stateless for some reason...
    # (ReformerConfig(), AutoModelForMaskedLM, (8, 4096), []), # not sure...
    # (BigBirdConfig(attention_type="block_sparse"), AutoModelForMaskedLM, (2, 1024), []), # not sure...
    # (AutoConfig.from_pretrained("distilbert-base-uncased"),  AutoModelForMaskedLM, (8, 512), []), # encounters inf as a global value
]

torch.manual_seed(42)
device = "cuda"


class NullContext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def get_cur_memory():
    torch.cuda.synchronize()

    gc.collect()
    torch.cuda.empty_cache()
    stats = torch.cuda.memory_stats()
    peak_bytes_requirement = stats["allocated_bytes.all.current"]
    print(f"Current memory requirement: {peak_bytes_requirement / 1024 ** 3:.2f} GB")
    return peak_bytes_requirement


def collect_results(model, prediction, loss):
    results = []
    results.append(prediction)
    results.append(loss)
    grads = dict()
    for name, param in model.named_parameters():
        grads[name + ".grad"] = clone_me(param.grad)
    results.append(grads)
    # for example in example_inputs:
    #     if isinstance(example, (tuple, list)):
    #         for inp in example:
    #             if isinstance(inp, torch.Tensor):
    #                 results.append(clone_me(inp.grad))
    #     else:
    #         if isinstance(example, torch.Tensor):
    #             results.append(clone_me(example.grad))
    return results


@torchdynamo.skip
def forward_pass(mod, inputs, collect_outputs=True):
    return mod(*inputs)


@torchdynamo.skip
def forward_and_backward_pass(mod, inputs, collect_outputs=True):
    mod.zero_grad(True)
    pred = mod(*inputs)
    loss = pred.loss.abs().sum()
    loss.backward()
    if collect_outputs:
        return collect_results(mod, pred, loss)
    return None


def check_correctness(mod, train_inputs):
    optimize_ctx = torchdynamo.optimize(aot_autograd_speedup_strategy)
    torch.manual_seed(1337)
    correct_result = forward_and_backward_pass(copy.deepcopy(mod), train_inputs)

    torch.manual_seed(1337)
    correct_rerun_result = forward_and_backward_pass(copy.deepcopy(mod), train_inputs)
    if not same(correct_result, correct_rerun_result):
        print("INCORRECT - Variation in Eager runs itself")
        return False

    torch.manual_seed(1337)
    torchdynamo.reset()
    try:
        with optimize_ctx:
            new_result = forward_and_backward_pass(mod, train_inputs)
    except Exception:
        print("ERROR")
        return False

    if not same(correct_result, new_result):
        print("INCORRECT")
        return False
    return True


def bench_model(name, mod, train_inputs):
    if name == "eager":
        optimize_ctx = NullContext()
    else:
        optimize_ctx = torchdynamo.optimize(aot_autograd_speedup_strategy)

    with optimize_ctx:
        m = None
        for i in range(5):
            out = mod(*train_inputs).loss.abs().sum()
            if i == 4:
                m = get_cur_memory()
            out.backward()
        iters = 20
        torch.cuda.synchronize()
        begin = time.time()
        for _ in range(iters):
            forward_and_backward_pass(mod, train_inputs, False)
        torch.cuda.synchronize()
    t = (time.time() - begin) / iters
    print(name, (time.time() - begin) / iters)
    return t, m


model_header, dtype_header, nh, th, mh, tp, mp = (
    "model",
    "dtype",
    "name",
    "time (s)",
    "mem (GB)",
    "time %",
    "mem %",
)


numerical_diffs = []
results = []
for config, model_type, input_size, not_supported_dtypes in benchmarks:
    for dtype in [torch.float, torch.half, torch.bfloat16]:
        if dtype in not_supported_dtypes:
            continue
        for attr in dir(config):
            if "drop" in attr:
                setattr(
                    config, attr, 1e-60
                )  # So we can check for correct gradients without eliminating the dropout computation
        model = model_type.from_config(config).to(device, dtype=dtype)
        input_ids = torch.randint(0, config.vocab_size, input_size).to(device)
        decoder_ids = torch.randint(0, config.vocab_size, input_size).to(device)

        train_inputs = (input_ids, decoder_ids)
        # train_inputs = {"input_ids": input_ids, "labels": decoder_ids}

        print("Did accuracy pass", check_correctness(model, train_inputs))
        model_name = type(model).__name__
        t, m = bench_model("eager", model, train_inputs)
        results.append(
            {
                model_header: model_name,
                dtype_header: str(dtype),
                nh: "eager",
                th: t,
                mh: m / 2**30,
            }
        )
        with torch.jit.fuser("fuser2"):
            t, m = bench_model("dynamo_aot", model, train_inputs)
        results.append(
            {
                model_header: model_name,
                dtype_header: str(dtype),
                nh: "dynamo_aot",
                th: t,
                mh: m / 2**30,
            }
        )

        # calculate relative improvements
        base_r = results[-2]
        for r in results[-2:]:
            r[tp] = round(100 * (r[th] - base_r[th]) / base_r[th])
            r[mp] = round(100 * (r[mh] - base_r[mh]) / base_r[mh])
        print(pd.DataFrame(results[-2:]).to_markdown(index=False, floatfmt=".3f"))

        print()

print(results)
for model_name, dtype, err in numerical_diffs:
    print(f"Numerical differences in {model_name} - {dtype} found")
    print(err)
    print()

print(pd.DataFrame(results).to_markdown(index=False, floatfmt=".3f"))
