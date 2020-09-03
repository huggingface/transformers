#!/usr/bin/env python3
from transformers.modeling_bert import BertScriptableModel
from transformers import BertConfig, BertModel, PyTorchBenchmark, PyTorchBenchmarkArguments
import torch


def get_model(torchscript=False, device="cpu", config=None, max_seq_length=None):
    input_ids = torch.ones((1, max_seq_length), device=device, dtype=torch.long)
    if not torchscript:
        model = BertModel(config).to(device).eval()
        traced_model = torch.jit.trace(model, input_ids)
        return traced_model
    model = BertScriptableModel(config).to(device).eval()
    return torch.jit.script(model)


def get_input_ids(input_tensor_type="single", config=None, batch_size=None, sequence_length=None, device="cpu"):
    if input_tensor_type == "single":
        return [torch.randint(config.vocab_size, (batch_size, sequence_length), dtype=torch.long, device=device)]
    elif input_tensor_type == "batched":
        num_batches = batch_size // 8
        sequence_lengths = [torch.randint(1, sequence_length, (1,)).item() for i in range(num_batches)]
        return [torch.randint(config.vocab_size, (10, sequence_length), dtype=torch.long, device=device) for sequence_length in sequence_lengths]
    elif input_tensor_type == "multiple":
        sequence_lengths = [torch.randint(1, sequence_length, (1,)).item() for i in range(batch_size)]
        return [torch.randint(config.vocab_size, (1, sequence_length), dtype=torch.long, device=device) for sequence_length in sequence_lengths]
    else:
        raise ValueError(f"{input_tensor_type} does not exist.")


def get_inference_func(device, config, sequence_length, batch_size, input_tensor_type, torchscript):
    model = get_model(torchscript, device, config, sequence_length)
    input_ids = get_input_ids(input_tensor_type=input_tensor_type, config=config, batch_size=batch_size, sequence_length=sequence_length, device=device)

    @torch.no_grad()
    def func():
        for i in input_ids:
            result = model(i)
        return result
    return func


def run_benchmark(batch_sizes, sequence_lengths, input_tensor_type="multiple", torchscript=True):
    config = BertConfig.from_pretrained("bert-base-uncased")
    args = PyTorchBenchmarkArguments(models=[f"Type: {input_tensor_type} - Script: {torchscript}"], no_memory=True, sequence_lengths=sequence_lengths, batch_sizes=batch_sizes, no_multi_process=True, repeat=1, torchscript=True, no_env_print=True)
    device = args.device
    benchmark = PyTorchBenchmark(args, configs=[config])

    def _prepare_inference_func(model_name, batch_size, sequence_length):
        return get_inference_func(device=device, config=config, sequence_length=sequence_length, batch_size=batch_size, input_tensor_type=input_tensor_type, torchscript=torchscript)

    benchmark._prepare_inference_func = _prepare_inference_func
    benchmark.run()


torch.manual_seed(0)
run_benchmark([500, 2500], [128, 512])
torch.manual_seed(0)
run_benchmark([500, 2500], [128, 512], torchscript=False)

torch.manual_seed(0)
run_benchmark([512, 4096], [128, 512], input_tensor_type="batched")
torch.manual_seed(0)
run_benchmark([512, 4096], [128, 512], torchscript=False, input_tensor_type="batched")
