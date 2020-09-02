#!/usr/bin/env python3
from transformers.modeling_bert import BertScriptableModel
from transformers import BertConfig, BertModel, PyTorchBenchmark, PyTorchBenchmarkArguments
import torch


def get_model(torchscript=False, device="cpu", config=None):
    if not torchscript:
        return BertModel(config).to(device).eval()
    model = BertScriptableModel(config).to(device).eval()
    return torch.jit.script(model)


def get_input_ids(input_tensor_type="single_tensor", config=None, batch_size=None, sequence_length=None, device="cpu"):
    if input_tensor_type == "single_tensor":
        return [torch.randint(config.vocab_size, (batch_size, sequence_length), dtype=torch.long, device=device)]
    elif input_tensor_type == "batched_tensors":
        num_batches = batch_size // 8
        sequence_lengths = [torch.randint(1, sequence_length, (1,)).item() for i in range(num_batches)]
        print("Seq Length", sequence_lengths)
        return [torch.randint(config.vocab_size, (10, sequence_length), dtype=torch.long, device=device) for sequence_length in sequence_lengths]
    elif input_tensor_type == "multiple_tensors":
        sequence_lengths = [torch.randint(1, sequence_length, (1,)).item() for i in range(batch_size)]
        print("Seq Length", sequence_lengths)
        return [torch.randint(config.vocab_size, (1, sequence_length), dtype=torch.long, device=device) for sequence_length in sequence_lengths]
    else:
        raise ValueError(f"{input_tensor_type} does not exist.")


def get_inference_func(device, config, sequence_length, batch_size, input_tensor_type, torchscript):
    model = get_model(torchscript, device, config)
    input_ids = get_input_ids(input_tensor_type=input_tensor_type, config=config, batch_size=batch_size, sequence_length=sequence_length, device=device)

    def func():
        for inputs in input_ids:
            result = model(inputs)
        return result

    return func


def run_benchmark(batch_sizes, sequence_lengths, input_tensor_type="multiple_tensors", torchscript=True):
    config = BertConfig.from_pretrained("bert-base-uncased")
    args = PyTorchBenchmarkArguments(models=[f"Type: {input_tensor_type} - Script: {torchscript}"], no_memory=True, sequence_lengths=sequence_lengths, batch_sizes=batch_sizes, no_multi_process=True)
    device = args.device
    benchmark = PyTorchBenchmark(args, configs=[config])

    def _prepare_inference_func(model_name, batch_size, sequence_length):
        return get_inference_func(device=device, config=config, sequence_length=sequence_length, batch_size=batch_size, input_tensor_type=input_tensor_type, torchscript=torchscript)

    benchmark._prepare_inference_func = _prepare_inference_func
    benchmark.run()


run_benchmark([10], [10])
run_benchmark([10], [10], torchscript=False)
