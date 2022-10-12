import argparse
import logging
import os

import torch
from torch.profiler import ProfilerActivity, profile, record_function


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-batches",
        type=int,
        default=50,
        help="",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="",
    )
    parser.add_argument(
        "--avg-seqlen",
        type=int,
        default=256,
        help="",
    )
    parser.add_argument(
        "--max-seqlen",
        type=int,
        default=256,
        help="",
    )
    parser.add_argument(
        "--seqlen-stdev",
        type=int,
        default=10,
        help="",
    )
    parser.add_argument(
        "--is-cuda",
        type=str2bool,
        default=True,
        help="",
    )
    parser.add_argument(
        "--is-large",
        type=str2bool,
        default=True,
        help="",
    )
    parser.add_argument(
        "--is-half",
        type=str2bool,
        default=True,
        help="",
    )
    parser.add_argument(
        "--use-mask",
        type=str2bool,
        default=True,
        help="",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=-1,
        help="num layers to run. if -1, then use all layers",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",  # + allows multiple args as lst
        default=["hf", "bt"],  # all possible benchmarks available.
        help="Choose choice of benchmarks.",
    )
    return parser


def numerical_test(tensors1, tensors2, rtol, atol):
    """
    truth_tensors is the source of truth.
    test_dict looks like
    [
        (name, out_tensors, atol, rtol),
        ...
    ]
    """
    assert len(tensors1) == len(tensors2)
    n_failures = 0
    max_diff = 0
    for tensor1, tensor2 in zip(tensors1, tensors2):
        # print(tensor1, tensor2)
        max_diff = max(max_diff, torch.max(torch.abs(tensor1 - tensor2)))
        if not torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol):
            n_failures += 1

    num_tested = len(tensors1)
    return n_failures, max_diff, num_tested


def benchmark_torch_function(is_cuda, f, inputs, masks, use_mask=True):
    """
    Benchmark torch on gpu or cpu
    """
    if use_mask and masks is not None:
        assert len(inputs) == len(masks)
    if not use_mask:
        masks = [None for _ in range(len(masks))]
    iters = len(inputs)
    if is_cuda:
        f(inputs[0], attention_mask=masks[0])
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for x, mask in zip(inputs, masks):
            f(x, attention_mask=mask)
        end_event.record()
        torch.cuda.synchronize()
        return (start_event.elapsed_time(end_event) * 1.0e-3) / iters
    else:
        import time

        f(inputs[0], attention_mask=masks[0])
        start_event = time.perf_counter()
        for x, mask in zip(inputs, masks):
            f(x, attention_mask=mask)
        end_event = time.perf_counter()
        return (end_event - start_event) / iters


def get_outputs(f, inputs, masks, use_mask=True):
    outputs = [f(x, attention_mask=mask if use_mask else None).logits for x, mask in zip(inputs, masks)]
    return outputs


def get_batch(batch_size, avg_seqlen, max_sequence_length, seqlen_stdev, vocab_size, pad_idx=0):
    mean_tensor = torch.Tensor([avg_seqlen]).expand(batch_size)
    stdev_tensor = torch.Tensor([seqlen_stdev]).expand(batch_size)
    lengths = torch.normal(mean_tensor, stdev_tensor).to(torch.int)
    lengths = torch.clamp(lengths, min=0, max=max_sequence_length)
    tokens = torch.full(
        (batch_size, max_sequence_length),
        pad_idx,
    )
    for i in range(batch_size):
        tokens[i, : lengths[i]] = torch.randint(
            pad_idx + 1,
            vocab_size - 1,
            size=(lengths[i],),
        )
    mask = torch.full(
        (batch_size, max_sequence_length),
        0,
    )
    for i in range(batch_size):
        mask[i, : lengths[i]] = 1
    return tokens, lengths, mask


def setup_logger(filename="log.csv"):

    # create log file
    if os.path.exists(filename):
        os.remove(filename)
        open(filename, "w").close()
    else:
        open(filename, "w").close()
    # create logger
    lgr = logging.getLogger("logger name")
    lgr.setLevel(logging.DEBUG)  # log all escalated at and above DEBUG
    # add a file handler
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)  # ensure all messages are logged to file

    # create a formatter and set the formatter for the handler.
    # frmt = logging.Formatter('%(asctime)s,%(name)s,%(levelname)s,%(message)s')
    frmt = logging.Formatter("%(message)s")
    fh.setFormatter(frmt)

    # add the Handler to the logger
    lgr.addHandler(fh)
    return lgr


def cut_layers(hf_encoder, n_layers):
    """
    Alters hf encoder in place to use less layers
    """
    assert n_layers <= len(hf_encoder.layer)
    hf_encoder.layer = torch.nn.ModuleList([hf_encoder.layer[i] for i in range(n_layers)])
    return hf_encoder


def benchmark(
    num_batches,
    batch_size,
    avg_seqlen,
    max_seqlen,
    seqlen_stdev,
    is_cuda,
    is_large,
    is_half,
    benchmarks,
    use_mask,
    num_layers,
):
    from transformers import AutoModelForSequenceClassification

    assert avg_seqlen <= max_seqlen

    log_file = "log.csv"
    lgr = setup_logger(filename=log_file)

    seed = 1234
    torch.manual_seed(seed)
    setup_str = (
        f"{num_batches} {batch_size} {avg_seqlen} {max_seqlen} {seqlen_stdev} {is_cuda} {is_large} {num_layers} {is_half} {benchmarks} {use_mask} {seed}"
    )
    lgr.info(setup_str)

    vocab_size = 25000
    numtype = torch.float16 if is_half else torch.float32
    model_name = "bert-large-cased" if is_large else "bert-base-cased"
    device = "cuda" if is_cuda else "cpu"

    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(device).to(numtype)
    if num_layers >= 0:
        model.bert.encoder = cut_layers(model.bert.encoder, num_layers)

    batches = [get_batch(batch_size, avg_seqlen, max_seqlen, seqlen_stdev, vocab_size) for _ in range(num_batches)]
    eval_inputs = torch.stack([batch[0] for batch in batches]).to(device)
    masks = torch.stack([batch[2] for batch in batches]).to(device)  # 0 == pad, 1 == keep

    # DEBUG
    # lgr.info(f"DEBUG")
    # eval_inputs = torch.Tensor([[[20464, 2069, 1]]]).to(device).to(torch.int)
    # masks = torch.Tensor([[[1, 1, 0]]]).to(device)
    # print(masks.shape, eval_inputs.shape)

    if "hf" in benchmarks:
        with torch.no_grad():
            hf_t = benchmark_torch_function(is_cuda, model, eval_inputs, masks, use_mask=use_mask)
            out = get_outputs(model, eval_inputs, masks, use_mask=use_mask)
        lgr.info(f"HF time per batch {hf_t}")

    model.bert.encoder.to_fast()

    if "bt" in benchmarks:
        with torch.no_grad():
            bt_t = benchmark_torch_function(is_cuda, model, eval_inputs, masks, use_mask=use_mask)
            out2 = get_outputs(model, eval_inputs, masks, use_mask=use_mask)
        lgr.info(f"BT time per batch {bt_t}")

    if "hf" in benchmarks and "bt" in benchmarks:
        lgr.info(f"BT as prop of HF {hf_t / bt_t}")
        n_failures, max_diff, n_tested = numerical_test(out, out2, 0, 1e-2)  # absolute difference of 0.01 logit
        if n_failures == 0:
            test_str = f"HF/BT PASS"
        else:
            test_str = f"HF/BT test FAIL {n_failures}/{n_tested}. Max diff is {max_diff}"
        lgr.info(test_str)

    print(f"Logged to {log_file}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    benchmark(
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        avg_seqlen=args.avg_seqlen,
        max_seqlen=args.max_seqlen,
        seqlen_stdev=args.seqlen_stdev,
        is_cuda=args.is_cuda,
        is_large=args.is_large,
        is_half=args.is_half,
        benchmarks=args.benchmarks,
        use_mask=args.use_mask,
        num_layers=args.num_layers,
    )
