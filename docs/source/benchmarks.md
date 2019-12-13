# Benchmarks

This section is dedicated to the Benchmarks done by the library, both by maintainers, contributors and users. These 
benchmark will help keep track of the preformance improvements that are brought to our models across versions.

## Benchmarking all models for inference

As of version 2.1 we have benchmarked all models for inference, across many different settings: using PyTorch, with
and without TorchScript, using TensorFlow, with and without XLA. All of those tests were done across CPUs (except for
TensorFlow XLA) and GPUs.

The approach is detailed in the [following blogpost](https://medium.com/huggingface/benchmarking-transformers-pytorch-and-tensorflow-e2917fb891c2)

The results are available [here](https://docs.google.com/spreadsheets/d/1sryqufw2D0XlUH4sq3e9Wnxu5EAQkaohzrJbd5HdQ_w/edit?usp=sharing).

## TF2 with mixed precision, XLA, Distribution (@tlkh)

This work was done by [Timothy Liu](https://github.com/tlkh).

There are very positive results to be gained from the various TensorFlow 2.0 features:

- Automatic Mixed Precision (AMP)
- XLA compiler
- Distribution strategies (multi-GPU)

The benefits are listed here (tested on CoLA, MRPC, SST-2):

- AMP: Between 1.4x to 1.6x decrease in overall time without change in batch size
- AMP+XLA: Up to 2.5x decrease in overall time on SST-2 (larger dataset)
- Distribution: Between 1.4x to 3.4x decrease in overall time on 4xV100
- Combined: Up to 5.7x decrease in overall training time, or 9.1x training throughput

The model quality (measured by the validation accuracy) fluctuates slightly. Taking an average of 4 training runs 
on a single GPU gives the following results:

- CoLA: AMP results in slighter lower acc (0.820 vs 0.824)
- MRPC: AMP results in lower acc (0.823 vs 0.835)
- SST-2: AMP results in slighter lower acc (0.918 vs 0.922)

However, in a distributed setting with 4xV100 (4x batch size), AMP can yield in better results:

CoLA: AMP results in higher acc (0.828 vs 0.812)
MRPC: AMP results in lower acc (0.817 vs 0.827)
SST-2: AMP results in slightly lower acc (0.926 vs 0.929)

The benchmark script is available [here](https://github.com/NVAITC/benchmarking/blob/master/tf2/bert_dist.py).

Note: on some tasks (e.g. MRPC), the dataset is too small. The overhead due to the model compilation with XLA as well
as the distribution strategy setup does not speed things up. The XLA compile time is also the reason why although throughput 
can increase a lot (e.g. 2.7x for single GPU), overall (end-to-end) training speed-up is not as fast (as low as 1.4x)

The benefits as seen on SST-2 (larger dataset) is much clear.

All results can be seen on this [Google Sheet](https://docs.google.com/spreadsheets/d/1538MN224EzjbRL239sqSiUy6YY-rAjHyXhTzz_Zptls/edit#gid=960868445).
