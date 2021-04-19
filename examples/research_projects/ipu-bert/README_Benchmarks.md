# BERT Training Benchmarking on IPUs using PyTorch

This README describes how to run PyTorch BERT models for throughput benchmarking on the Mk2 IPU.

## Preparation

Follow the installation instructions in applications/pytorch/bert/README.md.

Follow the instructions at the same location for obtaining and processing the dataset. Ensure the $DATASETS_DIR environment variable points to the location of the dataset.

Run the following commands from inside the applications/pytorch/bert/ directory.

## Benchmarks

### Pretrain BERT-Base Sequence Length 128

1 x IPU-POD16

```console
python bert.py --config pretrain_base_128 --training-steps 10
```

### Pretrain BERT-Base Sequence Length 384

1 x IPU-POD16

```console
python bert.py --config pretrain_base_384 --training-steps 10
```

### Pretrain BERT-Large Sequence Length 128

1 x IPU-POD16

```console
python bert.py --config pretrain_large_128 --training-steps 10
```

### Pretrain BERT-Large Sequence Length 384

1 x IPU-POD16

```console
python bert.py --config pretrain_large_384 --training-steps 10
```
