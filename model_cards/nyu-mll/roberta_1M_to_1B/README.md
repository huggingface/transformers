# RoBERTa Pretrained on Smaller Datasets

We pretrain RoBERTa on smaller datasets (1M, 10M, 100M, 1B tokens). We release 3 models with lowest perplexities for each pretraining data size out of 25 runs (or 10 in the case of 1B tokens). The pretraining data reproduces that of BERT: We combine English Wikipedia and a reproduction of BookCorpus using texts from smashwords in a ratio of approximately 3:1.

### Hyperparameters and Validation Perplexity

The hyperparameters and validation perplexities corresponding to each model are as follows:

| Model Name               | Training Size | Model Size | Max Steps | Batch Size | Validation Perplexity |
|--------------------------|---------------|------------|-----------|------------|-----------------------|
| [roberta-base-1B-1][link-roberta-base-1B-1]        | 1B            | BASE       | 100K      | 512        | 3.93                  |
| [roberta-base-1B-2][link-roberta-base-1B-2]        | 1B            | BASE       | 31K       | 1024       | 4.25                  |
| [roberta-base-1B-3][link-roberta-base-1B-3]        | 1B            | BASE       | 31K       | 4096       | 3.84                  |
| [roberta-base-100M-1][link-roberta-base-100M-1]      | 100M          | BASE       | 100K      | 512        | 4.99                  |
| [roberta-base-100M-2][link-roberta-base-100M-2]      | 100M          | BASE       | 31K       | 1024       | 4.61                  |
| [roberta-base-100M-3][link-roberta-base-100M-3]      | 100M          | BASE       | 31K       | 512        | 5.02                  |
| [roberta-base-10M-1][link-roberta-base-10M-1]       | 10M           | BASE       | 10K       | 1024       | 11.31                 |
| [roberta-base-10M-2][link-roberta-base-10M-2]       | 10M           | BASE       | 10K       | 512        | 10.78                 |
| [roberta-base-10M-3][link-roberta-base-10M-3]       | 10M           | BASE       | 31K       | 512        | 11.58                 |
| [roberta-med-small-1M-1][link-roberta-med-small-1M-1]   | 1M            | MED-SMALL  | 100K      | 512        | 153.38                |
| [roberta-med-small-1M-2][link-roberta-med-small-1M-2]   | 1M            | MED-SMALL  | 10K       | 512        | 134.18                |
| [roberta-med-small-1M-3][link-roberta-med-small-1M-3]   | 1M            | MED-SMALL  | 31K       | 512        | 139.39                |

The hyperparameters corresponding to model sizes mentioned above are as follows:

| Model Size | L  | AH | HS  | FFN  | P    |
|------------|----|----|-----|------|------|
| BASE       | 12 | 12 | 768 | 3072 | 125M |
| MED-SMALL  | 6  | 8  | 512 | 2048 | 45M  |

(AH = number of attention heads; HS = hidden size; FFN = feedforward network dimension; P = number of parameters.)

For other hyperparameters, we select:
- Peak Learning rate: 5e-4
- Warmup Steps: 6% of max steps
- Dropout: 0.1

[link-roberta-med-small-1M-1]: https://huggingface.co/nyu-mll/roberta-med-small-1M-1
[link-roberta-med-small-1M-2]: https://huggingface.co/nyu-mll/roberta-med-small-1M-2
[link-roberta-med-small-1M-3]: https://huggingface.co/nyu-mll/roberta-med-small-1M-3
[link-roberta-base-10M-1]: https://huggingface.co/nyu-mll/roberta-base-10M-1
[link-roberta-base-10M-2]: https://huggingface.co/nyu-mll/roberta-base-10M-2
[link-roberta-base-10M-3]: https://huggingface.co/nyu-mll/roberta-base-10M-3
[link-roberta-base-100M-1]: https://huggingface.co/nyu-mll/roberta-base-100M-1
[link-roberta-base-100M-2]: https://huggingface.co/nyu-mll/roberta-base-100M-2
[link-roberta-base-100M-3]: https://huggingface.co/nyu-mll/roberta-base-100M-3
[link-roberta-base-1B-1]: https://huggingface.co/nyu-mll/roberta-base-1B-1
[link-roberta-base-1B-2]: https://huggingface.co/nyu-mll/roberta-base-1B-2
[link-roberta-base-1B-3]: https://huggingface.co/nyu-mll/roberta-base-1B-3
