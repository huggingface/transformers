---
tags:
- exbert
---

## CS224n SQuAD2.0 Project Dataset
The goal of this model is to save CS224n students GPU time when establishing
baselines to beat for the [Default Final Project](http://web.stanford.edu/class/cs224n/project/default-final-project-handout.pdf).
The training set used to fine-tune this model is the same as
the [official one](https://rajpurkar.github.io/SQuAD-explorer/); however,
evaluation and model selection were performed using roughly half of the official
dev set, 6078 examples, picked at random. The data files can be found at
<https://github.com/elgeish/squad/tree/master/data> — this is the Winter 2020
version. Given that the official SQuAD2.0 dev set contains the project's test
set, students must make sure not to use the official SQuAD2.0 dev set in any way
— including the use of models fine-tuned on the official SQuAD2.0, since they
used the official SQuAD2.0 dev set for model selection.

<a href="https://huggingface.co/exbert/?model=elgeish/cs224n-squad2.0-albert-xxlarge-v1">
	<img width="300px" src="https://cdn-media.huggingface.co/exbert/button.png">
</a>

## Results
```json
{
  "exact": 85.93287265547877,
  "f1": 88.91258331187983,
  "total": 6078,
  "HasAns_exact": 84.36426116838489,
  "HasAns_f1": 90.58786301361013,
  "HasAns_total": 2910,
  "NoAns_exact": 87.37373737373737,
  "NoAns_f1": 87.37373737373737,
  "NoAns_total": 3168,
  "best_exact": 85.93287265547877,
  "best_exact_thresh": 0.0,
  "best_f1": 88.91258331187993,
  "best_f1_thresh": 0.0
}
```

## Notable Arguments
```json
{
  "do_lower_case": true,
  "doc_stride": 128,
  "fp16": false,
  "fp16_opt_level": "O1",
  "gradient_accumulation_steps": 24,
  "learning_rate": 3e-05,
  "max_answer_length": 30,
  "max_grad_norm": 1,
  "max_query_length": 64,
  "max_seq_length": 512,
  "model_name_or_path": "albert-xxlarge-v1",
  "model_type": "albert",
  "num_train_epochs": 4,
  "per_gpu_train_batch_size": 1,
  "save_steps": 1000,
  "seed": 42,
  "train_batch_size": 1,
  "version_2_with_negative": true,
  "warmup_steps": 814,
  "weight_decay": 0
}
```

## Environment Setup
```json
{
  "transformers": "2.5.1",
  "pytorch": "1.4.0=py3.6_cuda10.1.243_cudnn7.6.3_0",
  "python": "3.6.5=hc3d631a_2",
  "os": "Linux 4.15.0-1060-aws #62-Ubuntu SMP Tue Feb 11 21:23:22 UTC 2020 x86_64 x86_64 x86_64 GNU/Linux",
  "gpu": "Tesla V100-SXM2-16GB"
}
```

## How to Cite
```BibTeX
@misc{elgeish2020gestalt,
  title={Gestalt: a Stacking Ensemble for SQuAD2.0},
  author={Mohamed El-Geish},
  journal={arXiv e-prints},
  archivePrefix={arXiv},
  eprint={2004.07067},
  year={2020},
}
```

## Related Models
* [elgeish/cs224n-squad2.0-albert-base-v2](https://huggingface.co/elgeish/cs224n-squad2.0-albert-base-v2)
* [elgeish/cs224n-squad2.0-albert-large-v2](https://huggingface.co/elgeish/cs224n-squad2.0-albert-large-v2)
* [elgeish/cs224n-squad2.0-distilbert-base-uncased](https://huggingface.co/elgeish/cs224n-squad2.0-distilbert-base-uncased)
* [elgeish/cs224n-squad2.0-roberta-base](https://huggingface.co/elgeish/cs224n-squad2.0-roberta-base)
