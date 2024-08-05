# Self-training

This is an implementation of the self-training algorithm (without task augmentation) in the [EMNLP 2021](https://2021.emnlp.org/) paper: [STraTA: Self-Training with Task Augmentation for Better Few-shot Learning](https://arxiv.org/abs/2109.06270). Please check out https://github.com/google-research/google-research/tree/master/STraTA for the original codebase.

**Note**: The code can be used as a tool for automatic data labeling.

## Table of Contents

   * [Installation](#installation)
   * [Self-training](#self-training)
      * [Running self-training with a base model](#running-self-training-with-a-base-model)
      * [Hyperparameters for self-training](#hyperparameters-for-self-training)
      * [Distributed training](#distributed-training)
   * [Demo](#demo)
   * [How to cite](#how-to-cite)

## Installation
This repository is tested on Python 3.8+, PyTorch 1.10+, and the 🤗 Transformers 4.16+.

You should install all necessary Python packages in a [virtual environment](https://docs.python.org/3/library/venv.html). If you are unfamiliar with Python virtual environments, please check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

Below, we create a virtual environment with the [Anaconda Python distribution](https://www.anaconda.com/products/distribution) and activate it.
```sh
conda create -n strata python=3.9
conda activate strata
```
Next, you need to install 🤗 Transformers. Please refer to [🤗 Transformers installation page](https://github.com/huggingface/transformers#installation) for a detailed guide.
```sh
pip install transformers
```
Finally, install all necessary Python packages for our self-training algorithm.

```sh
pip install -r STraTA/selftraining/requirements.txt
```
This will install PyTorch as a backend.

## Self-training
### Running self-training with a base model
The following example code shows how to run our self-training algorithm with a base model (e.g., `BERT`) on the `SciTail` science entailment dataset, which has two classes `['entails', 'neutral']`. We assume that you have a data directory that includes some training data (e.g., `train.csv`), evaluation data (e.g., `eval.csv`), and unlabeled data (e.g., `infer.csv`).

```python
import os
from selftraining import selftrain

data_dir = '/path/to/your/data/dir'
parameters_dict = {
    'max_selftrain_iterations': 100,
    'model_name_or_path': '/path/to/your/base/model',  # could be the id of a model hosted by 🤗 Transformers
    'output_dir': '/path/to/your/output/dir',
    'train_file': os.path.join(data_dir, 'train.csv'),
    'infer_file': os.path.join(data_dir, 'infer.csv'),
    'eval_file': os.path.join(data_dir, 'eval.csv'),
    'eval_strategy': 'steps',
    'task_name': 'scitail',
    'label_list': ['entails', 'neutral'],
    'per_device_train_batch_size': 32,
    'per_device_eval_batch_size': 8,
    'max_length': 128,
    'learning_rate': 2e-5,
    'max_steps': 100000,
    'eval_steps': 1,
    'early_stopping_patience': 50,
    'overwrite_output_dir': True,
    'do_filter_by_confidence': False,
    # 'confidence_threshold': 0.3,
    'do_filter_by_val_performance': True,
    'finetune_on_labeled_data': False,
    'seed': 42,
}
selftrain(**parameters_dict)
```

**Note**: We checkpoint periodically during self-training. In case of preemptions, just re-run the above script and self-training will resume from the latest iteration.

### Hyperparameters for self-training
If you have development data, you might want to tune some hyperparameters for self-training.
Below are hyperparameters that could provide additional gains for your task.

  - `finetune_on_labeled_data`: If set to `True`, the resulting model from each self-training iteration is further fine-tuned on the original labeled data before the next self-training iteration. Intuitively, this would give the model a chance to "correct" ifself after being trained on pseudo-labeled data.
  - `do_filter_by_confidence`: If set to `True`, the pseudo-labeled data in each self-training iteration is filtered based on the model confidence. For instance, if `confidence_threshold` is set to `0.3`, pseudo-labeled examples with a confidence score less than or equal to `0.3` will be discarded. Note that `confidence_threshold` should be greater or equal to `1/num_labels`, where `num_labels` is the number of class labels. Filtering out the lowest-confidence pseudo-labeled examples could be helpful in some cases.
  - `do_filter_by_val_performance`: If set to `True`, the pseudo-labeled data in each self-training iteration is filtered based on the current validation performance. For instance, if your validation performance is 80% accuracy, you might want to get rid of 20% of the pseudo-labeled data with the lowest the confidence scores.

### Distributed training
We strongly recommend distributed training with multiple accelerators. To activate distributed training, please try one of the following methods:

1. Run `accelerate config` and answer to the questions asked. This will save a `default_config.yaml` file in your cache folder for 🤗 Accelerate. Now, you can run your script with the following command:

```sh
accelerate launch your_script.py --args_to_your_script
```

2. Run your script with the following command:

```sh
python -m torch.distributed.launch --nnodes="{$NUM_NODES}" --nproc_per_node="{$NUM_TRAINERS}" --your_script.py --args_to_your_script
```

3. Run your script with the following command:

```sh
torchrun --nnodes="{$NUM_NODES}" --nproc_per_node="{$NUM_TRAINERS}" --your_script.py --args_to_your_script
```

## Demo
Please check out `run.sh` to see how to perform our self-training algorithm with a `BERT` Base model on the SciTail science entailment dataset using 8 labeled examples per class. You can configure your training environment by specifying `NUM_NODES` and `NUM_TRAINERS` (number of processes per node). To launch the script, simply run `source run.sh`.

## How to cite
If you extend or use this code, please cite the [paper](https://arxiv.org/abs/2109.06270) where it was introduced:

```bibtex
@inproceedings{vu-etal-2021-strata,
    title = "{ST}ra{TA}: Self-Training with Task Augmentation for Better Few-shot Learning",
    author = "Vu, Tu  and
      Luong, Minh-Thang  and
      Le, Quoc  and
      Simon, Grady  and
      Iyyer, Mohit",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.462",
    doi = "10.18653/v1/2021.emnlp-main.462",
    pages = "5715--5731",
}
```
