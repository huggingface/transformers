# DeeBERT: Early Exiting for *BERT

This is the code base for the paper [DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference](https://www.aclweb.org/anthology/2020.acl-main.204/).




## Installation

First, install [pytorch](https://pytorch.org/) and the [transformer library](https://github.com/huggingface/transformers/blob/master/README.md)

There are a few other packages to install. Assuming we are in `transformers/examples/deebert`, simply run

```
pip install -r requirements.txt
```


## Usage

There are three scripts in the folder which can be run directly.

In each script, there are several things to modify before running:

* path to the GLUE dataset.
* path for saving fine-tuned models. Default: `./saved_models`.
* path for saving evaluation results. Default: `./results`. Results are printed to stdout and also saved to `npy` files in this directory to facilitate plotting figures and further analyses.
* model_type (bert or roberta)
* model_size (base or large)
* dataset (SST-2, MRPC, RTE, QNLI, QQP, or MNLI)

#### train_deebert.sh

This is for fine-tuning DeeBERT models.

#### eval_deebert.sh

This is for evaluating each exit layer for fine-tuned DeeBERT models.

#### entropy_eval.sh

This is for evaluating fine-tuned DeeBERT models, given a number of different early exit entropy thresholds.



## Citation

Please cite our paper if you find the resource useful:
```
@inproceedings{xin-etal-2020-deebert,
    title = "{D}ee{BERT}: Dynamic Early Exiting for Accelerating {BERT} Inference",
    author = "Xin, Ji  and
      Tang, Raphael  and
      Lee, Jaejun  and
      Yu, Yaoliang  and
      Lin, Jimmy",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.204",
    pages = "2246--2251",
    abstract = "Large-scale pre-trained language models such as BERT have brought significant improvements to NLP applications. However, they are also notorious for being slow in inference, which makes them difficult to deploy in real-time applications. We propose a simple but effective method, DeeBERT, to accelerate BERT inference. Our approach allows samples to exit earlier without passing through the entire model. Experiments show that DeeBERT is able to save up to {\textasciitilde}40{\%} inference time with minimal degradation in model quality. Further analyses show different behaviors in the BERT transformer layers and also reveal their redundancy. Our work provides new ideas to efficiently apply deep transformer-based models to downstream tasks. Code is available at https://github.com/castorini/DeeBERT.",
}
```

