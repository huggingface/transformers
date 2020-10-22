# LXMERT

## Model Description

[LXMERT](https://arxiv.org/abs/1908.07490) is a pre-trained multimodal transformer. The model takes an image and a sentence as input and compute cross-modal representions. The model is converted from [LXMERT github](https://github.com/airsplay/lxmert) by [Antonio Mendoza](https://avmendoza.info/) and is authored by [Hao Tan](https://www.cs.unc.edu/~airsplay/).

![](./lxmert_model-1.jpg?raw=True)

## Usage


## Training Data and Prodcedure
The model is jointly trained on multiple vision-and-language datasets.
We included two image captioning datsets (i.e., [MS COCO](http://cocodataset.org/#home), [Visual Genome](https://visualgenome.org/)) and three image-question answering datasets (i.e.,  [VQA](https://visualqa.org/), [GQA](https://cs.stanford.edu/people/dorarad/gqa/), [VG QA](https://github.com/yukezhu/visual7w-toolkit)). The model is pre-trained on the above datasets  for 20 epochs (roughly 670K iterations with batch size 256), which takes around 8 days on 4 Titan V cards. The details of training could be found in the [LXMERT paper](https://arxiv.org/pdf/1908.07490.pdf).

## Eval Results
| Split            | [VQA](https://visualqa.org/)     | [GQA](https://cs.stanford.edu/people/dorarad/gqa/)     | [NLVR2](http://lil.nlp.cornell.edu/nlvr/)  |
|-----------       |:----:   |:---:    |:------:|
| Local Validation | 69.90%  | 59.80%  | 74.95% |
| Test-Dev         | 72.42%  | 60.00%  | 74.45% (Test-P) |
| Test-Standard    | 72.54%  | 60.33%  | 76.18% (Test-U) |


## Reference
```bibtex
@inproceedings{tan2019lxmert,
  title={LXMERT: Learning Cross-Modality Encoder Representations from Transformers},
  author={Tan, Hao and Bansal, Mohit},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year={2019}
}
```


