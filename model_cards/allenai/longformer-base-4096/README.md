
# longformer-base-4096
[Longformer](https://arxiv.org/abs/2004.05150) is a transformer model for long documents. 

`longformer-base-4096` is a BERT-like model started from the RoBERTa checkpoint and pretrained for MLM on long documents. It supports sequences of length up to 4,096. 
 
Longformer uses a combination of a sliding window (local) attention and global attention. Global attention is user-configured based on the task to allow the model to learn task-specific representations.
Please refer to the examples in `modeling_longformer.py` and the paper for more details on how to set global attention.


### Citing

If you use `Longformer` in your research, please cite [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150).
```
@article{Beltagy2020Longformer,
  title={Longformer: The Long-Document Transformer},
  author={Iz Beltagy and Matthew E. Peters and Arman Cohan},
  journal={arXiv:2004.05150},
  year={2020},
}
```

`Longformer` is an open-source project developed by [the Allen Institute for Artificial Intelligence (AI2)](http://www.allenai.org).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.
