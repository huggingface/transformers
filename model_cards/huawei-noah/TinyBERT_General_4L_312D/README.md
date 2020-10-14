TinyBERT: Distilling BERT for Natural Language Understanding
======== 
TinyBERT is 7.5x smaller and 9.4x faster on inference than BERT-base and achieves competitive performances in the tasks of natural language understanding. It performs a novel transformer distillation at both the pre-training and task-specific learning stages. In general distillation, we use the original BERT-base without fine-tuning as the teacher and a large-scale text corpus as the learning data. By performing the Transformer distillation on the text from general domain, we obtain a general TinyBERT which provides a good initialization for the task-specific distillation. We here provide the general TinyBERT for your tasks at hand.

For more details about the techniques of TinyBERT, refer to our paper:
[TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)


Citation
========
If you find TinyBERT useful in your research, please cite the following paper:
```
@article{jiao2019tinybert,
  title={Tinybert: Distilling bert for natural language understanding},
  author={Jiao, Xiaoqi and Yin, Yichun and Shang, Lifeng and Jiang, Xin and Chen, Xiao and Li, Linlin and Wang, Fang and Liu, Qun},
  journal={arXiv preprint arXiv:1909.10351},
  year={2019}
}
```
