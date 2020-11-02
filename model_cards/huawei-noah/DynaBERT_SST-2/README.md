## DynaBERT: Dynamic BERT with Adaptive Width and Depth

* DynaBERT can flexibly adjust the size and latency by selecting adaptive width and depth, and 
the subnetworks of it have competitive performances as other similar-sized compressed models.
The training process of DynaBERT includes first training a width-adaptive BERT and then 
allowing both adaptive width and depth using knowledge distillation. 

* This code is modified based on the repository developed by Hugging Face: [Transformers v2.1.1](https://github.com/huggingface/transformers/tree/v2.1.1), and is released in [GitHub](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/DynaBERT).

### Reference
Lu Hou, Zhiqi Huang, Lifeng Shang, Xin Jiang, Xiao Chen, Qun Liu.
[DynaBERT: Dynamic BERT with Adaptive Width and Depth](https://arxiv.org/abs/2004.04037).
```
@inproceedings{hou2020dynabert,
  title = {DynaBERT: Dynamic BERT with Adaptive Width and Depth},
  author = {Lu Hou, Zhiqi Huang, Lifeng Shang, Xin Jiang, Xiao Chen, Qun Liu},  
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2020}
}
```
