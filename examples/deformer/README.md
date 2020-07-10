# **DeFormer**: **De**composing Pre-trained Trans**former**s for Faster Question Answering

This example is the [DeFormer paper](https://awk.ai/assets/deformer.pdf) (ACL 2020), adapted from the [author's codebase](https://github.com/StonyBrookNLP/deformer)

<img style="margin:auto;width:50%" src="https://awk.ai/assets/deformer-sketch.png" alt="deformer"/>

# TODOs

- [ ] use HF preprocessing (use HF nlp library)
- [ ] convert original TF DeFormer to HF version
- [ ] convert pre-trained checkpoints 
- [ ] compare and test accuracy for SQuAD, RACE, and BoolQ
- [ ] prepare demo and upload to model cards

## Citation

If you find our work useful to your research, please consider using the following citation:

````bib
@inproceedings{cao-etal-2020-deformer,
    title = "{D}e{F}ormer: Decomposing Pre-trained Transformers for Faster Question Answering",
    author = "Cao, Qingqing  and
      Trivedi, Harsh  and
      Balasubramanian, Aruna  and
      Balasubramanian, Niranjan",
    booktitle = "Proceedings of the 58th Annual Mdeformering of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.411",
    pages = "4487--4497",
}
````

