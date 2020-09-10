---
tags:
- exbert
license: apache-2.0
---

# ouBioBERT-Base, Uncased

Bidirectional Encoder Representations from Transformers for Biomedical Text Mining by Osaka University (ouBioBERT) is a language model based on the BERT-Base (Devlin, et al., 2019) architecture. We pre-trained ouBioBERT on PubMed abstracts from the PubMed baseline (ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline) via our method.  

The details of the pre-training procedure can be found in Wada, et al. (2020).  

## Evaluation

We evaluated the performance of ouBioBERT in terms of the biomedical language understanding evaluation (BLUE) benchmark (Peng, et al., 2019). The numbers are mean (standard deviation) on five different random seeds.  


| Dataset         |  Task Type                   |  Score       |
|:----------------|:-----------------------------|-------------:|
| MedSTS          |  Sentence similarity         |  84.9 (0.6)  |
| BIOSSES         |  Sentence similarity         |  92.3 (0.8)  |
| BC5CDR-disease  |  Named-entity recognition    |  87.4 (0.1)  |
| BC5CDR-chemical |  Named-entity recognition    |  93.7 (0.2)  |
| ShARe/CLEFE     |  Named-entity recognition    |  80.1 (0.4)  |
| DDI             |  Relation extraction         |  81.1 (1.5)  |
| ChemProt        |  Relation extraction         |  75.0 (0.3)  |
| i2b2 2010       |  Relation extraction         |  74.0 (0.8)  |
| HoC             |  Document classification     |  86.4 (0.5)  |
| MedNLI          |  Inference                   |  83.6 (0.7)  |
| **Total**       |  Macro average of the scores |**83.8 (0.3)**|


## Code for Fine-tuning
We made the source code for fine-tuning freely available at [our repository](https://github.com/sy-wada/blue_benchmark_with_transformers).

## Citation

If you use our work in your research, please kindly cite the following paper:  

```bibtex
@misc{2005.07202,
Author = {Shoya Wada and Toshihiro Takeda and Shiro Manabe and Shozo Konishi and Jun Kamohara and Yasushi Matsumura},
Title = {A pre-training technique to localize medical BERT and enhance BioBERT},
Year = {2020},
Eprint = {arXiv:2005.07202},
}
```

<a href="https://huggingface.co/exbert/?model=seiya/oubiobert-base-uncased&sentence=Coronavirus%20disease%20(COVID-19)%20is%20caused%20by%20SARS-COV2%20and%20represents%20the%20causative%20agent%20of%20a%20potentially%20fatal%20disease%20that%20is%20of%20great%20global%20public%20health%20concern.">
	<img width="300px" src="https://hf-dinosaur.huggingface.co/exbert/button.png">
</a>
