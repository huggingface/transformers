---
thumbnail: https://huggingface.co/front/thumbnails/allenai.png
---

# BioMed-RoBERTa-base

BioMed-RoBERTa-base is a language model based on the RoBERTa-base (Liu et. al, 2019) architecture. We adapt RoBERTa-base to 2.68 million scientific papers from the [Semantic Scholar](https://www.semanticscholar.org) corpus via continued pretraining. This amounts to 7.55B tokens and 47GB of data. We use the full text of the papers in training, not just abstracts.

Specific details of the adaptive pretraining procedure can be found in Gururangan et. al, 2020. 


## Evaluation

BioMed-RoBERTa achieves competitive performance to state of the art models on a number of NLP tasks in the biomedical domain (numbers are mean (standard deviation) over 3+ random seeds)


| Task         | Task Type           | RoBERTa-base | BioMed-RoBERTa-base |
|--------------|---------------------|--------------|---------------------|
| RCT-180K     | Text Classification | 86.4 (0.3)   | 86.9 (0.2)          |
| ChemProt     | Relation Extraction | 81.1 (1.1)   | 83.0 (0.7)          |
| JNLPBA       | NER                 | 74.3 (0.2)   | 75.2 (0.1)          |
| BC5CDR       | NER                 | 85.6 (0.1)   | 87.8 (0.1)          |
| NCBI-Disease | NER                 | 86.6 (0.3)   | 87.1 (0.8)          |

More evaluations TBD.

## Citation

If using this model, please cite the following paper:

```bibtex
@inproceedings{domains,
 author = {Suchin Gururangan and Ana Marasović and Swabha Swayamdipta and Kyle Lo and Iz Beltagy and Doug Downey and Noah A. Smith},
 title = {Don't Stop Pretraining: Adapt Language Models to Domains and Tasks},
 year = {2020},
 booktitle = {Proceedings of ACL},
}
```
