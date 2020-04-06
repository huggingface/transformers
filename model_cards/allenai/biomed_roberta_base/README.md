# BioMedical-RoBERTa-base

BioMedical-RoBERTa-base is a language model based on the RoBERTa-base (Liu et. al, 2019) architecture. We adapt RoBERTa-base to 2.68 million scientific papers from the [Semantic Scholar](https://www.semanticscholar.org) corpus via continued pretraining. This amounts to 7.55B tokens and 47GB of data. We use the full text of the papers in training, not just abstracts.

Specific details of the adaptive pretraining procedure can be found in Gururangan et. al, 2020. 

BioMed-RoBERTa achieves state-of-the-art or competitive performance on text classification, NER, and relation extraction tasks in the biomedical domain, consistently outperforming SciBERT (Beltagy et. al, 2019).

## Citation

If using this model, please cite the following paper:

@inproceedings{domains,
 author = {Suchin Gururangan and Ana Marasović and Swabha Swayamdipta and Kyle Lo and Iz Beltagy and Doug Downey and Noah A. Smith},
 title = {Don't Stop Pretraining: Adapt Language Models to Domains and Tasks},
 year = {2020},
 booktitle = {Proceedings of ACL},
}

