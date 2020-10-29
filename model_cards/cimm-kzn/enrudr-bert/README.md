---
language:
- ru
- en
---
## RuDR-BERT

EnRuDR-BERT - Multilingual, Cased, which pretrained on the raw part of the RuDReC corpus (1.4M reviews) and collecting of consumer comments on drug administration from [2]. Pre-training was based on the [original BERT code](https://github.com/google-research/bert) provided by Google. In particular, Multi-BERT was for used for initialization; vocabulary of Russian subtokens and parameters are the same as in Multi-BERT. Training details are described in our paper. \
   link: https://yadi.sk/d/-PTn0xhk1PqvgQ
   

## Citing & Authors

If you find this repository helpful, feel free to cite our publication:

[1] Tutubalina E, Alimova I, Miftahutdinov Z, et al. The Russian Drug Reaction Corpus and Neural Models for Drug Reactions and Effectiveness Detection in User Reviews.//Bioinformatics. - 2020. 
   
   preprint: https://arxiv.org/abs/2004.03659
```
@article{10.1093/bioinformatics/btaa675,
    author = {Tutubalina, Elena and Alimova, Ilseyar and Miftahutdinov, Zulfat and Sakhovskiy, Andrey and Malykh, Valentin and Nikolenko, Sergey},
    title = "{The Russian Drug Reaction Corpus and Neural Models for Drug Reactions and Effectiveness Detection in User Reviews}",
    journal = {Bioinformatics},
    year = {2020},
    month = {07},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btaa675},
    url = {https://doi.org/10.1093/bioinformatics/btaa675},
    note = {btaa675},
    eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btaa675/33539752/btaa675.pdf},
} 
```
[2] Tutubalina, EV and Miftahutdinov, Z Sh and Nugmanov, RI and Madzhidov, TI and Nikolenko, SI and Alimova, IS and Tropsha, AE Using semantic analysis of texts for the identification of drugs with similar therapeutic effects.//Russian Chemical Bulletin. – 2017. – Т. 66. – №. 11. – С. 2180-2189.
   [link to paper](https://www.researchgate.net/profile/Elena_Tutubalina/publication/323751823_Using_semantic_analysis_of_texts_for_the_identification_of_drugs_with_similar_therapeutic_effects/links/5bf7cfc3299bf1a0202cbc1f/Using-semantic-analysis-of-texts-for-the-identification-of-drugs-with-similar-therapeutic-effects.pdf)
```
@article{tutubalina2017using,
    title={Using semantic analysis of texts for the identification of drugs with similar therapeutic effects},
    author={Tutubalina, EV and Miftahutdinov, Z Sh and Nugmanov, RI and Madzhidov, TI and Nikolenko, SI and Alimova, IS and Tropsha, AE},
    journal={Russian Chemical Bulletin},
    volume={66},
    number={11},
    pages={2180--2189},
    year={2017},
    publisher={Springer}
}
```
