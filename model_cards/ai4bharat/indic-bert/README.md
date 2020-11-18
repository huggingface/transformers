---
language: en
license: mit
datasets:
- AI4Bharat IndicNLP Corpora
---

# IndicBERT

IndicBERT is a multilingual ALBERT model pretrained exclusively on 12 major Indian languages. It is pre-trained on our novel monolingual corpus of around 9 billion tokens and subsequently evaluated on a set of diverse tasks. IndicBERT has much fewer parameters than other multilingual models (mBERT, XLM-R etc.) while it also achieves a performance on-par or better than these models.

The 12 languages covered by IndicBERT are: Assamese, Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Oriya, Punjabi, Tamil, Telugu.

The code can be found [here](https://github.com/divkakwani/indic-bert). For more information, checkout our [project page](https://indicnlp.ai4bharat.org/) or our [paper](https://indicnlp.ai4bharat.org/papers/arxiv2020_indicnlp_corpus.pdf).



## Pretraining Corpus

We pre-trained indic-bert on AI4Bharat's monolingual corpus. The corpus has the following distribution of languages:


| Language          | as     | bn     | en     | gu     | hi     | kn     |         |
| ----------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------- |
| **No. of Tokens** | 36.9M  | 815M   | 1.34B  | 724M   | 1.84B  | 712M   |         |
| **Language**      | **ml** | **mr** | **or** | **pa** | **ta** | **te** | **all** |
| **No. of Tokens** | 767M   | 560M   | 104M   | 814M   | 549M   | 671M   | 8.9B    |



## Evaluation Results

IndicBERT is evaluated on IndicGLUE and some additional tasks. The results are summarized below. For more details about the tasks, refer our [official repo](https://github.com/divkakwani/indic-bert)

#### IndicGLUE

Task | mBERT | XLM-R | IndicBERT
-----| ----- | ----- | ------ 
News Article Headline Prediction | 89.58 | 95.52 | **95.87** 
Wikipedia Section Title Prediction| **73.66** | 66.33 | 73.31 
Cloze-style multiple-choice QA | 39.16 | 27.98 | **41.87** 
Article Genre Classification | 90.63 | 97.03 | **97.34** 
Named Entity Recognition (F1-score) | **73.24** | 65.93 | 64.47 
Cross-Lingual Sentence Retrieval Task | 21.46 | 13.74 | **27.12** 
Average | 64.62 | 61.09 | **66.66** 

#### Additional Tasks


Task | Task Type | mBERT | XLM-R | IndicBERT 
-----| ----- | ----- | ------ | ----- 
BBC News Classification | Genre Classification | 60.55 | **75.52** | 74.60 
IIT Product Reviews | Sentiment Analysis | 74.57 | **78.97** | 71.32 
IITP Movie Reviews | Sentiment Analaysis | 56.77 | **61.61** | 59.03 
Soham News Article | Genre Classification | 80.23 | **87.6** | 78.45 
Midas Discourse | Discourse Analysis | 71.20 | **79.94** | 78.44 
iNLTK Headlines Classification | Genre Classification | 87.95 | 93.38 | **94.52** 
ACTSA Sentiment Analysis | Sentiment Analysis | 48.53 | 59.33 | **61.18** 
Winograd NLI | Natural Language Inference | 56.34 | 55.87 | **56.34** 
Choice of Plausible Alternative (COPA) | Natural Language Inference | 54.92 | 51.13 | **58.33** 
Amrita Exact Paraphrase | Paraphrase Detection | **93.81** | 93.02 | 93.75 
Amrita Rough Paraphrase | Paraphrase Detection | 83.38 | 82.20 | **84.33** 
Average |  |  69.84 | **74.42** | 73.66 


\* Note: all models have been restricted to a max_seq_length of 128.



## Downloads

The model can be downloaded [here](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/models/indic-bert-v1.tar.gz). Both tf checkpoints and pytorch binaries are included in the archive. Alternatively, you can also download it from [Huggingface](https://huggingface.co/ai4bharat/indic-bert).



## Citing

If you are using any of the resources, please cite the following article:

```
@inproceedings{kakwani2020indicnlpsuite,
    title={{IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages}},
    author={Divyanshu Kakwani and Anoop Kunchukuttan and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
    year={2020},
    booktitle={Findings of EMNLP},
}
```

We would like to hear from you if:

- You are using our resources. Please let us know how you are putting these resources to use.
- You have any feedback on these resources.



## License

The IndicBERT code (and models) are released under the MIT License.

## Contributors

- Divyanshu Kakwani
- Anoop Kunchukuttan
- Gokul NC
- Satish Golla
- Avik Bhattacharyya
- Mitesh Khapra
- Pratyush Kumar

This work is the outcome of a volunteer effort as part of [AI4Bharat initiative](https://ai4bharat.org).



## Contact

- Anoop Kunchukuttan ([anoop.kunchukuttan@gmail.com](mailto:anoop.kunchukuttan@gmail.com))
- Mitesh Khapra ([miteshk@cse.iitm.ac.in](mailto:miteshk@cse.iitm.ac.in))
- Pratyush Kumar ([pratyush@cse.iitm.ac.in](mailto:pratyush@cse.iitm.ac.in))
