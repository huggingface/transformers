---
language: en
tags:
- legal
---

# LEGAL-BERT: The Muppets straight out of Law School

<img align="left" src="https://i.ibb.co/p3kQ7Rw/Screenshot-2020-10-06-at-12-16-36-PM.png" width="100"/> 

LEGAL-BERT is a family of BERT models for the legal domain, intended to assist legal NLP research, computational law, and legal technology applications.  To pre-train the different variations of LEGAL-BERT, we collected 12 GB of diverse English legal text from several fields (e.g., legislation, court cases,  contracts) scraped from publicly available resources. Sub-domains variants (CONTRACTS-, EURLEX-, ECHR-) and/or general LEGAL-BERT perform better than using BERT out of the box for domain-specific tasks. A light-weight model (33% the size of BERT-BASE) pre-trained from scratch on legal data with competitive perfomance is also available.
<br/><br/><br/><br/>

---

I. Chalkidis, M. Fergadiotis, P. Malakasiotis, N. Aletras and I. Androutsopoulos. "LEGAL-BERT: The Muppets straight out of Law School". In Findings of Empirical Methods in Natural Language Processing (EMNLP 2020) (Short Papers), to be held online, 2020. (https://arxiv.org/abs/2010.02559)

---

## Pre-training corpora

The pre-training corpora of LEGAL-BERT include:

* 116,062 documents of EU legislation, publicly available from EURLEX (http://eur-lex.europa.eu), the repository of EU Law running under the EU Publication Office.
    
* 61,826 documents of UK legislation, publicly available from the UK legislation portal (http://www.legislation.gov.uk).
    
* 19,867 cases from European Court of Justice (ECJ), also available from EURLEX.
    
* 12,554 cases from HUDOC, the repository of the European Court of Human Rights (ECHR) (http://hudoc.echr.coe.int/eng).
    
* 164,141 cases from various courts across the USA, hosted in the Case Law Access Project portal (https://case.law).
    
* 76,366 US contracts from EDGAR, the database of US Securities and Exchange Commission (SECOM) (https://www.sec.gov/edgar.shtml).

## Pre-training details

* We trained BERT using the official code provided in Google BERT's github repository (https://github.com/google-research/bert).
* We released a model similar to the English BERT-BASE model (12-layer, 768-hidden, 12-heads, 110M parameters).
* We chose to follow the same training set-up: 1 million training steps with batches of 256 sequences of length 512 with an initial learning rate 1e-4.
* We were able to use a single Google Cloud TPU v3-8 provided for free from [TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc), while also utilizing [GCP research credits](https://edu.google.com/programs/credits/research). Huge thanks to both Google programs for supporting us!
* Part of LEGAL-BERT is a light-weight model pre-trained from scratch on legal data, which achieves comparable performance to larger models, while being much more efficient (approximately 4 times faster) with a smaller environmental footprint.
## Models list

| Model name          | Model Path                            | Training corpora    |
| ------------------- | ------------------------------------  | ------------------- |
| CONTRACTS-BERT-BASE | `nlpaueb/bert-base-uncased-contracts` | US contracts        |
| EURLEX-BERT-BASE    | `nlpaueb/bert-base-uncased-eurlex`    | EU legislation      |
| ECHR-BERT-BASE      | `nlpaueb/bert-base-uncased-echr`      | ECHR cases          |
| LEGAL-BERT-BASE     | `nlpaueb/legal-bert-base-uncased`     | All                 |
| LEGAL-BERT-SMALL    | `nlpaueb/legal-bert-small-uncased`    | All                 |

## Load Pretrained Model

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
```

## Use LEBAL-BERT variants as Language Models

| Corpus                             | Model                               | Masked token | Predictions  |
| --------------------------------- | ---------------------------------- | ------------ | ------------ |
|  | **BERT-BASE-UNCASED**                 |
| (Contracts) | This [MASK] Agreement is between General Motors and John Murray . | employment | ('new', '0.09'), ('current', '0.04'), ('proposed', '0.03'), ('marketing', '0.03'), ('joint', '0.02')
| (ECHR) | The applicant submitted that her husband was subjected to treatment amounting to [MASK] whilst in the custody of Adana Security Directorate | torture | ('torture', '0.32'), ('rape', '0.22'), ('abuse', '0.14'), ('death', '0.04'), ('violence', '0.03')
| (EURLEX) | Establishing a system for the identification and registration of [MASK] animals and regarding the labelling of beef and beef products . | bovine | ('farm', '0.25'), ('livestock', '0.08'), ('draft', '0.06'), ('domestic', '0.05'), ('wild', '0.05')
|  | **CONTRACTS-BERT-BASE**                 |
| (Contracts) | This [MASK] Agreement is between General Motors and John Murray . | employment | ('letter', '0.38'), ('dealer', '0.04'), ('employment', '0.03'), ('award', '0.03'), ('contribution', '0.02')
| (ECHR) | The applicant submitted that her husband was subjected to treatment amounting to [MASK] whilst in the custody of Adana Security Directorate | torture | ('death', '0.39'), ('imprisonment', '0.07'), ('contempt', '0.05'), ('being', '0.03'), ('crime', '0.02')
| (EURLEX) | Establishing a system for the identification and registration of [MASK] animals and regarding the labelling of beef and beef products . | bovine | (('domestic', '0.18'), ('laboratory', '0.07'), ('household', '0.06'), ('personal', '0.06'), ('the', '0.04')
|  | **EURLEX-BERT-BASE**                 |
| (Contracts) | This [MASK] Agreement is between General Motors and John Murray . | employment | ('supply', '0.11'), ('cooperation', '0.08'), ('service', '0.07'), ('licence', '0.07'), ('distribution', '0.05')
| (ECHR) | The applicant submitted that her husband was subjected to treatment amounting to [MASK] whilst in the custody of Adana Security Directorate | torture | ('torture', '0.66'), ('death', '0.07'), ('imprisonment', '0.07'), ('murder', '0.04'), ('rape', '0.02')
| (EURLEX) | Establishing a system for the identification and registration of [MASK] animals and regarding the labelling of beef and beef products . | bovine | ('live', '0.43'), ('pet', '0.28'), ('certain', '0.05'), ('fur', '0.03'), ('the', '0.02')
|  | **ECHR-BERT-BASE**                 |
| (Contracts) | This [MASK] Agreement is between General Motors and John Murray . | employment | ('second', '0.24'), ('latter', '0.10'), ('draft', '0.05'), ('bilateral', '0.05'), ('arbitration', '0.04')
| (ECHR) | The applicant submitted that her husband was subjected to treatment amounting to [MASK] whilst in the custody of Adana Security Directorate | torture | ('torture', '0.99'), ('death', '0.01'), ('inhuman', '0.00'), ('beating', '0.00'), ('rape', '0.00')
| (EURLEX) | Establishing a system for the identification and registration of [MASK] animals and regarding the labelling of beef and beef products . | bovine | ('pet', '0.17'), ('all', '0.12'), ('slaughtered', '0.10'), ('domestic', '0.07'), ('individual', '0.05')
|  | **LEGAL-BERT-BASE**                |
| (Contracts) | This [MASK] Agreement is between General Motors and John Murray . | employment | ('settlement', '0.26'), ('letter', '0.23'), ('dealer', '0.04'), ('master', '0.02'), ('supplemental', '0.02')
| (ECHR) | The applicant submitted that her husband was subjected to treatment amounting to [MASK] whilst in the custody of Adana Security Directorate | torture | ('torture', '1.00'), ('detention', '0.00'), ('arrest', '0.00'), ('rape', '0.00'), ('death', '0.00')
| (EURLEX) | Establishing a system for the identification and registration of [MASK] animals and regarding the labelling of beef and beef products . | bovine | ('live', '0.67'), ('beef', '0.17'), ('farm', '0.03'), ('pet', '0.02'), ('dairy', '0.01')
|  | **LEGAL-BERT-SMALL**                |
| (Contracts) | This [MASK] Agreement is between General Motors and John Murray . | employment | ('license', '0.09'), ('transition', '0.08'), ('settlement', '0.04'), ('consent', '0.03'), ('letter', '0.03')
| (ECHR) | The applicant submitted that her husband was subjected to treatment amounting to [MASK] whilst in the custody of Adana Security Directorate | torture | ('torture', '0.59'), ('pain', '0.05'), ('ptsd', '0.05'), ('death', '0.02'), ('tuberculosis', '0.02')
| (EURLEX) | Establishing a system for the identification and registration of [MASK] animals and regarding the labelling of beef and beef products . | bovine | ('all', '0.08'), ('live', '0.07'), ('certain', '0.07'), ('the', '0.07'), ('farm', '0.05')



## Evaluation on downstream tasks

Consider the experiments in the article "LEGAL-BERT: The Muppets straight out of Law School". Chalkidis et al., 2018, (https://arxiv.org/abs/2010.02559)

## Author

Ilias Chalkidis on behalf of [AUEB's Natural Language Processing Group](http://nlp.cs.aueb.gr)

| Github: [@ilias.chalkidis](https://github.com/seolhokim) | Twitter: [@KiddoThe2B](https://twitter.com/KiddoThe2B) |
