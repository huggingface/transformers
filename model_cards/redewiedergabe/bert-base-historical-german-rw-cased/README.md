---
language: german
---

# Model description
## Dataset
Trained on fictional and non-fictional German texts written between 1840 and 1920:
* Narrative texts from Digitale Bibliothek (https://textgrid.de/digitale-bibliothek)
* Fairy tales and sagas from Grimm Korpus (https://www1.ids-mannheim.de/kl/projekte/korpora/archiv/gri.html)
* Newspaper and magazine article from Mannheimer Korpus Historischer Zeitungen und Zeitschriften (https://repos.ids-mannheim.de/mkhz-beschreibung.html)
* Magazine article from the journal „Die Grenzboten“ (http://www.deutschestextarchiv.de/doku/textquellen#grenzboten)
* Fictional and non-fictional texts from Projekt Gutenberg (https://www.projekt-gutenberg.org)

## Hardware used
1 Tesla P4 GPU

## Hyperparameters

| Parameter                     | Value    |
|-------------------------------|----------|
| Epochs                        | 3        |
| Gradient_accumulation_steps   | 1        |
| Train_batch_size              | 32       |
| Learning_rate                 | 0.00003  |
| Max_seq_len                   | 128      |

## Evaluation results: Automatic tagging of four forms of speech/thought/writing representation in historical fictional and non-fictional German texts

The language model was used in the task to tag direct, indirect, reported and free indirect speech/thought/writing representation in fictional and non-fictional German texts. The tagger is available and described in detail at https://github.com/redewiedergabe/tagger.

The tagging model was trained using the SequenceTagger Class of the Flair framework ([Akbik et al., 2019](https://www.aclweb.org/anthology/N19-4010)) which implements a BiLSTM-CRF architecture on top of a language embedding (as proposed by [Huang et al. (2015)](https://arxiv.org/abs/1508.01991)). 


Hyperparameters

| Parameter                     | Value      |
|-------------------------------|------------|
| Hidden_size                   | 256        |
| Learning_rate                 | 0.1        |
| Mini_batch_size               | 8          |
| Max_epochs                    | 150        |

Results are reported below in comparison to a custom trained flair embedding, which was stacked onto a custom trained fastText-model. Both models were trained on the same dataset.

|                | BERT       ||| FastText+Flair  |||Test data|
|----------------|----------|-----------|----------|------|-----------|--------|--------|
|                | F1       | Precision | Recall   | F1   | Precision | Recall ||
| Direct         | 0.80     | 0.86      | 0.74     | 0.84 | 0.90      | 0.79   |historical German, fictional & non-fictional|
| Indirect       | **0.76** | **0.79**  | **0.73** | 0.73 | 0.78      | 0.68   |historical German, fictional & non-fictional|
| Reported       | **0.58** | **0.69**  | **0.51** | 0.56 | 0.68      | 0.48   |historical German, fictional & non-fictional|
| Free indirect  | **0.57** | **0.80**  | **0.44** | 0.47 | 0.78      | 0.34   |modern German, fictional|

## Intended use:
Historical German Texts (1840 to 1920)

(Showed good performance with modern German fictional texts as well)

