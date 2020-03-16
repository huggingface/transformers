
# ClinicalBERT - Bio + Clinical BERT Model

The [Publicly Available Clinical BERT Embeddings](https://arxiv.org/abs/1904.03323) paper contains four unique clinicalBERT models: initialized with BERT-Base (`cased_L-12_H-768_A-12`) or BioBERT (`BioBERT-Base v1.0 + PubMed 200K + PMC 270K`) & trained on either all MIMIC notes or only discharge summaries. 

This model card describes the Bio+Clinical BERT model, which was initialized from [BioBERT](https://arxiv.org/abs/1901.08746) & trained on all MIMIC notes. 

## Pretraining Data
The `Bio_ClinicalBERT` model was trained on all notes from [MIMIC III](https://www.nature.com/articles/sdata201635), a database containing electronic health records from ICU patients at the Beth Israel Hospital in Boston, MA. For more details on MIMIC, see [here](https://mimic.physionet.org/). All notes from the `NOTEEVENTS` table were included (~880M words).

## Model Pretraining 

### Note Preprocessing
Each note in MIMIC was first split into sections using a rules-based section splitter (e.g. discharge summary notes were split into "History of Present Illness", "Family History", "Brief Hospital Course", etc. sections). Then each section was split into sentences using SciSpacy (`en core sci md` tokenizer). 

### Pretraining Procedures
The model was trained using code from [Google's BERT repository](https://github.com/google-research/bert) on a GeForce GTX TITAN X 12 GB GPU. Model parameters were initialized with BioBERT (`BioBERT-Base v1.0 + PubMed 200K + PMC 270K`).

### Pretraining Hyperparameters
We used a batch size of 32, a maximum sequence length of 128, and a learning rate of 5 · 10−5 for pre-training our models. The models trained on all MIMIC notes  were trained for 150,000 steps. The dup factor for duplicating input data with different masks was set to 5. All other default parameters were used (specifically, masked language model probability = 0.15
and max predictions per sequence = 20).

## How to use the model

Load the model via the transformers library:
```
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
```

## More Information

Refer to the original paper, [Publicly Available Clinical BERT Embeddings](https://arxiv.org/abs/1904.03323) (NAACL Clinical NLP Workshop 2019) for additional details and performance on NLI and NER tasks.

## Questions?

Post a Github issue on the [clinicalBERT repo](https://github.com/EmilyAlsentzer/clinicalBERT) or email emilya@mit.edu with any questions.

