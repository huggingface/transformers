# BioBERTpt(clin) - Clinical BERT Model for Portuguese Language

The [BioBERTpt - A Portuguese Neural Language Model for Clinical Named Entity Recognition](https://www.aclweb.org/anthology/2020.clinicalnlp-1.7/) paper contains clinical and biomedical BERT-based models for Portuguese Language, initialized with BERT-Multilingual-Cased & trained on clinical notes and biomedical literature. 

This model card describes the BioBERTpt(clin) model, a clinical version of BioBERTpt, trained on clinical narratives from electronic health records from Brazilian Hospitals. 

## How to use the model

Load the model via the transformers library:
```
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("pucpr/biobertpt-clin")
model = AutoModel.from_pretrained("pucpr/biobertpt-clin")
```

## More Information

Refer to the original paper, [BioBERTpt - A Portuguese Neural Language Model for Clinical Named Entity Recognition](https://www.aclweb.org/anthology/2020.clinicalnlp-1.7/) for additional details and performance on Portuguese NER tasks.

## Questions?

Post a Github issue on the [BioBERTpt repo](https://github.com/HAILab-PUCPR/BioBERTpt).
