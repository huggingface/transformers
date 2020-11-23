---
language: multilingual
datasets:
- tydiqa
pipeline_tag: question-answering
---

# mT5-small fine-tuned on TyDiQA for multilingual QA 🗺📖❓
[Google's mT5-small](https://huggingface.co/google/mt5-small) fine-tuned on [TyDi QA](https://huggingface.co/nlp/viewer/?dataset=tydiqa&config=secondary_task) (secondary task) for **multingual Q&A** downstream task.

## Details of mT5

[Google's mT5](https://github.com/google-research/multilingual-t5)

mT5 is pretrained on the [mC4](https://www.tensorflow.org/datasets/catalog/c4#c4multilingual) corpus, covering 101 languages:

Afrikaans, Albanian, Amharic, Arabic, Armenian, Azerbaijani, Basque, Belarusian, Bengali, Bulgarian, Burmese, Catalan, Cebuano, Chichewa, Chinese, Corsican, Czech, Danish, Dutch, English, Esperanto, Estonian, Filipino, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Haitian Creole, Hausa, Hawaiian, Hebrew, Hindi, Hmong, Hungarian, Icelandic, Igbo, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kurdish, Kyrgyz, Lao, Latin, Latvian, Lithuanian, Luxembourgish, Macedonian, Malagasy, Malay, Malayalam, Maltese, Maori, Marathi, Mongolian, Nepali, Norwegian, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Samoan, Scottish Gaelic, Serbian, Shona, Sindhi, Sinhala, Slovak, Slovenian, Somali, Sotho, Spanish, Sundanese, Swahili, Swedish, Tajik, Tamil, Telugu, Thai, Turkish, Ukrainian, Urdu, Uzbek, Vietnamese, Welsh, West Frisian, Xhosa, Yiddish, Yoruba, Zulu.

**Note**: mT5 was only pre-trained on mC4 excluding any supervised training. Therefore, this model has to be fine-tuned before it is useable on a downstream task.

Pretraining Dataset: [mC4](https://www.tensorflow.org/datasets/catalog/c4#c4multilingual)

Other Community Checkpoints: [here](https://huggingface.co/models?search=mt5)

Paper: [mT5: A massively multilingual pre-trained text-to-text transformer](https://arxiv.org/abs/2010.11934)

Authors: *Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, Colin Raffel* 


## Details of the dataset 📚 

**TyDi QA** is a question answering dataset covering 11 typologically diverse languages with 204K question-answer pairs. The languages of TyDi QA are diverse with regard to their typology -- the set of linguistic features that each language expresses -- such that we expect models performing well on this set to generalize across a large number of the languages in the world. It contains language phenomena that would not be found in English-only corpora. To provide a realistic information-seeking task and avoid priming effects, questions are written by people who want to know the answer, but don’t know the answer yet, (unlike SQuAD and its descendents) and the data is collected directly in each language without the use of translation (unlike MLQA and XQuAD).

| Dataset  | Task | Split | # samples |
| -------- | ----- |------| --------- |
| TyDi QA | GoldP  | train| 49881     |
| TyDi QA | GoldP  | valid| 5077      | 



## Results on validation dataset 📝

| Metric | # Value   |
| ------ | --------- |
| **EM** | **41.65** |



## Model in Action 🚀

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("mrm8488/mT5-small-finetuned-tydiqa-for-xqa")
model = AutoModelForCausalLM.from_pretrained("mrm8488/mT5-small-finetuned-tydiqa-for-xqa").to(device)

def get_response(question, context, max_length=32):
  input_text = 'question: %s  context: %s' % (question, context)
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'].to(device), 
               attention_mask=features['attention_mask'].to(device),
               max_length=max_length)

  return tokenizer.decode(output[0])
  
# Some examples in different languages

context = 'HuggingFace won the best Demo paper at EMNLP2020.'
question = 'What won HuggingFace?'
get_response(question, context)

context = 'HuggingFace ganó la mejor demostración con su paper en la EMNLP2020.'
question = 'Qué ganó HuggingFace?'
get_response(question, context)

context = 'HuggingFace выиграл лучшую демонстрационную работу на EMNLP2020.'
question = 'Что победило в HuggingFace?'
get_response(question, context)
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
