---
language: tr
---
# Turkish News Text Classification

    Turkish text classification model obtained by fine-tuning the Turkish bert model (dbmdz/bert-base-turkish-cased)

# Dataset

Dataset consists of 11 classes were obtained from https://www.trthaber.com/. The model was created using the most distinctive 6 classes.

Dataset can be accessed at https://github.com/gurkan08/datasets/tree/master/trt_11_category.

    label_dict = {
        'LABEL_0': 'ekonomi',
        'LABEL_1': 'spor',
        'LABEL_2': 'saglik',
        'LABEL_3': 'kultur_sanat',
        'LABEL_4': 'bilim_teknoloji',
        'LABEL_5': 'egitim'
    }

70% of the data were used for training and 30% for testing.

train f1-weighted score = %97

test f1-weighted score = %94

# Usage

    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("gurkan08/bert-turkish-text-classification")
    model = AutoModelForSequenceClassification.from_pretrained("gurkan08/bert-turkish-text-classification")

    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    text = ["Süper Lig'in 6. haftasında Sivasspor ile Çaykur Rizespor karşı karşıya geldi...",
    "Son 24 saatte 69 kişi Kovid-19 nedeniyle yaşamını yitirdi, 1573 kişi iyileşti"]

    out = nlp(text)
    
    label_dict = {
     'LABEL_0': 'ekonomi',
     'LABEL_1': 'spor',
     'LABEL_2': 'saglik',
     'LABEL_3': 'kultur_sanat',
     'LABEL_4': 'bilim_teknoloji',
     'LABEL_5': 'egitim'
    }

    results = []
    for result in out:
        result['label'] = label_dict[result['label']]
        results.append(result)
    print(results)

    # > [{'label': 'spor', 'score': 0.9992026090621948}, {'label': 'saglik', 'score': 0.9972177147865295}]
    
    
    
