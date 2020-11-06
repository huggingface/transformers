# Named Entity Recognition Model for Telugu

#### How to use

```python
from simpletransformers.ner import NERModel
model = NERModel('bert',
                 'kuppuluri/telugu_bertu_ner',
                 labels=[
                     'B-PERSON', 'I-ORG', 'B-ORG', 'I-LOC', 'B-MISC',
                     'I-MISC', 'I-PERSON', 'B-LOC', 'O'
                 ],
                 use_cuda=False,
                 args={"use_multiprocessing": False})

text = "విరాట్ కోహ్లీ కూడా అదే నిర్లక్ష్యాన్ని ప్రదర్శించి కేవలం ఒక పరుగుకే రనౌటై పెవిలియన్ చేరాడు ."
results = model.predict([text])
```

## Training data

Training data is from https://github.com/anikethjr/NER_Telugu

## Eval results

On the test set my results were

eval_loss = 0.0004407190410447974

f1_score = 0.999519076627124

precision = 0.9994389677005691

recall = 0.9995991983967936

