---
language: et
---
## Model Description

This model is based off **Sentence-Transformer's** `distiluse-base-multilingual-cased` multilingual model that has been extended to understand sentence embeddings in Estonian.

## Sentence-Transformers

This model can be imported directly via the SentenceTransformers package as shown below:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('kiri-ai/distiluse-base-multilingual-cased-et')
sentences = ['Here is a sample sentence','Another sample sentence']
embeddings = model.encode(sentences)

print("Sentence embeddings:")
print(embeddings)
```

## Fine-tuning

The fine-tuning and training processes were inspired by [sbert's](https://www.sbert.net/) multilingual training techniques which are available [here](https://www.sbert.net/examples/training/multilingual/README.html). The documentation shows and explains the step-by-step process of using parallel sentences to train models in a different language.

### Resources

The model was fine-tuned on English-Estonian parallel sentences taken from [OPUS](http://opus.nlpl.eu/) and [ParaCrawl](https://paracrawl.eu/).
