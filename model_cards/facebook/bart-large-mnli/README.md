---
license: mit
thumbnail: https://huggingface.co/front/thumbnails/facebook.png
pipeline_tag: zero-shot-classification
datasets:
- multi_nli
---

# bart-large-mnli

This is the checkpoint for [bart-large](https://huggingface.co/facebook/bart-large) after being trained on the [MultiNLI (MNLI)](https://huggingface.co/datasets/multi_nli) dataset.

Additional information about this model:
- The [bart-large](https://huggingface.co/facebook/bart-large) model page
- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
](https://arxiv.org/abs/1910.13461)
- [BART fairseq implementation](https://github.com/pytorch/fairseq/tree/master/fairseq/models/bart)

## NLI-based Zero Shot Text Classification

[Yin et al.](https://arxiv.org/abs/1909.00161) proposed a method for using pre-trained NLI models as a ready-made zero-shot sequence classifiers. The method works by posing the sequence to be classified as the NLI premise and to construct a hypothesis from each candidate label. For example, if we want to evaluate whether a sequence belongs to the class "politics", we could construct a hypothesis of `This text is about politics.`. The probabilities for entailment and contradiction are then converted to label probabilities.

This method is surprisingly effective in many cases, particularly when used with larger pre-trained models like BART and Roberta. See [this blog post](https://joeddav.github.io/blog/2020/05/29/ZSL.html) for a more expansive introduction to this and other zero shot methods, and see the code snippets below for examples of using this model for zero-shot classification both with Hugging Face's built-in pipeline and with native Transformers/PyTorch code.

#### With the zero-shot classification pipeline

The model can be loaded with the `zero-shot-classification` pipeline like so:

```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
```

You can then use this pipeline to classify sequences into any of the class names you specify.

```python
sequence_to_classify = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing']
classifier(sequence_to_classify, candidate_labels)
#{'labels': ['travel', 'dancing', 'cooking'],
# 'scores': [0.9938651323318481, 0.0032737774308770895, 0.002861034357920289],
# 'sequence': 'one day I will see the world'}
```

If more than one candidate label can be correct, pass `multi_class=True` to calculate each class independently:

```python
candidate_labels = ['travel', 'cooking', 'dancing', 'exploration']
classifier(sequence_to_classify, candidate_labels, multi_class=True)
#{'labels': ['travel', 'exploration', 'dancing', 'cooking'],
# 'scores': [0.9945111274719238,
#  0.9383890628814697,
#  0.0057061901316046715,
#  0.0018193122232332826],
# 'sequence': 'one day I will see the world'}
```


#### With manual PyTorch

```python
# pose sequence as a NLI premise and label as a hypothesis
from transformers import AutoModelForSequenceClassification, AutoTokenizer
nli_model = AutoModelForSequenceClassification.from_pretrained('joeddav/xlm-roberta-large-xnli')
tokenizer = AutoTokenizer.from_pretrained('joeddav/xlm-roberta-large-xnli')

premise = sequence
hypothesis = f'This example is {label}.'

# run through model pre-trained on MNLI
x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                     truncation_strategy='only_first')
logits = nli_model(x.to(device))[0]

# we throw away "neutral" (dim 1) and take the probability of
# "entailment" (2) as the probability of the label being true 
entail_contradiction_logits = logits[:,[0,2]]
probs = entail_contradiction_logits.softmax(dim=1)
prob_label_is_true = probs[:,1]
```
