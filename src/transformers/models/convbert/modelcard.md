<!-- ConvBERT model card -->

# ConvBERT

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

---

## Model Overview

ConvBERT is a lightweight and efficient NLP transformer model introduced by YituTech. It improves on the classic BERT architecture by incorporating **span-based dynamic convolutions** into the self-attention mechanism. This hybrid approach enables ConvBERT to model both local and global dependencies more effectively while reducing the computational cost.

The model performs exceptionally well on tasks such as **text classification**, **question answering**, and **sequence labeling**, making it suitable for deployment in real-time or edge environments. ConvBERT offers performance comparable to or better than BERT, but with fewer parameters and lower latency.

**Authors**: YituTech (Research team)  
**Contributors**: Hugging Face community  
**Visual Example**: *(image placeholder)*

---

## Model Details 

**Architecture**: ConvBERT is based on the Transformer encoder, similar to BERT, but introduces **span-based dynamic convolution** within its layers. Some self-attention heads are replaced with convolutional filters that dynamically select input spans, improving the modeling of local contexts.

**Training Objective**: ConvBERT uses the same masked language modeling (MLM) objective as BERT but is trained with an improved token masking strategy.

**Datasets Used**: ConvBERT is pre-trained on a combination of Wikipedia and BooksCorpus — the same corpora used for BERT pretraining.

**Pretraining Details**:
- MLM with whole-word masking
- Smaller model sizes (fewer parameters than RoBERTa or BERT-Large)
- Mixed attention/convolution blocks for speed

**Training Frameworks**:
- The architecture enables teacher-student knowledge distillation during fine-tuning for downstream tasks.
- No explicit teacher-student training in pretraining phase reported.

---

## Intended Use Cases

ConvBERT is designed for a variety of **NLP tasks**, including but not limited to:

- Sentiment Analysis
- Named Entity Recognition (NER)
- Question Answering
- Text Classification

The model is suitable for both **zero-shot inference** (using pipelines) and **fine-tuning** for specific downstream tasks. It is especially recommended when compute efficiency or real-time inference is important.

---

## Limitations and Warnings 

- ConvBERT may not perform as well as larger models like RoBERTa-Large on some high-resource benchmarks.
- The model inherits any **biases present in the BooksCorpus and Wikipedia**, such as social, gender, and geographic biases.
- Not suitable for tasks requiring reasoning over long documents unless specially fine-tuned.

Always evaluate model performance in your own application before production use.

---

##  How to Use 

You can use ConvBERT either through the Hugging Face `pipeline` API or directly with `AutoModel`:

### Using `pipeline`

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="YituTech/conv-bert-base")
print(classifier("ConvBERT is compact and powerful."))
```

### Using `AutoModel`

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("YituTech/conv-bert-base")
model = AutoModelForSequenceClassification.from_pretrained("YituTech/conv-bert-base")
inputs = tokenizer("ConvBERT balances speed and accuracy.", return_tensors="pt")
outputs = model(**inputs)
```

### CLI Usage

```bash
transformers-cli env
transformers-cli download YituTech/conv-bert-base
```

---

## Performance Metrics

ConvBERT outperforms BERT on the GLUE benchmark and performs comparably to RoBERTa-base while being faster.

- GLUE score: ~79.3 (ConvBERT) vs ~77.6 (BERT)
- SQuAD v1.1 F1: ~93.4
- Parameters: ~110M

---

## References and Resources

- Paper: https://arxiv.org/abs/2008.02496
- GitHub: https://github.com/yitu-opensource/ConvBERT
- Model on HF: https://huggingface.co/YituTech/conv-bert-base

### Citation

```
@article{jiang2020convbert,
  title={ConvBERT: Improving BERT with Span-based Dynamic Convolution},
  author={Jiang, Wei and Yu, Haihua and Ye, Zihan and Li, Peng and Li, Weiping and Lin, Chin-Yew},
  journal={arXiv preprint arXiv:2008.02496},
  year={2020}
}
```
