# ArlowGPT

This document provides an overview of **ArlowTokenizer**, a Byte-Pair Encoding (BPE) tokenizer designed for the **ArlowGPT** model, and the future **ArlowGPT** model architecture.

---

## **🔤 ArlowTokenizer Overview**
`ArlowTokenizer` is a **custom tokenizer** optimized for the ArlowGPT model. It is based on Byte-Pair Encoding (BPE) and supports fast tokenization using the `tokenizers` library.

### **✨ Features**
- **Fast Tokenization:** Built with Hugging Face's `tokenizers` library for high-speed processing.
- **GPT-style Special Tokens:** Supports `<|startoftext|>`, `<|endoftext|>`, `<|pad|>`, `<|unk|>`, `<|mask|>`.
- **Optimized for Pretraining:** Designed for large-scale training and inference.

---

### **Special Tokens**
ArlowTokenizer includes several special tokens:

| Token Name       | Token Symbol    | Description                  |
|-----------------|----------------|------------------------------|
| **Start Token**  | `<|startoftext|>` | Marks the beginning of a sequence. |
| **End Token**    | `<|endoftext|>`   | Marks the end of a sequence. |
| **Padding Token**| `<|pad|>`         | Used for sequence padding. |
| **Unknown Token**| `<|unk|>`         | Represents unknown tokens. |
| **Mask Token**   | `<|mask|>`        | Used for masked language modeling. |

---

## **🚀 ArlowGPT Model Overview**
> 🚧 **Note:** The full `ArlowGPT` model is under development and will be added in a future PR.

ArlowGPT is an optimized **causal language model** includes enhancements in:
- **Flash Attention 2**: Optimized for memory efficiency.
- **Grouped Query Attention (GQA)**: For better multi-head attention performance.
- **Rotary Embeddings (RoPE)**: For improved positional encoding.
- **Extended Vocabulary Size**: Up to **131,072 tokens** for richer text representation.
- **Cross Attention**: For easy vision encoder add on.

---

## **📜 Model Configuration**
The ArlowGPT model will use the following architecture:
- **Number of Layers:** *TBD*
- **Hidden Size:** *TBD*
- **Number of Attention Heads:** *TBD*
- **FFN Dimension:** *TBD*
- **Dropout Rate:** *TBD*

---

## ArlowTokenizer

[[autodoc]] ArlowTokenizer

---
## **📈 Future Work**
- ✅ ArlowTokenizer 
- 🚧 ArlowGPT Model Implementation (Coming Soon)
- 🚀 Pretraining & Evaluation on Large Datasets

---

### **🌟 Citation**

```
@misc{ArlowGPT2025,
  author = {Yuchen Xie},
  title = {ArlowGPT: A Custom Large Language Model},
  year = {2025},
  howpublished = {\url{https://github.com/huggingface/transformers}}
}
```

