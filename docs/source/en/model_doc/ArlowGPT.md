---
title: "ArlowGPT"
---

# ArlowGPT

ArlowGPT is a custom large language model architecture that will be integrated into Transformers in future releases. Currently, only `ArlowTokenizer` is available for tokenization and preprocessing.

## **ArlowTokenizer**

[`ArlowTokenizer`] is a Byte Pair Encoding (BPE)-based tokenizer optimized for large-scale language modeling. It is designed to work similarly to GPT-2 and GPT-3-style tokenizers.

### **Supported Special Tokens**
`ArlowTokenizer` comes with several special tokens for autoregressive generation:

| Token              | Purpose                          |
|--------------------|--------------------------------|
| `<|startoftext|>` | Beginning-of-sequence (BOS)   |
| `<|endoftext|>`   | End-of-sequence (EOS)         |
| `<|unk|>`         | Unknown token                 |
| `<|pad|>`         | Padding token                 |
| `<|mask|>`        | Masking token (for MLM tasks) |
| `<|im_start|>`    | Start of system message       |
| `<|im_end|>`      | End of system message         |

### **Usage Example**
You can load `ArlowTokenizer` directly or through `AutoTokenizer`:

