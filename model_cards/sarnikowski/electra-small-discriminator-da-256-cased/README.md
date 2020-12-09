---
language: da
license: cc-by-4.0
---

# Danish ELECTRA small (cased)

An [ELECTRA](https://arxiv.org/abs/2003.10555) model pretrained on a custom Danish corpus (~17.5gb). 
For details regarding data sources and training procedure, along with benchmarks on downstream tasks, go to: https://github.com/sarnikowski/danish_transformers/tree/main/electra 

## Usage

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("sarnikowski/electra-small-discriminator-da-256-cased")
model = AutoModel.from_pretrained("sarnikowski/electra-small-discriminator-da-256-cased")
```

## Questions?

If you have any questions feel free to open an issue on the [danish_transformers](https://github.com/sarnikowski/danish_transformers) repository, or send an email to p.sarnikowski@gmail.com
