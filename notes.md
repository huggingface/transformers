Potential issues with in transformers:
- beam sampling
- logit processor
  - penalties for repetitions -> introduce bias
-> slow (for loops / non vectorized)

Potentail TODOs:
1. improve DX of transformers to improve reach and simplify usage (typically on local devices)
2. better improve usage of torch within transformers (jit/torchscript, executorch, etc)
3. tokenizers
  - take papers and implement improvements (typically better Byte Pair Encoding)
  - maintenance work (improve python API, help out on various issues / improvements, etc)
