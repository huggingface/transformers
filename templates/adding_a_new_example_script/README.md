# How to add a new example script in 🤗Transformers

This folder provide a template for adding a new example script implementing a training or inference task with the models in the  🤗Transformers library.

Currently only examples for PyTorch are provided which are adaptations of the library's SQuAD examples which implement single-GPU and distributed training with gradient accumulation and mixed-precision (using NVIDIA's apex library) to cover a reasonable range of use cases.
