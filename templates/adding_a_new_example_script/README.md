# How to add a new example script in 🤗Transformers

This folder provide a template for adding a new example script implementing a training or inference task with the models in the  🤗Transformers library.
Add tests!


These folder can be put in a subdirectory under your example's name, like `examples/deebert`.


Best Practices: 
- use `Trainer`/`TFTrainer`
- write an @slow test that checks that your model can train on one batch and get a low loss.
    - this test should use cuda if it's available. (e.g. by checking `transformers.torch_device`)
- adding an `eval_xxx.py` script that can evaluate a pretrained checkpoint.  
- tweet about your new example with a carbon screenshot of how to run it and tag @huggingface
