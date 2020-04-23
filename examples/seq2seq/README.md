## Seq2Seq

Based on the script [`run_ner.py`](https://github.com/huggingface/transformers/blob/master/examples/ner/run_ner.py) for Pytorch 

This example fine-tunes a transformer model to format names from how an ASR engine might output them to how a human would write them.
This is obviously not a production use case, but it demonstrates the workflow for how to train a seq2seq model for text transformations.
Details and results provided by [@mgoldey](http://github.com/mgoldey).

### Environment

Pip requirements are stored in requirements.txt in this example.

```bash
python3 -m pip install -r requirements.txt
```

### Data preparation using Faker

Fake data can be made by running
```bash
python3 data_generator.py
```

By default, this uses a comparatively small training set of 10K examples and dev and test sets of 1K examples. 
You can change these values through command line flags.
```text
FLAGS
    --train_size=TRAIN_SIZE
    --dev_size=DEV_SIZE
    --test_size=TEST_SIZE
```

The training data format is given in paired input/output examples, as follows:
```text
input: nicole french
output: Nicole French
input: cody fuentes
output: Cody Fuentes
input: kathleen aguilar
output: Kathleen Aguilar
input: jessica wright m d
output: Jessica Wright MD
```

### Execution

To train a model for `GPT2`, simply run `bash run_gpt2.sh`. After 5 epochs, an f1 score of 0.643 was achieved.

### Missing features
- No test of Tensorflow version
- EncoderDecoderModel needs greater test coverage that considers the output and not just the shape of the output
- Max length should be enforced in data preparation
