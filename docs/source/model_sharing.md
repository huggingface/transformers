# Model upload and sharing

Starting with `v2.2.2`, you can now upload and share your fine-tuned models with the community, using the <abbr title="Command-line interface">CLI</abbr> that's built-in to the library.

**First, create an account on [https://huggingface.co/join](https://huggingface.co/join)**. Then:

```shell
transformers-cli login
# log in using the same credentials as on huggingface.co
```
Upload your model:
```shell
transformers-cli upload ./path/to/pretrained_model/

# ^^ Upload folder containing weights/tokenizer/config
# saved via `.save_pretrained()`

transformers-cli upload ./config.json [--filename foobar.json]

# ^^ Upload a single file
# (you can optionally override its filename)
```

Your model will then be accessible through its identifier:
```python
"username/model_name"
```

Anyone can load it from code:
```python
tokenizer = AutoTokenizer.from_pretrained("username/model_name")
model = AutoModel.from_pretrained("username/model_name")
```

Finally, list all your files on S3:
```shell
transformers-cli ls
# List all your S3 objects.
```

