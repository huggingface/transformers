# Modular transformers

`transformers` is an opinionated framework; our philosophy is defined in the following [conceptual guide](./philosophy).

The core of that philosophy is exemplified by the [single model, single file](https://huggingface.co/blog/transformers-design-philosophy)
aspect of the library. This component's downside is that it limits the inheritance and importability of components from
files to others in the toolkit.

As a result, model components tend to be repeated across many files. There are as many attention layers defined
in `transformers` as there are models, and a significant number of those are identical to each other. 
The unfortunate consequence is that independent implementations tend to diverge as fixes and changes get applied
to specific parts of the code.

In order to balance this issue, we introduced the concept of "copies" across the library. By adding a comment indicating
that code is a copy of another, we can enforce through CI and local commands that copies do not diverge. However,
while the complexity is low, this is often quite tedious to do.

And, finally, this contributes to adding a significant overhead to contributing models which we would like to remove.
This approach often requires model contributions to add modeling code (~1k lines), processor (~500 lines), tests, docs,
etc. Model contribution PRs rarely add less than 3-5k lines of code, with much of this code being boilerplate.

This raises the bar for contributions, and with Modular Transformers, we're aiming to lower the bar to a much more
acceptable point.

## What is it?

Modular Transformers introduces the concept of a "modular" file to a model folder. This modular file accepts code
that isn't typically accepted in modeling/processing files, as it allows importing from neighbouring models as well
as inheritance from classes to others.

This modular file defines models, processors, and the configuration class that would otherwise be defined in their
respective modules.

Finally, this feature introduces a new `linter` which will "unravel" the modular file into the "single model, single 
file" directory structure. These files will get auto-generated every time the script is run; reducing the required
contributions to the modular file, and therefore only to the changes between the contributed model and others.

Model users will end up importing and using the single-file interface, so no change is expected here. Doing this, we
hope to combine the best of both worlds: enabling simple contributions while sticking to our philosophy.

This is therefore a replacement for the `# Copied from` markers, and previously contributed models can be expected to
be moved to the new Modular Transformers format in the coming months.

### Details 

The "linter", which unravels the inheritance and creates all single-files from the modular file, will flatten the 
inheritance while trying to be invisible to Python users. At this time, the linter flattens a **single** level of
inheritance.

For example:
- If a configuration class inherits from another and adds/deletes an argument, the generated file will either directly 
  reference it (in case of addition) or completely remove it (in case of deletion).
- If a class inherits from another, for example: class GemmaModel(LlamaModel):, dependencies are automatically 
  inferred. All submodules will be automatically inferred from the superclass.

You should be able to write everything (the tokenizer, the image processor, the model, the config) in this `modular` 
file, and the corresponding files will be created for you. 

### Enforcement

[TODO] We are introducing a new test, that makes sure the generated content matches what is present in the `modular_xxxx.py`

### Examples

Here is a quick example with BERT and RoBERTa. The two models are intimately related: their modeling implementation 
differs solely by a change in the embedding layer.

Instead of redefining the model entirely, here is what the `modular_roberta.py` file looks like for the modeling &
configuration classes (for the sake of the example, the tokenizer is ignored at this time as very different).

```python
from torch import nn
from ..bert.configuration_bert import BertConfig
from ..bert.modeling_bert import (
    BertModel,
    BertEmbeddings,
    BertForMaskedLM
)

# The RoBERTa config is identical to BERT's config
class RobertaConfig(BertConfig):
  model_type = 'roberta'

# We redefine the embeddings here to highlight the padding ID difference, and we redefine the position embeddings
class RobertaEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config())

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

# The RoBERTa model is identical to the BERT model, except for the embedding layer. 
# We redefine the embeddings above, so here there is no need to do additional work
class RobertaModel(BertModel):
  def __init__(self, config):
    super().__init__(config)
    self.embeddings = RobertaEmbeddings(config)

      
# The heads now only need to redefine the model inside to the correct `RobertaModel`
class RobertaForMaskedLM(BertForMaskedLM):
  def __init__(self, config):
    super().__init__(config)
    self.model = RobertaModel(config)
```

Note that if you do not use the dependency that you defined, you will have the following error:

```bash
ValueError: You defined `RobertaEmbeddings` in the modular_roberta.py, it should be used
                                    when you define `BertModel`, as it is one of it's direct dependencies. Make sure
                                    you use it in the `__init__` function.
```

Additionally, you may find a list of examples here:

## What it is not

It is not a replacement for the modeling code (yet?), and if your model is not based on anything else that ever existed, then you can add a `modeling` file as usual.